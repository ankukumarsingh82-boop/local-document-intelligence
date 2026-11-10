"""SQLite full-text index. Search does not load a model."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ldi.query import acronyms_in, fold_for_fts, names_in
from ldi.spans import SpanRecord


@dataclass(frozen=True)
class Hit:
    path: str
    page: int | None
    text: str
    superseded: bool = False
    also_found_in: tuple[str, ...] = ()
    passage_hash: str = ""
    span_id: str = ""
    span_ids: str = ""
    source_lines: tuple[str, ...] = ()
    source_span_ids: tuple[str, ...] = ()
    pinned_span_ids: tuple[str, ...] = ()
    source_spans: tuple[SpanRecord, ...] = ()
    required_entities: tuple[str, ...] = ()
    comparison_sources: tuple[str, ...] = ()
    required_source: str = ""


class Index:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA busy_timeout = 2000")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gaps (
                path TEXT NOT NULL,
                page TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                path UNINDEXED,
                page UNINDEXED,
                body,
                tokenize = 'porter unicode61'
            );
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                doc_sha8 TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                extractor_version TEXT NOT NULL,
                source_type TEXT NOT NULL,
                page TEXT NOT NULL DEFAULT '',
                block INTEGER,
                line INTEGER,
                table_id TEXT NOT NULL DEFAULT '',
                row_key TEXT NOT NULL DEFAULT '',
                col INTEGER,
                header_path TEXT NOT NULL DEFAULT '',
                unit TEXT NOT NULL DEFAULT '',
                occurrence INTEGER NOT NULL,
                confidence REAL,
                bbox TEXT NOT NULL DEFAULT '',
                instruction_like INTEGER NOT NULL DEFAULT 0,
                cell_type TEXT NOT NULL DEFAULT '',
                period TEXT NOT NULL DEFAULT '',
                row_role TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS passages (
                passage_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                span_ids TEXT NOT NULL,
                text TEXT NOT NULL,
                doc_sha TEXT NOT NULL DEFAULT '',
                page_start TEXT NOT NULL DEFAULT '',
                page_end TEXT NOT NULL DEFAULT '',
                sheet TEXT NOT NULL DEFAULT '',
                row_start TEXT NOT NULL DEFAULT '',
                row_end TEXT NOT NULL DEFAULT '',
                heading_path TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                passage_hash TEXT NOT NULL DEFAULT '',
                chunker_version TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL DEFAULT '',
                aliases TEXT NOT NULL DEFAULT '',
                also_in TEXT NOT NULL DEFAULT '',
                duplicate_of TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS chunk_log (
                path TEXT NOT NULL,
                passage_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS provisions (
                provision_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                clause_key TEXT NOT NULL,
                value_text TEXT NOT NULL,
                value_span TEXT NOT NULL,
                effective_from TEXT NOT NULL DEFAULT '',
                effective_to TEXT NOT NULL DEFAULT '',
                superseded_by TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                source_span TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
                passage_id UNINDEXED,
                heading,
                body,
                tokenize = 'porter unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS passages_exact USING fts5(
                passage_id UNINDEXED,
                body,
                tokenize = "unicode61 tokenchars '-_./'"
            );
            CREATE TABLE IF NOT EXISTS passage_keys (
                passage_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                passage_hash TEXT NOT NULL DEFAULT '',
                span_ids TEXT NOT NULL DEFAULT '',
                page TEXT NOT NULL DEFAULT '',
                heading_path TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                fallback INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS term_aliases (
                path TEXT NOT NULL,
                alias TEXT NOT NULL,
                expansion TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS indexed_names (
                path TEXT NOT NULL,
                name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                passage_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                chunker_version TEXT NOT NULL,
                vector BLOB NOT NULL DEFAULT X'',
                PRIMARY KEY (passage_hash, model_id, chunker_version)
            );
            """
        )
        self._migrate_files()
        self._migrate_spans()
        self._migrate_passages()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM files")
            self.db.execute("DELETE FROM gaps")
            self.db.execute("DELETE FROM spans")
            self.db.execute("DELETE FROM passages")
            self.db.execute("DELETE FROM chunk_log")
            self.db.execute("DELETE FROM provisions")
            self.db.execute("DELETE FROM passages_fts")
            self.db.execute("DELETE FROM passages_exact")
            self.db.execute("DELETE FROM passage_keys")
            self.db.execute("DELETE FROM term_aliases")
            self.db.execute("DELETE FROM indexed_names")
            self.db.execute("DELETE FROM embeddings")

    def stats(self) -> tuple[int, int]:
        files = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(files), int(chunks)

    def is_unchanged(self, path: str, mtime_ns: int, size: int, sha256: str) -> bool:
        row = self.db.execute(
            "SELECT mtime_ns, bytes, sha256 FROM files WHERE path = ?",
            (path,),
        ).fetchone()
        return (
            row is not None
            and int(row[0]) == mtime_ns
            and int(row[1]) == size
            and row[2] == sha256
        )

    def sha256_for(self, path: str) -> str | None:
        row = self.db.execute(
            "SELECT sha256 FROM files WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])

    def replace_file(
        self,
        path: str,
        mtime_ns: int,
        size: int,
        chunks: list[tuple[int | None, str]],
        sha256: str,
    ) -> None:
        rows = [
            (path, "" if page is None else str(page), body)
            for page, body in chunks
            if body.strip()
        ]
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self.db.execute(
                """
                INSERT INTO files(path, mtime_ns, bytes, sha256) VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    bytes = excluded.bytes,
                    sha256 = excluded.sha256
                """,
                (path, mtime_ns, size, sha256),
            )
            self.db.executemany(
                "INSERT INTO chunks(path, page, body) VALUES (?, ?, ?)",
                rows,
            )

    def replace_spans(self, path: str, spans: list, passages: list) -> None:
        """Write the span table for one file. The full-text chunks stay as they are."""
        ids = [span.span_id for span in spans]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Span ids collide inside {path}.")
        with self.db:
            for span_id in ids:
                found = self.db.execute(
                    "SELECT path FROM spans WHERE span_id = ? AND path != ?",
                    (span_id, path),
                ).fetchone()
                if found is not None:
                    raise RuntimeError(f"Span id {span_id} already belongs to another document.")
            self.db.execute("DELETE FROM spans WHERE path = ?", (path,))
            self.db.execute("DELETE FROM passages WHERE path = ?", (path,))
            self.db.execute("DELETE FROM chunk_log WHERE path = ?", (path,))
            self.db.executemany(
                """
                INSERT INTO spans(
                    span_id, path, doc_sha8, content_hash, extractor_version,
                    source_type, page, block, line, table_id, row_key, col,
                    header_path, unit, occurrence, confidence, bbox,
                    instruction_like, cell_type, period, row_role, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        span.span_id,
                        path,
                        span.doc_sha8,
                        span.content_hash,
                        span.extractor_version,
                        span.source_type,
                        "" if span.page is None else str(span.page),
                        span.block,
                        span.line,
                        span.table_id,
                        span.row_key,
                        span.col,
                        span.header_path,
                        span.unit,
                        span.occurrence,
                        span.confidence,
                        span.bbox,
                        1 if span.instruction_like else 0,
                        span.cell_type,
                        span.period,
                        span.row_role,
                        span.text,
                    )
                    for span in spans
                ],
            )
            self.db.executemany(
                """
                INSERT INTO passages(
                    passage_id, path, span_ids, text, doc_sha, page_start, page_end,
                    sheet, row_start, row_end, heading_path, source_type,
                    passage_hash, chunker_version, label, aliases, also_in, duplicate_of
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        passage.passage_id,
                        path,
                        passage.span_ids,
                        passage.text,
                        passage.doc_sha,
                        "" if passage.page_start is None else str(passage.page_start),
                        "" if passage.page_end is None else str(passage.page_end),
                        passage.sheet,
                        "" if passage.row_start is None else str(passage.row_start),
                        "" if passage.row_end is None else str(passage.row_end),
                        passage.heading_path,
                        passage.source_type,
                        passage.passage_hash,
                        passage.chunker_version,
                        passage.label,
                        passage.aliases,
                        "",
                        "",
                    )
                    for passage in passages
                ],
            )
            self.db.executemany(
                """
                INSERT INTO chunk_log(path, passage_id, kind, detail)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (path, passage.passage_id, log.kind, log.detail)
                    for passage in passages
                    for log in passage.logs
                ],
            )
            self._index_passages(path, passages)

    def replace_provisions(self, doc_id: str, rows: list) -> None:
        """Replace one document's provision rows. Full-text chunks stay as they are."""
        with self.db:
            self.db.execute("DELETE FROM provisions WHERE doc_id = ?", (doc_id,))
            self.db.executemany(
                """
                INSERT INTO provisions(
                    provision_id, doc_id, clause_key, value_text, value_span,
                    effective_from, effective_to, superseded_by, status, source_span
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.provision_id,
                        doc_id,
                        row.clause_key,
                        row.value_text,
                        row.value_span,
                        row.effective_from,
                        row.effective_to,
                        row.superseded_by,
                        row.status,
                        row.source_span,
                    )
                    for row in rows
                ],
            )

    def forget_text(self, path: str) -> None:
        with self.db:
            self._delete_passage_index(path)
            self.db.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
            self.db.execute("DELETE FROM spans WHERE path = ?", (path,))
            self.db.execute("DELETE FROM passages WHERE path = ?", (path,))
            self.db.execute("DELETE FROM chunk_log WHERE path = ?", (path,))
            self.db.execute("DELETE FROM provisions WHERE doc_id = ?", (path,))

    def replace_gaps(self, path: str, gaps: list[tuple[int | None, str]]) -> None:
        rows = [
            (path, "" if page is None else str(page), reason) for page, reason in gaps
        ]
        with self.db:
            self.db.execute("DELETE FROM gaps WHERE path = ?", (path,))
            self.db.executemany(
                "INSERT INTO gaps(path, page, reason) VALUES (?, ?, ?)",
                rows,
            )

    def list_gaps(self) -> list[tuple[str, int | None, str]]:
        rows = self.db.execute(
            "SELECT path, page, reason FROM gaps ORDER BY path, page, reason"
        ).fetchall()
        return [
            (path, int(page) if page else None, reason) for path, page, reason in rows
        ]

    def purge_absent(self, seen: set[str], root: Path, project: Path) -> None:
        paths = {row[0] for row in self.db.execute("SELECT path FROM files")}
        paths.update(row[0] for row in self.db.execute("SELECT path FROM gaps"))
        root_resolved = root.resolve()
        with self.db:
            for stored in sorted(paths):
                if stored in seen or not _is_under(stored, root_resolved, project):
                    continue
                self._delete_passage_index(stored)
                self.db.execute("DELETE FROM chunks WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM files WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM gaps WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM spans WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM passages WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM chunk_log WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM provisions WHERE doc_id = ?", (stored,))

    def refresh_labels(self) -> None:
        files = self.db.execute(
            "SELECT path, sha256 FROM files ORDER BY path"
        ).fetchall()
        keeper: dict[str, str] = {}
        duplicate_of: dict[str, str | None] = {}
        for path, digest in files:
            if not digest:
                duplicate_of[path] = None
                continue
            current = keeper.get(digest)
            if current is None:
                keeper[digest] = path
                duplicate_of[path] = None
                continue
            if _looks_like_copy(current) and not _looks_like_copy(path):
                duplicate_of[current] = path
                duplicate_of[path] = None
                keeper[digest] = path
                continue
            duplicate_of[path] = current
        bodies: dict[str, list[str]] = {}
        for path, body in self.db.execute("SELECT path, body FROM chunks"):
            bodies.setdefault(path, []).append(body)
        names = {path: Path(path).name for path, _digest in files}
        superseded: set[str] = set()
        for path, parts in bodies.items():
            text = "\n".join(parts)
            if _SUPERSEDED_WORD.search(text):
                superseded.add(path)
            for match in _SUPERSEDES.finditer(text):
                target = match.group(1).strip(".,;\"'")
                for other, name in names.items():
                    if name == target:
                        superseded.add(other)
        with self.db:
            for path, _digest in files:
                self.db.execute(
                    """
                    UPDATE files
                    SET superseded = ?, duplicate_of = ?
                    WHERE path = ?
                    """,
                    (
                        1 if path in superseded else 0,
                        duplicate_of.get(path),
                        path,
                    ),
                )

    def dedupe_passages(self) -> None:
        """Collapse the same passage text across files. Copies inside one file stay."""
        rows = self.db.execute(
            "SELECT passage_id, path, text FROM passages"
        ).fetchall()
        groups: dict[str, list[tuple[str, str]]] = {}
        for passage_id, path, text in rows:
            if not str(text).strip():
                continue
            key = hashlib.sha256(
                unicodedata.normalize("NFC", str(text)).encode()
            ).hexdigest()
            groups.setdefault(key, []).append((str(passage_id), str(path)))
        with self.db:
            self.db.execute("UPDATE passages SET also_in = '', duplicate_of = ''")
            for items in groups.values():
                paths = {path for _passage_id, path in items}
                if len(paths) < 2:
                    continue
                survivor = sorted(paths, key=lambda path: (_looks_like_copy(path), path))[0]
                others = sorted(path for path in paths if path != survivor)
                also = "\n".join(others)
                survivor_ids = [
                    passage_id for passage_id, path in items if path == survivor
                ]
                for passage_id, path in items:
                    if path == survivor:
                        self.db.execute(
                            "UPDATE passages SET also_in = ? WHERE passage_id = ?",
                            (also, passage_id),
                        )
                        continue
                    self.db.execute(
                        "UPDATE passages SET duplicate_of = ? WHERE passage_id = ?",
                        (survivor_ids[0], passage_id),
                    )
                    self.db.execute(
                        "DELETE FROM passages_fts WHERE passage_id = ?",
                        (passage_id,),
                    )
                    self.db.execute(
                        "DELETE FROM passages_exact WHERE passage_id = ?",
                        (passage_id,),
                    )
                    self.db.execute(
                        "DELETE FROM passage_keys WHERE passage_id = ?",
                        (passage_id,),
                    )

    def search(self, match: str, limit: int) -> list[Hit]:
        window = max(limit, limit * 8)
        try:
            rows = self.db.execute(
                """
                SELECT path, page, body
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, window),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"The search could not run ({exc}).") from exc
        flags = self._file_flags()
        ordered: list[tuple[int, int, Hit]] = []
        for index, (path, page, body) in enumerate(rows):
            superseded, duplicate = flags.get(path, (False, None))
            if duplicate:
                continue
            ordered.append(
                (
                    1 if superseded else 0,
                    index,
                    Hit(
                        path=path,
                        page=int(page) if page else None,
                        text=body,
                        superseded=superseded,
                    ),
                )
            )
        ordered.sort()
        return [hit for _flag, _index, hit in ordered[:limit]]

    def search_question(self, question: str, limit: int) -> list[Hit]:
        terms = content_terms(question)
        if not terms:
            raise ValueError("That question has no words to search for.")
        widened = _with_aliases(terms)
        if len(widened) == 1:
            return _prefer_named_files(self.search(_join_terms(widened, "AND"), limit), terms)
        primary = self.search(_join_terms(terms, "AND"), limit)
        if len(primary) >= limit:
            return _prefer_named_files(primary, terms)
        extra = self.search(_join_terms(widened, "OR"), limit + len(primary))
        merged = list(primary)
        seen = {(hit.path, hit.page, hit.text) for hit in merged}
        for hit in extra:
            key = (hit.path, hit.page, hit.text)
            if key in seen:
                continue
            merged.append(hit)
            seen.add(key)
            if len(merged) >= limit:
                break
        return _prefer_named_files(merged, terms)

    def file_flags(self) -> dict[str, tuple[bool, str | None]]:
        return self._file_flags()

    def passage_count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM passage_keys").fetchone()
        return int(row[0]) if row else 0

    def document_frequency(self, token: str) -> int:
        from ldi.query import quote_token

        if not token:
            return 0
        try:
            row = self.db.execute(
                "SELECT count(*) FROM passages_fts WHERE passages_fts MATCH ?",
                (quote_token(token),),
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0

    def frequencies(self, tokens: tuple[str, ...]) -> dict[str, int]:
        return {token: self.document_frequency(token) for token in tokens}

    def alias_rows(self) -> tuple[tuple[str, str], ...]:
        rows = self.db.execute(
            "SELECT alias, expansion FROM term_aliases ORDER BY alias, expansion"
        ).fetchall()
        return tuple((str(alias), str(expansion)) for alias, expansion in rows)

    def indexed_names(self) -> tuple[str, ...]:
        rows = self.db.execute(
            "SELECT DISTINCT name FROM indexed_names ORDER BY name"
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def provisions_for(self, path: str) -> list:
        from ldi.provisions import Provision

        rows = self.db.execute(
            """
            SELECT provision_id, doc_id, clause_key, value_text, value_span,
                   effective_from, effective_to, superseded_by, status, source_span
            FROM provisions
            WHERE doc_id = ?
            ORDER BY provision_id
            """,
            (path,),
        ).fetchall()
        return [Provision(*row) for row in rows]

    def passage_row(self, passage_id: str) -> tuple | None:
        stored = self.db.execute(
            """
            SELECT path, span_ids, text, page_start, heading_path, passage_hash
            FROM passages
            WHERE passage_id = ?
            """,
            (passage_id,),
        ).fetchone()
        if stored is not None:
            return stored
        fallback = self.db.execute(
            """
            SELECT path, span_ids, body, page, heading_path, passage_hash
            FROM passage_keys
            WHERE passage_id = ? AND fallback = 1
            """,
            (passage_id,),
        ).fetchone()
        return fallback

    def lexical_ids(self, match: str, limit: int, heading_weight: float, body_weight: float) -> list[tuple[str, float]]:
        if not match.strip():
            return []
        try:
            rows = self.db.execute(
                """
                SELECT passage_id, bm25(passages_fts, ?, ?) AS score
                FROM passages_fts
                WHERE passages_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (heading_weight, body_weight, match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(passage_id), float(score)) for passage_id, score in rows]

    def lexical_ids_in(
        self, match: str, path: str, limit: int, heading_weight: float, body_weight: float
    ) -> list[tuple[str, float]]:
        """The same bm25 query, restricted to the passages of one file."""
        if not match.strip():
            return []
        try:
            rows = self.db.execute(
                """
                SELECT passages_fts.passage_id, bm25(passages_fts, ?, ?) AS score
                FROM passages_fts
                JOIN passage_keys ON passage_keys.passage_id = passages_fts.passage_id
                WHERE passages_fts MATCH ? AND passage_keys.path = ?
                ORDER BY score
                LIMIT ?
                """,
                (heading_weight, body_weight, match, path, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(passage_id), float(score)) for passage_id, score in rows]

    def exact_ids(self, match: str, limit: int) -> list[str]:
        if not match.strip():
            return []
        try:
            rows = self.db.execute(
                """
                SELECT passage_id
                FROM passages_exact
                WHERE passages_exact MATCH ?
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def _index_passages(self, path: str, passages: list) -> None:
        self._delete_passage_index(path)
        rows = list(passages)
        for passage in rows:
            self._add_passage_index(path, passage, False)
        uncovered = self._uncovered_chunks(path, rows)
        for passage in uncovered:
            self._add_passage_index(path, passage, True)
        text = "\n".join(passage.text for passage in [*rows, *uncovered])
        self.db.executemany(
            "INSERT INTO term_aliases(path, alias, expansion) VALUES (?, ?, ?)",
            [(path, alias, expansion) for alias, expansion in acronyms_in(text)],
        )
        self.db.executemany(
            "INSERT INTO indexed_names(path, name) VALUES (?, ?)",
            [(path, name) for name in names_in(text)],
        )

    def spans_for(self, path: str):
        """Source lines for one document, in read order."""
        from ldi.spans import SpanRecord

        rows = self.db.execute(
            """
            SELECT span_id, doc_sha8, content_hash, extractor_version, source_type,
                   page, block, line, table_id, row_key, col, header_path, unit,
                   occurrence, confidence, bbox, instruction_like, cell_type,
                   period, row_role, text
            FROM spans WHERE path = ?
            ORDER BY block, line, col, span_id
            """,
            (path,),
        ).fetchall()
        found = []
        for row in rows:
            page = row[5]
            found.append(
                SpanRecord(
                    span_id=row[0],
                    doc_sha8=row[1],
                    content_hash=row[2],
                    extractor_version=row[3],
                    source_type=row[4],
                    page=int(page) if page else None,
                    block=int(row[6] or 0),
                    line=int(row[7] or 0),
                    table_id=row[8] or "",
                    row_key=row[9] or "",
                    col=None if row[10] is None else int(row[10]),
                    header_path=row[11] or "",
                    unit=row[12] or "",
                    occurrence=int(row[13] or 0),
                    confidence=row[14],
                    bbox=row[15] or "",
                    instruction_like=bool(row[16]),
                    cell_type=row[17] or "",
                    period=row[18] or "",
                    row_role=row[19] or "",
                    text=row[20],
                )
            )
        return found

    def _uncovered_chunks(self, path: str, passages: list) -> list:
        from ldi.spans import PassageRecord

        covered = "\n".join(passage.text for passage in passages)
        found = []
        chunks = self.db.execute(
            "SELECT page, body FROM chunks WHERE path = ? ORDER BY rowid",
            (path,),
        ).fetchall()
        for page, body in chunks:
            lines = [line.strip() for line in str(body).splitlines() if line.strip()]
            if lines and all(line in covered for line in lines):
                continue
            if not str(body).strip():
                continue
            digest = hashlib.sha256(f"{path}\n{page}\n{body}".encode()).hexdigest()
            found.append(
                PassageRecord(
                    passage_id=digest,
                    span_ids="",
                    text=str(body),
                    page_start=int(page) if page else None,
                    passage_hash=hashlib.sha256(str(body).encode()).hexdigest(),
                )
            )
        return found

    def _add_passage_index(self, path: str, passage, fallback: bool) -> None:
        heading = fold_for_fts(passage.heading_path)
        body = _indexed_body(passage)
        stored_fallback = 1 if fallback else 0
        page = "" if passage.page_start is None else str(passage.page_start)
        self.db.execute(
            """
            INSERT INTO passage_keys(
                passage_id, path, passage_hash, span_ids, page, heading_path, body, fallback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                passage.passage_id,
                path,
                passage.passage_hash,
                passage.span_ids,
                page,
                passage.heading_path,
                passage.text if stored_fallback else "",
                stored_fallback,
            ),
        )
        self.db.execute(
            "INSERT INTO passages_fts(passage_id, heading, body) VALUES (?, ?, ?)",
            (passage.passage_id, heading, body),
        )
        self.db.execute(
            "INSERT INTO passages_exact(passage_id, body) VALUES (?, ?)",
            (passage.passage_id, _indexed_body(passage, heading=passage.heading_path)),
        )

    def _delete_passage_index(self, path: str) -> None:
        rows = self.db.execute(
            "SELECT passage_id, passage_hash FROM passage_keys WHERE path = ?",
            (path,),
        ).fetchall()
        for passage_id, passage_hash in rows:
            self.db.execute("DELETE FROM passages_fts WHERE passage_id = ?", (passage_id,))
            self.db.execute(
                "DELETE FROM passages_exact WHERE passage_id = ?",
                (passage_id,),
            )
            if passage_hash:
                self.db.execute(
                    "DELETE FROM embeddings WHERE passage_hash = ?",
                    (passage_hash,),
                )
        self.db.execute("DELETE FROM passage_keys WHERE path = ?", (path,))
        self.db.execute("DELETE FROM term_aliases WHERE path = ?", (path,))
        self.db.execute("DELETE FROM indexed_names WHERE path = ?", (path,))

    def _migrate_files(self) -> None:
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(files)")}
        if "sha256" not in columns:
            self.db.execute("ALTER TABLE files ADD COLUMN sha256 TEXT")
        if "superseded" not in columns:
            self.db.execute(
                "ALTER TABLE files ADD COLUMN superseded INTEGER NOT NULL DEFAULT 0"
            )
        if "duplicate_of" not in columns:
            self.db.execute("ALTER TABLE files ADD COLUMN duplicate_of TEXT")

    def _migrate_passages(self) -> None:
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(passages)")}
        additions = (
            "doc_sha",
            "page_start",
            "page_end",
            "sheet",
            "row_start",
            "row_end",
            "heading_path",
            "source_type",
            "passage_hash",
            "chunker_version",
            "label",
            "aliases",
            "also_in",
            "duplicate_of",
        )
        for name in additions:
            if name not in columns:
                self.db.execute(
                    f"ALTER TABLE passages ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_log (
                path TEXT NOT NULL,
                passage_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT NOT NULL
            )
            """
        )

    def _migrate_spans(self) -> None:
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(spans)")}
        if "period" not in columns:
            self.db.execute("ALTER TABLE spans ADD COLUMN period TEXT NOT NULL DEFAULT ''")
        if "row_role" not in columns:
            self.db.execute(
                "ALTER TABLE spans ADD COLUMN row_role TEXT NOT NULL DEFAULT ''"
            )

    def _file_flags(self) -> dict[str, tuple[bool, str | None]]:
        rows = self.db.execute(
            "SELECT path, superseded, duplicate_of FROM files"
        ).fetchall()
        return {path: (bool(flag), duplicate) for path, flag, duplicate in rows}


_STOPWORDS = frozenset(
    """
    a an the is are was were be been being
    of to for in on at by with from as and or
    what which who whom whose when where why how
    this that these those it its do does did
    can could has have had will would should
    if then than into about your you we they their
    there here but so any all each per via
    """.split()
)
_MAX_TERMS = 20
_SUPERSEDED_WORD = re.compile(r"\bSUPERSEDED\b", re.IGNORECASE)
_SUPERSEDES = re.compile(r"\bsupersedes\s+(\S+)", re.IGNORECASE)
_COPY_NAME = re.compile(r"copy|duplicate", re.IGNORECASE)


def _looks_like_copy(path: str) -> bool:
    return _COPY_NAME.search(Path(path).name) is not None


def _name_overlap(path: str, terms: list[str]) -> int:
    stem = Path(path).stem.lower().replace("_", " ").replace("-", " ")
    return sum(1 for term in terms if term.lower() in stem)


def _prefer_named_files(hits: list[Hit], terms: list[str]) -> list[Hit]:
    """Keep a superseded file behind a current one, and prefer a matching filename."""
    ranked = sorted(
        enumerate(hits),
        key=lambda item: (
            item[1].superseded,
            -_name_overlap(item[1].path, terms),
            item[0],
        ),
    )
    return [hit for _index, hit in ranked]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stored_file(stored: str, project: Path) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    return project / path


def _is_under(stored: str, root: Path, project: Path) -> bool:
    try:
        stored_file(stored, project).resolve().relative_to(root)
    except ValueError:
        return False
    return True


# "each year" and "pay" retrieve an annual fee. "paid" does not.
_SEARCH_ALIAS = {"year": "annual", "yearly": "annual", "pay": "fee", "payable": "fee"}


def _indexed_body(passage, heading: str = "") -> str:
    """Fold the passage, and its lookup aliases when source evidence is on.

    The stored passage text stays the original line. Aliases are not quoted.
    """
    from ldi import config

    text = passage.text
    if heading:
        text = f"{heading}\n{text}"
    if config.SOURCE_EVIDENCE and getattr(passage, "aliases", ""):
        text = f"{text}\n{passage.aliases}"
    return fold_for_fts(text)


def _with_aliases(terms: list[str]) -> list[str]:
    seen = {term.casefold() for term in terms}
    widened = list(terms)
    for term in terms:
        alias = _SEARCH_ALIAS.get(term.casefold())
        if alias and alias not in seen:
            widened.append(alias)
            seen.add(alias)
    return widened


def content_terms(question: str) -> list[str]:
    words = re.findall(r"\w+", question, flags=re.UNICODE)
    terms: list[str] = []
    for word in words:
        if len(word) <= 1 or word.lower() in _STOPWORDS:
            continue
        terms.append(word)
        if len(terms) >= _MAX_TERMS:
            break
    return terms


def match_query(question: str) -> str:
    terms = content_terms(question)
    if not terms:
        raise ValueError("That question has no words to search for.")
    return _join_terms(terms, "AND")


def match_query_any(question: str) -> str:
    terms = content_terms(question)
    if not terms:
        raise ValueError("That question has no words to search for.")
    return _join_terms(terms, "OR")


def _join_terms(terms: list[str], operator: str) -> str:
    return f" {operator} ".join('"' + term.replace('"', "") + '"' for term in terms)
