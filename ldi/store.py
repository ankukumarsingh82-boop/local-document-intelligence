"""SQLite full-text index. Search does not load a model."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    path: str
    page: int | None
    text: str
    superseded: bool = False


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
                text TEXT NOT NULL
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
            """
        )
        self._migrate_files()
        self._migrate_spans()

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
            self.db.execute("DELETE FROM provisions")

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
                "INSERT INTO passages(passage_id, path, span_ids, text) VALUES (?, ?, ?, ?)",
                [
                    (passage.passage_id, path, passage.span_ids, passage.text)
                    for passage in passages
                ],
            )

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
            self.db.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
            self.db.execute("DELETE FROM spans WHERE path = ?", (path,))
            self.db.execute("DELETE FROM passages WHERE path = ?", (path,))
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
                self.db.execute("DELETE FROM chunks WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM files WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM gaps WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM spans WHERE path = ?", (stored,))
                self.db.execute("DELETE FROM passages WHERE path = ?", (stored,))
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
