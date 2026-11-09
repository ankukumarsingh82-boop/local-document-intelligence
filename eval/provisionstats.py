"""Fill the provision table from stored spans. Full-text chunks stay as they are."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from eval.config import INDEX_ROOT, SEALED_FROM, TASK1_GOLD_PATH
from ldi.config import index_path
from ldi.provisions import extract_provisions, mentioned_pair, mentioned_values
from ldi.store import Index

STATS_PATH = Path(__file__).resolve().parent / "provision_stats.json"


class _Stored:
    def __init__(self, row: tuple) -> None:
        (
            self.span_id,
            self.text,
            page,
            self.block,
            self.line,
            self.table_id,
            self.row_key,
            col,
            self.header_path,
            self.source_type,
        ) = row
        self.page = int(page) if str(page).strip() else 0
        self.col = int(col) if col is not None else 0


def _fingerprint(index: Index) -> str:
    rows = index.db.execute(
        "SELECT path, page, body FROM chunks ORDER BY rowid"
    ).fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _spans(index: Index, path: str) -> list[_Stored]:
    rows = index.db.execute(
        """
        SELECT span_id, text, page, block, line, table_id, row_key, col,
               header_path, source_type
        FROM spans
        WHERE path = ?
        ORDER BY page, block, line, col, span_id
        """,
        (path,),
    ).fetchall()
    return [_Stored(row) for row in rows]


def fill_task1() -> dict:
    os.environ["LDI_ROOT"] = str(INDEX_ROOT / "task1")
    with Index(index_path()) as index:
        before = _fingerprint(index)
        files = [row[0] for row in index.db.execute("SELECT path FROM files ORDER BY path")]
        per_doc = []
        for path in files:
            rows = extract_provisions(path, _spans(index, path))
            index.replace_provisions(path, rows)
            linked = sum(1 for row in rows if row.superseded_by)
            per_doc.append(
                {
                    "doc": Path(path).name,
                    "provisions": len(rows),
                    "with_superseded_by": linked,
                }
            )
        after = _fingerprint(index)
        if after != before:
            raise SystemExit("Full-text rows changed while storing provisions.")
        questions = _questions(index)
    report = {"per_doc": per_doc, "questions": questions}
    STATS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _pair(quote: str) -> tuple[str, str] | None:
    found = mentioned_pair(quote)
    if found is not None:
        return found
    values = mentioned_values(quote)
    if len(values) >= 2:
        return values[0], values[-1]
    return None


def _questions(index: Index) -> list[dict]:
    report = []
    with TASK1_GOLD_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            number = int(row["id"].split("-")[1])
            if number >= SEALED_FROM:
                continue
            if row["type"] != "superseded-version":
                continue
            pair = None
            for part in row["evidence_quote"].split("|"):
                pair = _pair(part.strip()) or pair
            old_present = False
            current_present = False
            linked = False
            if pair is not None:
                old, new = pair
                stored = [
                    item
                    for item in index.db.execute(
                        "SELECT doc_id, value_text, superseded_by, provision_id FROM provisions"
                    )
                    if Path(item[0]).name == row["source_file"]
                ]
                stored = [(item[1], item[2], item[3]) for item in stored]
                old_ids = {item[2] for item in stored if item[0] == old}
                new_ids = {item[2] for item in stored if item[0] == new}
                old_present = bool(old_ids)
                current_present = bool(new_ids)
                linked = any(item[1] in new_ids for item in stored if item[2] in old_ids)
            report.append(
                {
                    "id": row["id"],
                    "source_file": row["source_file"],
                    "old_row": old_present,
                    "current_row": current_present,
                    "linked": linked,
                }
            )
    return report


def main() -> None:
    print(json.dumps(fill_task1(), indent=2))


if __name__ == "__main__":
    main()
