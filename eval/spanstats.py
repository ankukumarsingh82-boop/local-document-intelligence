"""Fill the span table from indexed files without rewriting full-text chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.config import INDEX_ROOT
from ldi.config import IMAGE_SUFFIXES, OCR_LOW_CONFIDENCE, index_path
from ldi.spans import build_spans
from ldi.store import Index

ROOTS = ("main", "stale1", "stale3", "task1")
STATS_PATH = Path(__file__).resolve().parent / "span_stats.json"


def _fingerprint(index: Index) -> str:
    rows = index.db.execute(
        "SELECT path, page, body FROM chunks ORDER BY rowid"
    ).fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _fill(root: Path) -> dict[str, int]:
    os.environ["LDI_ROOT"] = str(root)
    with Index(index_path()) as index:
        before = _fingerprint(index)
        files = index.db.execute("SELECT path, sha256 FROM files").fetchall()
        for path, digest in files:
            suffix = Path(path).suffix.lower()
            pages = [
                (int(page) if page else None, body)
                for page, body in index.db.execute(
                    "SELECT page, body FROM chunks WHERE path = ? ORDER BY rowid",
                    (path,),
                )
            ]
            kind = "ocr" if suffix in IMAGE_SUFFIXES else "text"
            spans, passages = build_spans(pages, digest or "", kind, path)
            index.replace_spans(path, spans, passages)
        after = _fingerprint(index)
        if after != before:
            raise SystemExit(f"Full-text rows changed in {root.name}.")
        counts = {
            row[0]: row[1]
            for row in index.db.execute(
                "SELECT source_type, COUNT(*) FROM spans GROUP BY source_type"
            )
        }
        instruction = index.db.execute(
            "SELECT COUNT(*) FROM spans WHERE instruction_like = 1"
        ).fetchone()[0]
        low = index.db.execute(
            "SELECT COUNT(*) FROM spans WHERE confidence IS NOT NULL AND confidence < ?",
            (OCR_LOW_CONFIDENCE,),
        ).fetchone()[0]
    return {
        "text": int(counts.get("text", 0)),
        "table": int(counts.get("table", 0)),
        "ocr": int(counts.get("ocr", 0)),
        "instruction_like": int(instruction),
        "low_confidence_ocr": int(low),
    }


def main() -> None:
    totals = {"text": 0, "table": 0, "ocr": 0, "instruction_like": 0, "low_confidence_ocr": 0}
    by_root = {}
    for name in ROOTS:
        counts = _fill(INDEX_ROOT / name)
        by_root[name] = counts
        for key, value in counts.items():
            totals[key] += value
    report = {"by_index": by_root, "totals": totals}
    STATS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
