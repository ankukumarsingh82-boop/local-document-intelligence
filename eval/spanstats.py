"""Fill the span table from indexed files without rewriting full-text chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.config import INDEX_ROOT
from ldi.config import IMAGE_SUFFIXES, OCR_LOW_CONFIDENCE, PDF_SUFFIXES, index_path, project_root
from ldi.pdftable import extract_tables, text_layer_empty
from ldi.spans import build_spans, ocr_low_confidence
from ldi.store import Index, stored_file

ROOTS = ("main", "stale1", "stale3", "task1")
STATS_PATH = Path(__file__).resolve().parent / "span_stats.json"


def _fingerprint(index: Index) -> str:
    rows = index.db.execute(
        "SELECT path, page, body FROM chunks ORDER BY rowid"
    ).fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _fill(root: Path) -> dict:
    os.environ["LDI_ROOT"] = str(root)
    empty_visible: list[dict] = []
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
            tables = []
            if suffix in PDF_SUFFIXES:
                file = stored_file(path, project_root())
                if file.is_file():
                    found = extract_tables(file)
                    blank = [table.page for table in found if table.empty]
                    if blank:
                        empty_visible.append({"doc": Path(path).name, "pages": blank})
                    tables = [table for table in found if not table.empty]
                    if text_layer_empty(file):
                        kind = "ocr"
            spans, passages = build_spans(pages, digest or "", kind, path, tables=tables)
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
        low = 0
        for source, confidence in index.db.execute(
            "SELECT source_type, confidence FROM spans"
        ):
            if ocr_low_confidence(source, confidence):
                low += 1
        per_doc = []
        if root.name == "task1":
            pdf_counts = {}
            for path, digest in files:
                if Path(path).suffix.lower() not in PDF_SUFFIXES:
                    continue
                file = stored_file(path, project_root())
                if not file.is_file():
                    continue
                found = extract_tables(file)
                pdf_counts[path] = sum(len(table.cells) for table in found if not table.empty)
            for path, table_count in index.db.execute(
                """
                SELECT files.path, COUNT(spans.span_id)
                FROM files
                LEFT JOIN spans ON spans.path = files.path AND spans.source_type = 'table'
                GROUP BY files.path
                ORDER BY files.path
                """
            ):
                per_doc.append(
                    {
                        "doc": Path(path).name,
                        "table_spans": int(table_count),
                        "pdf_cells": int(pdf_counts.get(path, 0)),
                    }
                )
    return {
        "text": int(counts.get("text", 0)),
        "table": int(counts.get("table", 0)),
        "ocr": int(counts.get("ocr", 0)),
        "instruction_like": int(instruction),
        "low_confidence_ocr": int(low),
        "task1_tables": per_doc,
        "visible_tables_with_no_cells": empty_visible,
    }


def main() -> None:
    totals = {
        "text": 0,
        "table": 0,
        "ocr": 0,
        "instruction_like": 0,
        "low_confidence_ocr": 0,
    }
    by_root = {}
    task1_tables: list[dict] = []
    empty_visible: list[dict] = []
    for name in ROOTS:
        counts = _fill(INDEX_ROOT / name)
        task1_tables.extend(counts.pop("task1_tables"))
        empty_visible.extend(counts.pop("visible_tables_with_no_cells"))
        by_root[name] = counts
        for key, value in counts.items():
            totals[key] += value
    report = {
        "by_index": by_root,
        "totals": totals,
        "task1_tables": task1_tables,
        "visible_tables_with_no_cells": empty_visible,
    }
    # The threshold is recorded so a null OCR score is counted with the rows above.
    report["ocr_low_confidence_below"] = OCR_LOW_CONFIDENCE
    STATS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"totals": totals, "empty_visible": empty_visible, "task1_tables": task1_tables}))


if __name__ == "__main__":
    main()
