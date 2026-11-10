"""Gold files must be indexed. OCR is not started from here."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.config import GOLD_EXTRA_PATH, GOLD_PATH, TASK1_GOLD_PATH
from ldi.ingest import FileCoverage, IngestReport, format_coverage


def gold_names_for(root_name: str) -> set[str]:
    """Filenames the named corpus is expected to contain."""
    names: set[str] = set()
    if root_name == "main":
        for row in json.loads(GOLD_PATH.read_text()):
            names.update(row.get("files") or [])
        if GOLD_EXTRA_PATH.exists():
            with GOLD_EXTRA_PATH.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    names.update(json.loads(row["files"]))
        names = {name for name in names if not name.startswith("docs/")}
    elif root_name == "task1":
        with TASK1_GOLD_PATH.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for part in row["source_file"].split("|"):
                    part = part.strip()
                    if part:
                        names.add(part)
    elif root_name in {"stale1", "stale3"}:
        names.add("policy.txt")
    return names


def missing_gold(report: IngestReport, names: set[str]) -> list[str]:
    status = {Path(item.path).name: item.status for item in report.coverage}
    missing = []
    for name in sorted(names):
        found = status.get(name)
        if found != "indexed":
            missing.append(f"{name} {found or 'absent'}")
    return missing


def print_coverage(report: IngestReport) -> None:
    print(format_coverage(report), flush=True)


__all__ = [
    "FileCoverage",
    "format_coverage",
    "gold_names_for",
    "missing_gold",
    "print_coverage",
]
