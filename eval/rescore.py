"""Score the saved baseline answers. This module does not load a model."""

from __future__ import annotations

import json
from pathlib import Path

from eval.config import RESULTS_DIR
from eval.goldload import synthetic_cases
from eval.metrics import is_fabricated, metric_for

FLOOR_PATH = Path(__file__).resolve().parent / "floor_ids.json"
RAW_PATH = RESULTS_DIR / "raw.jsonl"


def pass1_shown(path: Path = RAW_PATH) -> dict[str, str]:
    shown: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("pass") != 1:
                continue
            shown[row["id"]] = row["shown"]
    return shown


def score_saved(shown: dict[str, str]) -> list[dict]:
    """Non-OCR cases only. A missing saved answer is an error, not an unscored row."""
    rows = []
    for case in synthetic_cases():
        if case.category == "OCR":
            continue
        if case.id not in shown:
            raise KeyError(f"Saved answers have no pass-1 row for {case.id}")
        text = shown[case.id]
        rows.append(
            {
                "id": case.id,
                "metric": metric_for(case.gold, text),
                "fabrication": is_fabricated(text),
            }
        )
    return rows


def passing_ids(rows: list[dict]) -> list[str]:
    return sorted(row["id"] for row in rows if row["metric"] == "correct")


def write_floor(rows: list[dict], path: Path = FLOOR_PATH) -> list[str]:
    fabricated = [row["id"] for row in rows if row["fabrication"]]
    if fabricated:
        raise SystemExit(f"Fabrication is {len(fabricated)}: {', '.join(fabricated)}")
    ids = passing_ids(rows)
    path.write_text(json.dumps(ids, indent=2) + "\n")
    return ids


def main() -> None:
    rows = score_saved(pass1_shown())
    ids = write_floor(rows)
    print(f"{len(ids)}/{len(rows)}")
    for case_id in ids:
        print(case_id)


if __name__ == "__main__":
    main()
