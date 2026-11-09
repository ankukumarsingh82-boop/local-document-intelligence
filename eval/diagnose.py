"""Failure stage from ranks recorded during the run. No document text."""

from __future__ import annotations

import math
from collections import Counter

from eval.config import (
    CLAIMED_NON_OCR_CORRECT,
    CLAIMED_NON_OCR_TOTAL,
    CLAIMED_TASK1_CORRECT,
    CLAIMED_TASK1_TOTAL,
    MEDIAN_PERCENTILE,
    PRODUCT_HITS,
    TAIL_PERCENTILE,
)
from eval.metrics import METRICS, empty_counts


def stage(row: dict) -> str | None:
    metric = row["metric"]
    if row.get("timeout"):
        return None
    if metric in {"correct", "unscored"}:
        return None
    if row.get("cited_has"):
        return "scoring"
    if metric == "false_accept":
        return "generated"
    if not row.get("index_has"):
        return "extraction"
    if row.get("evidence_has"):
        return "generated"
    rank = row.get("rank")
    if rank is None or rank > PRODUCT_HITS:
        return "retrieval"
    return "gated"


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(percent / 100 * len(ordered)) - 1
    index = min(len(ordered) - 1, max(0, index))
    return round(ordered[index], 3)


def latency_block(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    block: dict[str, dict[str, float | None]] = {}
    for name in ("retrieve_s", "model_s", "render_s"):
        values = [float(row[name]) for row in rows]
        block[name] = {
            "median": percentile(values, MEDIAN_PERCENTILE),
            "p95": percentile(values, TAIL_PERCENTILE),
        }
    return block


def count_rows(rows: list[dict]) -> dict[str, int]:
    counts = empty_counts()
    for row in rows:
        if row.get("invalid"):
            counts["invalid_model_output"] += 1
        if row.get("fabrication"):
            counts["fabrication"] += 1
        if row.get("timeout"):
            counts["timeout"] += 1
            continue
        metric = row["metric"]
        if metric in METRICS:
            counts[metric] += 1
    return dict(counts)


def by_category(rows: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    return {category: count_rows(items) for category, items in sorted(grouped.items())}


def failing(rows: list[dict]) -> list[dict]:
    found = []
    for row in rows:
        if row.get("timeout"):
            continue
        if row["metric"] in {"correct", "unscored"}:
            continue
        found.append(
            {
                "id": row["id"],
                "category": row["category"],
                "stage": stage(row),
                "metric": row["metric"],
            }
        )
    return found


def false_not_found_target(rows: list[dict]) -> int:
    """Recoverable misses are the target gap. Extraction misses remain."""
    return sum(
        1
        for row in rows
        if row["metric"] == "false_not_found" and stage(row) == "extraction"
    )


def claimed_gap(non_ocr_correct: int, task1_correct: int) -> dict[str, int | bool]:
    return {
        "non_ocr_correct": non_ocr_correct,
        "non_ocr_total": CLAIMED_NON_OCR_TOTAL,
        "non_ocr_claimed": CLAIMED_NON_OCR_CORRECT,
        "non_ocr_reproduced": non_ocr_correct == CLAIMED_NON_OCR_CORRECT,
        "task1_correct": task1_correct,
        "task1_total": CLAIMED_TASK1_TOTAL,
        "task1_claimed": CLAIMED_TASK1_CORRECT,
        "task1_reproduced": task1_correct == CLAIMED_TASK1_CORRECT,
    }


def trap_counts(rows: list[dict]) -> dict[str, int]:
    codes: Counter[str] = Counter(row["judge"] for row in rows)
    result = dict(codes)
    result["cases"] = len(rows)
    result["acceptable"] = sum(1 for row in rows if row.get("trap_pass"))
    return result
