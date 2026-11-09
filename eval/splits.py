"""Seeded, category-stratified dev and held-out ids for the 72-case set."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from eval.config import HELD_PARTS, MIN_SPLIT_SIZE, SPLIT_PARTS, SPLIT_SEED

SPLIT_PATH = Path(__file__).resolve().parent / "splits.json"


def held_count(size: int) -> int:
    if size < MIN_SPLIT_SIZE:
        return 0
    count = (size * HELD_PARTS + SPLIT_PARTS // 2) // SPLIT_PARTS
    if count <= 0:
        count = 1
    if count >= size:
        count = size - 1
    return count


def assign(pairs: list[tuple[str, str]], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Map each id to dev or heldout. pairs are (id, category)."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id, category in pairs:
        grouped[category].append(case_id)
    rng = random.Random(seed)
    assigned: dict[str, str] = {}
    for category in sorted(grouped):
        ids = sorted(grouped[category])
        rng.shuffle(ids)
        cut = held_count(len(ids))
        for case_id in ids[:cut]:
            assigned[case_id] = "heldout"
        for case_id in ids[cut:]:
            assigned[case_id] = "dev"
    return assigned


def write_splits(pairs: list[tuple[str, str]], path: Path = SPLIT_PATH, seed: int = SPLIT_SEED) -> list[dict[str, str]]:
    assigned = assign(pairs, seed)
    rows = [{"id": case_id, "split": assigned[case_id]} for case_id, _category in sorted(pairs)]
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def read_splits(path: Path = SPLIT_PATH) -> dict[str, str]:
    rows = json.loads(path.read_text())
    return {row["id"]: row["split"] for row in rows}
