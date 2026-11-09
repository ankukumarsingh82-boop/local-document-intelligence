"""Load suite rows. Task 1 ids at the sealed boundary are discarded unread."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

from eval.config import (
    CASES_PATH,
    CATEGORY_MAP,
    GOLD_PATH,
    INDEX_ROOT,
    SEALED_FROM,
    STALE_ROOTS,
    TASK1_GOLD_PATH,
    TASK1_TYPE_MAP,
)
from eval.metrics import GoldView, gold_from_tag


@dataclass(frozen=True)
class SuiteCase:
    id: str
    category: str
    question: str
    gold: GoldView
    root_name: str


def synthetic_cases(cases_path=CASES_PATH, gold_path=GOLD_PATH) -> list[SuiteCase]:
    cases = json.loads(cases_path.read_text())
    gold_rows = {row["id"]: row for row in json.loads(gold_path.read_text())}
    loaded: list[SuiteCase] = []
    for case in cases:
        raw = case["category"]
        if raw not in CATEGORY_MAP:
            raise KeyError(f"Unmapped category {raw}")
        row = gold_rows.get(case["id"])
        if row is None:
            gold = GoldView(abstain=False, anchors=(), scored=False)
        else:
            anchors = tuple(anchor for anchor in row.get("anchors") or [] if anchor.strip())
            gold = gold_from_tag(row["tag"], anchors)
        loaded.append(
            SuiteCase(
                id=case["id"],
                category=CATEGORY_MAP[raw],
                question=case["q"],
                gold=gold,
                root_name=STALE_ROOTS.get(case["id"], "main"),
            )
        )
    return loaded


def task1_open_cases(path=TASK1_GOLD_PATH) -> list[SuiteCase]:
    """Read the open prefix. A sealed id is skipped before its fields are kept."""
    loaded: list[SuiteCase] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            number = int(row["id"].split("-")[1])
            if number >= SEALED_FROM:
                continue
            kind = row["type"]
            if kind not in TASK1_TYPE_MAP:
                raise KeyError(f"Unmapped task 1 type {kind}")
            anchors = tuple(
                part.strip() for part in row["evidence_quote"].split("|") if part.strip()
            )
            abstain = TASK1_TYPE_MAP[kind] == "absent" or not anchors
            gold = GoldView(
                abstain=abstain,
                anchors=() if abstain else anchors,
                scored=True,
            )
            loaded.append(
                SuiteCase(
                    id=row["id"],
                    category=TASK1_TYPE_MAP[kind],
                    question=row["question"],
                    gold=gold,
                    root_name="task1",
                )
            )
    return loaded


def index_dir(root_name: str) -> object:
    return INDEX_ROOT / root_name
