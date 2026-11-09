"""Load suite rows. Task 1 ids at the sealed boundary are discarded unread."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from eval.config import (
    CASES_PATH,
    CATEGORY_MAP,
    GOLD_EXTRA_PATH,
    GOLD_PATH,
    INDEX_ROOT,
    SEALED_FROM,
    STALE_ROOTS,
    TASK1_GOLD_PATH,
    TASK1_TYPE_MAP,
)
from eval.metrics import GoldView, gold_from_tag

EXTRA_MODES = frozenset({"spans", "summary", "stale"})
EXTRA_SOURCE = "added-step0.5"


@dataclass(frozen=True)
class SuiteCase:
    id: str
    category: str
    question: str
    gold: GoldView
    root_name: str


def extra_rows(path=GOLD_EXTRA_PATH) -> list[dict[str, str]]:
    """Rows added after the original gold file. They never replace an existing id."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def extra_gold(path=GOLD_EXTRA_PATH) -> dict[str, GoldView]:
    loaded: dict[str, GoldView] = {}
    for row in extra_rows(path):
        if row["source"] != EXTRA_SOURCE:
            raise ValueError(f"{row['id']} is missing the step 0.5 source mark")
        mode = row["mode"]
        if mode not in EXTRA_MODES:
            raise ValueError(f"Unknown gold mode {mode}")
        anchors = tuple(anchor for anchor in json.loads(row["anchors"]) if str(anchor).strip())
        if mode != "stale" and not anchors:
            raise ValueError(f"{row['id']} needs at least one span")
        loaded[row["id"]] = GoldView(
            abstain=False,
            anchors=anchors,
            scored=True,
            mode=mode,
        )
    return loaded


def synthetic_cases(
    cases_path=CASES_PATH,
    gold_path=GOLD_PATH,
    extra_path=GOLD_EXTRA_PATH,
) -> list[SuiteCase]:
    cases = json.loads(cases_path.read_text())
    gold_rows = {row["id"]: row for row in json.loads(gold_path.read_text())}
    extra = extra_gold(extra_path) if extra_path is not None and Path(extra_path).exists() else {}
    loaded: list[SuiteCase] = []
    for case in cases:
        raw = case["category"]
        if raw not in CATEGORY_MAP:
            raise KeyError(f"Unmapped category {raw}")
        row = gold_rows.get(case["id"])
        if row is not None:
            anchors = tuple(anchor for anchor in row.get("anchors") or [] if anchor.strip())
            gold = gold_from_tag(row["tag"], anchors)
        elif case["id"] in extra:
            gold = extra[case["id"]]
        else:
            gold = GoldView(abstain=False, anchors=(), scored=False)
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
