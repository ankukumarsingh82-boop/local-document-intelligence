"""Thresholds, seeds, and category names for the baseline. No document terms."""

from __future__ import annotations

from pathlib import Path

# Suite files stay outside the repo. The runner reads them and writes quotes
# only under RESULTS_DIR, which is gitignored.
SUITE_ROOT = Path("/Users/ankitkumarsingh/ldi-eval-occ")
CASES_PATH = SUITE_ROOT / "cases.json"
GOLD_PATH = SUITE_ROOT / "gold.json"
GOLD_EXTRA_PATH = Path(__file__).resolve().parent / "gold_extra.csv"
TASK1_GOLD_PATH = SUITE_ROOT / "task1" / "gold.csv"
INDEX_ROOT = SUITE_ROOT / "idx"

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SPLIT_SEED = 20261109
HELD_PARTS = 1
SPLIT_PARTS = 3
# Categories of at least this size keep one id on each side of the split.
MIN_SPLIT_SIZE = 2

TRAP_DEV_SEED = 20261109
TRAP_HELD_SEED = 20261110

# Product retrieval hands the model this many hits. Diagnosis also looks
# further down the same ranking.
PRODUCT_HITS = 4
RETRIEVAL_BANDS = (5, 10, 20)

# A case that is still running after this many seconds is a timeout.
# The retry is not started once the budget is spent.
CASE_TIMEOUT_S = 180

MEDIAN_PERCENTILE = 50
TAIL_PERCENTILE = 95

# Task 1 rows at this number and above are sealed.
SEALED_FROM = 27

# Claimed baseline, used only to report whether this run reproduced it.
CLAIMED_NON_OCR_CORRECT = 60
CLAIMED_NON_OCR_TOTAL = 67
CLAIMED_TASK1_CORRECT = 5
CLAIMED_TASK1_TOTAL = 26

# Tags with no gold span. The exact not-found sentence is the correct release.
ABSTAIN_TAGS = frozenset(
    {"absent", "near_miss", "absent_injection", "extraction_incomplete"}
)

# File category -> report category. Every category in the suite file is listed.
CATEGORY_MAP = {
    "lookup": "lookup",
    "false_premise": "lookup",
    "DOCX": "lookup",
    "number/table": "table",
    "Excel cell": "table",
    "version/superseded": "version",
    "stale/changed": "version",
    "DOCX diff": "compare",
    "summary coverage": "summary",
    "absent/abstain": "absent",
    "conflict": "conflict",
    "scan/OCR": "OCR",
    "injection": "trap",
}

TASK1_TYPE_MAP = {
    "lookup": "lookup",
    "superseded-version": "version",
    "table": "table",
    "cross-doc": "compare",
    "not-in-corpus": "absent",
}

REPORT_CATEGORIES = (
    "lookup",
    "table",
    "version",
    "compare",
    "summary",
    "absent",
    "conflict",
    "OCR",
    "trap",
)

STALE_ROOTS = {"STALE-01": "stale1", "STALE-03": "stale3"}

# Recorded for a later step. The embedding list stays empty, so this is unused.
RRF_K = 60
