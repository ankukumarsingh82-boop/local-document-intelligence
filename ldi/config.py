"""Sizes and paths for this MacBook Air."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RESERVE_BYTES = 25 * 1024**3
MAX_FILE_BYTES = 32 * 1024**2
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
PASSAGE_CHARS = 900
# The passage packer. Full-text search still uses CHUNK_CHARS above.
CHUNKER_VERSION = "1"
# Chosen on the dev split in eval/chunk_sweep.json. Hard max is the target plus 200.
PASSAGE_TARGET_CHARS = 1200
PASSAGE_MAX_CHARS = 1400
PASSAGE_OVERLAP_CHARS = 200
PASSAGE_MIN_CHARS = 30
PASSAGE_MAX_TOKENS = 384
PREVIEW_CHARS = 180
TOP_CHUNKS = 4
MAX_TOP_CHUNKS = 6
MAX_TOKENS = 384
PROBE_MAX_TOKENS = 128
MAX_KV_TOKENS = 2048
PREFILL_STEP = 256
METAL_CACHE_BYTES = 128 * 1024**2
SMALL_MODEL_METAL_CEILING = 2 * 1024**3

PRESSURE_NORMAL = 1
PRESSURE_WARNING = 2
PRESSURE_URGENT = 4
PRESSURE_CRITICAL = 8
LOADABLE_PRESSURE = frozenset({PRESSURE_NORMAL, PRESSURE_WARNING})

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst"})
PDF_SUFFIXES = frozenset({".pdf"})
DOCX_SUFFIXES = frozenset({".docx"})
XLSX_SUFFIXES = frozenset({".xlsx"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | XLSX_SUFFIXES | IMAGE_SUFFIXES
OCR_PIXELS = 1280 * 28 * 28
OCR_PROMPT = "OCR:"
EXTRACTOR_VERSION = "1"
# A stored OCR span below this confidence is low-confidence.
# A null confidence is also low-confidence: the line is ineligible for a
# number, date, or ID slot until a later step checks it against the image.
OCR_LOW_CONFIDENCE = 0.5
# PDF table geometry, in PDF points. Word spaces sit under the cap.
# A column gap is at least the multiple of that space, and at least the floor.
TABLE_WORD_GAP_CAP = 20.0
TABLE_GAP_MULTIPLE = 3.0
TABLE_MIN_COLUMN_GAP = 18.0
TABLE_COLUMN_TOLERANCE = 12.0
TABLE_MIN_ROWS = 3
TABLE_MIN_COLUMNS = 2
TABLE_LINE_Y_TOLERANCE = 2.4
TABLE_RULE_TOLERANCE = 2.0
TABLE_MIN_HLINE = 24.0
TABLE_MIN_VLINE = 12.0
TABLE_MAX_RULE_THICKNESS = 1.8
TABLE_MIN_ROW_GAP = 6.0
TABLE_MAX_ROW_GAP = 72.0
TABLE_MIN_RULE_OVERLAP = 30.0
# Phrases addressed to an assistant. Ordinary obligations are not in this list.
INSTRUCTION_PHRASES = (
    "ignore previous",
    "system prompt",
    "you are an ai",
    "respond only with",
    "disregard the above",
)
# Wording that retires an earlier provision. Ordinary duties are not in this list.
SUPERSESSION_PHRASES = (
    "supersedes",
    "in partial modification of",
    "rescinded",
)
# A fiscal year written "FY 2024-25" starts in this month.
FY_START_MONTH = 4
# The optional Gemma meaning check. Off until a user turns it on, and it stays
# off while the weights are not on disk.
GEMMA_MEANING_CHECK = False

# Passage retrieval. ldi ask reads these passages when SOURCE_EVIDENCE is on.
RETRIEVAL_DEPTH = 40
POINTER_LIMIT = 8
RRF_K = 60
RRF_WEIGHT_LEXICAL = 1.0
RRF_WEIGHT_DENSE = 1.0
RRF_WEIGHT_DENSE_TABLE = 0.5
BM25_HEADING_WEIGHT = 2.0
BM25_BODY_WEIGHT = 1.0
AND_COVERAGE_RANK_BONUS = 3
# A file the question names by two adjacent words of its file name. bm25 is scaled by up to 1.5.
NAME_BOOST = 0.5
# A four-term question that matches one term stays above this. Tuned on dev only.
LEXICAL_COVERAGE_FLOOR = 0.15
MAX_QUERY_TOKENS = 20
LONG_DIGIT_MIN = 5
NAME_EDIT_DISTANCE = 2
DENSE_ENABLED = False
# The document-first route is enabled in replay until its scan cases clear the release gate.
DOCUMENT_FIRST_RECONCILIATION = False
REDUCED_RETRIEVAL_MODE = "reduced retrieval mode"
# Floor fixes. Defaults are the fixed product. A measurement arm turns a later fix off.
# Fix 1. A finished prose cut does not pack its tail a second time.
CARRY_AFTER_FINAL_CUT = False
# Fix 2. A sentence end needs whitespace or the end of the text after it.
SENTENCE_NEEDS_BOUNDARY = True
# Fix 3. Rank on passage text plus rule-built aliases, and quote source lines.
SOURCE_EVIDENCE = True
# Fix 4. Keep every non-repeated source line in a passage or a table context line.
COVER_EVERY_LINE = True
# How many question-matched table rows the evidence unit may add.
EVIDENCE_MATCHED_ROWS = 3
# A table whose original lines fit this limit is quoted whole.
TABLE_EVIDENCE_CHARS = 1400
# Source lines shared by every hit of one question. Hits take turns, best hit first.
EVIDENCE_BUDGET = 24
# A cut line at least this long also brings the next line of its hit.
CONTINUATION_MIN_CHARS = 40

# One-character tokens are kept after these words.
STRUCTURAL_NOUNS = frozenset(
    {
        "annexure",
        "schedule",
        "clause",
        "section",
        "table",
        "part",
        "form",
        "appendix",
    }
)
INTENT_CURRENT = frozenset({"current", "latest", "still"})
INTENT_REVISED = frozenset({"revised"})
INTENT_HISTORICAL = frozenset({"earlier", "old"})
# Multi-word phrases are lifted first. Single words are not required match terms.
QUALIFIER_PHRASES = (
    "subject to",
    "provided that",
    "other than",
    "up to",
    "at least",
)
QUALIFIER_WORDS = frozenset({"not", "no", "except", "unless", "only"})
NEGATIONS = frozenset({"not", "no"})
# Function words. and, or, not, and near stay in the query so they can be quoted.
QUERY_STOPWORDS = frozenset(
    """
    a an the is are was were be been being
    of to for in on at by with from as
    what which who whom whose when where why how
    this that these those it its do does did
    can could has have had will would should
    if then than into about your you we they their
    there here but so any all each per via
    """.split()
)
# OR and dense queries only. Required coverage does not see these.
SEARCH_ALIASES = {
    "year": "annual",
    "yearly": "annual",
    "pay": "fee",
    "payable": "fee",
}
CURRENCY_FOLDS = {
    "rs": "inr",
    "inr": "inr",
    "rupee": "inr",
    "rupees": "inr",
}
UNIT_FOLDS = {
    "lakh": "lakh",
    "lakhs": "lakh",
    "lac": "lakh",
    "crore": "crore",
    "crores": "crore",
    "cr": "crore",
}
HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "dr", "shri", "smt", "prof", "sri"})
MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)
TABLE_CUES = frozenset(
    {"table", "total", "sum", "average", "column", "row", "percent", "percentage"}
)

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        ".next",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".cursor",
        "data",
    }
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    approx_bytes: int
    metal_limit_bytes: int
    label: str


# Allowlisted repos. Estimates sit above the real weight files so the
# disk check still leaves 25 GB free if the download is a little larger.
# 1b is 1280 MB from a measured 1.1 GB Metal peak, and it stays under
# the 2 GB line. 3b stays the default. Embeddings are not chat models.
MODELS: dict[str, ModelSpec] = {
    "1b": ModelSpec(
        key="1b",
        repo="mlx-community/granite-4.0-1b-4bit",
        approx_bytes=int(1.2 * 1024**3),
        metal_limit_bytes=1280 * 1024**2,
        label="Granite 4.0 1B, 4-bit",
    ),
    "3b": ModelSpec(
        key="3b",
        repo="mlx-community/Qwen2.5-3B-Instruct-4bit",
        approx_bytes=int(2.5 * 1024**3),
        metal_limit_bytes=4 * 1024**3,
        label="Qwen2.5 3B Instruct, 4-bit",
    ),
    "7b": ModelSpec(
        key="7b",
        repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        approx_bytes=int(5.5 * 1024**3),
        metal_limit_bytes=7 * 1024**3,
        label="Qwen2.5 7B Instruct, 4-bit",
    ),
}

EMBEDDINGS: dict[str, ModelSpec] = {
    "bge": ModelSpec(
        key="bge",
        repo="mlx-community/bge-small-en-v1.5-8bit",
        approx_bytes=64 * 1024**2,
        metal_limit_bytes=512 * 1024**2,
        label="BGE Small English v1.5, 8-bit",
    ),
}

# The 4 GB budget covers the 8-bit weights and the separate OCR environment.
# The Metal cap is the 2 GB small-model ceiling.
OCR_MODELS: dict[str, ModelSpec] = {
    "ocr": ModelSpec(
        key="ocr",
        repo="jmbarrancoidener/PaddleOCR-VL-1.6-mlx-8bit",
        approx_bytes=4 * 1024**3,
        metal_limit_bytes=SMALL_MODEL_METAL_CEILING,
        label="PaddleOCR-VL 1.6, 8-bit",
    ),
}


def setup_models() -> dict[str, ModelSpec]:
    return {**MODELS, **EMBEDDINGS, **OCR_MODELS}


def ocr_venv() -> Path:
    return project_root() / ".venv-ocr"


def ocr_python() -> Path:
    return ocr_venv() / "bin" / "python"

DEFAULT_MODEL = "3b"
CHAT_MODEL = "1b"


def gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.1f} GB"


def catalog_cap_from_peak(peak_bytes: int) -> int:
    """Round a measured Metal peak up to the next 256 MB, and stay at or below 2 GB."""
    step = 256 * 1024**2
    if peak_bytes <= 0:
        return step
    rounded = ((peak_bytes + step - 1) // step) * step
    if rounded <= peak_bytes:
        rounded += step
    return min(max(rounded, step), SMALL_MODEL_METAL_CEILING)


def project_root() -> Path:
    override = os.environ.get("LDI_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "ldi").is_dir():
            return candidate
    return cwd


def index_path() -> Path:
    return project_root() / "data" / "index.sqlite"


def gemma_available() -> bool:
    """True once Gemma has been downloaded. This loads nothing."""
    from gemma_flagger.service import available

    return available()


def meaning_check_state() -> dict:
    """What the settings page shows for the meaning check."""
    present = gemma_available()
    return {"available": present, "enabled": GEMMA_MEANING_CHECK and present}
