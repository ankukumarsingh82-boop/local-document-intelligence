"""Sweep passage targets on the dev split, then score the winner on held-out.

The product full-text index is not the thing being searched. Each target is
ingested into eval/results, and the new passage rows are ranked with the
existing question terms. Held-out is measured only after the dev choice.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True

from eval.config import RESULTS_DIR  # noqa: E402
from eval.goldload import synthetic_cases, task1_open_cases  # noqa: E402
from eval.provenance import collapse  # noqa: E402
from eval.splits import assign, read_splits  # noqa: E402
from ldi.chunker import install_token_counter, merge_overlapping, token_counter  # noqa: E402
from ldi.config import index_path  # noqa: E402
from ldi.ingest import ingest  # noqa: E402
from ldi.spans import PassageRecord  # noqa: E402
from ldi.store import Index  # noqa: E402

TARGETS = (500, 800, 1200)
CORPORA = {
    "main": Path("/Users/ankitkumarsingh/ldi-eval-occ/corpus"),
    "task1": Path("/Users/ankitkumarsingh/ldi-eval-occ/task1/documents"),
}
REPORT_PATH = Path(__file__).resolve().parent / "chunk_sweep.json"
SWEEP_ROOT = RESULTS_DIR / "chunk-sweep"


@dataclass(frozen=True)
class _Ranked:
    passage: PassageRecord
    label: str


def main() -> None:
    counter = token_counter()
    if counter is None:
        raise SystemExit("The 1b tokenizer is not on disk. This sweep does not download it.")
    install_token_counter(counter)
    cases = [
        case
        for case in synthetic_cases() + task1_open_cases()
        if case.root_name in CORPORA
        and case.gold.scored
        and not case.gold.abstain
        and case.gold.anchors
    ]
    splits = read_splits()
    missing = [(case.id, case.category) for case in cases if case.id not in splits]
    if missing:
        splits.update(assign(missing))
    dev = [case for case in cases if splits.get(case.id) == "dev"]
    heldout = [case for case in cases if splits.get(case.id) == "heldout"]
    if not dev or not heldout:
        raise SystemExit("The dev and held-out splits both need gold passages.")
    dev_reports = {}
    built: dict[int, dict[str, Path]] = {}
    for target in TARGETS:
        print(f"target {target}", flush=True)
        roots = _ingest(target)
        built[target] = roots
        dev_reports[str(target)] = _measure(roots, dev, counter)
        print(json.dumps({str(target): dev_reports[str(target)]}), flush=True)
    choice = min(
        TARGETS,
        key=lambda target: (
            -dev_reports[str(target)]["recall_at_10"],
            -dev_reports[str(target)]["recall_at_5"],
            dev_reports[str(target)]["duplicate_rate_top5"],
            dev_reports[str(target)]["mean_prompt_tokens"],
        ),
    )
    held = _measure(built[choice], heldout, counter)
    report = {
        "dev": dev_reports,
        "choice": {
            "target": choice,
            "hard_max": choice + 200,
            "rule": "Highest dev Recall@10, then Recall@5, then a lower duplicate rate, then fewer prompt tokens.",
        },
        "heldout": {str(choice): held},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["choice"]), flush=True)


def _ingest(target: int) -> dict[str, Path]:
    import ldi.config as config

    config.PASSAGE_TARGET_CHARS = target
    config.PASSAGE_MAX_CHARS = target + 200
    roots = {}
    for name, source in CORPORA.items():
        root = SWEEP_ROOT / str(target) / name
        os.environ["LDI_ROOT"] = str(root)
        path = index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        with Index(path) as index, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ingest(source, index)
        roots[name] = _passage_index(path, root / "passages.sqlite")
    return roots


def _passage_index(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    with Index(source) as origin, Index(dest) as index:
        rows = origin.db.execute(
            """
            SELECT path, page_start, text, span_ids, label, doc_sha, passage_hash
            FROM passages
            ORDER BY path, rowid
            """
        ).fetchall()
        grouped: dict[str, list[tuple[int | None, str]]] = defaultdict(list)
        for path, page, text, _span_ids, _label, _sha, _passage_hash in rows:
            grouped[path].append((int(page) if page else None, text))
        for path, chunks in grouped.items():
            # A shared hash would mark every later file as a duplicate and hide it.
            index.replace_file(path, 0, 1, chunks, hashlib.sha256(path.encode()).hexdigest())
        index.refresh_labels()
    return dest


def _measure(roots: dict[str, Path], cases, counter) -> dict[str, float | int]:
    correct_5 = 0
    correct_10 = 0
    duplicate = 0.0
    tokens = 0
    for case in cases:
        ranked = _ranked(roots[case.root_name], case.question)
        duplicate += _duplicate_rate(ranked[:5])
        merged_5 = merge_overlapping([item.passage for item in ranked[:5]])
        merged_10 = merge_overlapping([item.passage for item in ranked[:10]])
        if _covers(merged_5, case.gold.anchors):
            correct_5 += 1
        if _covers(merged_10, case.gold.anchors):
            correct_10 += 1
        tokens += _prompt_tokens(ranked[:5], counter)
    count = len(cases)
    return {
        "cases": count,
        "correct_at_5": correct_5,
        "correct_at_10": correct_10,
        "recall_at_5": round(correct_5 / count, 4),
        "recall_at_10": round(correct_10 / count, 4),
        "duplicate_rate_top5": round(duplicate / count, 4),
        "mean_prompt_tokens": round(tokens / count, 1),
    }


def _ranked(path: Path, question: str) -> list[_Ranked]:
    with Index(path) as index:
        try:
            hits = index.search_question(question, limit=10)
        except ValueError:
            return []
    source = path.parent / "data" / "index.sqlite"
    with Index(source) as origin:
        stored = origin.db.execute(
            """
            SELECT path, page_start, text, span_ids, label, passage_id, passage_hash,
                   doc_sha, heading_path, source_type
            FROM passages
            """
        ).fetchall()
    by_text: dict[tuple, list[PassageRecord]] = defaultdict(list)
    for (
        stored_path,
        page,
        text,
        span_ids,
        label,
        passage_id,
        passage_hash,
        doc_sha,
        heading,
        source_type,
    ) in stored:
        page_no = int(page) if page else None
        record = PassageRecord(
            passage_id=passage_id,
            span_ids=span_ids,
            text=text,
            doc_sha=doc_sha,
            page_start=page_no,
            page_end=page_no,
            heading_path=heading,
            source_type=source_type,
            passage_hash=passage_hash,
            label=label,
        )
        by_text[(stored_path, page_no, text)].append(record)
    ranked = []
    for hit in hits:
        bucket = by_text.get((hit.path, hit.page, hit.text))
        if not bucket:
            continue
        record = bucket.pop(0)
        ranked.append(_Ranked(record, record.label))
    return ranked


def _duplicate_rate(ranked: list[_Ranked]) -> float:
    seen: set[str] = set()
    duplicates = 0
    for item in ranked:
        ids = {span_id for span_id in item.passage.span_ids.split() if span_id}
        if ids & seen:
            duplicates += 1
        seen |= ids
    return duplicates / 5


def _covers(passages: list[PassageRecord], anchors: tuple[str, ...]) -> bool:
    blobs = [collapse(passage.text) for passage in passages]
    wanted = [collapse(anchor) for anchor in anchors if collapse(anchor)]
    return all(any(anchor in blob for blob in blobs) for anchor in wanted)


def _prompt_tokens(ranked: list[_Ranked], counter) -> int:
    if not ranked:
        return 0
    parts = [f"{item.label}\n{item.passage.text}" for item in ranked]
    try:
        return int(counter("\n\n".join(parts)))
    except Exception:
        return 0


if __name__ == "__main__":
    main()
