"""Re-index at the 1,200-character packer and score the dev split.

OCR is not run. A gold file that is not indexed makes this process exit 1
after the tables are printed. The model is loaded only for the floor check,
and only from weights already on disk.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True

from eval.config import RESULTS_DIR, SUITE_ROOT  # noqa: E402
from eval.coverage import gold_names_for, missing_gold, print_coverage  # noqa: E402
from eval.goldload import synthetic_cases, task1_open_cases  # noqa: E402
from eval.metrics import is_fabricated, metric_for  # noqa: E402
from eval.provenance import collapse  # noqa: E402
from eval.rescore import FLOOR_PATH  # noqa: E402
from eval.runner import _answer_at_root  # noqa: E402
from eval.splits import assign, read_splits  # noqa: E402
from ldi.chunker import merge_overlapping  # noqa: E402
from ldi.config import index_path  # noqa: E402
from ldi.ingest import IngestReport, ingest  # noqa: E402
from ldi.spans import PassageRecord  # noqa: E402
from ldi.store import Index  # noqa: E402

CORPORA = {
    "main": SUITE_ROOT / "corpus",
    "task1": SUITE_ROOT / "task1" / "documents",
    "stale1": SUITE_ROOT / "idx" / "stale1" / "docs",
    "stale3": SUITE_ROOT / "idx" / "stale3" / "docs",
}
BASELINE_MISSES = (
    "DOCX-02",
    "DOCX-03",
    "PQA-003",
    "PQA-006",
    "PQA-012",
    "PQA-015",
    "PQA-016",
    "XLSX-02",
    "XLSX-03",
    "XLSX-04",
)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _cases():
    return [
        case
        for case in synthetic_cases() + task1_open_cases()
        if case.root_name in CORPORA
        and case.gold.scored
        and not case.gold.abstain
        and case.gold.anchors
    ]


def _dev_held():
    cases = _cases()
    splits = read_splits()
    missing = [(case.id, case.category) for case in cases if case.id not in splits]
    if missing:
        splits.update(assign(missing))
    dev = [case for case in cases if splits.get(case.id) == "dev"]
    held = [case for case in cases if splits.get(case.id) == "heldout"]
    return dev, held


def publish_passages(index: Index) -> None:
    """Make full-text search read the passage rows for this index.

    A file the packer leaves with no passage keeps its source chunk. A
    one-line policy shorter than the passage minimum would otherwise vanish.
    """
    columns = {row[1] for row in index.db.execute("PRAGMA table_info(passages)")}
    alias_sql = "aliases" if "aliases" in columns else "''"
    drop_sql = "duplicate_of" if "duplicate_of" in columns else "''"
    before = index.db.execute("SELECT path, page, body FROM chunks").fetchall()
    rows = index.db.execute(
        f"""
        SELECT path, page_start, text, {alias_sql}, {drop_sql}
        FROM passages
        """
    ).fetchall()
    grouped: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for path, page, text, aliases, duplicate_of in rows:
        if duplicate_of:
            continue
        body = text if not aliases else text + "\n" + aliases
        grouped[path].append((int(page) if page else None, body))
    restored = [row for row in before if row[0] not in grouped]
    with index.db:
        index.db.execute("DELETE FROM chunks")
        index.db.executemany(
            "INSERT INTO chunks(path, page, body) VALUES (?, ?, ?)",
            [
                (path, "" if page is None else str(page), body)
                for path, chunks in grouped.items()
                for page, body in chunks
                if body.strip()
            ],
        )
        index.db.executemany(
            "INSERT INTO chunks(path, page, body) VALUES (?, ?, ?)",
            restored,
        )


def _keep_stale_copy(index: Index, name: str) -> None:
    """A changed policy stays stale: the index keeps the copy it had, not the file now on disk."""
    import sqlite3

    from ldi.store import sha256_file

    product = sqlite3.connect(
        f"file:{SUITE_ROOT / 'idx' / name / 'data' / 'index.sqlite'}?mode=ro",
        uri=True,
    )
    try:
        recorded = product.execute("SELECT sha256 FROM files").fetchone()
        stored = product.execute("SELECT body FROM chunks").fetchone()
    finally:
        product.close()
    if recorded is None or stored is None:
        return
    disk = (SUITE_ROOT / "idx" / name / "docs" / "policy.txt").resolve()
    if sha256_file(disk) == recorded[0]:
        return
    path = str(disk)
    stat = disk.stat()
    index.replace_file(path, stat.st_mtime_ns, stat.st_size, [(None, stored[0])], recorded[0])
    index.db.execute("UPDATE passages SET text = ? WHERE path = ?", (stored[0], path))


def _print_low_quality(root: Path) -> None:
    from ldi.chunker import LOW_QUALITY_MIN_CHARS, LOW_QUALITY_RATIO

    os.environ["LDI_ROOT"] = str(root)
    with Index(index_path()) as index:
        rows = index.db.execute(
            """
            SELECT path, detail FROM chunk_log
            WHERE kind = 'low_quality'
            ORDER BY path, detail
            """
        ).fetchall()
    print(
        f"low_quality threshold {LOW_QUALITY_RATIO:.2f} "
        f"on lines of at least {LOW_QUALITY_MIN_CHARS} characters "
        f"({len(rows)} flagged)",
        flush=True,
    )
    for path, detail in rows:
        print(f"{Path(path).name}\t{detail}", flush=True)


def _ingest_root(stage: int, name: str, source: Path) -> tuple[Path, IngestReport]:
    root = RESULTS_DIR / "stages" / str(stage) / name
    if root.exists():
        shutil.rmtree(root)
    os.environ["LDI_ROOT"] = str(root)
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Index(path) as index:
        report = ingest(source, index, ocr=False)
        publish_passages(index)
        if name in {"stale1", "stale3"}:
            _keep_stale_copy(index, name)
    return root, report


def _ranked(root: Path, question: str, limit: int = 10) -> list[PassageRecord]:
    os.environ["LDI_ROOT"] = str(root)
    with Index(index_path()) as index:
        try:
            hits = index.search_question(question, limit=limit)
        except ValueError:
            return []
        stored = index.db.execute(
            """
            SELECT path, page_start, text, span_ids, passage_id, passage_hash, source_type, aliases
            FROM passages
            """
        ).fetchall()
    by_text: dict[tuple, list[PassageRecord]] = defaultdict(list)
    for path, page, text, span_ids, passage_id, passage_hash, source_type, aliases in stored:
        page_no = int(page) if page else None
        by_text[(path, page_no, text)].append(
            PassageRecord(
                passage_id=passage_id,
                span_ids=span_ids,
                text=text,
                page_start=page_no,
                page_end=page_no,
                source_type=source_type,
                passage_hash=passage_hash,
                aliases=aliases or "",
            )
        )
    ranked = []
    for hit in hits:
        bucket = by_text.get((hit.path, hit.page, hit.text))
        if bucket:
            ranked.append(bucket.pop(0))
            continue
        matched = False
        # A searchable body may append alias lines after the quotable text.
        for (path, page, text), passages in by_text.items():
            if path == hit.path and page == hit.page and hit.text.startswith(text) and passages:
                ranked.append(passages.pop(0))
                matched = True
                break
        if not matched:
            # A file shorter than the passage minimum stays as its source chunk.
            ranked.append(
                PassageRecord(
                    passage_id="",
                    span_ids="",
                    text=hit.text,
                    page_start=hit.page,
                    page_end=hit.page,
                )
            )
    return ranked


def _covers(passages: list[PassageRecord], anchors: tuple[str, ...]) -> bool:
    blobs = [collapse(f"{passage.text}\n{passage.aliases}") for passage in passages]
    wanted = [collapse(anchor) for anchor in anchors if collapse(anchor)]
    return bool(wanted) and all(any(anchor in blob for blob in blobs) for anchor in wanted)


def score_split(roots: dict[str, Path], cases) -> dict:
    misses = []
    for case in cases:
        ranked = _ranked(roots[case.root_name], case.question, 10)
        if not _covers(merge_overlapping(ranked), case.gold.anchors):
            misses.append(case.id)
    count = len(cases)
    correct = count - len(misses)
    low, high = wilson(correct, count)
    return {
        "cases": count,
        "correct": correct,
        "recall": round(correct / count, 4) if count else None,
        "wilson95": [round(low, 4), round(high, 4)],
        "misses": misses,
    }


def floor_check(roots: dict[str, Path]) -> dict:
    from ldi.ask import _cap_metal
    from ldi.config import MODELS
    from ldi.guard import assess_load, machine_snapshot
    from ldi.weights import local_snapshot

    floor = set(json.loads(FLOOR_PATH.read_text()))
    cases = [case for case in synthetic_cases() if case.id in floor]
    missing_roots = sorted({case.root_name for case in cases if case.root_name not in roots})
    if missing_roots:
        raise SystemExit(f"No 1,200 index for {', '.join(missing_roots)}")
    spec = MODELS["1b"]
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    local = local_snapshot(spec.repo)
    if local is None:
        raise SystemExit("The 1b weights are not on disk. This run does not download them.")
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    _cap_metal(mx, spec.metal_limit_bytes)
    model, tokenizer = load(str(local), trust_remote_code=False)
    sampler = make_sampler(temp=0.0)
    missed = []
    fabricated = []
    shown_miss = {}
    try:
        for case in cases:
            released = _answer_at_root(
                case, roots[case.root_name], mx, generate_step, model, tokenizer, sampler
            )
            shown = released["shown"]
            if metric_for(case.gold, shown) != "correct":
                missed.append(case.id)
                shown_miss[case.id] = shown.splitlines()[0][:240]
            if is_fabricated(shown):
                fabricated.append(case.id)
    finally:
        del model
        del tokenizer
    return {
        "floor": len(cases),
        "floor_still_passing": len(cases) - len(missed),
        "floor_miss": missed,
        "fabrication": len(fabricated),
        "fabricated_ids": fabricated,
        "miss_first_line": shown_miss,
    }


def main(stage: int, *, heldout: bool = False, floor_only: bool = False) -> int:
    if heldout and floor_only:
        roots = {name: RESULTS_DIR / "stages" / str(stage) / name for name in CORPORA}
        dev, held = _dev_held()
        held_score = score_split(roots, held)
        print(json.dumps({"heldout": held_score}, indent=2), flush=True)
        out = RESULTS_DIR / "stages" / f"{stage}.json"
        payload = json.loads(out.read_text())
        payload["heldout"] = held_score
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return 0
    if floor_only:
        roots = {
            name: RESULTS_DIR / "stages" / str(stage) / name
            for name in CORPORA
        }
        for name in ("stale1", "stale3"):
            print(f"\n# {name}", flush=True)
            root, report = _ingest_root(stage, name, CORPORA[name])
            roots[name] = root
            print_coverage(report)
        checked = floor_check(roots)
        print(json.dumps({"floor": checked}, indent=2), flush=True)
        out = RESULTS_DIR / "stages" / f"{stage}.json"
        payload = json.loads(out.read_text()) if out.exists() else {"stage": stage}
        payload["floor"] = checked
        out.write_text(json.dumps(payload, indent=2) + "\n")
        failed = bool(checked["floor_miss"] or checked["fabrication"])
        return 1 if failed else 0
    dev, held = _dev_held()
    roots: dict[str, Path] = {}
    reports: dict[str, IngestReport] = {}
    missing: list[str] = []
    for name, source in CORPORA.items():
        print(f"\n# {name}", flush=True)
        root, report = _ingest_root(stage, name, source)
        roots[name] = root
        reports[name] = report
        print_coverage(report)
        _print_low_quality(root)
        gap = missing_gold(report, gold_names_for(name))
        for item in gap:
            missing.append(f"{name}: {item}")
    dev_score = score_split(roots, dev)
    previous_path = RESULTS_DIR / "stages" / f"{stage - 1}.json"
    if previous_path.exists():
        previous = set(json.loads(previous_path.read_text())["dev"]["misses"])
    elif stage == 1:
        previous = set(BASELINE_MISSES)
    else:
        previous = set()
    current = set(dev_score["misses"])
    dev_score["fixed"] = sorted(previous - current)
    dev_score["broken"] = sorted(current - previous)
    payload = {"stage": stage, "dev": dev_score, "gold_not_indexed": missing}
    print(json.dumps({"dev": dev_score, "gold_not_indexed": missing}, indent=2), flush=True)
    out = RESULTS_DIR / "stages" / f"{stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if heldout:
        held_score = score_split(roots, held)
        payload["heldout"] = held_score
        print(json.dumps({"heldout": held_score}, indent=2), flush=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        checked = floor_check(roots)
        payload["floor"] = checked
        print(json.dumps({"floor": checked}, indent=2), flush=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    failed = bool(missing)
    if not heldout and (checked["floor_miss"] or checked["fabrication"]):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    stage_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    score_held = "--heldout" in sys.argv
    floor_only = "--floor-only" in sys.argv
    raise SystemExit(main(stage_no, heldout=score_held, floor_only=floor_only))
