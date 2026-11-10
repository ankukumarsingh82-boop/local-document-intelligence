"""Compare the current search with the new retriever. Ask stays on the old path.

Open corpora are ingested into temporary indexes under eval/results. The suite
indexes are not opened. Dense search is a reversed lexical list, not a model.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True

from eval.config import (  # noqa: E402
    CASE_TIMEOUT_S,
    MEDIAN_PERCENTILE,
    PRODUCT_HITS,
    RESULTS_DIR,
    RETRIEVAL_BANDS,
    TAIL_PERCENTILE,
)
from eval.goldload import synthetic_cases, task1_open_cases  # noqa: E402
from eval.metrics import metric_for  # noqa: E402
from eval.provenance import collapse  # noqa: E402
from eval.splits import assign, read_splits  # noqa: E402
from ldi.ask import _cap_metal, _complete, format_stale, settle_answer, stale_paths  # noqa: E402
import ldi.config as config  # noqa: E402
from ldi.config import MODELS, index_path  # noqa: E402
from ldi.evidence import build_evidence  # noqa: E402
from ldi.guard import assess_load, machine_snapshot  # noqa: E402
from ldi.ingest import ingest  # noqa: E402
from ldi.retrieve import retrieve  # noqa: E402
from ldi.store import Hit, Index  # noqa: E402
from ldi.verify import format_not_found  # noqa: E402
from ldi.weights import local_snapshot  # noqa: E402

CORPORA = {
    "main": Path("/Users/ankitkumarsingh/ldi-eval-occ/corpus"),
    "task1": Path("/Users/ankitkumarsingh/ldi-eval-occ/task1/documents"),
}
REPORT_PATH = RESULTS_DIR / "retrieval_report.json"
INDEX_ROOT = RESULTS_DIR / "retrieval-report"
SETTINGS = ("legacy", "lexical", "hybrid")
CONFIG_PATH = ROOT / "ldi" / "config.py"


@dataclass
class Pack:
    hits: list[Hit]
    ranked: list[Hit]
    before: list[Hit]
    retrieve_s: float
    fuse_s: float | None
    gate_s: float | None
    fused: list[tuple[float, str]] = field(default_factory=list)
    fused_ids: list[tuple[str, float]] = field(default_factory=list)
    shown: str = ""
    metric: str = ""


def main() -> None:
    for name, source in CORPORA.items():
        if not source.is_dir():
            raise SystemExit(f"Missing corpus {name} at {source}")
    cases = [
        case
        for case in synthetic_cases() + task1_open_cases()
        if case.root_name in CORPORA and case.gold.scored
    ]
    splits = read_splits()
    missing = [(case.id, case.category) for case in cases if case.id not in splits]
    if missing:
        splits.update(assign(missing))
    dev = [case for case in cases if splits.get(case.id) == "dev"]
    heldout = [case for case in cases if splits.get(case.id) == "heldout"]
    if not dev or not heldout:
        raise SystemExit("The dev and held-out splits both need scored cases.")
    started_floor = config.LEXICAL_COVERAGE_FLOOR
    print(f"ingesting into {INDEX_ROOT}", flush=True)
    roots = _ingest()
    print(
        json.dumps({"coverage_floor": started_floor, "dev": len(dev), "heldout": len(heldout)}),
        flush=True,
    )
    dev_rows = _retrieve_all(roots, dev)
    tried = [started_floor]
    runtime = _load_model()
    try:
        _answer_all(dev, dev_rows, runtime, "dev")
        for _round in range(5):
            needed = _floor_block(dev, dev_rows)
            if needed is None or needed + 1e-12 >= config.LEXICAL_COVERAGE_FLOOR:
                break
            updated = math.floor((needed + 1e-9) * 1_000_000) / 1_000_000
            if updated >= config.LEXICAL_COVERAGE_FLOOR:
                break
            config.LEXICAL_COVERAGE_FLOOR = updated
            tried.append(updated)
            print(json.dumps({"coverage_floor": updated}), flush=True)
            refreshed = _retrieve_all(roots, dev, settings=("lexical", "hybrid"))
            for case in dev:
                for setting in ("lexical", "hybrid"):
                    previous = dev_rows[case.id][setting]
                    current = refreshed[case.id][setting]
                    if _hit_key(previous.hits) == _hit_key(current.hits):
                        current.shown = previous.shown
                        current.metric = previous.metric
                    dev_rows[case.id][setting] = current
            _answer_all(dev, dev_rows, runtime, "dev", settings=("lexical", "hybrid"))
        _write_floor(config.LEXICAL_COVERAGE_FLOOR)
        held_rows = _retrieve_all(roots, heldout)
        _answer_all(heldout, held_rows, runtime, "heldout")
    finally:
        _unload(runtime)
    report = {
        "coverage_floor": config.LEXICAL_COVERAGE_FLOOR,
        "floors_tried": tried,
        "dense_enabled": False,
        "ask_path": "search_question",
        "dev": _summary(dev, dev_rows),
        "heldout": _summary(heldout, held_rows),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)


def _ingest() -> dict[str, Path]:
    roots = {}
    for name, source in CORPORA.items():
        root = INDEX_ROOT / name
        os.environ["LDI_ROOT"] = str(root)
        path = index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)
        with (
            Index(path) as index,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            report = ingest(source, index)
        roots[name] = path
        print(
            f"ingested {name}: {report.files} files, {report.chunks} chunks, {report.skipped} skipped",
            flush=True,
        )
    return roots


def _retrieve_all(roots: dict[str, Path], cases, settings: tuple[str, ...] = SETTINGS) -> dict[str, dict[str, Pack]]:
    found: dict[str, dict[str, Pack]] = {}
    opened: dict[str, Index] = {}
    try:
        for name, path in roots.items():
            index = Index(path)
            index.__enter__()
            opened[name] = index
        for case in cases:
            index = opened[case.root_name]
            os.environ["LDI_ROOT"] = str(INDEX_ROOT / case.root_name)
            row = found.setdefault(case.id, {})
            lexical = None
            if "lexical" in settings or "hybrid" in settings:
                lexical = _safe_retrieve(index, case.question)
                if "lexical" in settings:
                    row["lexical"] = lexical
            if "hybrid" in settings:
                ordered = list(lexical.fused_ids) if lexical is not None else []
                fake = _fake_dense([passage_id for passage_id, _coverage in reversed(ordered)])
                row["hybrid"] = _safe_retrieve(index, case.question, dense=fake)
            if "legacy" in settings:
                row["legacy"] = _legacy(index, case.question)
    finally:
        for index in opened.values():
            index.__exit__(None, None, None)
    return found


def _safe_retrieve(index: Index, question: str, dense=None) -> Pack:
    try:
        found = retrieve(index, question, limit=PRODUCT_HITS, dense=dense)
    except ValueError:
        return Pack(hits=[], ranked=[], before=[], retrieve_s=0.0, fuse_s=0.0, gate_s=0.0)
    return _from_retrieval(index, found)


def _legacy(index: Index, question: str) -> Pack:
    started = time.perf_counter()
    try:
        hits = index.search_question(question, limit=RETRIEVAL_BANDS[-1])
    except ValueError:
        hits = []
    return Pack(
        hits=hits[:PRODUCT_HITS],
        ranked=hits,
        before=hits,
        retrieve_s=time.perf_counter() - started,
        fuse_s=None,
        gate_s=None,
    )


def _from_retrieval(index: Index, found) -> Pack:
    fused = []
    for passage_id, coverage in found.log["coverage"].items():
        stored = index.passage_row(passage_id)
        fused.append((float(coverage), "" if stored is None else str(stored[2])))
    return Pack(
        hits=list(found.hits),
        ranked=list(found.ranked),
        before=list(found.before_hash),
        retrieve_s=float(found.log["retrieve_s"]),
        fuse_s=float(found.log["fuse_s"]),
        gate_s=float(found.log["gate_s"]),
        fused=fused,
        fused_ids=[(passage_id, float(coverage)) for passage_id, coverage in found.log["coverage"].items()],
    )


def _answer_all(cases, rows, runtime, split: str, settings: tuple[str, ...] = SETTINGS) -> None:
    mx, generate_step, model, tokenizer, sampler = runtime
    total = len(cases)
    for number, case in enumerate(cases, start=1):
        os.environ["LDI_ROOT"] = str(INDEX_ROOT / case.root_name)
        for setting in settings:
            pack = rows[case.id][setting]
            if pack.metric:
                continue
            print(f"{split} {setting} {number}/{total} {case.id}", flush=True)
            pack.shown = _shown(case.question, pack.hits, _gaps(), mx, generate_step, model, tokenizer, sampler)
            pack.metric = "timeout" if pack.shown == "" else metric_for(case.gold, pack.shown)


def _shown(question, hits, gaps, mx, generate_step, model, tokenizer, sampler) -> str:
    if not hits:
        return format_not_found(gaps)
    cited = stale_paths(hits)
    if cited:
        return format_stale(cited)
    items = build_evidence(question, hits)
    if not items:
        return format_not_found(gaps)
    budget = time.perf_counter()

    def generate(reason: str) -> str:
        if time.perf_counter() - budget > CASE_TIMEOUT_S:
            raise TimeoutError(question)
        with contextlib.redirect_stdout(io.StringIO()):
            return _complete(mx, generate_step, model, tokenizer, sampler, question, hits, reason)

    try:
        _raw, released = settle_answer(generate, hits, question)
    except TimeoutError:
        return ""
    if released.paths:
        cited = stale_paths([Hit(path, None, "") for path in released.paths])
        if cited:
            return format_stale(cited)
    if released.kind in {"not_found", "retry"}:
        return format_not_found(gaps)
    return released.text


def _gaps() -> list[tuple[str, int | None, str]]:
    path = index_path()
    if not path.exists():
        return []
    with Index(path) as index:
        return index.list_gaps()


def _floor_block(cases, rows) -> float | None:
    """Lowest coverage that would put a gold passage into the top hits."""
    needed: float | None = None
    floor = config.LEXICAL_COVERAGE_FLOOR
    for case in cases:
        legacy = rows[case.id]["legacy"].metric
        if legacy == "false_not_found":
            continue
        anchors = [collapse(anchor) for anchor in case.gold.anchors if collapse(anchor)]
        if not anchors:
            continue
        for setting in ("lexical", "hybrid"):
            pack = rows[case.id][setting]
            if pack.metric != "false_not_found":
                continue
            candidate = _blocking_coverage(pack, anchors, floor)
            if candidate is None:
                continue
            needed = candidate if needed is None else min(needed, candidate)
    return needed


def _blocking_coverage(pack: Pack, anchors: list[str], floor: float) -> float | None:
    found = None
    for index, (coverage, text) in enumerate(pack.fused):
        if coverage + 1e-12 >= floor:
            continue
        folded = collapse(text)
        if not any(anchor in folded for anchor in anchors):
            continue
        ahead = sum(1 for earlier, _text in pack.fused[:index] if earlier + 1e-12 >= coverage)
        if ahead >= PRODUCT_HITS:
            continue
        found = coverage if found is None else min(found, coverage)
    return found


def _summary(cases, rows) -> dict:
    recall_cases = [case for case in cases if case.gold.anchors and not case.gold.abstain]
    settings = {}
    flipped = {}
    for setting in SETTINGS:
        packs = [rows[case.id][setting] for case in cases]
        correct = {
            band: sum(1 for case in recall_cases if _covers(rows[case.id][setting].ranked[:band], case.gold.anchors))
            for band in RETRIEVAL_BANDS
        }
        count = len(recall_cases)
        settings[setting] = {
            "cases": len(cases),
            "recall_cases": count,
            "correct_at_5": correct[5],
            "correct_at_10": correct[10],
            "correct_at_20": correct[20],
            "recall_at_5": _rate(correct[5], count),
            "recall_at_10": _rate(correct[10], count),
            "recall_at_20": _rate(correct[20], count),
            "false_not_found": sum(1 for pack in packs if pack.metric == "false_not_found"),
            "false_accept": sum(1 for pack in packs if pack.metric == "false_accept"),
            "timeout": sum(1 for pack in packs if pack.metric == "timeout"),
            "duplicate_rate_top5_before": _rate(sum(_dup_count(pack.before) for pack in packs), len(packs) * 5),
            "duplicate_rate_top5_after": _rate(sum(_dup_count(pack.ranked) for pack in packs), len(packs) * 5),
            "latency": {
                "retrieve": _latency([pack.retrieve_s for pack in packs]),
                "fuse": _latency([pack.fuse_s for pack in packs if pack.fuse_s is not None]),
                "gate": _latency([pack.gate_s for pack in packs if pack.gate_s is not None]),
            },
        }
        if setting != "legacy":
            flipped[setting] = [
                case.id
                for case in cases
                if rows[case.id]["legacy"].metric == "correct" and rows[case.id][setting].metric != "correct"
            ]
    return {"settings": settings, "flipped": flipped}


def _covers(hits: list[Hit], anchors: tuple[str, ...]) -> bool:
    blobs = [collapse(hit.text) for hit in hits]
    wanted = [collapse(anchor) for anchor in anchors if collapse(anchor)]
    return bool(wanted) and all(any(anchor in blob for blob in blobs) for anchor in wanted)


def _dup_count(hits: list[Hit]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for hit in hits[:5]:
        key = hit.passage_hash or hashlib.sha256(hit.text.encode()).hexdigest()
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _latency(values: list[float]) -> dict[str, float | None]:
    return {
        "median": _percentile(values, MEDIAN_PERCENTILE),
        "p95": _percentile(values, TAIL_PERCENTILE),
    }


def _percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(percent / 100 * len(ordered)) - 1
    index = min(len(ordered) - 1, max(0, index))
    return round(ordered[index], 6)


def _rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _hit_key(hits: list[Hit]) -> tuple[str, ...]:
    return tuple(hit.text for hit in hits)


def _print_summary(report: dict) -> None:
    print(json.dumps({"coverage_floor": report["coverage_floor"], "floors_tried": report["floors_tried"]}), flush=True)
    for split in ("dev", "heldout"):
        block = report[split]
        for setting, counts in block["settings"].items():
            print(
                json.dumps(
                    {
                        "split": split,
                        "setting": setting,
                        "recall_at_5": counts["recall_at_5"],
                        "recall_at_10": counts["recall_at_10"],
                        "recall_at_20": counts["recall_at_20"],
                        "false_not_found": counts["false_not_found"],
                        "false_accept": counts["false_accept"],
                        "duplicate_rate_top5_before": counts["duplicate_rate_top5_before"],
                        "duplicate_rate_top5_after": counts["duplicate_rate_top5_after"],
                        "latency": counts["latency"],
                    }
                ),
                flush=True,
            )
        for setting, ids in block["flipped"].items():
            for case_id in ids:
                print(f"flipped {split} {setting} {case_id}", flush=True)


def _write_floor(value: float) -> None:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    text = CONFIG_PATH.read_text()
    updated, count = re.subn(
        r"^LEXICAL_COVERAGE_FLOOR = .*$",
        f"LEXICAL_COVERAGE_FLOOR = {rendered}",
        text,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit("LEXICAL_COVERAGE_FLOOR is missing from ldi/config.py")
    if updated != text:
        CONFIG_PATH.write_text(updated)


def _fake_dense(ordered: list[str]):
    return _FakeDense(ordered)


def _FakeDense(ordered: list[str]):
    if _FakeDense.cls is None:
        path = ROOT / "tests" / "fake_dense.py"
        spec = importlib.util.spec_from_file_location("ldi_fake_dense", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Missing {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FakeDense.cls = module.FakeDenseRetriever
    return _FakeDense.cls(ordered)


_FakeDense.cls = None


def _load_model():
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
    started = time.perf_counter()
    model, tokenizer = load(str(local), trust_remote_code=False)
    print(json.dumps({"load_s": round(time.perf_counter() - started, 3)}), flush=True)
    return mx, generate_step, model, tokenizer, make_sampler(temp=0.0)


def _unload(runtime) -> None:
    if runtime is None:
        return
    mx = runtime[0]
    runtime_list = list(runtime)
    for index in range(len(runtime_list)):
        runtime_list[index] = None
    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear is not None:
        clear()


if __name__ == "__main__":
    main()
