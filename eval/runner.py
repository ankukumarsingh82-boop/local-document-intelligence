"""Run the current Granite path twice. Quotes stay in the gitignored results directory."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True

from eval.config import (  # noqa: E402
    CASE_TIMEOUT_S,
    CLAIMED_NON_OCR_TOTAL,
    INDEX_ROOT,
    PRODUCT_HITS,
    RESULTS_DIR,
    RETRIEVAL_BANDS,
)
from eval.diagnose import (  # noqa: E402
    by_category,
    claimed_gap,
    count_rows,
    failing,
    false_not_found_target,
    latency_block,
    trap_counts,
)
from eval.goldload import SuiteCase, index_dir, synthetic_cases, task1_open_cases  # noqa: E402
from eval.metrics import anchors_covered, is_fabricated, metric_for  # noqa: E402
from eval.provenance import collapse  # noqa: E402
from eval.splits import read_splits  # noqa: E402
from eval.traps import PASSING, Trap, all_traps, judge  # noqa: E402
from ldi.ask import _cap_metal, _complete, format_stale, settle_answer, stale_paths  # noqa: E402
from ldi.config import MODELS, TOP_CHUNKS, index_path  # noqa: E402
from ldi.evidence import build_evidence  # noqa: E402
from ldi.grammar import parse_answer_json  # noqa: E402
from ldi.guard import assess_load, machine_snapshot  # noqa: E402
from ldi.ingest import ingest  # noqa: E402
from ldi.store import Hit, Index  # noqa: E402
from ldi.verify import format_not_found  # noqa: E402
from ldi.weights import local_snapshot  # noqa: E402

PASSES = 2
COUNTS_PATH = Path(__file__).resolve().parent / "baseline_counts.json"


def _sysctl(name: str) -> str:
    result = subprocess.run(
        ["sysctl", "-n", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def hardware() -> dict[str, object]:
    snap = machine_snapshot()
    ram = _sysctl("hw.memsize")
    return {
        "machine": platform.machine(),
        "processor": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "system": platform.platform(),
        "ram_bytes": int(ram) if ram.isdigit() else None,
        "pressure": snap.pressure,
        "reclaimable_bytes": snap.reclaimable,
        "swap_used_bytes": snap.swap_used,
        "on_battery": snap.on_battery,
    }


def _bodies(root: Path) -> list[str]:
    os.environ["LDI_ROOT"] = str(root)
    path = index_path()
    if not path.exists():
        return []
    with Index(path) as index:
        return [row[0] for row in index.db.execute("SELECT body FROM chunks")]


def _contains(texts: list[str], anchor: str) -> bool:
    folded = collapse(anchor)
    if not folded:
        return False
    return any(folded in collapse(text) for text in texts)


def _rank(question: str, root: Path, anchors: tuple[str, ...]) -> int | None:
    if not anchors:
        return None
    os.environ["LDI_ROOT"] = str(root)
    with Index(index_path()) as index:
        try:
            hits = index.search_question(question, limit=RETRIEVAL_BANDS[-1])
        except ValueError:
            return None
    ranks = []
    for anchor in anchors:
        found = None
        folded = collapse(anchor)
        for number, hit in enumerate(hits, start=1):
            if folded in collapse(hit.text):
                found = number
                break
        if found is None:
            return None
        ranks.append(found)
    return max(ranks)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _release_case(case: SuiteCase, root: Path, generate) -> dict:
    os.environ["LDI_ROOT"] = str(root)
    started = time.perf_counter()
    with Index(index_path()) as index:
        try:
            hits = index.search_question(case.question, limit=PRODUCT_HITS)
        except ValueError:
            hits = []
        gaps = index.list_gaps()
    retrieve_s = time.perf_counter() - started
    evidence: list[dict[str, str]] = []
    if not hits:
        shown = format_not_found(gaps)
        return _timed(shown, "", evidence, retrieve_s, 0.0, 0.0, False)
    if stale_paths(hits):
        shown = format_stale(stale_paths(hits))
        return _timed(shown, "", evidence, retrieve_s, 0.0, 0.0, False)
    items = build_evidence(case.question, hits)
    evidence = [{"id": item.id, "locator": item.locator, "text": item.text} for item in items]
    if not items:
        shown = format_not_found(gaps)
        return _timed(shown, "", evidence, retrieve_s, 0.0, 0.0, False)
    invalid = False
    model_s = 0.0
    budget = time.perf_counter()

    def generate_timed(reason: str) -> str:
        nonlocal invalid, model_s
        if time.perf_counter() - budget > CASE_TIMEOUT_S:
            raise TimeoutError(case.id)
        clock = time.perf_counter()
        reply = generate(reason)
        model_s += time.perf_counter() - clock
        allowed = {item.id for item in items}
        if reply and parse_answer_json(reply, allowed) is None:
            invalid = True
        return reply

    render_clock = time.perf_counter()
    try:
        raw, released = settle_answer(generate_timed, hits, case.question)
    except TimeoutError:
        return _timed("", "", evidence, retrieve_s, model_s, 0.0, True, invalid)
    if time.perf_counter() - budget > CASE_TIMEOUT_S:
        return _timed("", "", evidence, retrieve_s, model_s, 0.0, True, invalid)
    if released.paths:
        cited = stale_paths([Hit(path, None, "") for path in released.paths])
        if cited:
            shown = format_stale(cited)
            render_s = max(0.0, time.perf_counter() - render_clock - model_s)
            return _timed(shown, raw, evidence, retrieve_s, model_s, render_s, False, invalid)
    shown = format_not_found(gaps) if released.kind == "not_found" else released.text
    render_s = max(0.0, time.perf_counter() - render_clock - model_s)
    return _timed(shown, raw, evidence, retrieve_s, model_s, render_s, False, invalid)


def _timed(shown, raw, evidence, retrieve_s, model_s, render_s, timeout, invalid=False) -> dict:
    return {
        "shown": shown,
        "raw": raw,
        "evidence": evidence,
        "retrieve_s": round(retrieve_s, 3),
        "model_s": round(model_s, 3),
        "render_s": round(render_s, 3),
        "timeout": timeout,
        "invalid": invalid,
    }


def _row_for_case(case: SuiteCase, split: str, pass_no: int, cold: bool, released: dict) -> dict:
    anchors = case.gold.anchors
    root = Path(index_dir(case.root_name))
    bodies = _bodies(root) if anchors else []
    index_has = bool(anchors) and all(_contains(bodies, anchor) for anchor in anchors)
    rank = _rank(case.question, root, anchors) if anchors else None
    evidence_texts = [item["text"] for item in released["evidence"]]
    evidence_has = bool(anchors) and all(_contains(evidence_texts, anchor) for anchor in anchors)
    shown = released["shown"]
    metric = "timeout" if released["timeout"] else metric_for(case.gold, shown)
    return {
        "id": case.id,
        "suite": "task1" if case.root_name == "task1" else "synthetic",
        "split": split,
        "category": case.category,
        "pass": pass_no,
        "cold": cold,
        "metric": metric,
        "scored": case.gold.scored,
        "index_has": index_has,
        "rank": rank,
        "evidence_has": evidence_has,
        "cited_has": anchors_covered(anchors, shown) if anchors else False,
        "fabrication": is_fabricated(shown),
        "invalid": released["invalid"],
        "timeout": released["timeout"],
        "retrieve_s": released["retrieve_s"],
        "model_s": released["model_s"],
        "render_s": released["render_s"],
        "shown_hash": _hash(shown),
        "judge": "",
        "trap_pass": False,
    }


def _prepare_trap(trap: Trap) -> Path:
    folder = RESULTS_DIR / "trap-docs" / trap.id
    root = RESULTS_DIR / "trap-index" / trap.id
    if folder.exists():
        shutil.rmtree(folder)
    if root.exists():
        shutil.rmtree(root)
    folder.mkdir(parents=True)
    for name, text in trap.files:
        if name in trap.selected:
            (folder / name).write_text(text)
    os.environ["LDI_ROOT"] = str(root)
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Index(path) as index, contextlib.redirect_stderr(io.StringIO()):
        ingest(folder, index)
    return root


def _row_for_trap(trap: Trap, pass_no: int, cold: bool, released: dict) -> dict:
    code = "timeout" if released["timeout"] else judge(trap, released["shown"])
    return {
        "id": trap.id,
        "suite": "trap",
        "split": trap.split,
        "category": "trap",
        "pass": pass_no,
        "cold": cold,
        "metric": code,
        "scored": True,
        "index_has": False,
        "rank": None,
        "evidence_has": False,
        "cited_has": False,
        "fabrication": is_fabricated(released["shown"]),
        "invalid": released["invalid"],
        "timeout": released["timeout"],
        "retrieve_s": released["retrieve_s"],
        "model_s": released["model_s"],
        "render_s": released["render_s"],
        "shown_hash": _hash(released["shown"]),
        "judge": code,
        "trap_pass": code in PASSING,
    }


def _trap_as_case(trap: Trap) -> SuiteCase:
    from eval.metrics import GoldView

    return SuiteCase(
        id=trap.id,
        category="trap",
        question=trap.question,
        gold=GoldView(abstain=False, anchors=(), scored=False),
        root_name=str(RESULTS_DIR / "trap-index" / trap.id),
    )


def run_passes(passes: int = PASSES) -> list[dict]:
    if PRODUCT_HITS != TOP_CHUNKS:
        raise SystemExit("PRODUCT_HITS must match the ask path's TOP_CHUNKS.")
    if not INDEX_ROOT.exists():
        raise SystemExit(f"Missing indexes at {INDEX_ROOT}")
    cases = synthetic_cases()
    task1 = task1_open_cases()
    splits = read_splits()
    traps = all_traps()
    spec = MODELS["1b"]
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    local = local_snapshot(spec.repo)
    if local is None:
        raise SystemExit("The 1b weights are not on disk. This run does not download them.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    facts_path = RESULTS_DIR / "facts.jsonl"
    raw_path = RESULTS_DIR / "raw.jsonl"
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    _cap_metal(mx, spec.metal_limit_bytes)
    load_clock = time.perf_counter()
    model, tokenizer = load(str(local), trust_remote_code=False)
    load_s = round(time.perf_counter() - load_clock, 3)
    print(json.dumps({"load_s": load_s}), flush=True)
    sampler = make_sampler(temp=0.0)
    rows: list[dict] = []
    try:
        with facts_path.open("w") as facts, raw_path.open("w") as raw:
            first = True
            for pass_no in range(1, passes + 1):
                for case in cases + task1:
                    released = _answer(case, mx, generate_step, model, tokenizer, sampler)
                    row = _row_for_case(
                        case,
                        splits.get(case.id, "task1"),
                        pass_no,
                        first,
                        released,
                    )
                    first = False
                    rows.append(row)
                    facts.write(json.dumps(row) + "\n")
                    facts.flush()
                    raw.write(json.dumps({"id": case.id, "pass": pass_no, "shown": released["shown"]}) + "\n")
                    print(
                        f"{case.id} pass {pass_no} {row['metric']} {row['model_s']}",
                        flush=True,
                    )
                for trap in traps:
                    _prepare_trap(trap)
                    stand_in = _trap_as_case(trap)
                    released = _answer_at_root(
                        stand_in,
                        RESULTS_DIR / "trap-index" / trap.id,
                        mx,
                        generate_step,
                        model,
                        tokenizer,
                        sampler,
                    )
                    row = _row_for_trap(trap, pass_no, False, released)
                    rows.append(row)
                    facts.write(json.dumps(row) + "\n")
                    facts.flush()
                    print(f"{trap.id} pass {pass_no} {row['judge']} {row['model_s']}", flush=True)
    finally:
        del model
        del tokenizer
    rows.append({"id": "_load", "suite": "meta", "pass": 0, "load_s": load_s})
    return rows


def _answer(case, mx, generate_step, model, tokenizer, sampler) -> dict:
    root = Path(index_dir(case.root_name))
    return _answer_at_root(case, root, mx, generate_step, model, tokenizer, sampler)


def _answer_at_root(case, root: Path, mx, generate_step, model, tokenizer, sampler) -> dict:
    def generate(reason: str) -> str:
        with contextlib.redirect_stdout(io.StringIO()):
            return _complete(
                mx,
                generate_step,
                model,
                tokenizer,
                sampler,
                case.question,
                _hits(case.question, root),
                reason,
            )

    return _release_case(case, root, generate)


def _hits(question: str, root: Path) -> list[Hit]:
    os.environ["LDI_ROOT"] = str(root)
    with Index(index_path()) as index:
        try:
            return index.search_question(question, limit=PRODUCT_HITS)
        except ValueError:
            return []


def summarise(rows: list[dict], chip: dict[str, object]) -> dict[str, object]:
    load_s = next((row.get("load_s") for row in rows if row["id"] == "_load"), None)
    rows = [row for row in rows if row["id"] != "_load"]
    first = [row for row in rows if row["pass"] == 1 and row["suite"] != "trap"]
    second = [row for row in rows if row["pass"] == 2 and row["suite"] != "trap"]
    by_id_pass: dict[tuple[str, int], str] = {}
    for row in rows:
        if row["suite"] == "trap":
            continue
        by_id_pass[(row["id"], row["pass"])] = row["shown_hash"]
    drifted = sorted(
        case_id
        for (case_id, pass_no), digest in by_id_pass.items()
        if pass_no == 1 and by_id_pass.get((case_id, 2)) not in {None, digest}
    )
    synthetic = [row for row in first if row["suite"] == "synthetic"]
    non_ocr = [row for row in synthetic if row["category"] != "OCR"]
    ocr = [row for row in synthetic if row["category"] == "OCR"]
    dev = [row for row in synthetic if row["split"] == "dev"]
    held = [row for row in synthetic if row["split"] == "heldout"]
    task1 = [row for row in first if row["suite"] == "task1"]
    traps_1 = [row for row in rows if row["suite"] == "trap" and row["pass"] == 1]
    non_ocr_correct = sum(1 for row in non_ocr if row["metric"] == "correct")
    task1_correct = sum(1 for row in task1 if row["metric"] == "correct")
    cold = [row for row in first if row["cold"]]
    warm = [row for row in first if not row["cold"]]
    dev_failing = failing(dev)
    task_failing = failing(task1)
    scoring_ids = [row["id"] for row in dev_failing + task_failing if row["stage"] == "scoring"]
    non_ocr_fn = [row for row in non_ocr if row["metric"] == "false_not_found"]
    task_fn = [row for row in task1 if row["metric"] == "false_not_found"]
    report: dict[str, object] = {
        "hardware": chip,
        "load_s": load_s,
        "passes": PASSES,
        "nondeterministic_ids": drifted,
        "latency_pass1_cold": latency_block(cold),
        "latency_pass1_warm": latency_block(warm),
        "latency_pass2_warm": latency_block(second),
        "synthetic_non_ocr": {
            "total": len(non_ocr),
            "claimed_total": CLAIMED_NON_OCR_TOTAL,
            "counts": count_rows(non_ocr),
            "by_category": by_category(non_ocr),
            "unscored_ids": sorted(row["id"] for row in non_ocr if row["metric"] == "unscored"),
            "not_correct_ids": sorted(
                row["id"] for row in non_ocr if row["metric"] not in {"correct", "unscored"}
            ),
        },
        "ocr": {
            "counts": count_rows(ocr),
            "ids": [
                {"id": row["id"], "metric": row["metric"]}
                for row in sorted(ocr, key=lambda item: item["id"])
            ],
        },
        "dev": {"counts": count_rows(dev), "by_category": by_category(dev)},
        "heldout": {"counts": count_rows(held), "by_category": by_category(held)},
        "task1_open": {
            "counts": count_rows(task1),
            "by_category": by_category(task1),
            "not_correct_ids": sorted(
                row["id"] for row in task1 if row["metric"] != "correct"
            ),
        },
        "traps": {
            "dev": trap_counts([row for row in traps_1 if row["split"] == "dev"]),
            "heldout": trap_counts([row for row in traps_1 if row["split"] == "heldout"]),
        },
        "diagnosis": dev_failing + task_failing,
        "scoring_ids": scoring_ids,
        "proposed_false_not_found": {
            "synthetic_non_ocr": false_not_found_target(non_ocr_fn),
            "task1_open": false_not_found_target(task_fn),
            "rule": "The target is the extraction-stage false_not_found count. Retrieval, gated, and generated misses are recoverable.",
        },
        "claimed": claimed_gap(non_ocr_correct, task1_correct),
    }
    return report


def main() -> None:
    chip = hardware()
    print(json.dumps({"hardware": chip}), flush=True)
    rows = run_passes()
    report = summarise(rows, chip)
    COUNTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["claimed"]), flush=True)
    print(json.dumps({"nondeterministic_ids": report["nondeterministic_ids"]}), flush=True)
    if report["scoring_ids"]:
        print(json.dumps({"scoring_ids": report["scoring_ids"]}), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
