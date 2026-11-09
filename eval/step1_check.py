"""One offline Granite pass. Shown text is compared with the saved baseline.

This module writes eval/results/step1.jsonl. It does not rewrite raw.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True

from eval.config import RESULTS_DIR  # noqa: E402
from eval.goldload import synthetic_cases, task1_open_cases  # noqa: E402
from eval.metrics import is_fabricated, metric_for  # noqa: E402
from eval.rescore import FLOOR_PATH, pass1_shown  # noqa: E402
from eval.runner import _answer  # noqa: E402
from ldi.ask import _cap_metal  # noqa: E402
from ldi.config import MODELS  # noqa: E402
from ldi.guard import assess_load, machine_snapshot  # noqa: E402
from ldi.weights import local_snapshot  # noqa: E402

COUNTS_PATH = RESULTS_DIR.parent / "step1_counts.json"


def main() -> None:
    baseline = pass1_shown()
    cases = synthetic_cases() + task1_open_cases()
    missing = [case.id for case in cases if case.id not in baseline]
    if missing:
        raise SystemExit(f"Baseline has no pass-1 answer for {', '.join(missing)}")
    floor = set(json.loads(FLOOR_PATH.read_text()))
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
    sampler = make_sampler(temp=0.0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    differed: list[str] = []
    fabricated: list[str] = []
    floor_miss: list[str] = []
    try:
        with (RESULTS_DIR / "step1.jsonl").open("w") as handle:
            for case in cases:
                released = _answer(case, mx, generate_step, model, tokenizer, sampler)
                shown = released["shown"]
                same = shown == baseline[case.id]
                metric = metric_for(case.gold, shown)
                made_up = is_fabricated(shown)
                if not same:
                    differed.append(case.id)
                if made_up:
                    fabricated.append(case.id)
                if case.id in floor and metric != "correct":
                    floor_miss.append(case.id)
                handle.write(
                    json.dumps(
                        {
                            "id": case.id,
                            "same": same,
                            "metric": metric,
                            "fabrication": made_up,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                print(f"{case.id} {'same' if same else 'DIFF'} {metric}", flush=True)
    finally:
        del model
        del tokenizer
    report = {
        "cases": len(cases),
        "byte_identical": len(cases) - len(differed),
        "differed": differed,
        "floor": len(floor),
        "floor_still_passing": len(floor) - len(floor_miss),
        "floor_miss": floor_miss,
        "fabrication": len(fabricated),
        "fabricated_ids": fabricated,
    }
    COUNTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if differed or floor_miss or fabricated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
