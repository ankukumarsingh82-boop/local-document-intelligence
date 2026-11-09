"""Load the 1B model once, measure real memory, and answer two questions."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ldi.ask import _cap_metal, _encode, _raise_if_memory
from ldi.config import (
    MAX_KV_TOKENS,
    MODELS,
    PREFILL_STEP,
    PROBE_MAX_TOKENS,
    ModelSpec,
    catalog_cap_from_peak,
    gb,
    index_path,
)
from ldi.guard import Machine, assess_load, machine_snapshot, pressure_label
from ldi.prompt import build_messages, format_passages
from ldi.store import Hit, Index
from ldi.weights import download, local_snapshot

FACT_QUESTION = "What is the replacement seal part?"
MISSING_QUESTION = "What is the warranty period for the north pump?"


@dataclass(frozen=True)
class AnswerMeasure:
    question: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    generate_seconds: float
    metal_peak: int


def probe() -> int:
    spec = MODELS["1b"]
    before = machine_snapshot()
    _print_machine("Before", before)
    assess_load(spec, before.pressure, before.reclaimable, before.swap_used)

    local = local_snapshot(spec.repo)
    if local is None:
        local = download(spec)
    weight_bytes = _directory_bytes(local)
    print(f"Weights on disk: {gb(weight_bytes)} at {local}")

    fact_hits = _hits(FACT_QUESTION)
    if not fact_hits:
        raise ValueError(
            "No indexed passage matched the sample question. Run ldi ingest documents."
        )
    missing_hits = _hits(MISSING_QUESTION) or fact_hits

    rss_before = _resident_bytes()
    print(f"Process RSS before load: {gb(rss_before)}")
    started = time.perf_counter()
    measures, rss_loaded, peak_loaded, rss_after = _run(
        spec,
        local,
        ((FACT_QUESTION, fact_hits), (MISSING_QUESTION, missing_hits)),
    )
    load_seconds = time.perf_counter() - started
    after = machine_snapshot()
    _print_machine("After", after)
    _print_report(
        spec,
        weight_bytes,
        rss_before,
        rss_loaded,
        peak_loaded,
        rss_after,
        load_seconds,
        measures,
        before,
        after,
    )
    return catalog_cap_from_peak(max(peak_loaded, *(item.metal_peak for item in measures)))


def _hits(question: str) -> list[Hit]:
    path = index_path()
    if not path.exists():
        raise ValueError("The index is empty. Run ldi ingest documents.")
    with Index(path) as index:
        return index.search_question(question, limit=4)


def _run(
    spec: ModelSpec,
    local: Path,
    jobs: tuple[tuple[str, list[Hit]], ...],
) -> tuple[list[AnswerMeasure], int, int, int]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        from ldi.guard import GuardError

        raise GuardError("This model runs with MLX on this Apple Silicon Mac.")

    import gc

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model = None
    tokenizer = None
    try:
        _cap_metal(mx, spec.metal_limit_bytes)
        reset = getattr(mx, "reset_peak_memory", None)
        if reset is not None:
            reset()
        print(f"Loading {spec.key}. Metal is capped at {gb(spec.metal_limit_bytes)}.")
        load_started = time.perf_counter()
        try:
            model, tokenizer = load(str(local), trust_remote_code=False)
            params = model.parameters() if hasattr(model, "parameters") else model
            mx.eval(params)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
        except RuntimeError as exc:
            _raise_if_memory(exc, spec)
            raise
        load_seconds = time.perf_counter() - load_started
        rss_loaded = _resident_bytes()
        peak_loaded = _metal_peak(mx)
        print(
            f"Load took {load_seconds:.2f}s. "
            f"RSS {gb(rss_loaded)}. Metal peak {gb(peak_loaded)}."
        )
        sampler = make_sampler(temp=0.0)
        measures: list[AnswerMeasure] = []
        for question, hits in jobs:
            print()
            print(format_passages(hits))
            print()
            print(f"Question: {question}")
            try:
                measures.append(
                    _answer(mx, generate_step, model, tokenizer, sampler, question, hits)
                )
            except RuntimeError as exc:
                _raise_if_memory(exc, spec)
                raise
        rss_after = _resident_bytes()
        return measures, rss_loaded, peak_loaded, rss_after
    finally:
        del model
        del tokenizer
        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()


def _answer(mx, generate_step, model, tokenizer, sampler, question: str, hits: list[Hit]) -> AnswerMeasure:
    messages = build_messages(question, hits)
    templated = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt = _encode(tokenizer, templated)
    prompt_tokens = int(prompt.size)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    eos = set(getattr(tokenizer, "eos_token_ids", []) or [])
    generated = 0
    prefill_seconds = 0.0
    started = time.perf_counter()
    first_token_at = 0.0
    for token, _logprobs in generate_step(
        prompt,
        model,
        max_tokens=PROBE_MAX_TOKENS,
        sampler=sampler,
        max_kv_size=MAX_KV_TOKENS,
        prefill_step_size=PREFILL_STEP,
    ):
        now = time.perf_counter()
        if generated == 0:
            first_token_at = now
            prefill_seconds = now - started
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if token_id in eos:
            break
        generated += 1
        detokenizer.add_token(token_id)
        piece = detokenizer.last_segment
        if piece:
            print(piece, end="", flush=True)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    finished = time.perf_counter()
    detokenizer.finalize()
    tail = detokenizer.last_segment
    if tail:
        print(tail, end="", flush=True)
    print()
    generate_seconds = max(finished - (first_token_at or finished), 0.0)
    text = detokenizer.text if hasattr(detokenizer, "text") else ""
    return AnswerMeasure(
        question=question,
        text=text.strip(),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        prefill_seconds=prefill_seconds,
        generate_seconds=generate_seconds,
        metal_peak=_metal_peak(mx),
    )


def _print_report(
    spec: ModelSpec,
    weight_bytes: int,
    rss_before: int,
    rss_loaded: int,
    peak_loaded: int,
    rss_after: int,
    load_seconds: float,
    measures: list[AnswerMeasure],
    before: Machine,
    after: Machine,
) -> None:
    overall_peak = max([peak_loaded, *(item.metal_peak for item in measures)], default=0)
    suggested = catalog_cap_from_peak(overall_peak)
    print()
    print("Measured")
    print(f"  Model: {spec.label}")
    print(f"  Quantization: 4-bit affine, group size 64")
    print(f"  Weights on disk: {gb(weight_bytes)}")
    print(f"  Metal ceiling for this run: {gb(spec.metal_limit_bytes)}")
    print(f"  Process RSS before load: {gb(rss_before)}")
    print(f"  Process RSS after weights evaluated: {gb(rss_loaded)}")
    print(f"  Process RSS after answers: {gb(rss_after)}")
    print(f"  Metal peak after weights evaluated: {gb(peak_loaded)}")
    print(f"  Metal peak after answers: {gb(overall_peak)}")
    print(f"  Catalog cap from this peak: {gb(suggested)}")
    print(f"  Load and answers wall time: {load_seconds:.2f}s")
    print(
        f"  Memory pressure: {pressure_label(before.pressure)} before, "
        f"{pressure_label(after.pressure)} after"
    )
    print(
        f"  Reclaimable: {_maybe_gb(before.reclaimable)} before, "
        f"{_maybe_gb(after.reclaimable)} after"
    )
    print(
        f"  Swap: {_maybe_gb(before.swap_used)} before, {_maybe_gb(after.swap_used)} after"
    )
    for item in measures:
        rate = item.generated_tokens / item.generate_seconds if item.generate_seconds else 0.0
        print()
        print(f"  Question: {item.question}")
        print(f"  Prompt tokens: {item.prompt_tokens}")
        print(f"  Generated tokens: {item.generated_tokens}")
        print(f"  Prefill: {item.prefill_seconds:.2f}s")
        print(f"  Generation: {item.generate_seconds:.2f}s, {rate:.1f} tokens/s")
        print(f"  Metal peak: {gb(item.metal_peak)}")
        print(f"  Answer: {item.text}")


def _print_machine(label: str, snap: Machine) -> None:
    print(
        f"{label}: pressure {pressure_label(snap.pressure)}, "
        f"reclaimable {_maybe_gb(snap.reclaimable)}, "
        f"swap {_maybe_gb(snap.swap_used)}"
    )


def _maybe_gb(value: int | None) -> str:
    if value is None:
        return "unreadable"
    return gb(value)


def _resident_bytes() -> int:
    raw = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        text=True,
    )
    return int(raw.strip().split()[0]) * 1024


def _metal_peak(mx) -> int:
    peak = getattr(mx, "get_peak_memory", None)
    if peak is None:
        return 0
    return int(peak())


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
