"""Answer from indexed passages with one local MLX model."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from ldi.config import (
    MAX_KV_TOKENS,
    MAX_TOKENS,
    METAL_CACHE_BYTES,
    PREFILL_STEP,
    ModelSpec,
    gb,
    index_path,
    project_root,
)
from ldi.evidence import build_evidence, conflicting, fallback_release, format_conflict, format_evidence
from ldi.grammar import json_processor
from ldi.guard import GuardError, assess_load, machine_snapshot
from ldi.prompt import build_messages
from ldi.store import Hit, Index, sha256_file, stored_file
from ldi.verify import Release, format_not_found, release_answer
from ldi.weights import local_snapshot


def retry_instruction(reason: str) -> str:
    return (
        "The previous JSON failed a check. "
        f"{reason} "
        "Reply again with one JSON object using status, evidence, and answer. "
        "Choose only ids from the evidence list. "
        "Use only words, numbers, and dates from the chosen evidence."
    )


class ModelNotReady(RuntimeError):
    pass


def answer(question: str, model_key: str, top: int) -> None:
    from ldi.config import MODELS

    spec = MODELS[model_key]
    hits = _hits(question, top)
    gaps = _gaps()
    if not hits:
        print(format_not_found(gaps))
        return
    stale = stale_paths(hits)
    if stale:
        print(format_stale(stale))
        return
    items = build_evidence(question, hits)
    if not items:
        print(format_not_found(gaps))
        return
    print(format_evidence(items))
    print()
    local = local_snapshot(spec.repo)
    if local is None:
        command = "ldi setup" if spec.key == "3b" else f"ldi setup --model {spec.key}"
        raise ModelNotReady(
            f"The {spec.key} weights are not on disk. Run `{command}` once while online. "
            "That download is the only network step. The question stays on this Mac."
        )
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    _generate(spec, local, question, hits, gaps)


def format_stale(paths: list[str]) -> str:
    return "Stale\nThe indexed copy does not match the file on disk: " + ", ".join(paths)


def stale_paths(hits: list[Hit]) -> list[str]:
    project = project_root()
    found: list[str] = []
    with Index(index_path()) as index:
        for name in dict.fromkeys(hit.path for hit in hits):
            recorded = index.sha256_for(name)
            if recorded is None:
                continue
            disk = stored_file(name, project)
            if not disk.is_file() or sha256_file(disk) != recorded:
                found.append(name)
    return found


def settle_answer(generate, hits: list[Hit], question: str = "") -> tuple[str, Release]:
    """Generate once, and once more when the citation check withholds the answer."""
    items = build_evidence(question, hits)
    reply = generate("")
    released = release_answer(reply, hits, question)
    if released.kind == "retry":
        reply = generate(released.reason)
        released = release_answer(reply, hits, question)
        if released.kind == "retry":
            pair = conflicting(question, items)
            if pair is not None:
                released = Release(
                    "conflict",
                    format_conflict(pair[0], pair[1]),
                    paths=(pair[0].path, pair[1].path),
                )
            else:
                released = fallback_release(question, items)
    return reply, released


def _gaps() -> list[tuple[str, int | None, str]]:
    path = index_path()
    if not path.exists():
        return []
    with Index(path) as index:
        return index.list_gaps()


def _hits(question: str, top: int) -> list[Hit]:
    path = index_path()
    if not path.exists():
        raise ValueError("The index is empty. Run ldi ingest on a folder of documents.")
    with Index(path) as index:
        return index.search_question(question, limit=top)


def _generate(
    spec: ModelSpec,
    local: Path,
    question: str,
    hits: list[Hit],
    gaps: list[tuple[str, int | None, str]],
) -> None:
    # stream_generate raises the Metal wired limit. Stay on generate_step and
    # keep the allocator under the cap for this model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if platform.system() != "Darwin" or platform.machine() != "arm64":
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
        print(f"Loading {spec.key}. Metal is capped at {gb(spec.metal_limit_bytes)}.")
        try:
            model, tokenizer = load(str(local), trust_remote_code=False)
        except RuntimeError as exc:
            _raise_if_memory(exc, spec)
            raise
        sampler = make_sampler(temp=0.0)

        def generate(reason: str) -> str:
            try:
                return _complete(
                    mx,
                    generate_step,
                    model,
                    tokenizer,
                    sampler,
                    question,
                    hits,
                    reason,
                )
            except RuntimeError as exc:
                _raise_if_memory(exc, spec)
                raise

        _reply, released = settle_answer(generate, hits, question)
        if released.paths:
            cited = stale_paths([Hit(path, None, "") for path in released.paths])
            if cited:
                print()
                print(format_stale(cited))
                return
        _print_released(released, gaps)
    finally:
        del model
        del tokenizer
        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()


def _cap_metal(mx, limit: int) -> None:
    memory = getattr(mx, "set_memory_limit", None) or getattr(
        getattr(mx, "metal", None), "set_memory_limit", None
    )
    if memory is None:
        raise GuardError("This MLX build has no memory cap. The model stays unloaded.")
    memory(limit)
    cache = getattr(mx, "set_cache_limit", None) or getattr(
        getattr(mx, "metal", None), "set_cache_limit", None
    )
    if cache is None:
        raise GuardError("This MLX build has no cache cap. The model stays unloaded.")
    cache(METAL_CACHE_BYTES)


def _print_released(
    released: Release,
    gaps: list[tuple[str, int | None, str]],
) -> None:
    print()
    if released.kind in {"not_found", "retry"}:
        print(format_not_found(gaps))
        return
    print(released.text)


def _complete(mx, generate_step, model, tokenizer, sampler, question, hits, reason: str) -> str:
    messages = build_messages(question, hits)
    if reason:
        messages = [
            {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + retry_instruction(reason),
            },
            messages[1],
        ]
    templated = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt = _encode(tokenizer, templated)
    items = build_evidence(question, hits)
    eos_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    if not eos_ids and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids = [tokenizer.eos_token_id]
    eos_id = int(eos_ids[0]) if eos_ids else None
    processor = json_processor(
        tokenizer,
        int(prompt.shape[0]),
        {item.id for item in items},
        eos_id,
    )
    return _stream_tokens(
        mx,
        generate_step,
        model,
        tokenizer,
        prompt,
        sampler,
        processors=[processor],
        echo=False,
    )


def _encode(tokenizer, prompt: str):
    import mlx.core as mx

    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not str(prompt).startswith(str(bos))
    token_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    return mx.array(token_ids)


def _stream_tokens(
    mx,
    generate_step,
    model,
    tokenizer,
    prompt,
    sampler,
    *,
    processors=None,
    echo: bool = True,
) -> str:
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    eos = set(getattr(tokenizer, "eos_token_ids", []) or [])
    peak_fn = getattr(mx, "get_peak_memory", None)
    peak = 0.0
    saw_token = False
    step = {
        "max_tokens": MAX_TOKENS,
        "sampler": sampler,
        "max_kv_size": MAX_KV_TOKENS,
        "prefill_step_size": PREFILL_STEP,
    }
    if processors:
        step["logits_processors"] = processors
    for token, _logprobs in generate_step(prompt, model, **step):
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if token_id in eos:
            break
        saw_token = True
        detokenizer.add_token(token_id)
        piece = detokenizer.last_segment
        if echo and piece:
            print(piece, end="", flush=True)
        if peak_fn is not None:
            peak = max(peak, peak_fn() / 1e9)
    detokenizer.finalize()
    tail = detokenizer.last_segment
    if echo and tail:
        print(tail, end="", flush=True)
    if echo:
        print()
    if saw_token and peak_fn is not None:
        print(f"\nPeak memory for this answer: {peak:.2f} GB")
    elif echo and not saw_token:
        print("The model produced an empty answer.")
    text = detokenizer.text.strip() if hasattr(detokenizer, "text") else ""
    return text if saw_token else ""


def _raise_if_memory(exc: RuntimeError, spec: ModelSpec) -> None:
    text = str(exc).lower()
    if "memory" in text or "metal" in text:
        raise GuardError(
            f"MLX stopped inside the {gb(spec.metal_limit_bytes)} Metal cap. "
            "The model is unloaded. Free some memory and try again."
        ) from exc
