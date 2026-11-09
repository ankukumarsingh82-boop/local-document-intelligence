"""Talk with one local model. The model loads once and unloads on exit."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from ldi.ask import _cap_metal, _encode, _raise_if_memory
from ldi.config import (
    CHAT_MODEL,
    MAX_KV_TOKENS,
    MODELS,
    PREFILL_STEP,
    ModelSpec,
    gb,
)
from ldi.guard import GuardError, assess_load, machine_snapshot
from ldi.weights import download, local_snapshot

CHAT_MAX_TOKENS = 256
CHAT_HISTORY = 6
SYSTEM = "You are a helpful assistant. Keep answers short and direct."


def chat(model_key: str = CHAT_MODEL, once: str | None = None) -> None:
    spec = MODELS[model_key]
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    local = local_snapshot(spec.repo)
    if local is None:
        local = download(spec)
    _session(spec, local, once)


def trim_history(messages: list[dict[str, str]], limit: int = CHAT_HISTORY) -> list[dict[str, str]]:
    system = [message for message in messages if message["role"] == "system"][:1]
    rest = [message for message in messages if message["role"] != "system"]
    return system + rest[-limit:]


def _session(spec: ModelSpec, local: Path, once: str | None) -> None:
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
        print(f"Loading {spec.label}. Metal is capped at {gb(spec.metal_limit_bytes)}.")
        try:
            model, tokenizer = load(str(local), trust_remote_code=False)
        except RuntimeError as exc:
            _raise_if_memory(exc, spec)
            raise
        sampler = make_sampler(temp=0.0)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
        if once is not None:
            _speak(spec, mx, generate_step, model, tokenizer, sampler, messages, once)
            return
        print(f"{spec.label}. The chat stays on this Mac. Exit with /quit.")
        while True:
            try:
                line = input("You: ")
            except EOFError:
                print()
                break
            text = line.strip()
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break
            try:
                _speak(spec, mx, generate_step, model, tokenizer, sampler, messages, text)
            except GuardError as exc:
                print(exc)
                messages[:] = trim_history(messages, limit=2)
                print("Earlier turns were dropped to stay inside the Metal cap.")
    finally:
        del model
        del tokenizer
        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()
        print("The model is unloaded.")


def _speak(spec, mx, generate_step, model, tokenizer, sampler, messages, text: str) -> None:
    messages.append({"role": "user", "content": text})
    messages[:] = trim_history(messages)
    try:
        reply = _reply(mx, generate_step, model, tokenizer, sampler, messages)
    except RuntimeError as exc:
        messages.pop()
        _raise_if_memory(exc, spec)
        raise
    if not reply:
        messages.pop()
        print("The model produced an empty answer.")
        return
    messages.append({"role": "assistant", "content": reply})
    print()


def _reply(mx, generate_step, model, tokenizer, sampler, messages) -> str:
    templated = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt = _encode(tokenizer, templated)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    eos = set(getattr(tokenizer, "eos_token_ids", []) or [])
    print("Model: ", end="", flush=True)
    for token, _logprobs in generate_step(
        prompt,
        model,
        max_tokens=CHAT_MAX_TOKENS,
        sampler=sampler,
        max_kv_size=MAX_KV_TOKENS,
        prefill_step_size=PREFILL_STEP,
    ):
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if token_id in eos:
            break
        detokenizer.add_token(token_id)
        piece = detokenizer.last_segment
        if piece:
            print(piece, end="", flush=True)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    detokenizer.finalize()
    tail = detokenizer.last_segment
    if tail:
        print(tail, end="", flush=True)
    print()
    text = detokenizer.text.strip() if hasattr(detokenizer, "text") else ""
    return text
