"""Run the five skill conditions once, then unload the model."""

from __future__ import annotations

import gc
import os
import platform

from ldi.ask import _cap_metal, _encode, _raise_if_memory
from ldi.cards import CARD, CONDITIONS, EXAMPLE, budget_problem, system_text
from ldi.chat import SYSTEM
from ldi.config import MODELS, PREFILL_STEP, gb
from ldi.desk import PROMPT_CEILING
from ldi.gold import CASES, GoldCase
from ldi.grammar import checks_processor
from ldi.guard import GuardError, assess_load, machine_snapshot
from ldi.score import Score, format_score, score_case
from ldi.weights import local_snapshot

EVAL_MODEL = "1b"
EVAL_MAX_TOKENS = 256


def exceeds_ceiling(token_count: int, ceiling: int = PROMPT_CEILING) -> bool:
    return token_count > ceiling


def user_text(case: GoldCase) -> str:
    return f"{case.passage.strip()}\n\nQuestion: {case.question}"


def evaluate() -> list[str]:
    spec = MODELS[EVAL_MODEL]
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    local = local_snapshot(spec.repo)
    if local is None:
        raise GuardError(
            "The 1b weights are not on disk. Run `ldi setup --model 1b` once while online. "
            "This measurement does not download them."
        )
    return _run(spec, local)


def _run(spec, local) -> list[str]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise GuardError("This model runs with MLX on this Apple Silicon Mac.")

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
        problem = budget_problem(
            _count(tokenizer, CARD),
            _count(tokenizer, EXAMPLE),
        )
        if problem is not None:
            raise GuardError(problem)
        sampler = make_sampler(temp=0.0, top_p=0.0, top_k=0, min_p=0.0)
        lines = [
            _condition(condition, model, tokenizer, sampler, mx, generate_step, spec)
            for condition in CONDITIONS
        ]
        for line in lines:
            print(line)
        return lines
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


def _condition(condition, model, tokenizer, sampler, mx, generate_step, spec) -> str:
    score = Score()
    system = system_text(condition, SYSTEM)
    for case in CASES:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text(case)},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = _encode(tokenizer, str(templated))
        length = int(prompt.shape[0]) if hasattr(prompt, "shape") else len(prompt)
        if exceeds_ceiling(length):
            score.not_run += 1
            continue
        processors = None
        if condition == "grammar":
            eos_ids = getattr(tokenizer, "eos_token_ids", None) or []
            if isinstance(eos_ids, int):
                eos = eos_ids
            elif eos_ids:
                eos = int(next(iter(eos_ids)))
            else:
                eos = None
            processors = [checks_processor(tokenizer, length, eos)]
        try:
            reply = _complete(
                mx,
                generate_step,
                model,
                tokenizer,
                prompt,
                sampler,
                processors,
            )
        except RuntimeError as exc:
            _raise_if_memory(exc, spec)
            raise
        score.add(score_case(case, reply))
    return format_score(condition, score)


def _complete(mx, generate_step, model, tokenizer, prompt, sampler, processors) -> str:
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    eos = set(getattr(tokenizer, "eos_token_ids", []) or [])
    for token, _logprobs in generate_step(
        prompt,
        model,
        max_tokens=EVAL_MAX_TOKENS,
        sampler=sampler,
        logits_processors=processors,
        max_kv_size=PROMPT_CEILING,
        prefill_step_size=PREFILL_STEP,
    ):
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if token_id in eos:
            break
        detokenizer.add_token(token_id)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    detokenizer.finalize()
    if hasattr(detokenizer, "text"):
        return detokenizer.text.strip()
    return ""


def _count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
