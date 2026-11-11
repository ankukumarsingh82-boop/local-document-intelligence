"""One local chat: a single loaded model, its prompt, and attached text."""

from __future__ import annotations

import gc
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ldi.ask import _cap_metal, _encode, _raise_if_memory
from ldi.chat import CHAT_HISTORY, CHAT_MAX_TOKENS, SYSTEM, trim_history
from ldi.config import (
    CHAT_MODEL,
    DENSE_ENABLED,
    MAX_KV_TOKENS,
    MODELS,
    PREFILL_STEP,
    REDUCED_RETRIEVAL_MODE,
    SUPPORTED_SUFFIXES,
    TEXT_SUFFIXES,
    ModelSpec,
    gb,
)
from ldi.guard import GuardError, load_block_reason, machine_snapshot, pressure_label
from ldi import config
from ldi.ingest import read_file, text_from_upload
from ldi.ocr import recognize
from ldi.verify import generation_system, verify
from ldi.weights import local_snapshot

# The attached text itself stops at 2,048 tokens. The template wrapper and the
# short thread sit on top of that, and the cache has to hold the whole prompt
# or the document at the front is discarded. 4,096 is the ceiling for a turn.
PROMPT_CEILING = 4096
MAX_OUTPUT_TOKENS = 1024


@dataclass
class Generation:
    """Knobs for the next reply. Temperature 0 picks the most likely word."""

    temperature: float = 0.0
    max_tokens: int = CHAT_MAX_TOKENS
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    document_tokens: int = MAX_KV_TOKENS
    history_messages: int = CHAT_HISTORY

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "document_tokens": self.document_tokens,
            "history_messages": self.history_messages,
        }

_SYSTEM_MARKERS = (
    ("<|start_of_role|>system<|end_of_role|>", "<|end_of_text|>"),
    ("<|im_start|>system\n", "<|im_end|>"),
    ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
)


def uses_documents(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None) or ""
    return "documents" in template


def document_kwargs(tokenizer, passages: list[dict]) -> dict:
    if not passages or not uses_documents(tokenizer):
        return {}
    return {
        "documents": [
            {"title": item["title"], "text": item["text"]} for item in passages
        ]
    }


def question_text(question: str, passages: list[dict], tokenizer) -> str:
    if not passages or uses_documents(tokenizer):
        return question
    attached = "\n\n".join(f"{item['title']}\n{item['text']}" for item in passages)
    return f"{attached}\n\n{question}"


def _template_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in messages]


def system_text(rendered: str) -> str:
    for start, end in _SYSTEM_MARKERS:
        if start in rendered:
            return rendered.split(start, 1)[1].split(end, 1)[0]
    return rendered


def render_system(tokenizer, base: str, passages: list[dict]) -> str:
    messages = [
        {"role": "system", "content": base},
        {"role": "user", "content": "Hello"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
        **document_kwargs(tokenizer, passages),
    )
    return system_text(str(rendered))


def clip_documents(
    files: list[tuple[str, str]],
    limit: int,
    encode=None,
    decode=None,
) -> tuple[list[dict], str]:
    """Keep at most `limit` tokens. Extra text is cut from the end."""
    if limit < 1:
        raise ValueError("The token limit must be at least 1.")
    kept: list[dict] = []
    used = 0
    notes: list[str] = []
    estimated = encode is None
    for name, text in files:
        if estimated:
            total = max(1, (len(text) + 3) // 4)
            piece, taken, note = _clip_estimate(name, text, total, limit - used)
        else:
            ids = _ids(encode(text))
            piece, taken, note = _clip_ids(name, text, ids, limit - used, decode)
        if note:
            notes.append(note)
        if taken <= 0:
            continue
        used += taken
        kept.append({"title": name, "text": piece, "tokens": taken})
    if estimated and files:
        notes.append("Token counts are estimated until the weights can be read.")
    return kept, " ".join(notes)


def _clip_estimate(name: str, text: str, total: int, room: int) -> tuple[str, int, str]:
    if room <= 0:
        return "", 0, f"{name} was left out. The token limit was already full."
    if total > room:
        return (
            text[: room * 4],
            room,
            f"{name} kept about {room} of {total} tokens. The rest was cut from the end.",
        )
    return text, total, ""


def _clip_ids(
    name: str,
    text: str,
    ids: list[int],
    room: int,
    decode,
) -> tuple[str, int, str]:
    total = len(ids)
    if total == 0:
        return "", 0, f"{name} has no tokens."
    if room <= 0:
        return "", 0, f"{name} was left out. The token limit was already full."
    if total > room:
        return (
            decode(ids[:room]),
            room,
            f"{name} kept {room} of {total} tokens. The rest was cut from the end.",
        )
    return text, total, ""


def _ids(encoded) -> list[int]:
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return [int(item) for item in encoded]


class Desk:
    """Holds at most one model. Ingest and search never come through here."""

    def __init__(self) -> None:
        self.base = SYSTEM
        self.generation = Generation()
        self.files: list[tuple[str, str]] = []
        self.history: list[dict[str, str]] = []
        self.model = None
        self.tokenizer = None
        self.spec: ModelSpec | None = None
        self._mx = None
        self._generate_step = None
        self._sampler = None
        self._make_sampler = None
        self._preview: dict[str, object] = {}
        self._preview_error = ""

    def state(self) -> dict:
        snap = machine_snapshot()
        tokenizer, key = self._reading_tokenizer()
        passages, kept = self._passages(tokenizer)
        rendered = self.base
        uses = False
        if tokenizer is not None:
            uses = uses_documents(tokenizer)
            try:
                rendered = render_system(tokenizer, self.base, passages)
            except Exception as exc:
                kept = f"{kept} The template could not be rendered. {exc}".strip()
        note = _template_note(key, uses, bool(passages))
        if self._preview_error and key is None:
            note = self._preview_error
        return {
            "base": self.base,
            "rendered": rendered,
            "template_note": note,
            "kept": kept,
            "passages": passages,
            "uses_documents": uses,
            "template_model": key,
            "loaded": self.spec.key if self.spec is not None else None,
            "messages": list(self.history),
            "generation": self.generation.as_dict(),
            "memory": pressure_label(snap.pressure),
            "retrieval_mode": "" if DENSE_ENABLED else REDUCED_RETRIEVAL_MODE,
            "meaning_check": config.meaning_check_state(),
            "models": self._models(snap),
        }

    def set_prompt(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("The system prompt needs a sentence.")
        if len(cleaned) > 2000:
            raise ValueError("Keep the system prompt to a short paragraph.")
        self.base = cleaned

    def set_generation(self, payload: dict) -> None:
        current = self.generation
        self.generation = Generation(
            temperature=_setting(payload, "temperature", current.temperature, 0, 2, "Temperature"),
            max_tokens=int(
                _setting(
                    payload,
                    "max_tokens",
                    current.max_tokens,
                    1,
                    MAX_OUTPUT_TOKENS,
                    "Output tokens",
                    whole=True,
                )
            ),
            top_p=_setting(payload, "top_p", current.top_p, 0, 1, "Top-p"),
            top_k=int(
                _setting(payload, "top_k", current.top_k, 0, 200, "Top-k", whole=True)
            ),
            min_p=_setting(payload, "min_p", current.min_p, 0, 1, "Min-p"),
            document_tokens=int(
                _setting(
                    payload,
                    "document_tokens",
                    current.document_tokens,
                    64,
                    MAX_KV_TOKENS,
                    "Attached text",
                    whole=True,
                )
            ),
            history_messages=int(
                _setting(
                    payload,
                    "history_messages",
                    current.history_messages,
                    0,
                    24,
                    "Messages kept",
                    whole=True,
                )
            ),
        )
        self._apply_sampler()

    def attach_payload(self, payload: dict) -> None:
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Name the file.")
        encoded = payload.get("file_base64", payload.get("pdf_base64"))
        if encoded:
            import base64

            try:
                data = base64.b64decode(str(encoded), validate=False)
            except Exception as exc:
                raise ValueError("That file could not be read.") from exc
            self._attach_bytes(name, data)
            return
        if "text" in payload:
            data = str(payload.get("text", "")).encode("utf-8")
        else:
            raise ValueError("The file had no text.")
        self.attach_text(name, text_from_upload(name, data))

    def _attach_bytes(self, name: str, data: bytes) -> None:
        suffix = Path(name).suffix.lower()
        if suffix in TEXT_SUFFIXES:
            self.attach_text(name, text_from_upload(name, data))
            return
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / Path(name).name
            path.write_bytes(data)
            pages, _gaps, scans = read_file(path)
            if scans:
                key = self.spec.key if self.spec is not None else None
                if key is not None:
                    self.unload()
                try:
                    texts = recognize(scans)
                finally:
                    if key is not None:
                        self.load(key)
                for scan, text in zip(scans, texts):
                    if text:
                        pages.append((scan.page, text))
            parts = [text for _page, text in pages if text.strip()]
            if not parts:
                raise ValueError(f"{Path(name).name} has no text.")
            self.attach_text(name, "\n\n".join(parts))

    def attach_text(self, name: str, text: str) -> None:
        safe = Path(name).name
        if Path(safe).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Attach a .txt, .md, .pdf, or image file.")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError(f"{safe} has no text.")
        self.files = [item for item in self.files if item[0] != safe]
        self.files.append((safe, cleaned))

    def detach(self, name: str) -> None:
        safe = Path(name).name
        kept = [item for item in self.files if item[0] != safe]
        if len(kept) == len(self.files):
            raise ValueError(f"{safe} is not attached.")
        self.files = kept

    def clear(self) -> None:
        self.history.clear()

    def load(self, key: str) -> None:
        if key not in MODELS:
            raise ValueError("That model is not available.")
        spec = MODELS[key]
        local = local_snapshot(spec.repo)
        if local is None:
            raise ValueError(_missing_weights(spec))
        snap = machine_snapshot()
        assess_reason = load_block_reason(
            spec, snap.pressure, snap.reclaimable, snap.swap_used
        )
        if assess_reason is not None:
            raise GuardError(assess_reason)
        if self.model is not None:
            self.unload()
        self._ensure_runtime()
        _cap_metal(self._mx, spec.metal_limit_bytes)
        try:
            model, tokenizer = self._load_weights(str(local))
        except RuntimeError as exc:
            _raise_if_memory(exc, spec)
            raise
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self._apply_sampler()
        self._preview.clear()
        print(f"Loaded {spec.label}. Metal is capped at {gb(spec.metal_limit_bytes)}.")

    def unload(self) -> None:
        had_model = self.model is not None or self.tokenizer is not None
        self.model = None
        self.tokenizer = None
        self.spec = None
        self._sampler = None
        self._preview.clear()
        gc.collect()
        if self._mx is not None:
            clear = getattr(self._mx, "clear_cache", None) or getattr(
                getattr(self._mx, "metal", None), "clear_cache", None
            )
            if clear is not None:
                clear()
        if had_model:
            print("Unloaded the model.")

    def ask(self, text: str) -> None:
        question = text.strip()
        if not question:
            raise ValueError("Write a message first.")
        if self.model is None or self.tokenizer is None or self.spec is None:
            raise GuardError("Load a model before sending a question.")
        passages, _kept = self._passages(self.tokenizer)
        history = list(self.history)
        prompt = None
        messages: list[dict[str, str]] = []
        system = generation_system(self.base, bool(passages))
        while True:
            user = question_text(question, passages, self.tokenizer)
            messages = self._trim(
                [
                    {"role": "system", "content": system},
                    *history,
                    {"role": "user", "content": user},
                ]
            )
            try:
                templated = self.tokenizer.apply_chat_template(
                    _template_messages(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                    **document_kwargs(self.tokenizer, passages),
                )
            except TypeError:
                if not passages or not uses_documents(self.tokenizer):
                    raise
                user = question_text(question, passages, _NeverDocuments())
                messages = self._trim(
                    [
                        {"role": "system", "content": system},
                        *history,
                        {"role": "user", "content": user},
                    ]
                )
                templated = self.tokenizer.apply_chat_template(
                    _template_messages(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                )
            prompt = _encode(self.tokenizer, str(templated))
            length = _prompt_length(prompt)
            if length <= PROMPT_CEILING:
                break
            if history:
                history = []
                continue
            raise GuardError(
                "That message and the attached text are longer than this Mac will send. "
                "Remove a file or clear the chat."
            )
        try:
            reply = self._complete(
                prompt,
                max(length + self.generation.max_tokens, self.generation.document_tokens),
            )
        except RuntimeError as exc:
            try:
                _raise_if_memory(exc, self.spec)
            except GuardError:
                self.unload()
                raise
            raise
        if not reply:
            raise ValueError("The model produced an empty answer.")
        source = "\n\n".join(item["text"] for item in passages)
        stored = [item for item in messages if item["role"] != "system"]
        if stored and stored[-1]["role"] == "user":
            stored[-1] = {"role": "user", "content": question}
        stored.append(
            {
                "role": "assistant",
                "content": reply,
                "checks": [item.as_dict() for item in verify(reply, source)],
            }
        )
        self.history = [
            item
            for item in self._trim([{"role": "system", "content": self.base}, *stored])
            if item["role"] != "system"
        ]

    def _passages(self, tokenizer) -> tuple[list[dict], str]:
        if tokenizer is None:
            return clip_documents(self.files, self.generation.document_tokens)
        return clip_documents(
            self.files,
            self.generation.document_tokens,
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            tokenizer.decode,
        )

    def _models(self, snap) -> list[dict]:
        loaded = self.spec.key if self.spec is not None else None
        rows = []
        for spec in MODELS.values():
            on_disk = local_snapshot(spec.repo) is not None
            reason = None if not on_disk else load_block_reason(
                spec, snap.pressure, snap.reclaimable, snap.swap_used
            )
            if not on_disk:
                detail = _missing_weights(spec)
            elif spec.key == loaded:
                detail = "Weights are on disk."
            elif reason is not None:
                detail = reason
            else:
                detail = "Weights are on disk."
            rows.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "metal": gb(spec.metal_limit_bytes),
                    "on_disk": on_disk,
                    "loaded": spec.key == loaded,
                    "can_load": on_disk and reason is None and spec.key != loaded,
                    "detail": detail,
                }
            )
        return rows

    def _reading_tokenizer(self):
        if self.tokenizer is not None and self.spec is not None:
            return self.tokenizer, self.spec.key
        return self._preview_tokenizer()

    def _preview_tokenizer(self):
        key = _preview_key()
        if key is None:
            return None, None
        if key not in self._preview:
            try:
                _offline()
                from mlx_lm.utils import load_tokenizer

                local = local_snapshot(MODELS[key].repo)
                self._preview[key] = load_tokenizer(str(local))
                self._preview_error = ""
            except Exception as exc:
                self._preview_error = f"The chat template could not be read. {exc}"
                return None, None
        return self._preview[key], key

    def _ensure_runtime(self) -> None:
        _offline()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise GuardError("This model runs with MLX on this Apple Silicon Mac.")
        if self._mx is None:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            self._mx = mx
            self._load_weights = lambda path: load(path, trust_remote_code=False)
            self._generate_step = generate_step
            self._make_sampler = make_sampler

    def _apply_sampler(self) -> None:
        if self._make_sampler is None:
            return
        settings = self.generation
        self._sampler = self._make_sampler(
            temp=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            min_p=settings.min_p,
        )

    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        limit = self.generation.history_messages
        if limit <= 0:
            return [message for message in messages if message["role"] == "system"][:1]
        return trim_history(messages, limit)

    def _complete(self, prompt, kv_size: int, logits_processors=None) -> str:
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        eos = set(getattr(self.tokenizer, "eos_token_ids", []) or [])
        for token, _logprobs in self._generate_step(
            prompt,
            self.model,
            max_tokens=self.generation.max_tokens,
            sampler=self._sampler,
            logits_processors=logits_processors,
            max_kv_size=kv_size,
            prefill_step_size=PREFILL_STEP,
        ):
            token_id = int(token.item()) if hasattr(token, "item") else int(token)
            if token_id in eos:
                break
            detokenizer.add_token(token_id)
        if hasattr(self._mx, "synchronize"):
            self._mx.synchronize()
        detokenizer.finalize()
        if hasattr(detokenizer, "text"):
            return detokenizer.text.strip()
        return ""


class _NeverDocuments:
    chat_template = ""


def _template_note(key: str | None, uses: bool, has_files: bool) -> str:
    if key is None:
        return "The full template text appears when weights are on disk."
    label = MODELS[key].label
    if has_files and uses:
        return (
            f"{label} adds its document instruction and the attached text after the base sentence. "
            "That addition comes from the chat template."
        )
    if has_files:
        return f"{label} puts the attached text in the question."
    return f"{label} receives the base sentence."


def _missing_weights(spec: ModelSpec) -> str:
    if spec.key == "3b":
        return "No weights on disk yet. ldi setup downloads them."
    return f"No weights on disk yet. ldi setup --model {spec.key} downloads them."


def _preview_key() -> str | None:
    if local_snapshot(MODELS[CHAT_MODEL].repo) is not None:
        return CHAT_MODEL
    for spec in MODELS.values():
        if local_snapshot(spec.repo) is not None:
            return spec.key
    return None


def _offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _prompt_length(prompt) -> int:
    shape = getattr(prompt, "shape", None)
    if shape:
        return int(shape[-1])
    return len(prompt)


def _setting(payload, key: str, default, low, high, label: str, *, whole: bool = False):
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} needs a number.")
    if whole and int(value) != value:
        raise ValueError(f"{label} needs a whole number.")
    number = int(value) if whole else float(value)
    if number < low or number > high:
        raise ValueError(f"{label} must be from {low} to {high}.")
    return number
