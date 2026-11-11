"""Score a changed pair with Gemma 3 270M hidden states plus a trained probe.

Gemma never generates text here. The probe reads one hidden state and returns a
score. The score can only add a "meaning may have changed" flag. It never edits
a passage and never touches a citation.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np

_ASSETS = Path(__file__).parent / "assets"
_PROBE = "probe_legal_v1.npz"
_CONFIG = "frozen_config.json"
_LAYER = 9
_HIDDEN = 640

_MISSING = (
    "Gemma 3 270M is not on this Mac, so the meaning check stays off. "
    "Download it once with `python scripts/download_gemma.py`."
)


class MeaningFlagger:
    """One loaded Gemma model and the frozen probe that reads its hidden state."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"{_MISSING} Expected the weights at {self.model_dir}.")
        config = json.loads((_ASSETS / _CONFIG).read_text())
        self._verify_probe(config["probe_sha256"])
        self.prompt = config["prompt"]
        self.tau = float(config["tau"])
        probe = np.load(_ASSETS / _PROBE)
        self._mu = np.asarray(probe["mu"], dtype=np.float64)
        self._sd = np.asarray(probe["sd"], dtype=np.float64)
        self._w = np.asarray(probe["w"], dtype=np.float64)
        self._b = float(np.asarray(probe["b"], dtype=np.float64).reshape(-1)[0])
        self._sp, self._model = _load_model(self.model_dir)

    def score(self, old: str, new: str) -> float:
        """The probe score for one pair. The same pair always returns the same number."""
        from gemma_flagger.assets import reference_rules

        old_text = unicodedata.normalize("NFC", old)
        new_text = unicodedata.normalize("NFC", new)
        hidden = self._hidden(self.prompt.format(old=old_text, new=new_text))
        rule_flag, _detail = reference_rules.classify(old_text, new_text)
        keyword_flag, _keyword_detail = reference_rules.classify(
            old_text,
            new_text,
            reference_rules.CLOSED | reference_rules.EXT,
            numwords=True,
        )
        features = np.concatenate(
            [
                hidden,
                np.array(
                    [rule_flag == "flagged", keyword_flag == "flagged"],
                    dtype=np.float64,
                ),
            ]
        )
        return float(((features - self._mu) / self._sd) @ self._w + self._b)

    def check(self, old: str, new: str) -> dict:
        """Score one pair. ``needs_check`` is only ever added, never removed."""
        from gemma_flagger.assets import reference_rules

        old_text = unicodedata.normalize("NFC", old)
        new_text = unicodedata.normalize("NFC", new)
        rule_flag, _detail = reference_rules.classify(old_text, new_text)
        keyword_flag, _keyword_detail = reference_rules.classify(
            old_text,
            new_text,
            reference_rules.CLOSED | reference_rules.EXT,
            numwords=True,
        )
        scored = self.score(old, new)
        probe_flag = scored >= self.tau
        return {
            "score": scored,
            "probe_flag": probe_flag,
            "rule_flag": rule_flag == "flagged",
            "kw_flag": keyword_flag == "flagged",
            "needs_check": rule_flag == "flagged" or keyword_flag == "flagged" or probe_flag,
        }

    def _hidden(self, text: str) -> np.ndarray:
        import torch

        bos = self._sp.piece_to_id("<bos>")
        start = self._sp.piece_to_id("<start_of_turn>")
        end = self._sp.piece_to_id("<end_of_turn>")
        ids = (
            [bos, start]
            + self._sp.encode("user\n" + text)
            + [end]
            + self._sp.encode("\n")
            + [start]
            + self._sp.encode("model\n")
        )
        with torch.inference_mode():
            output = self._model(
                input_ids=torch.tensor([ids]),
                output_hidden_states=True,
            )
        hidden = output.hidden_states[_LAYER][0, -1].detach().to(torch.float32).cpu().numpy()
        return np.asarray(hidden, dtype=np.float64)

    @staticmethod
    def _verify_probe(expected: str) -> None:
        digest = hashlib.sha256((_ASSETS / _PROBE).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{_PROBE} does not match the probe this flagger was built for. "
                f"Expected sha256 {expected}, found {digest}."
            )


def _load_model(model_dir: Path):
    """Load the tokenizer and the model on CPU. The model never generates."""
    import sentencepiece as spm
    from transformers import AutoModelForCausalLM

    tokenizer = model_dir / "tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"{_MISSING} {tokenizer} is not there.")
    processor = spm.SentencePieceProcessor()
    processor.load(str(tokenizer))
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="float32")
    model.to("cpu")
    model.eval()
    return processor, model
