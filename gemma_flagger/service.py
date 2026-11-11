"""One shared flagger, loaded the first time a meaning check is asked for.

The model stays unloaded until then, and a missing download leaves the check
off instead of breaking the app.
"""

from __future__ import annotations

import os
from pathlib import Path

from gemma_flagger.flagger import MeaningFlagger

_flagger: MeaningFlagger | None = None
_failed = False


def gemma_dir() -> Path:
    return Path(os.environ.get("GEMMA_DIR", "models/gemma-3-270m-it"))


def available() -> bool:
    """True once the weights have been downloaded. This loads nothing."""
    return (gemma_dir() / "model.safetensors").is_file()


def get_flagger() -> MeaningFlagger | None:
    """The loaded flagger, or None when the weights are absent or failed to load."""
    global _flagger, _failed
    if _flagger is not None or _failed:
        return _flagger
    if not available():
        return None
    try:
        _flagger = MeaningFlagger(gemma_dir())
    except Exception:
        _failed = True
        return None
    return _flagger


def reset() -> None:
    """Drop the loaded flagger. Tests use this to start from a clean state."""
    global _flagger, _failed
    _flagger = None
    _failed = False
