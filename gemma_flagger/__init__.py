"""Optional "meaning may have changed" check for version comparison.

The Gemma weights are not part of this repo. A user downloads them with
``python scripts/download_gemma.py``. Every function here works with the
weights absent, and the rest of the app never imports this package unless the
meaning check is switched on.
"""

from __future__ import annotations

__all__ = ["MeaningFlagger"]


def __getattr__(name: str):
    if name == "MeaningFlagger":
        from gemma_flagger.flagger import MeaningFlagger

        return MeaningFlagger
    raise AttributeError(name)
