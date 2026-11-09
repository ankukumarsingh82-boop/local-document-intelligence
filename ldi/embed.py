"""BGE stays on disk until a small embedding runtime exists.

mlx-embeddings imports mlx-vlm. That package installs OpenCV, an audio
library, datasets, and a web server. This Mac does not install that stack,
and this module does not load the embedding weights.
"""

from __future__ import annotations

from ldi.guard import GuardError

REFUSAL = (
    "The BGE file can stay on disk. The similarity check was not run. "
    "mlx-embeddings imports mlx-vlm, and that package installs a second runtime: "
    "OpenCV, an audio library, datasets, and a web server. This Mac will not install it."
)


def embed_check() -> None:
    raise GuardError(REFUSAL)
