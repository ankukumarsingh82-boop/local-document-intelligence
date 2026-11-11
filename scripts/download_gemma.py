"""Download Gemma 3 270M once. The weights are not part of this repo.

The model is gated. Accept the Gemma Terms of Use at
https://huggingface.co/google/gemma-3-270m-it and run ``huggingface-cli login``
before this script. ``--mirror`` uses an ungated copy of the same weights.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_REPO = "google/gemma-3-270m-it"
_REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
_MIRROR_REPO = "unsloth/gemma-3-270m-it"
_MIRROR_REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
_WEIGHTS = "model.safetensors"
_WEIGHTS_SHA256 = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=f"Download {_MIRROR_REPO} instead of the gated Google repo.",
    )
    args = parser.parse_args(argv)
    repo, revision = (_MIRROR_REPO, _MIRROR_REVISION) if args.mirror else (_REPO, _REVISION)
    target = Path(os.environ.get("GEMMA_DIR", "models/gemma-3-270m-it"))
    print(f"Downloading {repo} at {revision} into {target}.")
    print("The Gemma weights stay on this Mac and are not added to the repo.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Run `uv sync` and try again.",
            file=sys.stderr,
        )
        return 1
    try:
        snapshot_download(repo, revision=revision, local_dir=target)
    except Exception as exc:
        print(
            f"The download failed: {exc}\n"
            "Accept the Gemma Terms of Use on the model page and run "
            "`huggingface-cli login`, or rerun with --mirror.",
            file=sys.stderr,
        )
        return 1
    weights = target / _WEIGHTS
    if not weights.is_file():
        print(f"The download finished without {weights}.", file=sys.stderr)
        return 1
    digest = _sha256(weights)
    if digest != _WEIGHTS_SHA256:
        print(
            f"{weights} is not the Gemma 3 270M this check was built for.\n"
            f"Expected sha256 {_WEIGHTS_SHA256}, found {digest}.",
            file=sys.stderr,
        )
        return 1
    print(f"Verified {weights.name}: sha256 {digest}.")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
