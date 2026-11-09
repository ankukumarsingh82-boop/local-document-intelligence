"""Local weight files. Downloading is the only networked step."""

from __future__ import annotations

import os
from pathlib import Path

from ldi.config import ModelSpec
from ldi.guard import assess_download, assess_setup, machine_snapshot


def cache_dir(repo: str) -> Path:
    slug = "models--" + repo.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / slug


def local_snapshot(repo: str) -> Path | None:
    snapshots = cache_dir(repo) / "snapshots"
    if not snapshots.is_dir():
        return None
    found = [
        path
        for path in snapshots.iterdir()
        if path.is_dir()
        and (path / "config.json").is_file()
        and _has_weights(path)
    ]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


def _has_weights(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(path.glob("*.npz"))


def download(spec: ModelSpec) -> Path:
    existing = local_snapshot(spec.repo)
    if existing is not None:
        print(f"Weights are already on disk: {existing}")
        return existing

    snap = machine_snapshot()
    assess_setup(snap.pressure)
    assess_download(spec.approx_bytes, snap.disk_free)

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from huggingface_hub import snapshot_download

    print(
        f"Downloading {spec.repo}. "
        "This is the one network step. Documents and questions stay on this Mac."
    )
    path = Path(snapshot_download(repo_id=spec.repo, max_workers=1))
    print(f"Weights are on disk: {path}")
    return path
