"""Scan pages with PaddleOCR-VL. The chat model is unloaded before this runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ldi.config import (
    IMAGE_SUFFIXES,
    OCR_MODELS,
    OCR_PIXELS,
    PDF_SUFFIXES,
    gb,
    ocr_python,
    ocr_venv,
    project_root,
)
from ldi.guard import GuardError, assess_download, assess_load, assess_setup, machine_snapshot
from ldi.weights import download, local_snapshot

MISSING = "No OCR weights on disk yet. ldi setup --model ocr downloads them."


@dataclass(frozen=True)
class Scan:
    path: Path
    page: int | None


def classify_page(text: str, image_count: int) -> str:
    """A real text layer is the page. An image with no text is a scan. Empty is blank."""
    if text.strip():
        return "text"
    if image_count > 0:
        return "scan"
    return "blank"


def image_count(page: object) -> int:
    try:
        images = page.images  # type: ignore[attr-defined]
    except Exception:
        return 0
    try:
        return len(images)
    except TypeError:
        return 0


def environment_ready() -> bool:
    return ocr_python().is_file() and (ocr_venv() / ".ldi-ready").is_file()


def setup_ocr() -> None:
    spec = OCR_MODELS["ocr"]
    if local_snapshot(spec.repo) is None or not environment_ready():
        snap = machine_snapshot()
        assess_setup(snap.pressure)
        assess_download(spec.approx_bytes, snap.disk_free)
    download(spec)
    _ensure_environment()


def recognize(scans: list[Scan]) -> list[str]:
    """Read every scan with one model load. The caller unloads the chat model first.

    `recognize.confidence` is one list of per-line scores per page. A missing
    PaddleOCR line score is None, which later counts as low confidence.
    """
    if not scans:
        recognize.confidence = []
        return []
    spec = OCR_MODELS["ocr"]
    local = local_snapshot(spec.repo)
    if local is None or not environment_ready():
        raise ValueError(MISSING)
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    pages = "page" if len(scans) == 1 else "pages"
    print(
        f"Reading {len(scans)} scanned {pages} with {spec.label}. "
        f"Metal is capped at {gb(spec.metal_limit_bytes)}."
    )
    payload = {
        "model": str(local),
        "items": [{"path": str(scan.path), "page": scan.page} for scan in scans],
    }
    with tempfile.TemporaryDirectory() as directory:
        job = Path(directory) / "job.json"
        result_path = Path(directory) / "result.json"
        payload["result"] = str(result_path)
        job.write_text(json.dumps(payload))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root())
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["LDI_OCR_METAL_BYTES"] = str(spec.metal_limit_bytes)
        env["LDI_OCR_PIXELS"] = str(OCR_PIXELS)
        try:
            result = subprocess.run(
                [str(ocr_python()), "-m", "ldi.ocr_worker", str(job)],
                cwd=str(project_root()),
                env=env,
                capture_output=True,
                text=True,
                timeout=max(1200, 1200 * len(scans)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardError("The OCR process stopped before it finished.") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            tail = detail[-1] if detail else "the OCR process failed."
            raise GuardError(f"The OCR process stopped. {tail}")
        try:
            parsed = json.loads(result_path.read_text())
            texts = parsed["pages"]
            scores = parsed.get("confidence")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise GuardError("The OCR process returned nothing readable.") from exc
    if not isinstance(texts, list) or len(texts) != len(scans):
        raise GuardError("The OCR process returned nothing readable.")
    if not isinstance(scores, list) or len(scores) != len(texts):
        scores = [None] * len(texts)
    recognize.confidence = [_score_row(row) for row in scores]
    print("The OCR model is unloaded.")
    return [str(item).strip() for item in texts]


def _score_row(row: object) -> list[float | None] | None:
    if not isinstance(row, list):
        return None
    scores: list[float | None] = []
    for item in row:
        if item is None:
            scores.append(None)
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        scores.append(float(item))
    return scores


def _ensure_environment() -> None:
    if environment_ready():
        print("OCR environment is ready.")
        return
    root = ocr_venv()
    python = ocr_python()
    if not python.is_file():
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(root)],
            check=False,
        )
        if created.returncode != 0 or not python.is_file():
            raise GuardError("The OCR environment could not be created.")
    print("Installing the OCR environment. It stays out of the main environment.")
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "mlx-vlm>=0.3.11",
            "pypdfium2>=4.30",
            "pillow",
        ],
        check=False,
    )
    if installed.returncode != 0:
        raise GuardError("The OCR environment could not be installed.")
    (root / ".ldi-ready").write_text("mlx-vlm\n")
    print("OCR environment is ready.")
