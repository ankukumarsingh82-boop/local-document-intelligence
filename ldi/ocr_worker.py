"""Read scanned pages. This module runs only inside the OCR environment."""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
from pathlib import Path

from ldi.config import MAX_KV_TOKENS, METAL_CACHE_BYTES, OCR_PROMPT


def main() -> int:
    if len(sys.argv) != 2:
        print("The OCR worker needs a job file.", file=sys.stderr)
        return 2
    job = json.loads(Path(sys.argv[1]).read_text())
    try:
        import mlx.core as mx
        from mlx_vlm import generate, load
    except Exception as exc:
        print(f"The OCR environment is missing a library. {exc}", file=sys.stderr)
        return 2
    try:
        _cap_metal(mx, int(os.environ["LDI_OCR_METAL_BYTES"]))
    except (KeyError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    pixels = int(os.environ.get("LDI_OCR_PIXELS", "1003520"))
    model = None
    try:
        model, processor = load(job["model"], trust_remote_code=True)
        read = [
            _read(generate, model, processor, _image(item, pixels), pixels)
            for item in job["items"]
        ]
        pages = [text for text, _scores in read]
        confidence = [scores for _text, scores in read]
    except Exception as exc:
        print(f"The OCR model stopped. {exc}", file=sys.stderr)
        return 1
    finally:
        del model
        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()
    Path(job["result"]).write_text(
        json.dumps({"pages": pages, "confidence": confidence})
    )
    return 0


def _cap_metal(mx, limit: int) -> None:
    memory = getattr(mx, "set_memory_limit", None) or getattr(
        getattr(mx, "metal", None), "set_memory_limit", None
    )
    if memory is None:
        raise RuntimeError("This MLX build has no memory cap. The model stays unloaded.")
    memory(limit)
    cache = getattr(mx, "set_cache_limit", None) or getattr(
        getattr(mx, "metal", None), "set_cache_limit", None
    )
    if cache is None:
        raise RuntimeError("This MLX build has no cache cap. The model stays unloaded.")
    cache(METAL_CACHE_BYTES)


def _image(item: dict, pixels: int):
    path = Path(item["path"])
    page = item.get("page")
    if page is None:
        from PIL import Image

        return _fit(Image.open(path).convert("RGB"), pixels)
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        sheet = document[int(page) - 1]
        area = max(float(sheet.get_width()) * float(sheet.get_height()), 1.0)
        scale = min((pixels / area) ** 0.5, 4.0)
        rendered = sheet.render(scale=scale).to_pil().convert("RGB")
    finally:
        document.close()
    return _fit(rendered, pixels)


def _fit(image, pixels: int):
    width, height = image.size
    if width * height <= pixels:
        return image
    scale = (pixels / (width * height)) ** 0.5
    resized = (
        max(28, int(width * scale) // 28 * 28),
        max(28, int(height * scale) // 28 * 28),
    )
    return image.resize(resized)


def _read(generate, model, processor, image, pixels: int) -> tuple[str, list[float | None]]:
    prompt = _prompt(processor, model)
    fitted = _fit(image, pixels)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        image_path = Path(handle.name)
    fitted.save(image_path)
    try:
        result = generate(
            model,
            processor,
            prompt,
            image=str(image_path),
            max_tokens=1024,
            temperature=0.0,
            max_kv_size=MAX_KV_TOKENS,
            verbose=False,
        )
    finally:
        image_path.unlink(missing_ok=True)
    text = _result_text(result)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return text, _line_confidence(result, len(lines))


def _result_text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    for name in ("text", "output"):
        value = getattr(result, name, None)
        if isinstance(value, str):
            return value.strip()
    return str(result).strip()


def _line_confidence(result, count: int) -> list[float | None]:
    """PaddleOCR-VL on MLX returns text, not a detection score. Missing scores stay null."""
    for name in ("line_confidences", "confidences"):
        value = getattr(result, name, None)
        if not isinstance(value, list) or len(value) != count:
            continue
        scores: list[float | None] = []
        for item in value:
            if item is None:
                scores.append(None)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                scores.append(float(item))
            else:
                break
        else:
            return scores
    return [None] * count


def _prompt(processor, model) -> str:
    try:
        from mlx_vlm.prompt_utils import apply_chat_template

        config = getattr(model, "config", None)
        rendered = apply_chat_template(processor, config, OCR_PROMPT, num_images=1)
    except Exception:
        return OCR_PROMPT
    if isinstance(rendered, str) and rendered.strip():
        return rendered
    return OCR_PROMPT


if __name__ == "__main__":
    raise SystemExit(main())
