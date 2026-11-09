"""Split document text into short passages."""

from __future__ import annotations

import re
import unicodedata

from ldi.config import CHUNK_CHARS, CHUNK_OVERLAP

_HYPHEN_BREAK = re.compile(r"([^\W\d_])-\r?\n([^\W\d_])")


def normalise_text(text: str) -> str:
    """Join words split by a line-break hyphen, and fold spaces and accents."""
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ")
    for _ in range(5):
        joined = _HYPHEN_BREAK.sub(r"\1\2", cleaned)
        if joined == cleaned:
            break
        cleaned = joined
    return cleaned


def chunk_text(
    text: str,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    cleaned = normalise_text(text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            paragraph = cleaned.rfind("\n\n", start + size // 2, end)
            space = cleaned.rfind(" ", start + size // 2, end)
            if paragraph != -1:
                end = paragraph
            elif space != -1:
                end = space
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
