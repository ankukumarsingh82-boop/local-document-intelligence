"""Build a short prompt from a handful of passages."""

from __future__ import annotations

import re

from ldi.config import PREVIEW_CHARS
from ldi.evidence import build_evidence, format_evidence
from ldi.store import Hit
from ldi.verify import conflict_banner

ASK_SYSTEM = (
    "You answer questions about the user's own files. "
    "Use only the evidence items in the next message. "
    "Reply with one JSON object and nothing else. "
    "Use the keys status, evidence, and answer, in that order. "
    "status is answered, not_found, or conflict. "
    "evidence is an array of ids copied from the evidence list. "
    "answer uses only words, numbers, and dates that appear in the chosen evidence. "
    "When the question asks you to add, subtract, multiply, or divide, "
    "add a calc string made only of those ids, the operators + - * / , and parentheses. "
    "A calculator object is accepted in that calc field. "
    "Do not write a number of your own in the calc string, and do not write the result. "
    "Do not write a quote. The program adds the citation."
)

SYSTEM = (
    "You answer questions about the user's own files. "
    "Use only the passages in the next message. "
    "When a passage supports a statement, name the file. "
    "When the passages do not contain the answer, say exactly: "
    "I could not find supporting evidence in the indexed text. "
    "Keep the answer short."
)


def format_passages(hits: list[Hit]) -> str:
    lines: list[str] = []
    banner = conflict_banner(hits)
    if banner:
        lines.append(banner)
        lines.append("")
    lines.append("Passages")
    for number, hit in enumerate(hits, start=1):
        preview = re.sub(r"\s+", " ", hit.text).strip()
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS].rstrip() + "…"
        lines.append(f"  {number}. {_where(hit)}")
        lines.append(f"     {preview}")
    return "\n".join(lines)


def _where(hit: Hit) -> str:
    place = hit.path if hit.page is None else f"{hit.path}, page {hit.page}"
    if hit.superseded:
        return f"{place} [superseded]"
    return place


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    items = build_evidence(question, hits)
    user = format_evidence(items) + f"\n\nQuestion: {question.strip()}"
    return [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user", "content": user},
    ]
