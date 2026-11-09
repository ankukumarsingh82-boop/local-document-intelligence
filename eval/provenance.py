"""Fail a rendered answer whose facts are not in a cited span or a traced sum."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from ldi.chunk import normalise_text
from ldi.evidence import number_values
from ldi.verify import NOT_FOUND, _NUMBER

_DATE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4})\b"
)
_ID = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")
_CLAUSE = re.compile(
    r"\b(?:clause|section|article|annexure|schedule|paragraph)\s+\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_CITE = re.compile(r'- cite: "(?P<quote>.*)"\n  (?P<locator>.*)')


@dataclass(frozen=True)
class Span:
    locator: str
    text: str


@dataclass(frozen=True)
class Fact:
    kind: str
    surface: str


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", normalise_text(text)).strip()


def cited_spans(shown: str) -> list[Span]:
    found = [
        Span(match.group("locator").strip(), match.group("quote"))
        for match in _CITE.finditer(shown)
    ]
    if found:
        return found
    lines = shown.splitlines()
    if not lines or lines[0].strip() != "Conflict":
        return []
    body = [line for line in lines[1:] if line.strip()]
    spans: list[Span] = []
    for index in range(0, len(body) - 1, 2):
        spans.append(Span(body[index].strip(), body[index + 1].strip()))
    return spans


def answer_text(shown: str) -> str:
    """The claim, without citation lines or the coverage note."""
    if shown.startswith(NOT_FOUND) or shown.startswith("Conflict") or shown.startswith("Stale"):
        return ""
    head = shown.split("\n- cite:", 1)[0]
    return head.split("\nCoverage:", 1)[0].strip()


def facts_in(text: str) -> list[Fact]:
    found: list[Fact] = []
    for match in _NUMBER.finditer(text):
        found.append(Fact("number", match.group(0)))
    for pattern, kind in ((_DATE, "date"), (_ID, "id"), (_CLAUSE, "clause"), (_NAME, "name")):
        for match in pattern.finditer(text):
            found.append(Fact(kind, match.group(0)))
    return found


def _supported_number(surface: str, spans: list[Span]) -> bool:
    folded = collapse(surface)
    blob = "\n".join(collapse(span.text) for span in spans)
    if folded and folded in blob:
        return True
    try:
        value = Decimal(re.sub(r"[^\d.+-]", "", surface.replace(",", "")))
    except Exception:
        return False
    pool = [number for span in spans for number in number_values(span.text)]
    if value in pool:
        return True
    unique = list(dict.fromkeys(pool))
    if len(unique) >= 2 and sum(unique, Decimal(0)) == value:
        return True
    for left in unique:
        for right in unique:
            if left == right:
                continue
            difference = left - right
            if difference == value or abs(difference) == value:
                return True
    return False


def unsupported(shown: str) -> list[Fact]:
    """Facts in the claim that no cited span or traced sum accounts for."""
    spans = cited_spans(shown)
    missing: list[Fact] = []
    for fact in facts_in(answer_text(shown)):
        if fact.kind == "number":
            if not _supported_number(fact.surface, spans):
                missing.append(fact)
            continue
        blob = "\n".join(collapse(span.text) for span in spans)
        if collapse(fact.surface).casefold() not in blob.casefold():
            missing.append(fact)
    return missing


def fabricated(shown: str) -> bool:
    return bool(unsupported(shown))
