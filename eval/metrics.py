"""Score a release against gold anchors. Gold text is not rewritten."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from eval.config import ABSTAIN_TAGS
from eval.provenance import cited_spans, collapse, fabricated
from ldi.verify import NOT_FOUND

METRICS = (
    "correct",
    "wrong",
    "false_not_found",
    "false_accept",
    "unscored",
)


@dataclass(frozen=True)
class GoldView:
    """What the scorer is allowed to know. Anchors are the gold file's own spans."""

    abstain: bool
    anchors: tuple[str, ...]
    scored: bool


def anchors_covered(anchors: tuple[str, ...], shown: str) -> bool:
    if not anchors:
        return False
    spans = [collapse(span.text) for span in cited_spans(shown)]
    if not spans:
        return False
    return all(
        any(collapse(anchor) in span for span in spans) for anchor in anchors if collapse(anchor)
    )


def metric_for(gold: GoldView, shown: str) -> str:
    if not gold.scored:
        return "unscored"
    abstained = shown.startswith(NOT_FOUND)
    if gold.abstain or not gold.anchors:
        return "correct" if abstained else "false_accept"
    if anchors_covered(gold.anchors, shown):
        return "correct"
    if abstained:
        return "false_not_found"
    return "wrong"


def gold_from_tag(tag: str, anchors: tuple[str, ...]) -> GoldView:
    abstain = tag in ABSTAIN_TAGS or not anchors
    return GoldView(abstain=abstain, anchors=() if abstain else anchors, scored=True)


def empty_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for name in METRICS:
        counts[name] = 0
    counts["fabrication"] = 0
    counts["invalid_model_output"] = 0
    counts["timeout"] = 0
    return counts


def add_case(counts: Counter[str], metric: str, *, fabrication: bool, invalid: bool, timeout: bool) -> None:
    if timeout:
        counts["timeout"] += 1
        return
    counts[metric] += 1
    if fabrication:
        counts["fabrication"] += 1
    if invalid:
        counts["invalid_model_output"] += 1


def is_fabricated(shown: str) -> bool:
    return fabricated(shown)
