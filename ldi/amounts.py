"""Read an amount written after a currency marker without altering its source quote."""

from __future__ import annotations

import re
from decimal import Decimal

_SMALL = {
    word: number for number, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    word: number * 10 for number, word in enumerate(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2
    )
}
_SCALES = {"crore": 10_000_000, "million": 1_000_000, "lakh": 100_000, "thousand": 1_000}
_CURRENCY = re.compile(r"\brupees\s+([a-z][a-z\s-]{0,120})", re.IGNORECASE)


def written_rupees(text: str) -> Decimal | None:
    """Read one explicit rupee amount. Unrecognized phrases stay unparsed."""
    values = set()
    for match in _CURRENCY.finditer(text):
        tokens = re.findall(r"[a-z]+", match[1].casefold())
        if "only" in tokens:
            tokens = tokens[:tokens.index("only")]
        value = _cardinal(tokens)
        if value is not None:
            values.add(Decimal(value))
    return next(iter(values)) if len(values) == 1 else None


def _cardinal(tokens: list[str]) -> int | None:
    if not tokens or tokens[0] == "and" or tokens[-1] == "and":
        return None
    words = [word for word in tokens if word != "and"]
    total = 0
    group = []
    previous_scale = float("inf")
    for word in words:
        scale = _SCALES.get(word)
        if scale is None:
            group.append(word)
            continue
        value = _under_thousand(group)
        if not value or scale >= previous_scale:
            return None
        total += value * scale
        previous_scale = scale
        group = []
    tail = _under_thousand(group)
    return total + tail if tail is not None else None


def _under_thousand(words: list[str]) -> int | None:
    hundreds = 0
    if len(words) >= 2 and words[1] == "hundred" and 1 <= _SMALL.get(words[0], 0) <= 9:
        hundreds = _SMALL[words[0]] * 100
        words = words[2:]
    if not words:
        return hundreds
    if len(words) == 1:
        value = _SMALL.get(words[0], _TENS.get(words[0]))
        return hundreds + value if value is not None else None
    if len(words) == 2 and words[0] in _TENS and 1 <= _SMALL.get(words[1], 0) <= 9:
        return hundreds + _TENS[words[0]] + _SMALL[words[1]]
    return None
