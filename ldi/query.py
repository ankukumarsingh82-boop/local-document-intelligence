"""Turn a question into a quoted query frame. This module does not search."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from ldi.config import (
    CURRENCY_FOLDS,
    HONORIFICS,
    INTENT_CURRENT,
    INTENT_HISTORICAL,
    INTENT_REVISED,
    LONG_DIGIT_MIN,
    MAX_QUERY_TOKENS,
    MONTHS,
    NAME_EDIT_DISTANCE,
    NEGATIONS,
    QUALIFIER_PHRASES,
    QUALIFIER_WORDS,
    QUERY_STOPWORDS,
    SEARCH_ALIASES,
    STRUCTURAL_NOUNS,
    TABLE_CUES,
    UNIT_FOLDS,
)

_TOKEN = re.compile(
    r"₹|%|\d{1,3}(?:,\d{2,3})+(?:\.\d+)?%?|\d+(?:\.\d+)?%?|"
    r"[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)*"
)
_ROMAN = re.compile(
    r"^(?=[IVXLCDM]+$)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_ACRONYM = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+\(([A-Z][A-Z0-9]{2,})\)")
_POSSESSIVE = re.compile(r"^[A-Z][a-z]+['’]s?$")
_NUMBER = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(frozen=True)
class RawToken:
    text: str
    norm: str
    short_kept: bool


@dataclass(frozen=True)
class QueryFrame:
    question: str
    intent: str
    as_of: str
    required: tuple[str, ...]
    aliases: tuple[str, ...]
    ids: tuple[str, ...]
    names: tuple[str, ...]
    designators: tuple[str, ...]
    numbers: tuple[str, ...]
    negations: tuple[str, ...]
    qualifiers: tuple[str, ...]
    table_numeric: bool

    @property
    def hard_entities(self) -> bool:
        return bool(self.ids or self.names)

    @property
    def entity_route(self) -> bool:
        return bool(self.ids or self.names or self.designators or self.numbers)


def idf(df: int, total: int) -> float:
    """Smooth inverse document frequency. Higher means rarer."""
    return math.log((total + 1) / (df + 1)) + 1.0


def quote_token(token: str) -> str:
    """One FTS5 literal. AND, OR, NOT, and NEAR cannot act as operators."""
    return '"' + token.replace('"', '""') + '"'


def or_query(tokens: tuple[str, ...] | list[str]) -> str:
    return " OR ".join(quote_token(token) for token in tokens)


def phrase_query(phrase: str) -> str:
    parts = [part for part in phrase.split() if part]
    if len(parts) <= 1:
        return quote_token(parts[0] if parts else phrase)
    return '"' + " ".join(parts).replace('"', '""') + '"'


def fold_for_fts(text: str) -> str:
    """Fold numbers and currency. Quotable passage text is left unchanged."""
    folded = unicodedata.normalize("NFC", text).replace("\u00a0", " ").replace("₹", " inr ")
    folded = re.sub(r"(?<=\d),(?=\d)", "", folded)
    folded = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", folded)
    tokens = []
    for raw in _TOKEN.findall(folded):
        if raw == "%":
            continue
        norm = _normal(raw)
        if norm:
            tokens.append(norm)
    return " ".join(tokens)


def acronyms_in(text: str) -> tuple[tuple[str, str], ...]:
    """Pairs of (acronym, full name), both casefolded."""
    found = []
    for match in _ACRONYM.finditer(text):
        expansion = " ".join(part.casefold() for part in match.group(1).split())
        alias = match.group(2).casefold()
        if expansion and alias:
            found.append((alias, expansion))
    return tuple(found)


def names_in(text: str) -> tuple[str, ...]:
    return _names(tokenize(text), ())


def parse_question(
    question: str,
    *,
    today: date | None = None,
    frequencies: dict[str, int] | None = None,
    passage_count: int = 0,
    alias_rows: tuple[tuple[str, str], ...] = (),
    indexed_names: tuple[str, ...] = (),
) -> QueryFrame:
    masked, phrases = _mask_phrases(question)
    raw = tokenize(masked)
    intent = _intent(raw)
    names = _names(raw, indexed_names)
    designators = _designators(raw)
    name_tokens = {part for name in names for part in name.split()}
    designator_tokens = {part for phrase in designators for part in phrase.split()}
    kept: list[str] = []
    negations: list[str] = []
    qualifiers: list[str] = list(phrases)
    ids: list[str] = []
    numbers: list[str] = []
    for token in raw:
        if token.norm in INTENT_CURRENT | INTENT_REVISED | INTENT_HISTORICAL:
            continue
        if token.norm in QUALIFIER_WORDS:
            qualifiers.append(token.norm)
            if token.norm in NEGATIONS:
                negations.append(token.norm)
            continue
        if token.norm in QUERY_STOPWORDS:
            continue
        if len(token.norm) <= 1 and not token.short_kept and token.norm not in designator_tokens:
            continue
        if token.norm in name_tokens or token.norm in designator_tokens or _is_id(token.norm):
            if _is_id(token.norm) and token.norm not in ids:
                ids.append(token.norm)
        elif _is_number(token.norm) and token.norm not in numbers:
            numbers.append(token.norm)
        kept.append(token.norm)
    protected = set(ids) | name_tokens | designator_tokens
    required = _cap(kept, protected, frequencies or {}, passage_count)
    aliases = _aliases(required, alias_rows)
    table = bool(TABLE_CUES.intersection(required) or "table" in designator_tokens)
    return QueryFrame(
        question=question,
        intent=intent,
        as_of=_as_of(intent, today),
        required=required,
        aliases=aliases,
        ids=tuple(dict.fromkeys(ids)),
        names=names,
        designators=designators,
        numbers=tuple(dict.fromkeys(numbers)),
        negations=tuple(dict.fromkeys(negations)),
        qualifiers=tuple(dict.fromkeys(qualifiers)),
        table_numeric=table,
    )


def tokenize(text: str) -> tuple[RawToken, ...]:
    folded = unicodedata.normalize("NFC", text).replace("\u00a0", " ").replace("₹", " ₹ ")
    folded = re.sub(r"(?<=\d),(?=\d)", "", folded)
    folded = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", folded)
    pieces = [piece for piece in _TOKEN.findall(folded) if piece != "%"]
    tokens: list[RawToken] = []
    for index, piece in enumerate(pieces):
        norm = _normal(piece)
        if not norm:
            continue
        previous = _normal(pieces[index - 1]) if index else ""
        nxt = _normal(pieces[index + 1]) if index + 1 < len(pieces) else ""
        percent = piece.endswith("%") or (index + 1 < len(pieces) and pieces[index + 1] == "%")
        currency = previous in CURRENCY_FOLDS or nxt in CURRENCY_FOLDS or piece.casefold().rstrip(".") in CURRENCY_FOLDS
        short = (
            norm.isdigit()
            or previous in STRUCTURAL_NOUNS
            or percent
            or currency
            or norm in CURRENCY_FOLDS.values()
        )
        tokens.append(RawToken(piece, norm, short))
    return tuple(tokens)


def slot_supported(frame: QueryFrame, text: str) -> bool:
    """Exact name and ID slots only. Unfilled slots and qualifiers are not enforced."""
    folded = fold_for_fts(text)
    words = set(folded.split())
    for ident in frame.ids:
        if ident not in words:
            return False
    for name in frame.names:
        if name not in folded:
            return False
    return True


def _normal(raw: str) -> str:
    if raw in {"%", "₹"}:
        return "inr" if raw == "₹" else ""
    body = raw[:-1] if raw.endswith("%") else raw
    key = body.casefold().replace("’", "'")
    if key.endswith("'s"):
        key = key[:-2]
    elif key.endswith("'"):
        key = key[:-1]
    if key in CURRENCY_FOLDS:
        return CURRENCY_FOLDS[key]
    if key in UNIT_FOLDS:
        return UNIT_FOLDS[key]
    return key


def _mask_phrases(question: str) -> tuple[str, tuple[str, ...]]:
    masked = question
    found: list[str] = []
    for phrase in sorted(QUALIFIER_PHRASES, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        if pattern.search(masked):
            found.append(phrase)
            masked = pattern.sub(" ", masked)
    return masked, tuple(found)


def _intent(tokens: tuple[RawToken, ...]) -> str:
    for token in tokens:
        if token.norm in INTENT_CURRENT:
            return "current"
        if token.norm in INTENT_REVISED:
            return "revised"
        if token.norm in INTENT_HISTORICAL:
            return "historical"
    return ""


def _as_of(intent: str, today: date | None) -> str:
    if intent in {"current", "revised"}:
        return (today or date.today()).isoformat()
    return ""


def _is_id(token: str) -> bool:
    if re.fullmatch(rf"\d{{{LONG_DIGIT_MIN},}}", token):
        return True
    return bool(re.search(r"[a-z]", token) and re.search(r"\d", token))


def _is_number(token: str) -> bool:
    return _NUMBER.fullmatch(token) is not None and not _is_id(token)


def _is_roman(token: str) -> bool:
    return _ROMAN.fullmatch(token) is not None and bool(token)


def _designators(tokens: tuple[RawToken, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token.norm not in STRUCTURAL_NOUNS:
            continue
        nxt = tokens[index + 1]
        if _is_roman(nxt.norm) or _is_number(nxt.norm) or _is_id(nxt.norm) or (
            len(nxt.norm) == 1 and nxt.norm.isalpha()
        ):
            phrase = f"{token.norm} {nxt.norm}"
            if phrase not in found:
                found.append(phrase)
    return tuple(found)


def _names(tokens: tuple[RawToken, ...], indexed: tuple[str, ...]) -> tuple[str, ...]:
    found: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.norm in HONORIFICS and index + 1 < len(tokens) and _personal(tokens[index + 1]):
            words = [tokens[index + 1].norm]
            cursor = index + 2
            while cursor < len(tokens) and cursor < index + 4 and _personal(tokens[cursor]):
                words.append(tokens[cursor].norm)
                cursor += 1
            _add(found, " ".join(words))
            index = cursor
            continue
        possessive = _POSSESSIVE.fullmatch(token.text)
        if possessive is not None:
            _add(found, token.norm.split("'")[0].split("’")[0])
            index += 1
            continue
        if token.norm in {"of", "for"}:
            words = []
            cursor = index + 1
            while cursor < len(tokens) and _personal(tokens[cursor]):
                words.append(tokens[cursor].norm)
                cursor += 1
            if len(words) >= 2:
                _add(found, " ".join(words))
                index = cursor
                continue
        index += 1
    if indexed:
        for start, token in enumerate(tokens):
            if token.norm in QUERY_STOPWORDS:
                continue
            for width in (1, 2, 3):
                window = tokens[start : start + width]
                if len(window) != width:
                    continue
                if any(part.norm in QUERY_STOPWORDS for part in window):
                    continue
                if width == 1 and not _personal(token):
                    continue
                surface = " ".join(part.norm for part in window)
                if any(_near_name(surface, known) for known in indexed):
                    _add(found, surface)
    return tuple(found)


def _personal(token: RawToken) -> bool:
    if token.norm in QUERY_STOPWORDS or token.norm in STRUCTURAL_NOUNS or token.norm in MONTHS:
        return False
    if token.norm in HONORIFICS or _is_id(token.norm) or _is_number(token.norm):
        return False
    text = token.text.strip("%")
    return len(text) > 1 and text[:1].isupper() and any(char.islower() for char in text)


def _near_name(surface: str, known: str) -> bool:
    """Edit distance may only classify a mention. It does not rewrite the name."""
    if surface == known:
        return True
    return _distance(surface, known) <= NAME_EDIT_DISTANCE


def _distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > NAME_EDIT_DISTANCE:
        return NAME_EDIT_DISTANCE + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            cost = 0 if a == b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _add(found: list[str], name: str) -> None:
    cleaned = " ".join(part for part in name.split() if part)
    if cleaned and cleaned not in found:
        found.append(cleaned)


def _cap(
    tokens: list[str],
    protected: set[str],
    frequencies: dict[str, int],
    total: int,
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    optional = [token for token in ordered if token not in protected]
    optional.sort(key=lambda token: (-idf(frequencies.get(token, 0), total), token))
    room = max(0, MAX_QUERY_TOKENS - sum(1 for token in ordered if token in protected))
    chosen = {token for token in ordered if token in protected}
    chosen.update(optional[:room])
    return tuple(token for token in ordered if token in chosen)


def _aliases(required: tuple[str, ...], rows: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    extra: list[str] = []
    seen = set(required)
    expansions: dict[str, list[str]] = {}
    for alias, expansion in rows:
        expansions.setdefault(alias, []).append(expansion)
    for token in required:
        mapped = SEARCH_ALIASES.get(token)
        if mapped and mapped not in seen:
            extra.append(mapped)
            seen.add(mapped)
        for expansion in expansions.get(token, []):
            for part in expansion.split():
                if part in seen or part in QUERY_STOPWORDS:
                    continue
                extra.append(part)
                seen.add(part)
    required_set = set(required)
    for alias, expansion in rows:
        parts = [part for part in expansion.split() if part not in QUERY_STOPWORDS]
        if parts and all(part in required_set for part in parts) and alias not in seen:
            extra.append(alias)
            seen.add(alias)
    return tuple(extra)
