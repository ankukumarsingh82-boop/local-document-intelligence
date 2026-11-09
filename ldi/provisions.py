"""Provision rows and the version resolver.

The resolver is not on the answer path. A later step is what calls it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from ldi.config import FY_START_MONTH, SUPERSESSION_PHRASES

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_LONG_DATE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})\b",
    re.IGNORECASE,
)
# A leading 1 read as O or l, as in "Olst July,2023".
_BROKEN_ORDINAL = re.compile(
    r"\b[0oOlI]{1,2}(?:st|nd|rd|th)\s+([A-Za-z]+),?\s*(\d{4})\b",
    re.IGNORECASE,
)
_FY_RANGE = re.compile(r"\bFY\s*'?(\d{4})\s*[-/]\s*(\d{2,4})\b", re.IGNORECASE)
_RATE = (
    r"(?:rs\.?\s*)?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\s*%"
    r"|(?:rs\.?\s*)\d[\d,]*\s*/\s*-"
)
_ENHANCED = re.compile(
    rf"enhanced from\s+(?:the\s+existing\s+rate\s+of\s+)?(?P<old>{_RATE})\s+to\s+(?P<new>{_RATE})",
    re.IGNORECASE,
)
_EFFECT = re.compile(
    r"(?:with effect from|w\.?\s*e\.?\s*f\.?)\s*(?P<rest>.+)$",
    re.IGNORECASE,
)
_OF_THE = re.compile(r"\bof the\s+([A-Za-z][A-Za-z ]{1,40})", re.IGNORECASE)
_AMOUNT = re.compile(_RATE, re.IGNORECASE)
_RANGE_WORDS = ("less than", "more than", "exceed")


@dataclass(frozen=True)
class Provision:
    provision_id: str
    doc_id: str
    clause_key: str
    value_text: str
    value_span: str
    effective_from: str
    effective_to: str
    superseded_by: str
    status: str
    source_span: str


@dataclass(frozen=True)
class Resolution:
    """current carries one row. conflicting carries both claimants and picks neither."""

    status: str
    rows: tuple[Provision, ...]


def parse_dates(text: str) -> list[str]:
    """ISO dates in the order they appear. An impossible day is left out."""
    found: list[tuple[int, str]] = []
    for match in _NUMERIC_DATE.finditer(text):
        iso = _iso(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if iso:
            found.append((match.start(), iso))
    for match in _LONG_DATE.finditer(text):
        month = _MONTHS.get(match.group(2).casefold())
        if month is None:
            continue
        iso = _iso(int(match.group(1)), month, int(match.group(3)))
        if iso:
            found.append((match.start(), iso))
    for match in _BROKEN_ORDINAL.finditer(text):
        month = _MONTHS.get(match.group(1).casefold())
        if month is None:
            continue
        iso = _iso(1, month, int(match.group(2)))
        if iso:
            found.append((match.start(), iso))
    for match in _FY_RANGE.finditer(text):
        year = int(match.group(1))
        found.append((match.start(), f"{year:04d}-{FY_START_MONTH:02d}-01"))
    found.sort(key=lambda item: item[0])
    dates: list[str] = []
    for _pos, iso in found:
        if iso not in dates:
            dates.append(iso)
    return dates


def extract_provisions(doc_id: str, spans: list) -> list[Provision]:
    ordered = sorted(spans, key=_order)
    rows = []
    rows.extend(_enhancements(doc_id, ordered))
    rows.extend(_revision_pairs(doc_id, ordered))
    _apply_phrases(rows, ordered)
    _chain_same_value(rows)
    return _dedupe(rows)


def resolve(rows: list[Provision], clause_key: str, as_of: str) -> Resolution:
    """Pick the row in force on as_of. A filename is not a reason to pick."""
    chosen = [row for row in rows if row.clause_key == clause_key]
    if _cycle(chosen):
        return Resolution("conflicting", tuple(chosen))
    chosen = [replace(row) for row in chosen]
    _chain_same_value(chosen)
    active = [row for row in chosen if _in_force(row, as_of) and not _replaced(row, chosen, as_of)]
    if len(active) == 1:
        return Resolution("current", (active[0],))
    if len(active) > 1:
        return Resolution("conflicting", tuple(active))
    return Resolution("current", ())


def _iso(day: int, month: int, year: int) -> str:
    if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _order(span) -> tuple:
    return (
        getattr(span, "page", None) or 0,
        getattr(span, "block", 0) or 0,
        getattr(span, "line", 0) or 0,
        getattr(span, "col", None) or 0,
        getattr(span, "span_id", ""),
    )


def _window(spans: list, index: int) -> str:
    text = spans[index].text
    for follower in spans[index + 1 : index + 3]:
        if len(text) > 400:
            break
        text = f"{text} {follower.text}"
    return text


def _enhancements(doc_id: str, spans: list) -> list[Provision]:
    rows: list[Provision] = []
    for index, span in enumerate(spans):
        window = _window(spans, index)
        match = _ENHANCED.search(window)
        if match is None or match.start() > len(span.text) + 20:
            continue
        if span.text.casefold().find("enhanced from") < 0:
            continue
        old = _normal_value(match.group("old"))
        new = _normal_value(match.group("new"))
        if not old or not new or old == new:
            continue
        effect = _EFFECT.search(window[match.end() :])
        when = ""
        if effect is not None:
            when = _first_date(effect.group("rest"))
            if not when:
                when = _repaired_date(effect.group("rest"), spans, getattr(span, "page", None))
        clause = _clause_key(window[match.end() :])
        source = span.span_id
        old_row = _row(doc_id, clause, old, source, "", when, "", "superseded", source, "old")
        new_id = _identity(doc_id, clause, new, when, source, "new")
        old_row = replace(old_row, superseded_by=new_id)
        new_row = _row(doc_id, clause, new, source, when, "", "", "current", source, "new")
        rows.extend((old_row, new_row))
    return rows


def _revision_pairs(doc_id: str, spans: list) -> list[Provision]:
    header_date = ""
    armed = False
    for span in spans:
        folded = span.text.casefold()
        if "existing" in folded and "revised" in folded:
            armed = True
            dates = parse_dates(span.text)
            if dates:
                header_date = dates[0]
    if not armed:
        return []
    rows: list[Provision] = []
    rows.extend(_text_pairs(doc_id, spans, header_date))
    rows.extend(_cell_pairs(doc_id, spans, header_date))
    return rows


def _text_pairs(doc_id: str, spans: list, when: str) -> list[Provision]:
    rows: list[Provision] = []
    for span in spans:
        if getattr(span, "source_type", "text") == "table":
            continue
        pair = _two_amounts(span.text)
        if pair is None:
            continue
        old, new = pair
        clause = _pair_key(span.text, old, new)
        source = span.span_id
        rows.append(_row(doc_id, clause, old, source, "", when, "", "superseded", source, "old"))
        new_id = _identity(doc_id, clause, new, when, source, "new")
        rows[-1] = replace(rows[-1], superseded_by=new_id)
        rows.append(_row(doc_id, clause, new, source, when, "", "", "current", source, "new"))
    return rows


def _cell_pairs(doc_id: str, spans: list, when: str) -> list[Provision]:
    groups: dict[tuple, list] = {}
    for span in spans:
        if getattr(span, "source_type", "") != "table":
            continue
        if not _one_amount(span.text):
            continue
        key = (
            getattr(span, "table_id", ""),
            getattr(span, "line", 0),
            getattr(span, "row_key", ""),
        )
        groups.setdefault(key, []).append(span)
    rows: list[Provision] = []
    for grouped in groups.values():
        if len(grouped) != 2:
            continue
        grouped.sort(key=lambda span: getattr(span, "col", 0) or 0)
        old = _one_amount(grouped[0].text)
        new = _one_amount(grouped[1].text)
        if not old or not new or old == new:
            continue
        label = getattr(grouped[0], "row_key", "") or getattr(grouped[0], "header_path", "")
        clause = _pair_key(label, old, new)
        source = grouped[0].span_id
        rows.append(
            _row(doc_id, clause, old, grouped[0].span_id, "", when, "", "superseded", source, "old")
        )
        new_id = _identity(doc_id, clause, new, when, source, "new")
        rows[-1] = replace(rows[-1], superseded_by=new_id)
        rows.append(
            _row(
                doc_id,
                clause,
                new,
                grouped[1].span_id,
                when,
                "",
                "",
                "current",
                source,
                "new",
            )
        )
    return rows


def _one_amount(text: str) -> str:
    found = [item for item in (_normal_value(match) for match in _AMOUNT.findall(text)) if item]
    if len(found) != 1:
        return ""
    return found[0]


def _two_amounts(text: str) -> tuple[str, str] | None:
    folded = text.casefold()
    if "enhanced from" in folded or any(word in folded for word in _RANGE_WORDS):
        return None
    found = [_normal_value(match) for match in _AMOUNT.findall(text)]
    found = [item for item in found if item]
    if len(found) != 2 or found[0] == found[1]:
        return None
    return found[0], found[1]


def _clause_key(window: str) -> str:
    match = _OF_THE.search(window)
    if match is None:
        return "rate"
    words = match.group(1)
    words = re.split(r"\b(?:with|w\.?\s*e\.?\s*f|including)\b", words, maxsplit=1, flags=re.I)[0]
    cleaned = re.sub(r"\s+", " ", words).strip(" .,:;").casefold()
    return cleaned or "rate"


def _pair_key(text: str, old: str, new: str) -> str:
    label = _AMOUNT.sub(" ", text)
    label = re.sub(r"\s+", " ", label).strip(" .,:;-").casefold()
    label = label[:40]
    return label or f"{old}-{new}"


def _normal_value(text: str) -> str:
    cleaned = text.casefold().replace(" ", "")
    cleaned = cleaned.replace("rs.", "rs").replace("/-", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip(" .")
    if cleaned.startswith("rs"):
        cleaned = cleaned[2:]
    return cleaned


def _first_date(text: str) -> str:
    dates = parse_dates(text)
    return dates[0] if dates else ""


def _repaired_date(text: str, spans: list, page) -> str:
    """An impossible day beside a month uses a numeric date of that month on the page."""
    match = re.search(r"\b(\d{1,3})\s+([A-Za-z]+),?\s*(\d{4})?", text, re.IGNORECASE)
    if match is None:
        return ""
    month = _MONTHS.get(match.group(2).casefold())
    if month is None or int(match.group(1)) <= 31:
        return ""
    year = int(match.group(3)) if match.group(3) else 0
    same_page = [
        span.text
        for span in spans
        if page is None or (getattr(span, "page", None) or 0) == page
    ]
    for iso in parse_dates(" ".join(same_page)):
        if int(iso[5:7]) != month:
            continue
        if year and int(iso[:4]) != year:
            continue
        return iso
    return ""


def _chain_same_value(rows: list[Provision]) -> None:
    for index, row in enumerate(rows):
        if row.status != "current" or row.superseded_by:
            continue
        for other in rows:
            if other is row or other.clause_key != row.clause_key:
                continue
            if other.value_text != row.value_text or not other.superseded_by:
                continue
            if other.effective_to and row.effective_from and other.effective_to < row.effective_from:
                continue
            rows[index] = replace(
                row,
                effective_to=other.effective_to,
                superseded_by=other.superseded_by,
                status="superseded",
            )
            break


def _row(
    doc_id: str,
    clause: str,
    value: str,
    value_span: str,
    starts: str,
    ends: str,
    link: str,
    status: str,
    source: str,
    role: str,
) -> Provision:
    return Provision(
        provision_id=_identity(doc_id, clause, value, starts or ends, source, role),
        doc_id=doc_id,
        clause_key=clause,
        value_text=value,
        value_span=value_span,
        effective_from=starts,
        effective_to=ends,
        superseded_by=link,
        status=status,
        source_span=source,
    )


def _identity(doc_id: str, clause: str, value: str, when: str, source: str, role: str) -> str:
    material = "\n".join((doc_id, clause, value, when, source, role))
    return hashlib.sha256(material.encode()).hexdigest()


def _in_force(row: Provision, as_of: str) -> bool:
    if row.effective_from and row.effective_from > as_of:
        return False
    if row.effective_to and row.effective_to <= as_of:
        return False
    return True


def _replaced(row: Provision, rows: list[Provision], as_of: str) -> bool:
    if not row.superseded_by:
        return False
    target = next((item for item in rows if item.provision_id == row.superseded_by), None)
    if target is None:
        return True
    return _in_force(target, as_of) or bool(target.effective_from and target.effective_from <= as_of)


def _cycle(rows: list[Provision]) -> bool:
    by_id = {row.provision_id: row for row in rows}
    for start in rows:
        seen = set()
        current = start.superseded_by
        while current:
            if current in seen:
                return True
            if current == start.provision_id:
                return True
            seen.add(current)
            nxt = by_id.get(current)
            if nxt is None:
                break
            current = nxt.superseded_by
    return False


def mentioned_values(text: str) -> list[str]:
    values: list[str] = []
    for match in _AMOUNT.findall(text):
        value = _normal_value(match)
        if value and value not in values:
            values.append(value)
    return values


def mentioned_pair(text: str) -> tuple[str, str] | None:
    """The old and new values named in one sentence, with no document context."""
    match = _ENHANCED.search(text)
    if match is not None:
        old = _normal_value(match.group("old"))
        new = _normal_value(match.group("new"))
        if old and new and old != new:
            return old, new
    return _two_amounts(text)


def phrase_hits(text: str) -> bool:
    folded = text.casefold()
    return any(phrase in folded for phrase in SUPERSESSION_PHRASES)


def supersession_date(text: str) -> str:
    """The date named after a supersession phrase. An obligation sentence has none."""
    if not phrase_hits(text):
        return ""
    folded = text.casefold()
    start = min(folded.find(phrase) for phrase in SUPERSESSION_PHRASES if phrase in folded)
    return _first_date(text[start:])


def _apply_phrases(rows: list[Provision], spans: list) -> None:
    for span in spans:
        when = supersession_date(span.text)
        if not when:
            continue
        for index, row in enumerate(rows):
            if row.effective_from != when or row.superseded_by:
                continue
            later = [
                item
                for item in rows
                if item.clause_key == row.clause_key
                and item.provision_id != row.provision_id
                and item.effective_from
                and item.effective_from > when
            ]
            if len(later) != 1:
                continue
            rows[index] = replace(
                row,
                effective_to=later[0].effective_from,
                superseded_by=later[0].provision_id,
                status="superseded",
            )


def _dedupe(rows: list[Provision]) -> list[Provision]:
    kept: list[Provision] = []
    seen: dict[tuple, str] = {}
    for row in rows:
        key = (row.clause_key, row.value_text, row.effective_from, row.effective_to, row.status)
        if key in seen:
            continue
        seen[key] = row.provision_id
        kept.append(row)
    rewritten = []
    for row in kept:
        link = row.superseded_by
        if link and link not in {item.provision_id for item in kept}:
            target = next((item for item in rows if item.provision_id == link), None)
            if target is not None:
                key = (
                    target.clause_key,
                    target.value_text,
                    target.effective_from,
                    target.effective_to,
                    target.status,
                )
                link = seen.get(key, link)
        rewritten.append(replace(row, superseded_by=link))
    return rewritten
