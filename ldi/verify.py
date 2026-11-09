"""Check a model answer with ordinary Python. The model is not called again."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from ldi.chunk import normalise_text
from ldi.store import Hit

CHECK_INSTRUCTION = (
    "End the answer with a Checks block. Use the file's own numbers and words.\n"
    "Checks:\n"
    "- calc: <left> + <right> = <result>\n"
    "- date: <start> + <days> days = <end>\n"
    "- date: <start> to <end> = <days> days\n"
    "- within: <date> <= <start> + <days> days = <true or false>\n"
    '- cite: "exact words from the file"'
)

NOT_FOUND = "I could not find supporting evidence in the indexed text"
# A line that tells the model what to say is not evidence, even when the file contains it.
_INJECTED = re.compile(
    r"\b(?:ignore\s+(?:all\s+)?previous\s+instructions|system\s+note\s+to\s+ai|disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions)\b",
    re.IGNORECASE,
)

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_SHORT_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_NUMBER = re.compile(
    r"(?<![\d.])([+-]?(?:\d{1,3}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s*(lakh|crore|%|percent))?(?![\d])",
    re.IGNORECASE,
)
_PERCENT_OF = re.compile(
    r"^(?P<pct>.+?)\s*(?:%|percent)\s+of\s+(?P<base>.+?)\s*=\s*(?P<result>.+)$",
    re.IGNORECASE,
)
_CALC = re.compile(
    r"^(?P<left>.+?)\s*(?P<op>[+\-*/])\s*(?P<right>.+?)\s*=\s*(?P<result>.+)$"
)
_ISO = r"\d{4}-\d{2}-\d{2}"
_DEADLINE = re.compile(
    rf"^(?P<start>{_ISO})\s*\+\s*(?P<days>\d+)\s+days\s*=\s*(?P<end>{_ISO})$",
    re.IGNORECASE,
)
_ELAPSED = re.compile(
    rf"^(?P<start>{_ISO})\s+to\s+(?P<end>{_ISO})\s*=\s*(?P<count>\d+)\s+days$",
    re.IGNORECASE,
)
_BUSINESS = re.compile(
    rf"^(?P<start>{_ISO})\s+to\s+(?P<end>{_ISO})\s*=\s*(?P<count>\d+)\s+business days"
    rf"(?:\s+holidays\s+(?P<holidays>.+))?$",
    re.IGNORECASE,
)
_WITHIN = re.compile(
    rf"^(?P<left>{_ISO})\s*<=\s*(?P<start>{_ISO})\s*\+\s*(?P<days>\d+)\s+days"
    rf"(?:\s*=\s*(?P<claim>true|false))?$",
    re.IGNORECASE,
)
_LINE = re.compile(r"^-\s*(calc|date|within|cite):\s*(.*)$", re.IGNORECASE)
_LOCATOR = re.compile(
    r"(?:page\s+\d+|\d+|[\w.\- ]+\.(?:txt|md|markdown|rst|pdf))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Check:
    kind: str
    status: str
    wrote: str
    computed: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "status": self.status,
            "wrote": self.wrote,
            "computed": self.computed,
            "detail": self.detail,
        }


def generation_system(base: str, has_source: bool) -> str:
    if not has_source:
        return base
    return f"{base}\n\n{CHECK_INSTRUCTION}"


def verify(reply: str, source: str) -> list[Check]:
    """Score the Checks block. A missing block is Insufficient, not a pass."""
    found = re.search(r"Checks[ \t]*:", reply, flags=re.IGNORECASE)
    if found is None:
        return [_missing()]
    body = reply[found.end() :]
    prose = reply[: found.start()]
    present = _numbers(source)
    checks: list[Check] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            if checks:
                break
            continue
        match = _LINE.match(line)
        if match is None:
            if checks:
                break
            continue
        kind = match.group(1).lower()
        payload = match.group(2).strip()
        checks.append(_one(kind, payload, source, present, prose))
    if not checks:
        return [_missing()]
    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["Verification"]
    for check in checks:
        lines.append(
            f"  {check.kind}: {check.status}. "
            f"Model wrote {check.wrote}. Python computed {check.computed}."
        )
    return "\n".join(lines)


def _missing() -> Check:
    return Check("reply", "Insufficient", "", "", "No Checks block")


def _block(reply: str) -> str | None:
    match = re.search(r"Checks[ \t]*:", reply, flags=re.IGNORECASE)
    if match is None:
        return None
    return reply[match.end() :]


def _one(kind: str, payload: str, source: str, present: set[Decimal], prose: str = "") -> Check:
    if kind == "calc":
        return _calc(payload, present)
    if kind == "date":
        return _date(payload, present, source)
    if kind == "within":
        return _within(payload, present, source)
    if kind == "cite":
        return _cite(payload, source, prose)
    return Check(kind, "Insufficient", payload, "", "That check could not be read.")


def _calc(payload: str, present: set[Decimal]) -> Check:
    percent = _PERCENT_OF.match(payload)
    if percent is not None:
        pct = _operand(percent.group("pct"))
        base = _operand(percent.group("base"))
        wrote = _operand(percent.group("result"))
        if pct is None or base is None or wrote is None or pct[1] == "percent":
            return Check("calc", "Insufficient", payload, "", "That calculation could not be read.")
        operands = (pct[0], base[0])
        computed = base[0] * pct[0] / Decimal(100)
        return _scored("calc", wrote[0], computed, operands, present)
    calc = _CALC.match(payload)
    if calc is None:
        return Check("calc", "Insufficient", payload, "", "That calculation could not be read.")
    left = _operand(calc.group("left"))
    right = _operand(calc.group("right"))
    wrote = _operand(calc.group("result"))
    if left is None or right is None or wrote is None:
        return Check("calc", "Insufficient", payload, "", "That calculation could not be read.")
    op = calc.group("op")
    if op == "/" and right[0] == 0:
        return _missing_or_fail(left[0], right[0], present, "calc", _text(wrote[0]), "undefined")
    computed = _apply(left[0], left[1], right[0], right[1], op)
    if computed is None:
        return Check("calc", "Insufficient", _text(wrote[0]), "", "That calculation could not be read.")
    return _scored("calc", wrote[0], computed, (left[0], right[0]), present)


def _date(payload: str, present: set[Decimal], source: str) -> Check:
    business = _BUSINESS.match(payload)
    if business is not None:
        return _business(business, present, source)
    elapsed = _ELAPSED.match(payload)
    if elapsed is not None:
        start = _iso(elapsed.group("start"))
        end = _iso(elapsed.group("end"))
        if start is None or end is None or not _dates_in(source, start, end):
            return Check("date", "Insufficient", elapsed.group("count"), "", "A date is not in the file.")
        computed = Decimal((end - start).days)
        wrote = Decimal(elapsed.group("count"))
        return _compared("date", wrote, computed)
    deadline = _DEADLINE.match(payload)
    if deadline is None:
        return Check("date", "Insufficient", payload, "", "That date check could not be read.")
    start = _iso(deadline.group("start"))
    wrote = _iso(deadline.group("end"))
    days = Decimal(deadline.group("days"))
    if start is None or wrote is None or days not in present or not _dates_in(source, start):
        return Check(
            "date",
            "Insufficient",
            deadline.group("end"),
            "",
            "A date or day count is not in the file.",
        )
    computed = start + timedelta(days=int(days))
    status = "Pass" if computed == wrote else "Fail"
    return Check("date", status, wrote.isoformat(), computed.isoformat())


def _business(match: re.Match[str], present: set[Decimal], source: str) -> Check:
    start = _iso(match.group("start"))
    end = _iso(match.group("end"))
    holidays = _holiday_dates(match.group("holidays") or "")
    if start is None or end is None or holidays is None:
        return Check("date", "Insufficient", match.group("count"), "", "That date check could not be read.")
    needed = [start, end, *holidays]
    if not _dates_in(source, *needed):
        return Check("date", "Insufficient", match.group("count"), "", "A date is not in the file.")
    wrote = Decimal(match.group("count"))
    computed = Decimal(_business_days(start, end, set(holidays)))
    return _compared("date", wrote, computed)


def _within(payload: str, present: set[Decimal], source: str) -> Check:
    match = _WITHIN.match(payload)
    if match is None:
        return Check("within", "Insufficient", payload, "", "That deadline check could not be read.")
    left = _iso(match.group("left"))
    start = _iso(match.group("start"))
    days = Decimal(match.group("days"))
    if left is None or start is None or days not in present or not _dates_in(source, left, start):
        return Check(
            "within",
            "Insufficient",
            match.group("claim") or "true",
            "",
            "A date or day count is not in the file.",
        )
    actual = left <= start + timedelta(days=int(days))
    wrote = (match.group("claim") or "true").lower()
    computed = "true" if actual else "false"
    status = "Pass" if wrote == computed else "Fail"
    return Check("within", status, wrote, computed)


def _cite(payload: str, source: str, prose: str = "") -> Check:
    if len(payload) < 2 or payload[0] not in "\"'" or payload[-1] != payload[0]:
        return Check("cite", "Insufficient", payload, "", "A citation needs a quoted phrase.")
    quote = payload[1:-1].strip()
    if not quote:
        return Check("cite", "Insufficient", "", "", "A citation needs a quoted phrase.")
    if _LOCATOR.fullmatch(quote):
        return Check("cite", "Fail", quote, "absent", "A file name or page number is not a citation.")
    found = _collapse(quote) in _collapse(source)
    if not found:
        return Check("cite", "Fail", quote, "absent")
    if _INJECTED.search(_covering_sentence(source, quote)):
        return Check(
            "cite",
            "Fail",
            quote,
            "present",
            "The quoted words sit inside an instruction.",
        )
    if prose.strip() and not _quote_supports(prose, quote):
        return Check("cite", "Fail", quote, "present", "The quote does not support the answer.")
    return Check("cite", "Pass", quote, "present")


def _covering_sentence(source: str, quote: str) -> str:
    """The source sentence that contains the quote, not only the quoted words."""
    needle = _collapse(quote)
    parts = re.split(r"(?<=[.!?])\s+", source.strip())
    for part in parts:
        if needle and needle in _collapse(part):
            return part
    return source


def _quote_supports(prose: str, quote: str) -> bool:
    """The words before Checks have to agree with the quote, not merely sit beside it."""
    if _INJECTED.search(quote):
        return False
    if any(number not in _numbers(quote) for number in _numbers(prose)):
        return False
    if _content_words(prose) & _content_words(quote) and _negative(prose) != _negative(quote):
        return False
    return True


def _scored(
    kind: str,
    wrote: Decimal,
    computed: Decimal,
    operands: tuple[Decimal, ...],
    present: set[Decimal],
) -> Check:
    if any(item not in present for item in operands):
        return Check(kind, "Insufficient", _text(wrote), _text(computed), "An operand is not in the file.")
    return _compared(kind, wrote, computed)


def _missing_or_fail(
    left: Decimal,
    right: Decimal,
    present: set[Decimal],
    kind: str,
    wrote: str,
    computed: str,
) -> Check:
    if left not in present or right not in present:
        return Check(kind, "Insufficient", wrote, computed, "An operand is not in the file.")
    return Check(kind, "Fail", wrote, computed)


def _compared(kind: str, wrote: Decimal, computed: Decimal) -> Check:
    status = "Pass" if wrote == computed else "Fail"
    return Check(kind, status, _text(wrote), _text(computed))


def _apply(left: Decimal, left_kind: str, right: Decimal, right_kind: str, op: str) -> Decimal | None:
    if left_kind == "percent":
        left = left / Decimal(100)
    if right_kind == "percent":
        right = right / Decimal(100)
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/" and right != 0:
        return left / right
    return None


def _operand(text: str) -> tuple[Decimal, str] | None:
    match = _NUMBER.fullmatch(text.strip())
    if match is None:
        return None
    try:
        head = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    suffix = (match.group(2) or "").lower()
    if suffix == "lakh":
        return head * Decimal(100000), "plain"
    if suffix == "crore":
        return head * Decimal(10000000), "plain"
    if suffix in {"%", "percent"}:
        return head, "percent"
    return head, "plain"


def _numbers(text: str) -> set[Decimal]:
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        try:
            head = Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        suffix = (match.group(2) or "").lower()
        found.add(head)
        if suffix == "lakh":
            found.add(head * Decimal(100000))
        elif suffix == "crore":
            found.add(head * Decimal(10000000))
    return found


def _iso(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _holiday_dates(text: str) -> list[date] | None:
    if not text.strip():
        return []
    holidays: list[date] = []
    for part in text.split(","):
        parsed = _iso(part.strip())
        if parsed is None:
            return None
        holidays.append(parsed)
    return holidays


def _dates_in(source: str, *values: date) -> bool:
    folded = source.lower()
    return all(any(form in folded for form in _date_forms(value)) for value in values)


def _date_forms(value: date) -> list[str]:
    month = _MONTHS[value.month - 1]
    short = _SHORT_MONTHS[value.month - 1]
    day = str(value.day)
    year = str(value.year)
    return [
        form.lower()
        for form in (
            value.isoformat(),
            f"{day} {month} {year}",
            f"{day} {short} {year}",
            f"{month} {day}, {year}",
            f"{short} {day}, {year}",
            f"{day}/{value.month:02d}/{year}",
            f"{day}/{value.month}/{year}",
            f"{day}-{value.month:02d}-{year}",
        )
    ]


def _business_days(start: date, end: date, holidays: set[date]) -> int:
    if end < start:
        return -_business_days(end, start, holidays)
    count = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5 and cursor not in holidays:
            count += 1
    return count


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", normalise_text(text)).strip()


def _text(value: Decimal) -> str:
    if value == value.to_integral():
        return format(value.quantize(Decimal(1)), "f")
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


_ABSTAIN = re.compile(
    r"(do(?:es)? not contain|not in the (?:file|passage|document|indexed)|"
    r"cannot find|can't find|could not find|no indexed passage)",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(?:not|no|never|without|excludes|excluded|excluding|exclude|cannot|can't)\b",
    re.IGNORECASE,
)
_CONTENT_STOP = frozenset(
    "the a an is are was were be of to for in on at by with from and or "
    "it this that do does did has have had".split()
)
_TEXTUAL_DATE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Release:
    kind: str
    text: str
    reason: str = ""
    paths: tuple[str, ...] = ()


def format_not_found(gaps: list[tuple[str, int | None, str]]) -> str:
    lines = [NOT_FOUND]
    if not gaps:
        lines.append("Coverage: no files or pages were skipped.")
        return "\n".join(lines)
    parts: list[str] = []
    for path, page, reason in gaps:
        if page is None:
            parts.append(f"{path} ({reason})")
        else:
            parts.append(f"{path} page {page} ({reason})")
    lines.append("Coverage: " + "; ".join(parts))
    return "\n".join(lines)


def format_excerpt(hits: list[Hit]) -> str:
    if not hits:
        return NOT_FOUND
    lines: list[str] = []
    for hit in hits:
        where = hit.path if hit.page is None else f"{hit.path}, page {hit.page}"
        body = _collapse(hit.text)
        if len(body) > 900:
            body = body[:900].rstrip() + "…"
        lines.append(where)
        lines.append(body)
    return "\n".join(lines)


def release_answer(reply: str, hits: list[Hit], question: str = "") -> Release:
    """Release a code-built citation, or withhold the reply for one retry.

    The model's words are not the quote. The first abstention is one retry.
    A second failure becomes a same-field conflict, one overlapping sentence,
    or the exact not-found line.
    """
    if not hits:
        return Release("not_found", NOT_FOUND)
    from ldi.evidence import build_evidence, release_from_evidence

    items = build_evidence(question, hits)
    if not items:
        return Release("not_found", NOT_FOUND)
    return release_from_evidence(reply, items, question)


def passage_integers(text: str) -> set[int]:
    found: set[int] = set()
    for match in _NUMBER.finditer(text):
        try:
            head = Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        suffix = (match.group(2) or "").lower()
        if suffix == "lakh":
            head *= Decimal(100000)
        elif suffix == "crore":
            head *= Decimal(10000000)
        if head == head.to_integral():
            found.add(int(head))
    return found


def conflict_banner(hits: list[Hit]) -> str:
    grouped: list[tuple[str, set[int]]] = []
    for hit in hits:
        if hit.superseded:
            continue
        numbers = passage_integers(hit.text)
        if not numbers:
            continue
        where = hit.path if hit.page is None else f"{hit.path}, page {hit.page}"
        grouped.append((where, numbers))
    if len(grouped) < 2:
        return ""
    if len({frozenset(numbers) for _, numbers in grouped}) < 2:
        return ""
    lines = ["Conflict"]
    for where, numbers in grouped:
        shown = ", ".join(str(number) for number in sorted(numbers))
        lines.append(f"{where}: {shown}")
    lines.append("The passages disagree. Show both values.")
    return "\n".join(lines)


def _prose(reply: str) -> str:
    match = re.search(r"Checks[ \t]*:", reply, flags=re.IGNORECASE)
    if match is None:
        return reply
    return reply[: match.start()]


def _sentences(prose: str) -> list[str]:
    cleaned = prose.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _supported(prose: str, checks: list[Check], source: str) -> bool:
    cites = [item for item in checks if item.kind == "cite" and item.status == "Pass"]
    sentences = _sentences(prose)
    if not cites or not sentences:
        return False
    if any(not _sentence_ok(sentence, cites) for sentence in sentences):
        return False
    if not _dates_ok(prose, checks, source):
        return False
    return _numbers_ok(prose, checks, source)


def _sentence_ok(sentence: str, cites: list[Check]) -> bool:
    for cite in cites:
        quote = cite.wrote
        if _contradicts(sentence, quote):
            continue
        if _content_words(sentence) & _content_words(quote):
            return True
        if quote and _collapse(quote) in _collapse(sentence):
            return True
    return False


def _contradicts(sentence: str, quote: str) -> bool:
    if not (_content_words(sentence) & _content_words(quote)):
        return False
    return _negative(sentence) != _negative(quote)


def _negative(text: str) -> bool:
    return _NEGATIVE.search(text) is not None


def _content_words(text: str) -> set[str]:
    words = {word.lower() for word in re.findall(r"[^\W\d_]+", text)}
    return {word for word in words if len(word) > 3 and word not in _CONTENT_STOP}


def _dates_ok(prose: str, checks: list[Check], source: str) -> bool:
    return all(_date_allowed(value, checks, source) for value in _prose_dates(prose))


def _date_allowed(value: date, checks: list[Check], source: str) -> bool:
    if _dates_in(source, value):
        return True
    for check in checks:
        if check.status != "Pass":
            continue
        for token in (check.wrote, check.computed):
            if _iso(token) == value:
                return True
    return False


def _prose_dates(text: str) -> list[date]:
    found: list[date] = []
    for match in re.finditer(r"\d{4}-\d{2}-\d{2}", text):
        parsed = _iso(match.group(0))
        if parsed is not None:
            found.append(parsed)
    for match in _TEXTUAL_DATE.finditer(text):
        parsed = _month_date(match.group(1), match.group(2), match.group(3))
        if parsed is not None:
            found.append(parsed)
    return found


def _month_date(day: str, month: str, year: str) -> date | None:
    try:
        month_index = [name.lower() for name in _MONTHS].index(month.lower()) + 1
        return date(int(year), month_index, int(day))
    except ValueError:
        return None


def _numbers_ok(prose: str, checks: list[Check], source: str) -> bool:
    allowed = _numbers(source)
    for check in checks:
        if check.kind != "calc" or check.status != "Pass":
            continue
        for token in (check.wrote, check.computed):
            parsed = _operand(token)
            if parsed is not None:
                allowed.add(parsed[0])
    masked = _mask_allowed_dates(prose, checks, source)
    for match in _NUMBER.finditer(masked):
        parsed = _operand(match.group(0))
        if parsed is None:
            continue
        if parsed[0] not in allowed:
            return False
    return True


def _mask_allowed_dates(prose: str, checks: list[Check], source: str) -> str:
    masked = prose
    for value in _prose_dates(prose):
        if not _date_allowed(value, checks, source):
            continue
        masked = masked.replace(value.isoformat(), " ")
        for form in _date_forms(value):
            masked = re.sub(re.escape(form), " ", masked, flags=re.IGNORECASE)
    return masked
