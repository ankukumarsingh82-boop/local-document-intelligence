"""Turn retrieved passages into evidence ids, then build the citation in code."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ldi.chunk import normalise_text
from ldi.grammar import parse_answer_json
from ldi.store import Hit, content_terms
from ldi.verify import (
    NOT_FOUND,
    Release,
    _INJECTED,
    _NUMBER,
    _content_words,
    _dates_in,
    _negative,
    _prose_dates,
)

EVIDENCE_LIMIT = 12
SENTENCE_CHARS = 320
_CELL_LINE = re.compile(
    r"^(?P<cell>[A-Za-z0-9][^!\n]{0,40}![A-Z]+[0-9]+)\s+(?P<body>\S.*)$"
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CALC_QUESTION = re.compile(
    r"\b(?:add|added|adding|summed|plus|difference|minus|subtract|subtracted|multiplied|divided)\b",
    re.IGNORECASE,
)
# A year or a second benefit in the same policy is not a second value for the asked field.
_GENERIC_FIELD = frozenset(
    """
    day days week weeks month months year years
    paid leave policy employee employees company
    current version section page number per under period
    must within many before after same this sheet
    service services firm provide provides
    """.split()
)
_PERIOD = re.compile(r"\b(?:q[1-4]|fy\d{2}|\d{4})\b", re.IGNORECASE)
_ADD = re.compile(r"\b(?:add|added|adding|plus)\b", re.IGNORECASE)
_SUBTRACT = re.compile(r"\b(?:difference|minus|subtract|subtracted)\b", re.IGNORECASE)
_CHANGE = re.compile(r"\b(?:change|changed|changing)\b", re.IGNORECASE)
_PERIOD_HEADER = re.compile(r"Q[1-4]\s+FY\d{2}", re.IGNORECASE)
_GRID_TOKEN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_BOILER = frozenset(
    "fictional synthetic test data record real company inr note clause vendor".split()
)
# "each year" and "pay" are the same field as an annual fee. "paid" is not.
# "planned" is the Budget column, not the Actual column.
_ALIAS = {
    "year": "annual",
    "yearly": "annual",
    "pay": "fee",
    "payable": "fee",
    "planned": "budget",
}
_ROLES = frozenset(
    """
    chief financial officer president director manager head founder
    counsel secretary treasurer controller ceo cfo cto coo
    """.split()
)
_SHEET_CELL = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)!([A-Z]{1,3}[0-9]{1,4})\b")
_BARE_CELL = re.compile(r"\b([A-Z]{1,3}[0-9]{1,4})\b")
_NAMED_SHEET = re.compile(r"\b(Budget|Headcount)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Evidence:
    id: str
    path: str
    locator: str
    text: str
    superseded: bool = False


def build_evidence(question: str, hits: list[Hit], limit: int = EVIDENCE_LIMIT) -> list[Evidence]:
    """Split current passages into short items and keep the ones the question names."""
    if any(hit.source_lines for hit in hits):
        return _evidence_from_source(question, hits, limit)
    drafts: list[tuple[str, str, str]] = []
    for hit in hits:
        if hit.superseded:
            continue
        drafts.extend((hit.path, locator, text) for text, locator in _pieces(hit))
    terms = {term.casefold() for term in content_terms(question)}
    ranked = sorted(
        enumerate(drafts),
        key=lambda item: (-_overlap(terms, item[1][2]), item[0]),
    )
    chosen = [drafts[index] for index, _draft in ranked[:limit]]
    return [
        Evidence(f"E{number}", path, locator, text)
        for number, (path, locator, text) in enumerate(chosen, start=1)
    ]


_CLAUSE_HEADING = re.compile(
    r"^(?:(?:clause|section)\s+)?(?:\d+(?:\.\d+)+|annexure\s+\S+|schedule\s+\S+)\b",
    re.IGNORECASE,
)


def attach_source(question: str, hits: list[Hit], load_spans) -> list[Hit]:
    """Replace each hit's evidence lines with the original source spans."""
    attached = []
    for hit in hits:
        lines, ids, pinned = _unit_for_hit(question, hit, load_spans(hit.path))
        attached.append(
            replace(
                hit,
                source_lines=tuple(lines),
                source_span_ids=tuple(ids),
                pinned_span_ids=tuple(pinned),
            )
        )
    return attached


def _evidence_from_source(question: str, hits: list[Hit], limit: int) -> list[Evidence]:
    """Quote source lines. Pinned rows stay even when they pass the line cap."""
    terms = {term.casefold() for term in content_terms(question)}
    pinned: list[tuple[str, str, str]] = []
    rest: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        if hit.superseded:
            continue
        if not hit.source_lines:
            for text, locator in _pieces(hit):
                key = (hit.path, locator)
                if key in seen:
                    continue
                seen.add(key)
                rest.append((hit.path, locator, text))
            continue
        pin_ids = set(hit.pinned_span_ids)
        for text, span_id in zip(hit.source_lines, hit.source_span_ids):
            key = (hit.path, span_id)
            if key in seen:
                continue
            seen.add(key)
            row = (hit.path, _span_place(hit, span_id), text)
            if span_id in pin_ids:
                pinned.append(row)
            else:
                rest.append(row)
    room = limit - len(pinned)
    if room < 0:
        chosen = pinned
    else:
        order = sorted(range(len(rest)), key=lambda index: (-_overlap(terms, rest[index][2]), index))
        chosen = pinned + [rest[index] for index in sorted(order[:room])]
    return [
        Evidence(f"E{number}", path, locator, text)
        for number, (path, locator, text) in enumerate(chosen, start=1)
    ]


def _span_place(hit: Hit, span_id: str) -> str:
    place = hit.path if hit.page is None else f"{hit.path}, page {hit.page}"
    return f"{place}, span {span_id}"


def _unit_for_hit(question: str, hit: Hit, spans) -> tuple[list[str], list[str], list[str]]:
    wanted = {item for item in f"{hit.span_ids} {hit.span_id}".split() if item}
    chosen = [span for span in spans if span.span_id in wanted]
    if not chosen:
        return [], [], []
    table_ids = {span.table_id for span in chosen if span.table_id}
    if table_ids:
        return _table_unit(question, [span for span in spans if span.table_id in table_ids])
    return _clause_unit(question, spans, chosen)


def _table_unit(question: str, cells) -> tuple[list[str], list[str], list[str]]:
    from ldi import config
    from ldi.spans import table_header_lines

    by_line: dict[int, list] = {}
    for cell in cells:
        by_line.setdefault(cell.line, []).append(cell)
    ordered = []
    meta = []
    for line in sorted(by_line):
        row = sorted(by_line[line], key=lambda cell: (cell.col or 0, cell.span_id))
        ordered.append(row)
        meta.append((line, [(cell.text, cell.cell_type or "text") for cell in row]))
    titles, headers = table_header_lines(meta)
    title_set, header_set = set(titles), set(headers)
    title_rows, header_rows, body_rows = [], [], []
    for row in ordered:
        line = row[0].line
        if line in title_set:
            title_rows.append(row)
        elif line in header_set:
            header_rows.append(row)
        else:
            body_rows.append(row)
    terms = [term.casefold() for term in content_terms(question)]
    matched = _best_rows(body_rows, terms, config.EVIDENCE_MATCHED_ROWS)
    whole = "\n".join(cell.text for row in ordered for cell in row)
    use = ordered if len(whole) <= config.TABLE_EVIDENCE_CHARS else title_rows + header_rows + matched
    pin_ids = {cell.span_id for row in header_rows + matched for cell in row}
    return _span_lines(use, pin_ids)


def _best_rows(rows: list, terms: list[str], limit: int) -> list:
    """Up to `limit` body rows, preferring a term that is rare inside the table."""
    if limit <= 0 or not rows:
        return []
    folded = [" ".join(cell.text for cell in row).casefold() for row in rows]
    rarity = {
        term: 1.0 / (1 + sum(term in text for text in folded))
        for term in terms
    }
    scored = []
    for index, text in enumerate(folded):
        score = sum(rarity[term] for term in terms if term in text)
        if score > 0:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(index for _score, index in scored[:limit])
    return [rows[index] for index in chosen]


def _clause_unit(question: str, spans, chosen) -> tuple[list[str], list[str], list[str]]:
    prose = [span for span in spans if not span.table_id]
    prose.sort(key=lambda span: (span.block, span.line, span.col or 0, span.span_id))
    positions = {span.span_id: index for index, span in enumerate(prose)}
    indexes = [positions[span.span_id] for span in chosen if span.span_id in positions]
    if not indexes:
        return _span_lines([chosen], {span.span_id for span in chosen})
    start = min(indexes)
    while start > 0 and _clause_level(prose[start].text) is None:
        start -= 1
    if _clause_level(prose[start].text) is None:
        # No enclosing heading. Quote the retrieved lines, not the whole document.
        window = [prose[index] for index in range(min(indexes), max(indexes) + 1)]
    else:
        level = _clause_level(prose[start].text) or 0
        end = max(indexes) + 1
        while end < len(prose):
            later = _clause_level(prose[end].text)
            if later is not None and level and later <= level:
                break
            end += 1
        window = prose[start:end]
    terms = [term.casefold() for term in content_terms(question)]
    hit_ids = {span.span_id for span in chosen}

    def overlap(span) -> int:
        folded = span.text.casefold()
        return sum(1 for term in terms if term in folded)

    ranked = sorted(window, key=lambda span: (-overlap(span), prose.index(span)))
    pinned: list[str] = []

    def take(span) -> None:
        if span.span_id in pinned or len(pinned) >= EVIDENCE_LIMIT:
            return
        pinned.append(span.span_id)

    if window and _clause_level(window[0].text) is not None:
        take(window[0])
    for span in ranked:
        if span.span_id in hit_ids and overlap(span) > 0:
            take(span)
    for span in ranked:
        if overlap(span) > 0:
            take(span)
    if not pinned:
        for span in chosen:
            take(span)
    return _span_lines([window], set(pinned))


def _clause_level(text: str) -> int | None:
    if _CLAUSE_HEADING.match(text.strip()) is None:
        return None
    numbers = re.match(
        r"^(?:(?:clause|section)\s+)?(\d+(?:\.\d+)+)",
        text.strip(),
        re.IGNORECASE,
    )
    if numbers:
        return numbers.group(1).count(".") + 1
    return 1


def _span_lines(rows: list, pinned: set[str]) -> tuple[list[str], list[str], list[str]]:
    lines: list[str] = []
    ids: list[str] = []
    kept: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for span in row:
            if span.span_id in seen:
                continue
            seen.add(span.span_id)
            lines.append(span.text)
            ids.append(span.span_id)
            if span.span_id in pinned:
                kept.append(span.span_id)
    return lines, ids, kept


def format_evidence(items: list[Evidence]) -> str:
    blocks = [f"[{item.id}] {item.locator}\n{item.text}" for item in items]
    return "Evidence:\n\n" + "\n\n".join(blocks)


def render_citation(answer: str, items: list[Evidence]) -> str:
    """The quote is the evidence sentence, so it matches the index."""
    lines = [answer.strip()]
    for item in items:
        lines.append(f'- cite: "{item.text}"')
        lines.append(f"  {item.locator}")
    return "\n".join(lines)


def format_conflict(left: Evidence, right: Evidence) -> str:
    return "\n".join(
        [
            "Conflict",
            left.locator,
            left.text,
            right.locator,
            right.text,
        ]
    )


def conflicting(question: str, items: list[Evidence]) -> tuple[Evidence, Evidence] | None:
    """Two current values of one field. A shared word from another field does not count."""
    ranked: list[tuple[Evidence, set[int]]] = []
    for item in items:
        if item.superseded or _multi_number_grid(item.text):
            continue
        numbers = integer_values(item.text)
        if len(numbers) != 1:
            continue
        ranked.append((item, numbers))
    for index, (left, left_numbers) in enumerate(ranked):
        for right, right_numbers in ranked[index + 1 :]:
            if left_numbers == right_numbers or not _same_field(left, right, items):
                continue
            if _units_disagree(left.text, right.text):
                continue
            if not _question_selects(question, left, right, items):
                continue
            return left, right
    return None


def fallback_release(question: str, items: list[Evidence]) -> Release:
    located = _cell_from_question(question, items)
    if located is not None:
        return located
    asked = _asked_cells(question)
    pool = items
    if asked:
        pool = [item for item in items if _is_asked_cell(item, asked)]
        if not pool:
            return Release("not_found", NOT_FOUND)
    ranked = sorted(
        enumerate(pool),
        key=lambda pair: (-_overlap_score(question, pair[1].text, items), pair[0]),
    )
    for _index, item in ranked:
        if item.superseded or _multi_number_grid(item.text) or _INJECTED.search(item.text):
            continue
        if not _sentence_overlaps(question, item.text, items):
            continue
        if not _periods_covered(question, [_period_blob(item.text, items)]):
            continue
        return Release("answer", render_citation(item.text, [item]), paths=(item.path,))
    return Release("not_found", NOT_FOUND)


def release_from_evidence(reply: str, items: list[Evidence], question: str) -> Release:
    by_id = {item.id: item for item in items}
    parsed = parse_answer_json(reply, set(by_id))
    pair = conflicting(question, items)
    arithmetic = _CALC_QUESTION.search(question) is not None
    if arithmetic and parsed is not None and parsed.status != "not_found" and parsed.calc:
        outcome = _calculation(parsed.calc, by_id)
        if outcome is not None:
            cited = [by_id[item_id] for item_id in dict.fromkeys(re.findall(r"E\d+", parsed.calc))]
            if any(_shares(question, item.text) for item in cited):
                text, paths = outcome
                return Release("answer", text, paths=paths)
    if arithmetic:
        computed = _operation_from_question(question, items)
        if computed is not None:
            return computed
    changed = _change_from_question(question, items)
    if changed is not None:
        return changed
    located = _cell_from_question(question, items)
    if located is not None:
        return located
    if parsed is not None and (parsed.status == "not_found" or not parsed.evidence):
        return Release(
            "retry",
            "",
            reason="Choose an evidence id that answers the question.",
        )
    if pair is not None and not arithmetic:
        return Release(
            "conflict",
            format_conflict(pair[0], pair[1]),
            paths=(pair[0].path, pair[1].path),
        )
    if parsed is None:
        return Release(
            "retry",
            "",
            reason="The reply was not one JSON object with status, evidence, and answer.",
        )
    if any(item_id not in by_id for item_id in parsed.evidence):
        return Release("retry", "", reason="An evidence id is not in the list.")
    chosen = [by_id[item_id] for item_id in dict.fromkeys(parsed.evidence)]
    chosen = [
        item
        for item in chosen
        if _sentence_overlaps(question, item.text, items) and not _INJECTED.search(item.text)
    ]
    if len(chosen) > 1:
        scores = [(_overlap_score(question, item.text, items), item) for item in chosen]
        best = max(score for score, _item in scores)
        if best > 0:
            chosen = [item for score, item in scores if score == best]
    asked = _asked_cells(question)
    if asked and any(_is_asked_cell(item, asked) for item in items):
        cells = [item for item in chosen if _is_asked_cell(item, asked)]
        if not cells:
            return Release("retry", "", reason="The question asks for a spreadsheet cell.")
        chosen = cells
    if not chosen:
        return Release(
            "retry",
            "",
            reason="The chosen evidence does not name the subject of the question.",
        )
    if any(item.superseded for item in chosen):
        return Release("retry", "", reason="A chosen evidence id is superseded.")
    if not any(_shares(question, item.text) for item in chosen):
        return Release(
            "retry",
            "",
            reason="The chosen evidence does not share a word with the question.",
        )
    if not _periods_covered(question, [_period_blob(item.text, items) for item in chosen]):
        return Release(
            "retry",
            "",
            reason="A period in the question is not in the chosen evidence.",
        )
    if not _mentions_focus(question, [item.text for item in chosen]):
        return Release(
            "retry",
            "",
            reason="The chosen evidence does not name the subject of the question.",
        )
    if parsed.calc:
        outcome = _calculation(parsed.calc, by_id)
        if outcome is None:
            return Release(
                "retry",
                "",
                reason="The calculation must use evidence ids that each contain one number.",
            )
        text, paths = outcome
        return Release("answer", text, paths=paths)
    answer = normalise_text(parsed.answer).strip()
    if chosen and not _content_words(answer) and not number_values(answer):
        answer = chosen[0].text
    reason = _gate(answer, chosen, question, items)
    if reason:
        return Release("retry", "", reason=reason)
    paths = tuple(dict.fromkeys(item.path for item in chosen))
    return Release("answer", render_citation(answer, chosen), paths=paths)


def number_values(text: str) -> set[Decimal]:
    body = _body(text)
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(body):
        start = match.start(1)
        if start > 0 and (body[start - 1].isalpha() or body[start - 1] == "-"):
            continue
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


def integer_values(text: str) -> set[int]:
    """Written integers only. A decimal such as 3.40 crore is not a second fee."""
    body = _body(text)
    found: set[int] = set()
    for match in _NUMBER.finditer(body):
        start = match.start(1)
        if start > 0 and (body[start - 1].isalpha() or body[start - 1] == "-"):
            continue
        token = match.group(1)
        if "." in token:
            continue
        try:
            found.add(int(token.replace(",", "")))
        except ValueError:
            continue
    return found


def sole_number(text: str) -> Decimal | None:
    values = number_values(text)
    if len(values) != 1:
        return None
    return next(iter(values))


def evaluate_expression(expr: str, values: dict[str, Decimal]) -> Decimal | None:
    tokens = _tokenize(expr, set(values))
    if not tokens:
        return None
    try:
        value, index = _eval_expr(tokens, 0, values)
    except (IndexError, InvalidOperation, ZeroDivisionError):
        return None
    if index != len(tokens):
        return None
    return value


def _tokenize(expr: str, allowed: set[str]) -> list[str] | None:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
            continue
        if expr[i] in "+-*/()":
            tokens.append(expr[i])
            i += 1
            continue
        match = re.match(r"E\d+", expr[i:])
        if match is None or match.group(0) not in allowed:
            return None
        tokens.append(match.group(0))
        i += match.end()
    return tokens


def _eval_expr(tokens: list[str], index: int, values: dict[str, Decimal]) -> tuple[Decimal, int]:
    value, index = _eval_term(tokens, index, values)
    while index < len(tokens) and tokens[index] in "+-":
        op = tokens[index]
        right, index = _eval_term(tokens, index + 1, values)
        value = value + right if op == "+" else value - right
    return value, index


def _eval_term(tokens: list[str], index: int, values: dict[str, Decimal]) -> tuple[Decimal, int]:
    value, index = _eval_factor(tokens, index, values)
    while index < len(tokens) and tokens[index] in "*/":
        op = tokens[index]
        right, index = _eval_factor(tokens, index + 1, values)
        if op == "/" and right == 0:
            raise ZeroDivisionError
        value = value * right if op == "*" else value / right
    return value, index


def _eval_factor(tokens: list[str], index: int, values: dict[str, Decimal]) -> tuple[Decimal, int]:
    if index >= len(tokens):
        raise IndexError
    if tokens[index] == "(":
        value, index = _eval_expr(tokens, index + 1, values)
        if index >= len(tokens) or tokens[index] != ")":
            raise IndexError
        return value, index + 1
    if tokens[index] not in values:
        raise IndexError
    return values[tokens[index]], index + 1


def _calculation(expr: str, by_id: dict[str, Evidence]) -> tuple[str, tuple[str, ...]] | None:
    id_list = re.findall(r"E\d+", expr)
    if not id_list or any(item_id not in by_id for item_id in id_list):
        return None
    values: dict[str, Decimal] = {}
    for item_id in dict.fromkeys(id_list):
        number = sole_number(by_id[item_id].text)
        if number is None:
            return None
        values[item_id] = number
    result = evaluate_expression(expr, values)
    if result is None:
        return None
    chosen = [by_id[item_id] for item_id in dict.fromkeys(id_list)]
    if any(item.superseded for item in chosen):
        return None
    paths = tuple(dict.fromkeys(item.path for item in chosen))
    return render_citation(_format_decimal(result), chosen), paths


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _gate(answer: str, chosen: list[Evidence], question: str = "", items: list[Evidence] | None = None) -> str:
    if not answer:
        return "The answer was empty."
    source = normalise_text("\n".join(item.text for item in chosen))
    for value in _prose_dates(answer):
        if not _dates_in(source, value):
            return "A date in the answer is not in the chosen evidence."
    masked = answer
    for value in _prose_dates(answer):
        if _dates_in(source, value):
            masked = masked.replace(value.isoformat(), " ")
    allowed_numbers: set[Decimal] = set()
    allowed_words: set[str] = set()
    for item in chosen:
        allowed_numbers |= number_values(item.text)
        allowed_words |= _content_words(item.text)
    for number in number_values(masked):
        if number not in allowed_numbers:
            return "A number in the answer is not in the chosen evidence."
    placed = _placed_number(question, masked, chosen, items or chosen)
    if placed:
        return placed
    if _content_words(answer) - allowed_words:
        return "A word in the answer is not in the chosen evidence."
    if _polarity_conflict(answer, chosen):
        return "The answer reverses a negation in the chosen evidence."
    return ""


def _polarity_conflict(answer: str, chosen: list[Evidence]) -> bool:
    answer_words = _content_words(answer)
    answer_negative = _negative(answer)
    for item in chosen:
        if not (answer_words & _content_words(item.text)):
            continue
        if _negative(item.text) != answer_negative:
            return True
    return False


def _pieces(hit: Hit) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    prose: list[str] = []

    def flush() -> None:
        if not prose:
            return
        text = " ".join(prose)
        prose.clear()
        for sentence in _SENTENCE.split(text):
            clipped = _clip(sentence)
            if clipped:
                pieces.append((clipped, _place(hit, None)))

    for line in hit.text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _CELL_LINE.match(stripped)
        if match:
            flush()
            clipped = _clip(stripped)
            if clipped:
                pieces.append((clipped, _place(hit, match.group("cell"))))
            continue
        prose.append(stripped)
    flush()
    return pieces


def _place(hit: Hit, cell: str | None) -> str:
    if cell:
        return f"{hit.path}, {cell}"
    if hit.page is None:
        return hit.path
    return f"{hit.path}, page {hit.page}"


def _clip(text: str, limit: int = SENTENCE_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit // 2:
        return cut[:space]
    return cut


def _body(text: str) -> str:
    match = _CELL_LINE.match(text.strip())
    if match:
        return match.group("body")
    return text


def _fold_token(token: str) -> str:
    return unicodedata.normalize("NFC", token).casefold()


def _words(text: str) -> set[str]:
    return {_fold_token(word) for word in re.findall(r"\w+", text)}


def _overlap(terms: set[str], text: str) -> int:
    return sum(1 for term in terms if _term_matches(term, text))


def _shares(question: str, text: str) -> bool:
    terms = {term.casefold() for term in content_terms(question)}
    return bool(terms & _words(text))


def _mentions_focus(question: str, texts: list[str]) -> bool:
    """The question's subject, including a named party, has to appear in the field."""
    terms = _focus_terms(question, drop_names=False)
    if not terms:
        return True
    return any(_text_has_term(text, term) for text in texts for term in terms)


def _described(text: str, items: list[Evidence] | None = None) -> str:
    parsed = _parse_cell(text)
    if parsed is None or not items:
        return text
    sheet, column, row, _body = parsed
    header = _lookup_cell(items, sheet, "A", row) or ""
    column_header = _column_header(items, sheet, column, row) or ""
    return f"{text} {header} {column_header}"


def _sentence_overlaps(question: str, text: str, items: list[Evidence] | None = None) -> bool:
    """Most subject words have to match. A column header does not count as one of them."""
    described = _support_text(text, items)
    if not _coordinates_covered(question, described):
        return False
    terms = _subject_terms(question)
    if not terms:
        names = [name for name in _name_tokens(question) if len(name) >= 3]
        if names:
            return all(_term_matches(name, described) for name in names)
        return bool(_asked_cells(question))
    matched = [term for term in terms if _term_in_scope(term, described, text, items)]
    longest = max(len(term) for term in terms)
    distinctive = [term for term in terms if len(term) >= longest - 1]
    if any(term not in matched for term in distinctive):
        return False
    # A single shared word is not enough once the question names several subjects.
    return len(terms) < 3 or len(matched) * 2 > len(terms)


def _overlap_score(question: str, text: str, items: list[Evidence] | None = None) -> int:
    terms = _subject_terms(question)
    described = _support_text(text, items)
    return sum(1 for term in terms if _term_in_scope(term, described, text, items))


def _support_text(text: str, items: list[Evidence] | None) -> str:
    """The cell plus the labels that name it. One label is not a whole subject."""
    return _described(text, items)


def _file_key(item: Evidence) -> str:
    raw = item.path or item.locator
    if ", " in raw:
        head, tail = raw.rsplit(", ", 1)
        if tail.lower().startswith("page ") or "!" in tail:
            raw = head
    return raw


def _term_in_scope(
    term: str, described: str, text: str, items: list[Evidence] | None
) -> bool:
    if _term_matches(term, described):
        return True
    owner = next((item for item in items or [] if item.text == text), None)
    if owner is None:
        return False
    key = _file_key(owner)
    if not key:
        return False
    stem = Path(key).stem.replace("_", " ").replace("-", " ")
    if _term_matches(term, stem):
        return True
    for other in items or []:
        if _file_key(other) != key or not _heading(other.text, stem):
            continue
        if _term_matches(term, other.text):
            return True
    return False


def _heading(text: str, stem: str) -> bool:
    """A title line of this file, not another sentence that happens to be short."""
    if re.search(r"\d", text) or ":" in text:
        return False
    words = re.findall(r"[A-Za-z]+", text)
    if not words or len(words) > 12:
        return False
    return any(len(part) >= 4 and _term_matches(part, text) for part in stem.split())


def _period_blob(text: str, items: list[Evidence] | None) -> str:
    """A period may sit in the page heading rather than in the footnote itself."""
    described = _described(text, items)
    owner = next((item for item in items or [] if item.text == text), None)
    if owner is None:
        return described
    page = _page_of(owner)
    if page is None:
        return described
    parts = [described]
    for other in items or []:
        if other.text == text or other.path != owner.path or _page_of(other) != page:
            continue
        if _PERIOD.search(other.text):
            parts.append(other.text)
    return " ".join(parts)


def _page_of(item: Evidence) -> str | None:
    match = re.search(r"\bpage\s+(\d+)\b", item.locator, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _focus_terms(question: str, *, drop_names: bool) -> list[str]:
    names = _name_tokens(question) if drop_names else set()
    terms = [term for term in content_terms(question) if term.casefold() not in names]
    long = [term for term in terms if len(term) >= 4]
    return long or terms


def _name_tokens(question: str) -> set[str]:
    """A person's name is two or more proper words. Cloud on its own is a row label."""
    tokens = re.findall(r"\w+", question)
    flags = [_proper_token(index, token) for index, token in enumerate(tokens)]
    names: set[str] = set()
    for index, token in enumerate(tokens):
        if not flags[index]:
            continue
        neighbor = (index > 0 and flags[index - 1]) or (
            index + 1 < len(flags) and flags[index + 1]
        )
        if neighbor:
            names.add(_fold_token(token))
    return names


def _proper_token(index: int, token: str) -> bool:
    if index == 0 or _PERIOD.fullmatch(token):
        return False
    if _fold_token(token) in _ROLES:
        return False
    titled = len(token) > 1 and token[0].isupper() and token[1:].islower()
    return token.isupper() or titled


def _stem(word: str) -> str:
    folded = _fold_token(word)
    if len(folded) > 3 and folded.endswith("s") and not folded.endswith("ss"):
        return folded[:-1]
    return folded


def _same_word(left: str, right: str) -> bool:
    one, other = _stem(left), _stem(right)
    if one == other:
        return True
    if len(one) > 4 and one.startswith("un") and _stem(one[2:]) == other:
        return True
    if len(other) > 4 and other.startswith("un") and _stem(other[2:]) == one:
        return True
    return _inflected_pair(one, other)


def _inflected_pair(one: str, other: str) -> bool:
    shared = 0
    for left_char, right_char in zip(one, other):
        if left_char != right_char:
            break
        shared += 1
    # govern/governing, terminate/termination, cover/covered.
    # internal does not match international, and word1 does not match word19.
    if shared < 5:
        return False
    return _ending(one, one[shared:]) and _ending(other, other[shared:])


def _ending(stem: str, rest: str) -> bool:
    if rest in _INFLECTIONS:
        return True
    # transferring is transfer plus a doubled consonant and -ing.
    if len(rest) < 2 or rest[1:] not in _REAL_SUFFIXES:
        return False
    cut = len(stem) - len(rest)
    return cut > 0 and stem[cut - 1] == rest[0]


_INFLECTIONS = frozenset(
    {"", "e", "d", "s", "ed", "es", "ly", "er", "or", "al", "ing", "ion", "tion", "ation", "able", "ment", "ies"}
)
_REAL_SUFFIXES = _INFLECTIONS - {""}


def _text_has_term(text: str, term: str) -> bool:
    return any(_same_word(term, word) for word in _words(text))


def _generic_stems() -> set[str]:
    return {_stem(word) for word in _GENERIC_FIELD}


def _content_stems(text: str) -> set[str]:
    generic = _generic_stems()
    stems: set[str] = set()
    for word in re.findall(r"[A-Za-z]+", text):
        if len(word) < 3:
            continue
        stem = _stem(word)
        if stem in generic or stem in _BOILER:
            continue
        stems.add(stem)
    return stems


def _near_number_stems(text: str) -> set[str]:
    body = _body(text)
    matches = []
    for match in _NUMBER.finditer(body):
        start = match.start(1)
        if start > 0 and (body[start - 1].isalpha() or body[start - 1] == "-"):
            continue
        if "." in match.group(1):
            continue
        matches.append(match)
    if len(matches) != 1:
        return set()
    match = matches[0]
    window = body[max(0, match.start() - 80) : min(len(body), match.end() + 40)]
    return _content_stems(window)


def _parse_cell(text: str) -> tuple[str, str, int, str] | None:
    match = _CELL_LINE.match(text.strip())
    if match is None:
        return None
    sheet, coord = match.group("cell").rsplit("!", 1)
    column = re.match(r"[A-Z]+", coord)
    row = re.search(r"\d+", coord)
    if column is None or row is None:
        return None
    return sheet, column.group(0), int(row.group(0)), match.group("body")


def _lookup_cell(items: list[Evidence], sheet: str, column: str, row: int) -> str | None:
    for item in items:
        parsed = _parse_cell(item.text)
        if parsed is None:
            continue
        found_sheet, found_column, found_row, body = parsed
        if found_sheet.casefold() != sheet.casefold() or found_column != column or found_row != row:
            continue
        if re.search(r"[A-Za-z]", body):
            return body
    return None


def _same_field(left: Evidence, right: Evidence, items: list[Evidence]) -> bool:
    left_cell = _parse_cell(left.text)
    right_cell = _parse_cell(right.text)
    if left_cell is not None and right_cell is not None:
        return _same_cell_field(left_cell, right_cell, items)
    if left_cell is not None or right_cell is not None:
        return False
    return len(_near_number_stems(left.text) & _near_number_stems(right.text)) >= 2


def _same_cell_field(
    left: tuple[str, str, int, str],
    right: tuple[str, str, int, str],
    items: list[Evidence],
) -> bool:
    left_sheet, left_column, left_row, _left_body = left
    right_sheet, right_column, right_row, _right_body = right
    if left_sheet.casefold() != right_sheet.casefold():
        return False
    left_header = _lookup_cell(items, left_sheet, "A", left_row)
    right_header = _lookup_cell(items, right_sheet, "A", right_row)
    if not left_header or not right_header:
        return False
    if _content_stems(left_header) != _content_stems(right_header):
        return False
    left_column_header = _column_header(items, left_sheet, left_column, left_row) or ""
    right_column_header = _column_header(items, right_sheet, right_column, right_row) or ""
    return _content_stems(left_column_header) == _content_stems(right_column_header)


def _column_header(items: list[Evidence], sheet: str, column: str, row: int) -> str | None:
    """The label above a value, which is often the second row rather than row 1."""
    found: tuple[int, str] | None = None
    for item in items:
        parsed = _parse_cell(item.text)
        if parsed is None:
            continue
        found_sheet, found_column, found_row, body = parsed
        if found_sheet.casefold() != sheet.casefold() or found_column != column or found_row >= row:
            continue
        if not re.search(r"[A-Za-z]", body):
            continue
        if found is None or found_row > found[0]:
            found = (found_row, body)
    if found is None:
        return None
    return found[1]


def _question_selects(
    question: str, left: Evidence, right: Evidence, items: list[Evidence]
) -> bool:
    """The question has to name this field, not only a party that appears beside it."""
    shared = _near_number_stems(left.text) & _near_number_stems(right.text)
    shared -= {_stem(name) for name in _name_tokens(question)}
    if len(shared) < 2:
        return False
    stems = _question_stems(question)
    return len(shared & stems) >= 2


def _question_stems(question: str) -> set[str]:
    stems: set[str] = set()
    for term in content_terms(question):
        stem = _stem(term)
        stems.add(stem)
        alias = _ALIAS.get(stem)
        if alias:
            stems.add(alias)
    return stems


def _subject_terms(question: str) -> list[str]:
    names = _name_tokens(question)
    generic = _generic_stems()
    filler = {
        "much",
        "during",
        "end",
        "about",
        "into",
        "required",
        "available",
        "receive",
        "far",
        "advance",
        "formula",
        "total",
        "compare",
        "against",
    }
    terms: list[str] = []
    for term in content_terms(question):
        if term.casefold() in names or term.casefold() in filler or _PERIOD.fullmatch(term):
            continue
        if _stem(term) in generic:
            continue
        terms.append(term)
    return terms


def _term_matches(term: str, text: str) -> bool:
    if _text_has_term(text, term):
        return True
    alias = _ALIAS.get(_stem(term))
    if alias is None:
        return False
    return any(_stem(word) == alias or _ALIAS.get(_stem(word)) == alias for word in _words(text))


def _parse_grid(text: str) -> tuple[list[str], list[tuple[str, list[Decimal]]]] | None:
    matches = list(_PERIOD_HEADER.finditer(text))
    runs: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    for match in matches:
        if current and text[current[-1].end() : match.start()].strip():
            if len(current) >= 2:
                runs.append(current)
            current = [match]
        else:
            current.append(match)
    if len(current) >= 2:
        runs.append(current)
    if not runs:
        return None
    headers = [match.group(0) for match in runs[0]]
    rest = text[runs[0][-1].end() :]
    tokens = _GRID_TOKEN.findall(rest)
    rows: list[tuple[str, list[Decimal]]] = []
    index = 0
    width = len(headers)
    while index < len(tokens):
        if tokens[index][0].isdigit():
            break
        words: list[str] = []
        while index < len(tokens) and tokens[index][0].isalpha():
            words.append(tokens[index])
            index += 1
        numbers: list[Decimal] = []
        while index < len(tokens) and tokens[index][0].isdigit() and len(numbers) < width:
            numbers.append(Decimal(tokens[index]))
            index += 1
        if words and len(numbers) == width:
            rows.append((" ".join(words), numbers))
            continue
        break
    if not rows:
        return None
    return headers, rows


def _multi_number_grid(text: str) -> bool:
    grid = _parse_grid(text)
    if grid is None:
        return False
    _headers, rows = grid
    return sum(len(numbers) for _label, numbers in rows) > 1


def _header_selected(header: str, question: str) -> bool:
    tokens = [match.group(0).casefold() for match in _PERIOD.finditer(header)]
    present = {match.group(0).casefold() for match in _PERIOD.finditer(question)}
    return bool(tokens) and all(token in present for token in tokens)


def _match_row(question: str, rows: list[tuple[str, list[Decimal]]]) -> int | None:
    hits = [
        index
        for index, (label, _numbers) in enumerate(rows)
        if any(_text_has_term(label, term) for term in content_terms(question))
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _selected_columns(question: str, headers: list[str]) -> list[int]:
    return [index for index, header in enumerate(headers) if _header_selected(header, question)]


def _change_from_question(question: str, items: list[Evidence]) -> Release | None:
    if _CHANGE.search(question) is None:
        return None
    for item in items:
        grid = _parse_grid(item.text)
        if grid is None:
            continue
        headers, rows = grid
        columns = _selected_columns(question, headers)
        row = _match_row(question, rows)
        if row is None or len(columns) != 2:
            continue
        ordered = sorted(
            columns,
            key=lambda index: question.casefold().find(headers[index].split()[0].casefold()),
        )
        earlier, later = ordered
        value = rows[row][1][later] - rows[row][1][earlier]
        return Release(
            "answer",
            render_citation(_format_decimal(value), [item]),
            paths=(item.path,),
        )
    return None


def _cell_from_question(question: str, items: list[Evidence]) -> Release | None:
    """One row and one column name a cell. The model's number is not required."""
    found: Release | None = None
    for item in items:
        if item.superseded:
            continue
        grid = _parse_grid(item.text)
        if grid is None:
            continue
        headers, rows = grid
        columns = _selected_columns(question, headers)
        row = _match_row(question, rows)
        if row is None or len(columns) != 1:
            continue
        value = rows[row][1][columns[0]]
        release = Release(
            "answer",
            render_citation(_format_decimal(value), [item]),
            paths=(item.path,),
        )
        if found is not None and found.text.splitlines()[0] != release.text.splitlines()[0]:
            return None
        found = release
    return found


def _placed_number(
    question: str, answer: str, chosen: list[Evidence], items: list[Evidence]
) -> str:
    numbers = number_values(answer)
    if not numbers:
        return ""
    for item in chosen:
        grid = _parse_grid(item.text)
        if grid is None:
            continue
        headers, rows = grid
        columns = _selected_columns(question, headers)
        row = _match_row(question, rows)
        if row is None or len(columns) != 1:
            continue
        cell = rows[row][1][columns[0]]
        if numbers != {cell}:
            return "The number is not in the asked row and column."
    if _wrong_sheet_cell(question, chosen, items):
        return "The number is not in the asked row and column."
    return ""


def _asked_cells(question: str) -> list[tuple[str | None, str]]:
    """A question that names Headcount!B3 or Budget sheet B3 is asking for that cell."""
    explicit = [
        (match.group(1), match.group(2).upper()) for match in _SHEET_CELL.finditer(question)
    ]
    if explicit:
        return explicit
    sheets = _NAMED_SHEET.findall(question)
    cells = _BARE_CELL.findall(question)
    if sheets and cells:
        return [(sheets[0], cells[0].upper())]
    return []


def _is_asked_cell(item: Evidence, asked: list[tuple[str | None, str]]) -> bool:
    parsed = _parse_cell(item.text)
    if parsed is None or not asked:
        return False
    sheet, column, row, _body = parsed
    coord = f"{column}{row}".upper()
    for asked_sheet, asked_coord in asked:
        if coord != asked_coord:
            continue
        if asked_sheet is None or sheet.casefold() == asked_sheet.casefold():
            return True
    return False


def _coordinates_covered(question: str, text: str) -> bool:
    asked = _asked_cells(question)
    if not asked:
        return True
    for sheet, coord in asked:
        if re.search(rf"\b{re.escape(coord)}\b", text) is None:
            return False
        if sheet is not None and re.search(rf"\b{re.escape(sheet)}\b", text, re.IGNORECASE) is None:
            return False
    return True


def _wrong_sheet_cell(question: str, chosen: list[Evidence], items: list[Evidence]) -> bool:
    asked = _asked_cells(question)
    if asked and any(_is_asked_cell(item, asked) for item in items):
        return any(not _is_asked_cell(item, asked) for item in chosen)
    scored: list[tuple[str, int]] = []
    for item in items:
        parsed = _parse_cell(item.text)
        if parsed is None or parsed[1] == "A":
            continue
        sheet, column, row, _body = parsed
        header = _lookup_cell(items, sheet, "A", row) or ""
        column_header = _column_header(items, sheet, column, row) or ""
        if not header and not column_header:
            continue
        score = _overlap_score(question, f"{header} {column_header} {sheet}")
        scored.append((item.id, score))
    if not scored:
        return False
    best = max(score for _item_id, score in scored)
    if best <= 0:
        return False
    scores = dict(scored)
    for item in chosen:
        if _parse_cell(item.text) is None or item.id not in scores:
            continue
        if scores[item.id] < best:
            return True
    return False


def _named_pair(question: str, items: list[Evidence]) -> tuple[Evidence, Evidence] | None:
    if _ADD.search(question):
        opening = _cell_for(items, question, ("opening", "start"))
        added = _cell_for(items, question, ("hire", "hires", "net"))
        if opening is not None and added is not None and opening.id != added.id:
            return opening, added
    soles = [
        item
        for item in items
        if not item.superseded
        and sole_number(item.text) is not None
        and (_sentence_overlaps(question, item.text, items) or _shares(question, item.text))
    ]
    if len(soles) < 2:
        return None
    return soles[0], soles[1]


def _cell_for(items: list[Evidence], question: str, words: tuple[str, ...]) -> Evidence | None:
    found: Evidence | None = None
    for item in items:
        parsed = _parse_cell(item.text)
        if parsed is None or sole_number(item.text) is None:
            continue
        sheet, _column, row, _body = parsed
        header = _lookup_cell(items, sheet, "A", row) or ""
        blob = f"{item.text} {header}"
        if any(_text_has_term(blob, word) for word in words) and _text_has_term(blob, "headcount"):
            if found is not None:
                return None
            found = item
    if found is None:
        return None
    if "headcount" not in question.casefold() and not _shares(question, found.text):
        return None
    return found


def _measure(text: str) -> str | None:
    """The unit written after the one integer, so weeks are not compared with days."""
    body = _body(text)
    found: list[str | None] = []
    for match in _NUMBER.finditer(body):
        start = match.start(1)
        if start > 0 and (body[start - 1].isalpha() or body[start - 1] == "-"):
            continue
        if "." in match.group(1):
            continue
        unit = re.match(r"\s*(days?|weeks?|months?|years?)\b", body[match.end() :], re.IGNORECASE)
        if unit is None:
            found.append(None)
            continue
        word = unit.group(1).casefold()
        found.append(word[:-1] if word.endswith("s") else word)
    if len(found) != 1:
        return None
    return found[0]


def _units_disagree(left: str, right: str) -> bool:
    one, other = _measure(left), _measure(right)
    if one is None and other is None:
        return False
    return one != other


def _periods_covered(question: str, texts: list[str]) -> bool:
    needed = {match.group(0).casefold() for match in _PERIOD.finditer(question)}
    if not needed:
        return True
    present: set[str] = set()
    for text in texts:
        present.update(match.group(0).casefold() for match in _PERIOD.finditer(text))
    return needed <= present


def _operation_from_question(question: str, items: list[Evidence]) -> Release | None:
    """Add or subtract two values the question names, even when they are different rows."""
    if _ADD.search(question) is None and _SUBTRACT.search(question) is None:
        return None
    pair = _named_pair(question, items)
    if pair is None:
        return None
    left = sole_number(pair[0].text)
    right = sole_number(pair[1].text)
    if left is None or right is None:
        return None
    if _ADD.search(question):
        value = left + right
    else:
        value = abs(left - right)
    paths = tuple(dict.fromkeys(item.path for item in pair))
    return Release("answer", render_citation(_format_decimal(value), list(pair)), paths=paths)
