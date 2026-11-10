"""Turn retrieved passages into evidence ids, then build the citation in code."""

from __future__ import annotations

import re
import unicodedata
from calendar import month_name
from contextvars import ContextVar
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ldi import config
from ldi.amounts import written_rupees
from ldi.chunk import normalise_text
from ldi.grammar import parse_answer_json
from ldi.query import COMPARISON, document_version, fold_for_fts
from ldi.spans import SpanRecord, instruction_like
from ldi.store import Hit, content_terms
from ldi.verify import (
    NOT_FOUND,
    Release,
    _INJECTED,
    _NEGATIVE,
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
_SENTENCE_END = (".", "?", "!", ";", ":")
_AROUND_STEPS = 3
_CLAUSE_REFERENCE = re.compile(
    r"\b(?:section|clause|article|rule|paragraph)\s+(\d+(?:\.\d+)*)(?![\d.])",
    re.IGNORECASE,
)
_AMOUNT_QUESTION = re.compile(
    r"\b(?:how (?:many|much|long)|amount|expenditure|igst|gst rate|total cost)\b", re.IGNORECASE
)
_ENTITLEMENT_QUESTION = re.compile(r"\b(?:available|entitled|allowed)\b", re.I)
_FORMULA_QUESTION = re.compile(r"\b(?:what|which|show)\s+(?:is\s+|the\s+)?(?:the\s+)?formula\b", re.IGNORECASE)
_RECONCILE = re.compile(r"\b(?:match(?:es)?|reconcil(?:e|es)|make up|sum to)\b", re.I)
_RECONCILE_SOURCE = re.compile(r"\b(?:ledger|invoice|quotation|payment)\b", re.I)
_RATE_BAND = re.compile(
    r"(?<![\d-])(?P<band>\d+\s*-\s*\d+)\s+(?:Rs\.?|INR|₹)\s*(?P<old>[\d,]+)(?:\s*/\s*-)?"
    r"\s+(?P=band)\s+(?:Rs\.?|INR|₹)\s*(?P<new>[\d,]+)(?![\d.,])", re.I
)
_RATE_HEADER = re.compile(r"\bexisting rates?\s+revised rates?\b", re.I)
_PERSON_ID_QUESTION = re.compile(
    r"\b(?:employee|staff|member)\s+(?:id|number)\s+(?:for|of)\s+([\w .’'-]+)\??\s*$", re.I,
)
_PERSON_ID_VALUE = r"(?:[A-Za-z]+[-/]?)?\d+"
# Words after a negation cue that an answer must reuse before a negative line contradicts it.
_NEGATED_SPAN = 3
# After the model has abstained, a fallback quote may only match through grammatical endings.
_STRICT_WORDS: ContextVar[bool] = ContextVar("strict_words", default=False)


@dataclass(frozen=True)
class Evidence:
    id: str
    path: str
    locator: str
    text: str
    superseded: bool = False
    span: SpanRecord | None = None
    required_entities: tuple[str, ...] = ()
    comparison_sources: tuple[str, ...] = ()
    required_source: str = ""


_SourceLine = tuple[str, str, str, SpanRecord | None]


def build_evidence(question: str, hits: list[Hit], limit: int | None = None) -> list[Evidence]:
    """Split current passages into short items and keep the ones the question names."""
    items = _build_evidence(question, hits, limit)
    context = {
        hit.path: hit for hit in hits
        if hit.required_entities or hit.comparison_sources or hit.required_source
    }
    if not context:
        return items
    return [
        replace(
            item, required_entities=context[item.path].required_entities,
            comparison_sources=context[item.path].comparison_sources,
            required_source=context[item.path].required_source,
        ) if item.path in context else item
        for item in items
    ]


def _build_evidence(question: str, hits: list[Hit], limit: int | None) -> list[Evidence]:
    if any(hit.source_lines for hit in hits):
        from ldi import config

        budget = config.EVIDENCE_BUDGET if limit is None else limit
        return _evidence_from_source(question, hits, budget)
    if limit is None:
        limit = EVIDENCE_LIMIT
    drafts: list[tuple[str, str, str]] = []
    for hit in hits:
        if hit.superseded:
            continue
        drafts.extend((hit.path, locator, text) for text, locator in _hit_pieces(hit))
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
        spans = load_spans(hit.path)
        ledger = (
            _ledger_row(question, spans)
            if config.DOCUMENT_FIRST_RECONCILIATION and hit.comparison_sources
            and Path(hit.path).suffix.lower() == ".xlsx"
            and re.search(r"\bledger\b", question, re.I)
            else None
        )
        if ledger is None:
            lines, ids, pinned = _unit_for_hit(question, hit, spans)
        else:
            row, _amount = ledger
            lines, ids, pinned = _span_lines([row], {span.span_id for span in row})
        by_id = {span.span_id: span for span in spans if isinstance(span, SpanRecord)}
        attached.append(
            replace(
                hit,
                source_lines=tuple(lines),
                source_span_ids=tuple(ids),
                pinned_span_ids=tuple(pinned),
                source_spans=tuple(by_id[span_id] for span_id in ids if span_id in by_id),
            )
        )
    return attached


def _evidence_from_source(question: str, hits: list[Hit], limit: int) -> list[Evidence]:
    """Quote source lines. Hits take turns so one hit's pinned rows cannot crowd out the rest."""
    from ldi import config

    terms = _question_terms(question)
    monetary = bool(re.search(r"\b(?:invoice|receipt|payment|ledger|gst|igst|quotation)\b", question, re.I))
    seen: set[tuple[str, str]] = set()
    per_hit: list[list[_SourceLine]] = []
    queues: list[list[int]] = []
    for hit in hits:
        if hit.superseded:
            continue
        lines: list[_SourceLine] = []
        pinned_ids = set(hit.pinned_span_ids)
        if not hit.source_lines:
            for text, locator in _hit_pieces(hit):
                key = (hit.path, f"{locator}\n{text}")
                if key in seen:
                    continue
                seen.add(key)
                lines.append((hit.path, locator, text, None))
        else:
            by_id = {span.span_id: span for span in hit.source_spans}
            for text, span_id in zip(hit.source_lines, hit.source_span_ids):
                key = (hit.path, span_id)
                if key in seen:
                    continue
                seen.add(key)
                span = by_id.get(span_id)
                place = replace(hit, page=span.page) if span is not None else hit
                lines.append((hit.path, _span_place(place, span_id), text, span))
        if not lines:
            continue
        lines = _join_source_lines(lines)
        pinned = [
            pos for pos, (_, locator, _, _) in enumerate(lines)
            if pinned_ids.intersection(locator.rsplit(", span ", 1)[-1].split())
        ]
        scores = [
            _overlap(terms, text) + (4 if monetary and _labeled_amount(text, span) else 0)
            for _path, _locator, text, span in lines
        ]
        pinned_set = set(pinned)
        pinned.sort(key=lambda index: (-scores[index], index))
        matched = sorted(
            (index for index in range(len(lines)) if index not in pinned_set and scores[index] > 0),
            key=lambda index: (-scores[index], index),
        )
        rest = [index for index in range(len(lines)) if index not in pinned_set and scores[index] == 0]
        per_hit.append(lines)
        queues.append(pinned + matched + rest)
    taken: set[tuple[int, int]] = set()
    cursors = [0] * len(queues)
    while len(taken) < limit:
        progressed = False
        for hit_index, queue in enumerate(queues):
            if len(taken) >= limit:
                break
            while cursors[hit_index] < len(queue) and (hit_index, queue[cursors[hit_index]]) in taken:
                cursors[hit_index] += 1
            if cursors[hit_index] >= len(queue):
                continue
            line_index = queue[cursors[hit_index]]
            cursors[hit_index] += 1
            taken.add((hit_index, line_index))
            progressed = True
            text = per_hit[hit_index][line_index][2]
            following = line_index + 1
            if (
                len(taken) < limit
                and len(text) >= config.CONTINUATION_MIN_CHARS
                and not text.rstrip().endswith((".", ":", ";", "!", "?", ")"))
                and following < len(per_hit[hit_index])
                and (
                    (per_hit[hit_index][line_index][3] is None
                     and per_hit[hit_index][following][3] is None)
                    or _adjacent(
                        per_hit[hit_index][line_index][3], per_hit[hit_index][following][3]
                    )
                )
            ):
                taken.add((hit_index, following))
        if not progressed:
            break
    chosen = [per_hit[hit_index][line_index] for hit_index, line_index in sorted(taken)]
    return [
        Evidence(f"E{number}", path, locator, text, span=span)
        for number, (path, locator, text, span) in enumerate(chosen, start=1)
    ]


def _join_source_lines(lines: list[_SourceLine]) -> list[_SourceLine]:
    """Keep a wrapped PDF sentence or a real pipe-table row in one quote."""
    joined: list[_SourceLine] = []
    previous: SpanRecord | None = None
    for line in lines:
        path, locator, text, span = line
        if joined and previous is not None and span is not None:
            _path, earlier, quote, first = joined[-1]
            pipe_row = (
                Path(path).suffix.lower() in {".docx", ".md", ".markdown", ".txt"}
                and span.table_id.startswith("table-")
                and span.table_id == previous.table_id and span.line == previous.line
                and span.col is not None and previous.col is not None
                and span.col == previous.col + 1
            )
            wrapped = (
                Path(path).suffix.lower() == ".pdf"
                and _adjacent(previous, span) and len(previous.text) >= 30
                and not quote.rstrip().endswith(_SENTENCE_END + (")",))
                and bool(re.match(r"[a-z\d]", text))
            )
            if (pipe_row or wrapped) and len(quote) + len(text) <= SENTENCE_CHARS:
                separator = " | " if pipe_row else " "
                if wrapped and re.search(r"[a-z]-$", quote) and text[0].islower():
                    quote, separator = quote[:-1], ""
                joined[-1] = (path, f"{earlier} {span.span_id}", f"{quote}{separator}{text}", first)
                previous = span
                continue
        joined.append(line)
        previous = span
    return joined


def _question_terms(question: str) -> set[str]:
    """Content words plus every digit run, so "3-5" can pin the line that says it."""
    terms = {term.casefold() for term in content_terms(question)}
    terms.update(re.findall(r"\d+", question))
    return terms


_AMOUNT_LABEL = re.compile(
    r"\b(?:igst|cgst|sgst|net amt|(?:grand |invoice )?total|amount paid|rupees)\b", re.I
)


def _labeled_amount(text: str, span: SpanRecord | None) -> bool:
    label = f"{span.row_key} {span.header_path} {text}" if span is not None else text
    return _AMOUNT_LABEL.search(label) is not None and (
        bool(number_values(text)) or written_rupees(text) is not None
    )


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
    terms = sorted(_question_terms(question))
    matched = _best_rows(body_rows, terms, config.EVIDENCE_MATCHED_ROWS)
    whole = "\n".join(cell.text for row in ordered for cell in row)
    use = ordered if len(whole) <= config.TABLE_EVIDENCE_CHARS else title_rows + header_rows + matched
    pin_ids = {cell.span_id for row in header_rows + matched for cell in row}
    return _span_lines(use, pin_ids)


def _best_rows(rows: list, terms: list[str], limit: int) -> list:
    """Up to `limit` body rows, preferring a term that is rare inside the table."""
    if limit <= 0 or not rows:
        return []
    joined = [" ".join(cell.text for cell in row) for row in rows]
    present = [{term for term in terms if _term_matches(term, text)} for text in joined]
    rarity = {
        term: 1.0 / (1 + sum(term in found for found in present))
        for term in terms
    }
    scored = []
    for index, found in enumerate(present):
        score = sum(rarity[term] for term in found)
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
    terms = sorted(_question_terms(question))
    hit_ids = {span.span_id for span in chosen}
    scores: dict[str, int] = {}

    def overlap(span) -> int:
        if span.span_id not in scores:
            scores[span.span_id] = sum(1 for term in terms if _term_matches(term, span.text))
        return scores[span.span_id]

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
    scoped = _eligible_items(question, items)
    ranked: list[tuple[Evidence, set[Decimal]]] = []
    for item in scoped:
        if _multi_number_grid(item.text):
            continue
        numbers = _conflict_values(item.text)
        if len(numbers) != 1:
            continue
        ranked.append((item, numbers))
    for index, (left, left_numbers) in enumerate(ranked):
        for right, right_numbers in ranked[index + 1 :]:
            if left.path == right.path:
                continue
            if left_numbers == right_numbers or not _same_field(left, right, items):
                continue
            if _units_disagree(left.text, right.text):
                continue
            if not _question_selects(question, left, right, items):
                continue
            left_period = _period_tokens(left.text) - {_format_decimal(value) for value in left_numbers}
            right_period = _period_tokens(right.text) - {_format_decimal(value) for value in right_numbers}
            if left_period and right_period and left_period != right_period:
                continue
            if not _periods_covered(question, [_period_blob(question, item, scoped) for item in (left, right)]):
                continue
            if _one_sentence_off_topic(question, left, right, items):
                continue
            return left, right
    return None


def _conflict_values(text: str) -> set[Decimal]:
    body = re.sub(
        r"^\s*(?:clause|section|rule|paragraph)\s+\d+(?:\.\d+)*(?=[^:\n]{0,50}:)",
        " ", text, flags=re.I,
    )
    return number_values(body)


def _eligible_items(question: str, items: list[Evidence]) -> list[Evidence]:
    references = _CLAUSE_REFERENCE.findall(question)
    paths = {item.path for item in items}
    if len(paths) > 1:
        from ldi.retrieve import explicit_source, source_names

        named = explicit_source(question, source_names(paths, question))
        if named:
            paths = {named}
    return [
        item for item in items
        if item.path in paths
        and (not item.required_source or item.path == item.required_source)
        and not item.superseded
        and not _INJECTED.search(item.text)
        and not instruction_like(item.text)
        and all(f" {entity} " in f" {fold_for_fts(item.text)} " for entity in item.required_entities)
        and all(_mentions_clause(ref, item, items) for ref in references)
    ]


def _mentions_clause(ref: str, item: Evidence, items: list[Evidence]) -> bool:
    if ref in _CLAUSE_REFERENCE.findall(item.text):
        return True
    if item.span is None:
        return bool(re.match(rf"\s*{re.escape(ref)}[.)]?\s+", item.text))
    heading = next((
        other for other in reversed(items[:items.index(item) + 1])
        if other.path == item.path and other.span is not None
        and other.span.page == item.span.page and _clause_level(other.text) is not None
    ), None)
    return heading is not None and (
        ref in _CLAUSE_REFERENCE.findall(heading.text)
        or bool(re.match(rf"\s*{re.escape(ref)}[.)]?\s+", heading.text))
    )


def conflict_release(question: str, items: list[Evidence]) -> Release | None:
    duration = _semester_leave_comparison(question, _eligible_items(question, items))
    if duration is not None:
        return duration
    pair = conflicting(question, items)
    if pair is None:
        return None
    left, right = pair
    older = document_version(left.path)
    newer = document_version(right.path)
    if (
        _CHANGE.search(question) and older and newer
        and older[0] == newer[0] and older[1] != newer[1]
        and all(
            re.search(rf"(?<![\d.]){re.escape('.'.join(map(str, version[1])))}(?![\d.])", question)
            for version in (older, newer)
        )
    ):
        ordered = sorted(zip((older, newer), pair), key=lambda entry: entry[0][1])
        values = [_format_decimal(next(iter(_conflict_values(item.text)))) for _, item in ordered]
        cited = [item for _, item in ordered]
        return Release("answer", render_citation(f"Changed from {values[0]} to {values[1]}.", cited),
                       paths=tuple(item.path for item in cited))
    return Release("conflict", format_conflict(left, right), paths=(left.path, right.path))


def _semester_leave_comparison(question: str, items: list[Evidence]) -> Release | None:
    subject = re.search(r"\b([a-z]+ leave)\b", question, re.I)
    if subject is None or not re.search(r"\bsemester\b", question, re.I):
        return None
    sides = next((item.comparison_sources for item in items if item.comparison_sources), ())
    if len(sides) != 2:
        return None
    values = []
    for path in sides:
        found = {}
        for item in items:
            if item.path != path or subject[1].casefold() not in item.text.casefold():
                continue
            match = re.search(
                r"\b(\d+)\s+(?:working\s+)?days?\s+(?:in|during|per)\s+(?:a\s+|the\s+)?semester\b",
                item.text, re.I,
            )
            if match and not _negative(item.text[max(0, match.start() - 28) : match.end()]):
                found.setdefault(int(match[1]), item)
        if len(found) != 1:
            return None
        values.append(next(iter(found.items())))
    (one, left), (other, right) = values
    if one != other:
        return Release("conflict", format_conflict(left, right), paths=sides)
    return Release("answer", render_citation(f"{one} days per semester.", [left, right]), paths=sides)


def _one_sentence_off_topic(
    question: str, left: Evidence, right: Evidence, items: list[Evidence]
) -> bool:
    """Two line fragments can share a lead-in while their sentences say different things."""
    left_sentence = _sentence_around(left, items)
    right_sentence = _sentence_around(right, items)
    if left_sentence == left.text and right_sentence == right.text:
        return False
    left_on = _sentence_overlaps(question, left_sentence, items)
    right_on = _sentence_overlaps(question, right_sentence, items)
    return left_on != right_on


def _sentence_around(item: Evidence, items: list[Evidence]) -> str:
    """The item plus neighbouring lines of its page until the sentence ends."""
    try:
        at = next(index for index, other in enumerate(items) if other is item)
    except StopIteration:
        return item.text
    parts = [item.text]
    index = at
    for _step in range(_AROUND_STEPS):
        before = index - 1
        if before < 0 or not _adjacent_items(items[before], items[index]):
            break
        if items[before].text.rstrip().endswith(_SENTENCE_END):
            break
        parts.insert(0, items[before].text)
        index = before
    index = at
    for _step in range(_AROUND_STEPS):
        if items[index].text.rstrip().endswith(_SENTENCE_END):
            break
        after = index + 1
        if after >= len(items) or not _adjacent_items(items[index], items[after]):
            break
        parts.append(items[after].text)
        index = after
    return " ".join(parts)


def _adjacent(left: SpanRecord | None, right: SpanRecord | None) -> bool:
    return (
        left is not None and right is not None
        and left.page == right.page and left.source_type == right.source_type
        and not left.table_id and not right.table_id
        and not left.instruction_like and not right.instruction_like
        and right.line == left.line + 1 and not right.blank_before
        and _clause_level(right.text) is None
    )


def _adjacent_items(left: Evidence, right: Evidence) -> bool:
    return left.path == right.path and _adjacent(left.span, right.span)


def fallback_release(
    question: str, items: list[Evidence]
) -> Release:
    token = _STRICT_WORDS.set(True)
    try:
        return _fallback(question, items)
    finally:
        _STRICT_WORDS.reset(token)


def _fallback(question: str, items: list[Evidence]) -> Release:
    items = _eligible_items(question, items)
    person = _PERSON_ID_QUESTION.search(question)
    if person:
        return _person_id(person[1], items) or Release("not_found", NOT_FOUND)
    if _requires_reconciliation(question):
        return _reconcile(question, items) or Release("not_found", NOT_FOUND)
    if _requires_address_comparison(question, items):
        return _compare_addresses(items) or Release("not_found", NOT_FOUND)
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
        subject = question
        if re.search(r"\bapproval of (?:the\s+)?[A-Z]", item.text):
            subject = re.sub(r"\bwho approves\b", "approval", question, flags=re.I)
        if not _sentence_overlaps(subject, item.text, items):
            continue
        if not _supplies_value(question, item):
            continue
        if not _periods_covered(question, [_period_blob(question, item, items)]):
            continue
        item = _full_statement(question, item, items)
        return Release("answer", render_citation(item.text, [item]), paths=(item.path,))
    return Release("not_found", NOT_FOUND)


def _supplies_value(question: str, item: Evidence) -> bool:
    if _ENTITLEMENT_QUESTION.search(question) and len(item.text) < 100:
        body = item.text.strip()
        heading = _CLAUSE_HEADING.match(body)
        words = re.findall(r"[A-Za-z]+", body)
        if heading or (words and all(word.istitle() for word in words) and not body.endswith(".")):
            if heading:
                body = body[heading.end():]
            if not number_values(body) and not re.search(r"\b(?:may|must|shall|is|are|can)\b", body, re.I):
                return False
    if _FORMULA_QUESTION.search(question):
        return _parse_cell(item.text) is not None and _body(item.text).startswith("=")
    if _AMOUNT_QUESTION.search(question):
        if Path(item.path).suffix.lower() == ".xlsx" and (
            _parse_cell(item.text) is not None or item.span is not None and item.span.table_id
        ):
            return bool(re.fullmatch(r"(?:₹|INR\s*|Rs\.?\s*)?[+-]?[\d,]+(?:\.\d+)?%?", _body(item.text), re.I))
        if item.span is not None and item.span.table_id and item.span.cell_type == "text":
            value = item.text.partition(" | ")[2] or _body(item.text)
            if not re.match(r"\s*(?:[+-]?\d|₹|INR\b|Rs\b)", value, re.I):
                return False
        return bool(number_values(item.text))
    if re.match(r"\s*when\b", question, re.IGNORECASE):
        return bool(_prose_dates(item.text))
    if _parse_cell(item.text) is not None:
        return bool(_asked_cells(question) or number_values(item.text))
    return True


def release_from_evidence(reply: str, items: list[Evidence], question: str) -> Release:
    by_id = {item.id: item for item in items}
    parsed = parse_answer_json(reply, set(by_id))
    conflict = conflict_release(question, items)
    items = _eligible_items(question, items)
    by_id = {item.id: item for item in items}
    person = _PERSON_ID_QUESTION.search(question)
    if person:
        return _person_id(person[1], items) or Release(
            "retry", "", reason="The evidence does not identify an exact person and ID row."
        )
    if _requires_reconciliation(question):
        return _reconcile(question, items) or Release(
            "retry", "", reason="The evidence does not establish each side of the reconciliation."
        )
    if _requires_address_comparison(question, items):
        return _compare_addresses(items) or Release(
            "retry", "", reason="The evidence does not establish both addresses."
        )
    arithmetic = _CALC_QUESTION.search(question) is not None
    if arithmetic and parsed is not None and parsed.status == "answered" and parsed.calc:
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
    if conflict is not None and not arithmetic:
        return conflict
    if parsed is None:
        return Release(
            "retry",
            "",
            reason="The reply was not one JSON object with status, evidence, and answer.",
        )
    if parsed.status != "answered":
        return Release("retry", "", reason="The cited items do not establish a conflict.")
    chosen = [by_id[item_id] for item_id in dict.fromkeys(parsed.evidence) if item_id in by_id]
    chosen = [
        item
        for item in chosen
        if not _INJECTED.search(item.text)
        and (
            _sentence_overlaps(question, item.text, items, slack=1)
            or _sentence_overlaps(question, _sentence_around(item, items), items, slack=1)
        )
    ]
    asked = _asked_cells(question)
    if asked and any(_is_asked_cell(item, asked) for item in items):
        cells = [item for item in chosen if _is_asked_cell(item, asked)]
        if not cells:
            return Release("retry", "", reason="The question asks for a spreadsheet cell.")
        chosen = cells
    if (_AMOUNT_QUESTION.search(question) or _FORMULA_QUESTION.search(question)
        or _ENTITLEMENT_QUESTION.search(question)
        or re.match(r"\s*when\b", question, re.IGNORECASE)) and not any(
        _supplies_value(question, item) for item in chosen
    ):
        return Release("retry", "", reason="The cited items do not supply the requested value.")
    if not chosen:
        return Release(
            "retry",
            "",
            reason="The chosen evidence does not name the subject of the question.",
        )
    if not any(_shares(question, item.text) for item in chosen):
        return Release(
            "retry",
            "",
            reason="The chosen evidence does not share a word with the question.",
        )
    if not _periods_covered(question, [_period_blob(question, item, items) for item in chosen]):
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
    chosen = list(dict.fromkeys(_full_statement(question, item, items) for item in chosen))
    paths = tuple(dict.fromkeys(item.path for item in chosen))
    return Release("answer", render_citation(answer, chosen), paths=paths)


def _requires_reconciliation(question: str) -> bool:
    return bool(
        config.DOCUMENT_FIRST_RECONCILIATION and _RECONCILE.search(question) and (
            re.search(r"\bledger\b", question, re.I)
            or len({word.casefold() for word in _RECONCILE_SOURCE.findall(question)}) >= 2
        )
    )


def _person_id(name: str, items: list[Evidence]) -> Release | None:
    """Read only an ID on the requested person's own labelled table row."""
    identity = " ".join(unicodedata.normalize("NFC", name).casefold().split()).rstrip(".")
    explicit = re.compile(
        rf"\b(?:employee|staff|member)\s+(?:id|number)\s+(?:for|of)\s+"
        rf"{re.escape(identity)}\s*(?::|\bis\b)\s*({_PERSON_ID_VALUE})(?![\w/-])", re.I,
    )
    found = {}
    for item in items:
        match = explicit.search(" ".join(unicodedata.normalize("NFC", item.text).split()))
        if match is not None:
            found.setdefault(match[1], item)
            continue
        label, separator, value = item.text.partition(" | ")
        if not separator or " ".join(unicodedata.normalize("NFC", label).casefold().split()) != identity:
            continue
        if not re.fullmatch(_PERSON_ID_VALUE, value):
            continue
        if item.span is None or not item.span.table_id:
            continue
        if any(
            header.path == item.path and header.span is not None
            and header.span.table_id == item.span.table_id
            and header.span.line < item.span.line
            and re.fullmatch(r"(?:employee )?name\s+\|\s+(?:(?:employee|staff|member) )?(?:id|number)",
                             header.text, re.I)
            for header in items
        ):
            found.setdefault(value, item)
    if len(found) != 1:
        return None
    value, item = next(iter(found.items()))
    return Release("answer", render_citation(value, [item]), paths=(item.path,))


def _requires_address_comparison(question: str, items: list[Evidence]) -> bool:
    return bool(
        config.DOCUMENT_FIRST_RECONCILIATION and re.search(r"\baddress\b", question, re.I)
        and COMPARISON.search(question) and any(item.comparison_sources for item in items)
    )


def _compare_addresses(items: list[Evidence]) -> Release | None:
    sources = next((item.comparison_sources for item in items if item.comparison_sources), ())
    if len(sources) != 2:
        return None
    addresses = []
    for path in sources:
        found = {}
        for item in items:
            if item.path != path:
                continue
            match = re.search(
                r"\b(?:place of business|office|address)\s+(?:is\s+|at\s+|:\s*)+"
                r"(\d+[^()\n]{8,140}?\b\d{5}(?:-\d{4})?\b)",
                item.text, re.I,
            )
            if match:
                found.setdefault(fold_for_fts(match[1]), item)
        if len(found) != 1:
            return None
        addresses.append(next(iter(found.items())))
    (left, first), (right, second) = addresses
    answer = "The addresses match." if left == right else "The addresses differ."
    return Release("answer", render_citation(answer, [first, second]), paths=sources)


def _reconcile(question: str, items: list[Evidence]) -> Release | None:
    from ldi.retrieve import _comparison_sources, source_names

    paths = {item.path for item in items}
    sides = _comparison_sources(question, source_names(paths, question), dict.fromkeys(paths, (False, None)))
    if len(sides) != 2:
        return None
    sum_requested = re.search(r"\b(?:make up|sum to)\b", question, re.I) is not None
    selected = []
    context: list[Evidence] = []
    for side, path in enumerate(sides):
        if side == 0 and Path(path).suffix.lower() == ".xlsx":
            by_span = {item.span: item for item in items if item.path == path and item.span is not None}
            ledger = _ledger_row(question, list(by_span))
            if ledger is not None:
                spans, value = ledger
                context = [by_span[span] for span in spans[:-1]]
                selected.append([(value, by_span[spans[-1]])])
                continue
        amounts: dict[str | None, dict[Decimal, Evidence]] = {}
        for item in items:
            if item.path != path or not _labeled_amount(item.text, item.span):
                continue
            amount = sole_number(item.text)
            if amount is None:
                amount = written_rupees(item.text)
            if amount is not None:
                page = _page_of(item) if sum_requested and side == 1 else None
                amounts.setdefault(page, {}).setdefault(amount, item)
        if not amounts or any(len(page) != 1 for page in amounts.values()):
            return None
        selected.append([next(iter(page.items())) for page in amounts.values()])
    (left, first), = selected[0]
    right = sum((amount for amount, _ in selected[1]), Decimal(0))
    if sum_requested:
        if len(selected[1]) < 2 or left != right:
            return None
        parts = [_format_decimal(amount) for amount, _ in selected[1]]
        answer = f"{' + '.join(parts)} = {_format_decimal(left)}"
    else:
        answer = "The amounts match." if left == right else "The amounts differ."
    cited = [first, *context, *(item for _, item in selected[1])]
    return Release("answer", render_citation(answer, cited), paths=sides)


def _ledger_row(question: str, spans: list[SpanRecord]) -> tuple[list[SpanRecord], Decimal] | None:
    """Bind the year, category, value and scale to the same indexed worksheet row."""
    from ldi.retrieve import _ledger_category
    from ldi.spans import _column_index, cell_type

    category = _ledger_category(question)
    period = re.search(r"\b\d{4}\s*[-–/]\s*\d{2,4}\b", question)
    if not category or period is None:
        return None
    originals = {span.span_id: span for span in spans}
    cells = []
    for span in spans:
        if span.col is not None:
            cells.append(span)
            continue
        parsed = _parse_cell(span.text)
        if parsed is not None:
            sheet, column, row, body = parsed
            cells.append(replace(
                span, table_id=sheet, col=_column_index(column), line=row,
                text=body, cell_type=cell_type(body),
            ))
    years = [
        span for span in cells if span.col == 1
        and re.match(r"\s*(?:year|fiscal year)\b", span.text, re.I)
    ]
    result = []
    for year in years:
        if fold_for_fts(period[0]) not in fold_for_fts(year.text):
            continue
        stop = min(
            (later.line for later in years if later.table_id == year.table_id and later.line > year.line),
            default=float("inf"),
        )
        block = [span for span in cells if span.table_id == year.table_id and year.line < span.line < stop]
        for label in block:
            if label.col not in (1, 2) or len(_words(label.text)) > 4:
                continue
            if _ledger_category(label.text) != category:
                continue
            for value in block:
                if value.line != label.line or value.cell_type != "num" or value.col is None:
                    continue
                if value.col <= 2:
                    continue
                units = [
                    span for span in block if span.col == value.col and span.line < value.line
                    and re.search(r"\bINR\s+in\s+lakhs?\b", span.text, re.I)
                ]
                amount = sole_number(value.text)
                if units and amount is not None:
                    unit = max(units, key=lambda span: span.line)
                    cited = [originals[span.span_id] for span in (year, label, unit, value)]
                    result.append((cited, amount * 100_000))
    return result[0] if len(result) == 1 else None


def _full_statement(question: str, item: Evidence, items: list[Evidence]) -> Evidence:
    """Prefer the full prose statement when it repeats a cited table row's fact."""
    if item.span is None or not item.span.table_id or " | " not in item.text:
        return item
    label, value = item.text.split(" | ", 1)
    label = fold_for_fts(label)
    value = fold_for_fts(value)
    if not label or not value or not number_values(item.text):
        return item
    for other in items:
        if other.path != item.path or (other.span is not None and other.span.table_id):
            continue
        body = f" {fold_for_fts(other.text)} "
        if (
            " | " not in other.text and f" {label} " in body and f" {value} " in body
            and number_values(other.text) == number_values(item.text)
            and _sentence_overlaps(question, other.text, items)
            and not _polarity_conflict(item.text, [other])
        ):
            return other
    return item


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
        item_negative = _negative(item.text)
        if answer_negative and not item_negative:
            return True
        if item_negative and not answer_negative and _reuses_negated(answer_words, item.text):
            return True
    return False


def _reuses_negated(answer_words: set[str], text: str) -> bool:
    """A negated word only contradicts an answer that says that word."""
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text)
    for index, word in enumerate(words):
        if not _NEGATIVE.fullmatch(word):
            continue
        following = {later.lower() for later in words[index + 1 : index + 1 + _NEGATED_SPAN]}
        if following & answer_words:
            return True
    return False


def _hit_pieces(hit: Hit) -> list[tuple[str, str]]:
    # The fallback passage still carries its page. Preserve short physical lines on
    # scan-like pages so an undelimited OCR block cannot swallow a late total.
    lines = [line.strip() for line in hit.text.splitlines() if line.strip()]
    if (
        hit.page is not None and hit.span_ids == "" and len(lines) >= 3
        and not any(line.endswith((".", "!", "?")) for line in lines)
    ):
        return [(_clip(line), _place(hit, None)) for line in lines]
    return _pieces(hit)


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
    owners = {item.path for item in items if item.text == text}
    if len(owners) != 1:
        return text
    items = [item for item in items if item.path in owners]
    sheet, column, row, _body = parsed
    header = _lookup_cell(items, sheet, "A", row) or ""
    column_header = _column_header(items, sheet, column, row) or ""
    return f"{text} {header} {column_header}"


def _sentence_overlaps(
    question: str, text: str, items: list[Evidence] | None = None, slack: int = 0
) -> bool:
    """Most subject words have to match. A column header does not count as one of them.

    slack lets evidence the model chose miss one distinctive word when there are three or more.
    """
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
    missing = sum(1 for term in distinctive if term not in matched)
    allowed = slack if len(distinctive) >= 3 else 0
    if missing > allowed:
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


def _period_blob(question: str, owner: Evidence, items: list[Evidence]) -> str:
    """Use source-specific heading or confirmed filename context for periods."""
    from ldi.retrieve import source_names

    parts = [_described(owner.text, [item for item in items if item.path == owner.path])]
    named = source_names({item.path for item in items}, question)
    if owner.path in named:
        parts.extend(word for word in named[owner.path][1] if _PERIOD.fullmatch(word))
    page = _page_of(owner)
    if page is None:
        return " ".join(parts)
    for other in items:
        if other is owner or other.path != owner.path or _page_of(other) != page:
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
    if _STRICT_WORDS.get():
        return _inflected_pair(one, other)
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
    endings = _GRAMMATICAL if _STRICT_WORDS.get() else _INFLECTIONS
    if rest in endings:
        return True
    # transferring is transfer plus a doubled consonant and -ing.
    if len(rest) < 2 or rest[1:] not in endings - {""}:
        return False
    cut = len(stem) - len(rest)
    return cut > 0 and stem[cut - 1] == rest[0]


_INFLECTIONS = frozenset(
    {"", "e", "d", "s", "ed", "es", "ly", "er", "or", "al", "ing", "ion", "tion", "ation", "able", "ment", "ies"}
)
# manager is not managing, and approved is not approval.
_GRAMMATICAL = frozenset({"", "e", "d", "s", "ed", "es", "ing", "ies"})


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
        return _same_cell_field(
            left_cell, right_cell,
            [item for item in items if item.path == left.path],
            [item for item in items if item.path == right.path],
        )
    if left_cell is not None or right_cell is not None:
        return False
    return len(_near_number_stems(left.text) & _near_number_stems(right.text)) >= 2


def _same_cell_field(
    left: tuple[str, str, int, str],
    right: tuple[str, str, int, str],
    left_items: list[Evidence],
    right_items: list[Evidence],
) -> bool:
    left_sheet, left_column, left_row, _left_body = left
    right_sheet, right_column, right_row, _right_body = right
    if left_sheet.casefold() != right_sheet.casefold():
        return False
    left_header = _lookup_cell(left_items, left_sheet, "A", left_row)
    right_header = _lookup_cell(right_items, right_sheet, "A", right_row)
    if not left_header or not right_header:
        return False
    if _content_stems(left_header) != _content_stems(right_header):
        return False
    left_column_header = _column_header(left_items, left_sheet, left_column, left_row) or ""
    right_column_header = _column_header(right_items, right_sheet, right_column, right_row) or ""
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
    rate = _paired_rate(question, items)
    if rate is not None:
        return rate
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


def _paired_rate(question: str, items: list[Evidence]) -> Release | None:
    """Select a labelled old or revised rate from a row containing both columns."""
    direction = {word.casefold() for word in re.findall(r"\b(?:existing|revised)\b", question, re.I)}
    band = re.search(r"\blevels?\s+(\d+\s*-\s*\d+)\b", question, re.I)
    subjects = set(re.findall(r"\b[A-Z]{4,}\b", question))
    if len(direction) != 1 or band is None or not subjects:
        return None
    column = "new" if direction == {"revised"} else "old"
    period = re.search(
        rf"\b({'|'.join(month_name[1:])})\s+(\d{{4}})\b", question, re.I,
    )
    found: dict[str, tuple[Evidence, Evidence]] = {}
    for item in items:
        for match in _RATE_BAND.finditer(item.text):
            if match["band"].replace(" ", "") != band[1].replace(" ", ""):
                continue
            scopes = re.findall(r"\b[Rr]ate of ([A-Z]{4,})\b", item.text[:match.start()])
            if not scopes or scopes[-1] not in subjects:
                continue
            header = next((
                other for other in items
                if other.path == item.path and _page_of(other) == _page_of(item)
                and _RATE_HEADER.search(other.text)
                and _rate_period_matches(period, other.text)
            ), None)
            if header is not None:
                found.setdefault(match[column].replace(",", ""), (item, header))
    if len(found) != 1:
        return None
    value, pair = next(iter(found.items()))
    cited = list(dict.fromkeys(pair))
    return Release("answer", render_citation(value, cited), paths=(pair[0].path,))


def _rate_period_matches(period: re.Match[str] | None, text: str) -> bool:
    if period is None:
        return True
    month = next(number for number, name in enumerate(month_name) if name.casefold() == period[1].casefold())
    return bool(re.search(rf"\b\d{{1,2}}[./-]0?{month}[./-]{period[2]}\b", text))


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
    needed = _period_tokens(question)
    if not needed:
        return True
    present: set[str] = set()
    for text in texts:
        present.update(_period_tokens(text))
    return needed <= present


def _period_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _PERIOD.finditer(text)}


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
