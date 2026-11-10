"""Stable span ids stored beside the full-text index. Search does not read them.

Ids are path-keyed. The document key is the first 8 hex digits of the SHA-256
of the stored path. Renaming or moving the file changes every span id from
that file. The file's content hash stays on the files row and is not part of
the id. Two copies of the same line stay distinct because of the occurrence
index.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from ldi.config import EXTRACTOR_VERSION, INSTRUCTION_PHRASES, OCR_LOW_CONFIDENCE

_CELL = re.compile(r"^(?P<sheet>[^!\n]+)!(?P<col>[A-Z]+)(?P<row>\d+) (?P<value>.*)$")
_UNIT = re.compile(
    r"\b(?:inr|rs|rupees?|lakhs?|lac|crores?|cr|days?|weeks?|percent)\b|₹|%",
    re.IGNORECASE,
)
_PERIOD = re.compile(
    r"\bQ[1-4]\s*FY\s*'?\d{2,4}(?:\s*[-/]\s*\d{2,4})?\b"
    r"|\bFY\s*'?\d{2,4}(?:\s*[-/]\s*\d{2,4})?\b",
    re.IGNORECASE,
)
_ROW_ROLE = re.compile(
    r"^(?:sub-?\s*total|(?:grand\s+)?total)\b",
    re.IGNORECASE,
)
_EXTRA_INSTRUCTION = re.compile(
    r"ignore\s+all\s+previous"
    r"|system\s+note\s+to\s+(?:an?\s+)?ai"
    r"|disregard\s+(?:all\s+)?(?:previous|prior)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_DATE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4})\b"
)
_NULL = frozenset({"-", "na", "n/a", "—", "–"})
_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")


@dataclass(frozen=True)
class SpanRecord:
    span_id: str
    doc_sha8: str
    content_hash: str
    extractor_version: str
    source_type: str
    page: int | None
    block: int
    line: int
    table_id: str
    row_key: str
    col: int | None
    header_path: str
    unit: str
    occurrence: int
    confidence: float | None
    bbox: str
    instruction_like: bool
    cell_type: str
    period: str
    row_role: str
    text: str
    blank_before: bool = False


@dataclass(frozen=True)
class PassageRecord:
    passage_id: str
    span_ids: str
    text: str
    doc_sha: str = ""
    page_start: int | None = None
    page_end: int | None = None
    sheet: str = ""
    row_start: int | None = None
    row_end: int | None = None
    heading_path: str = ""
    source_type: str = ""
    passage_hash: str = ""
    chunker_version: str = ""
    label: str = ""
    logs: tuple = ()
    aliases: str = ""


@dataclass(frozen=True)
class _Draft:
    text: str
    source_type: str
    page: int | None
    block: int
    line: int
    table_id: str = ""
    row_key: str = ""
    col: int | None = None
    header_path: str = ""
    unit: str = ""
    confidence: float | None = None
    bbox: str = ""
    cell_type: str = ""
    period: str = ""
    row_role: str = ""
    row: int = 0
    blank_before: bool = False


def normalise_span(text: str) -> str:
    """NFC, then one space where any whitespace ran."""
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def span_id(text: str, doc_sha8: str, occurrence: int) -> str:
    """Hash the line, the stable document key, and which copy this is."""
    material = f"{normalise_span(text)}\n{doc_sha8}\n{occurrence}"
    return hashlib.sha256(material.encode()).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalise_span(text).encode()).hexdigest()


def instruction_like(text: str) -> bool:
    """True for a line addressed to an assistant. An obligation is not one."""
    folded = normalise_span(text).casefold()
    if any(phrase in folded for phrase in INSTRUCTION_PHRASES):
        return True
    return _EXTRA_INSTRUCTION.search(folded) is not None


def cell_type(value: str) -> str:
    stripped = value.strip()
    if stripped.casefold() in _NULL:
        return "null"
    body = stripped[:-1].strip() if stripped.endswith("*") else stripped
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
    if _DATE.fullmatch(body):
        return "date"
    if _NUMBER.fullmatch(body.replace(" ", "")):
        return "num"
    return "text"


def unit_of(value: str) -> str:
    matches = list(_UNIT.finditer(value))
    if not matches:
        return ""
    token = matches[-1].group(0).casefold()
    if token in {"₹", "rs", "rupee", "rupees", "inr"}:
        return "inr"
    if token == "%":
        return "percent"
    if token in {"lakh", "lakhs", "lac"}:
        return "lakh"
    if token in {"crore", "crores", "cr"}:
        return "crore"
    if token == "days":
        return "day"
    if token == "weeks":
        return "week"
    return token


def period_of(value: str) -> str:
    match = _PERIOD.search(value)
    return match.group(0) if match else ""


def document_key(path: str, digest: str) -> str:
    """Eight hex chars of the stored path. A rename changes this key and every id."""
    if path:
        return hashlib.sha256(path.encode()).hexdigest()[:8]
    return digest[:8]


def ocr_low_confidence(source_type: str, confidence: float | None) -> bool:
    """Null OCR confidence is low confidence, so the line cannot fill a number, date, or ID slot later."""
    if source_type != "ocr":
        return False
    if confidence is None:
        return True
    return confidence < OCR_LOW_CONFIDENCE


def build_spans(
    pages: list[tuple[int | None, str]],
    digest: str,
    source_type: str = "text",
    path: str = "",
    merges: list[tuple[str, int, int, int, int]] | None = None,
    tables: list | None = None,
    confidences: list[list[float | None] | None] | None = None,
    page_sources: list[str] | None = None,
) -> tuple[list[SpanRecord], list[PassageRecord]]:
    drafts: list[_Draft] = []
    for index, (page, text) in enumerate(pages):
        scores = None
        if confidences is not None and index < len(confidences):
            scores = confidences[index]
        kind = source_type
        if page_sources is not None and index < len(page_sources):
            kind = page_sources[index]
        drafts.extend(
            _drafts_for_page(
                page,
                text,
                kind,
                len(drafts),
                merges or [],
                scores,
            )
        )
    drafts.extend(drafts_from_tables(tables or []))
    spans = _identify(drafts, document_key(path, digest))
    from ldi.chunker import pack_passages

    return spans, pack_passages(spans, digest=digest, path=path)


def span_is_fresh(span: SpanRecord) -> bool:
    return span.span_id == span_id(span.text, span.doc_sha8, span.occurrence)


def drafts_from_tables(tables: list) -> list[_Draft]:
    """Turn detected PDF cells into table drafts. The full-text chunks stay untouched."""
    drafts: list[_Draft] = []
    for table in tables:
        if table.empty or not table.cells:
            continue
        cells = [
            (cell.row, cell.col, _column_letters(cell.col), cell.text)
            for cell in table.cells
        ]
        drafts.extend(
            _table_drafts(
                table.page,
                table.table_id,
                cells,
                "text",
                len(drafts),
                caption=table.caption,
            )
        )
    return drafts


def _drafts_for_page(
    page: int | None,
    text: str,
    source_type: str,
    block_base: int,
    merges: list[tuple[str, int, int, int, int]],
    confidences: list[float | None] | None = None,
) -> list[_Draft]:
    lines = [line.strip() for line in text.splitlines()]
    cells = [_CELL.fullmatch(line) for line in lines if line]
    if cells and all(match is not None for match in cells):
        return _sheet_drafts(
            page,
            [match for match in cells if match is not None],
            source_type,
            block_base,
            merges,
        )
    drafts: list[_Draft] = []
    block = block_base
    line_number = 0
    index = 0
    caption = ""
    scores = iter(confidences or [])
    pending_blank = False
    while index < len(lines):
        if not lines[index]:
            pending_blank = True
            index += 1
            continue
        if "|" in lines[index]:
            run = []
            while index < len(lines) and lines[index] and "|" in lines[index]:
                run.append(lines[index])
                index += 1
            drafts.extend(_pipe_drafts(page, run, source_type, block, line_number, caption))
            line_number += len(run)
            block += 1
            caption = ""
            pending_blank = False
            continue
        caption = lines[index]
        score = next(scores, None) if source_type == "ocr" else None
        drafts.append(
            _Draft(
                text=lines[index],
                source_type=source_type,
                page=page,
                block=block,
                line=line_number,
                confidence=score,
                blank_before=pending_blank,
            )
        )
        pending_blank = False
        line_number += 1
        index += 1
        block += 1
    return drafts


def _sheet_drafts(page, matches, source_type: str, block: int, merges) -> list[_Draft]:
    grouped: dict[str, list[tuple[int, int, str, str]]] = {}
    for match in matches:
        sheet = match.group("sheet")
        column = _column_index(match.group("col"))
        row = int(match.group("row"))
        grouped.setdefault(sheet, []).append((row, column, match.group("col"), match.group("value")))
    drafts: list[_Draft] = []
    for offset, (sheet, cells) in enumerate(sorted(grouped.items())):
        covered = [item[1:] for item in merges if item[0] == sheet]
        drafts.extend(
            _table_drafts(
                page,
                f"{sheet}",
                _expand_merges(cells, covered),
                source_type,
                block + offset,
            )
        )
    return drafts


def _expand_merges(
    cells: list[tuple[int, int, str, str]],
    merges: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, str, str]]:
    """Copy a merged cell's value onto the covered coordinates. The source line stays one line."""
    present = {(row, column) for row, column, _letters, _value in cells}
    values = {(row, column): value for row, column, _letters, value in cells}
    expanded = list(cells)
    for top, left, bottom, right in merges:
        anchor = values.get((top, left))
        if not anchor:
            continue
        for row in range(top, bottom + 1):
            for column in range(left, right + 1):
                if (row, column) in present:
                    continue
                expanded.append((row, column, _column_letters(column), anchor))
                present.add((row, column))
    return expanded


def _pipe_drafts(
    page, lines: list[str], source_type: str, block: int, line: int, caption: str = ""
) -> list[_Draft]:
    rows = []
    for raw in lines:
        if _SEPARATOR.match(raw.strip()):
            continue
        parts = [part.strip() for part in raw.strip().strip("|").split("|")]
        rows.append(parts)
    cells: list[tuple[int, int, str, str]] = []
    for row_number, parts in enumerate(rows, start=1):
        for column, value in enumerate(parts, start=1):
            if value:
                cells.append((row_number, column, _column_letters(column), value))
    if not cells:
        return []
    return _table_drafts(
        page,
        f"table-{block}",
        cells,
        source_type if source_type == "ocr" else "table",
        block,
        line,
        caption,
    )


def _table_drafts(
    page: int | None,
    table_id: str,
    cells: list[tuple[int, int, str, str]],
    source_type: str,
    block: int,
    line: int = 0,
    caption: str = "",
) -> list[_Draft]:
    header_row = _header_row(cells)
    headers = {
        column: value
        for row, column, _letters, value in cells
        if header_row >= 0 and row == header_row
    }
    keys = {row: value for row, column, _letters, value in cells if column == 1}
    roles = _row_roles(cells)
    drafts: list[_Draft] = []
    for row, column, _letters, value in cells:
        header = headers.get(column, "")
        if row == header_row:
            path = f"{table_id}/{value}"
            unit = unit_of(value) or unit_of(caption)
            period = period_of(value) or period_of(caption)
        else:
            path = f"{table_id}/{header}" if header else table_id
            unit = unit_of(header) or unit_of(caption)
            period = period_of(value) or period_of(header) or period_of(caption)
        drafts.append(
            _Draft(
                text=value,
                source_type="table" if source_type != "ocr" else "ocr",
                page=page,
                block=block,
                line=line + row,
                table_id=table_id,
                row_key="" if column == 1 else keys.get(row, ""),
                col=column,
                header_path=path,
                unit=unit,
                cell_type=cell_type(value),
                period=period,
                row_role=roles.get(row, ""),
                row=row,
            )
        )
    return drafts


def _row_roles(cells: list[tuple[int, int, str, str]]) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for row, _column, _letters, value in cells:
        grouped.setdefault(row, []).append(value)
    roles: dict[int, str] = {}
    for row, values in grouped.items():
        for value in values:
            match = _ROW_ROLE.match(value.strip())
            if match is None:
                continue
            roles[row] = "subtotal" if match.group(0).casefold().startswith("sub") else "total"
            break
    return roles


def table_header_lines(
    rows: list[tuple[int, list[tuple[str, str]]]],
) -> tuple[list[int], list[int]]:
    """Return title lines and header lines.

    A merged title has one distinct text label and a later row that is wider
    or that names two columns. That title is not a header. Following rows
    whose cells are all text are the header, including a second header row.
    The last of those is the column header.
    """
    ordered = sorted(rows, key=lambda item: item[0])
    parsed = []
    for line, cells in ordered:
        labels = [text.strip() for text, kind in cells if kind == "text" and text.strip()]
        nonempty = [(text, kind) for text, kind in cells if text.strip()]
        parsed.append((line, labels, nonempty, len(nonempty)))
    titles: list[int] = []
    headers: list[int] = []
    for index, (line, labels, nonempty, width) in enumerate(parsed):
        if not nonempty:
            continue
        all_text = all(kind == "text" for _text, kind in nonempty)
        distinct = len(set(labels))
        later = parsed[index + 1 :]
        later_wider = any(item[3] > width for item in later)
        later_header = any(
            len(set(item[1])) >= 2 and all(kind == "text" for _text, kind in item[2])
            for item in later
        )
        if all_text and distinct <= 1 and (later_wider or later_header):
            titles.append(line)
            continue
        if all_text and (distinct >= 2 or not headers):
            # A second header row may name the columns. A later row with a
            # digit is a body row, even when every cell is text.
            has_digit = any(any(char.isdigit() for char in text) for text, _kind in nonempty)
            if headers and (has_digit or len(headers) >= 2):
                break
            headers.append(line)
            continue
        break
    return titles, headers


def _header_row(cells: list[tuple[int, int, str, str]]) -> int:
    from ldi import config

    if config.SOURCE_EVIDENCE:
        grouped: dict[int, list[tuple[str, str]]] = {}
        for row, _column, _letters, value in cells:
            grouped.setdefault(row, []).append((value, cell_type(value)))
        _titles, headers = table_header_lines(list(grouped.items()))
        return headers[-1] if headers else -1
    by_row: dict[int, list[str]] = {}
    for row, _column, _letters, value in cells:
        by_row.setdefault(row, []).append(value)
    for row in sorted(by_row):
        labels = [value.strip() for value in by_row[row] if cell_type(value) == "text" and value.strip()]
        # A title repeated across a merged row is not a header.
        if len(set(labels)) >= 2:
            return row
    return min(row for row, _column, _letters, _value in cells)


def _identify(drafts: list[_Draft], sha8: str) -> list[SpanRecord]:
    seen: dict[str, int] = {}
    records: list[SpanRecord] = []
    for draft in drafts:
        normalised = normalise_span(draft.text)
        if not normalised:
            continue
        occurrence = seen.get(normalised, 0)
        seen[normalised] = occurrence + 1
        records.append(
            SpanRecord(
                span_id=span_id(draft.text, sha8, occurrence),
                doc_sha8=sha8,
                content_hash=content_hash(draft.text),
                extractor_version=EXTRACTOR_VERSION,
                source_type=draft.source_type,
                page=draft.page,
                block=draft.block,
                line=draft.line,
                table_id=draft.table_id,
                row_key=draft.row_key,
                col=draft.col,
                header_path=draft.header_path,
                unit=draft.unit,
                occurrence=occurrence,
                confidence=draft.confidence,
                bbox=draft.bbox,
                instruction_like=instruction_like(draft.text),
                cell_type=draft.cell_type,
                period=draft.period,
                row_role=draft.row_role,
                text=unicodedata.normalize("NFC", draft.text),
                blank_before=draft.blank_before,
            )
        )
    if len({record.span_id for record in records}) != len(records):
        raise RuntimeError("Span ids collide inside one document.")
    return records


def _column_index(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - 64)
    return value


def _column_letters(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters
