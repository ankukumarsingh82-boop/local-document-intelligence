"""Find table cells from PDF text positions and ruled lines.

The page text used for full-text search is left as it was. These cells are
stored only as spans.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from ldi.config import (
    TABLE_COLUMN_TOLERANCE,
    TABLE_GAP_MULTIPLE,
    TABLE_LINE_Y_TOLERANCE,
    TABLE_MAX_ROW_GAP,
    TABLE_MAX_RULE_THICKNESS,
    TABLE_MIN_COLUMN_GAP,
    TABLE_MIN_COLUMNS,
    TABLE_MIN_HLINE,
    TABLE_MIN_ROWS,
    TABLE_MIN_RULE_OVERLAP,
    TABLE_MIN_ROW_GAP,
    TABLE_MIN_VLINE,
    TABLE_RULE_TOLERANCE,
    TABLE_WORD_GAP_CAP,
)


@dataclass(frozen=True)
class DetectedCell:
    row: int
    col: int
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectedTable:
    page: int
    table_id: str
    caption: str
    cells: tuple[DetectedCell, ...]
    empty: bool


def extract_tables(path: Path) -> list[DetectedTable]:
    """One entry per visible table. An empty grid has empty=True and no cells."""
    try:
        import pypdfium2 as pdfium
        from pypdfium2 import raw
    except ImportError:
        return []
    try:
        document = pdfium.PdfDocument(str(path))
    except Exception:
        return []
    tables: list[DetectedTable] = []
    try:
        for number, page in enumerate(document, start=1):
            try:
                chars = _characters(page)
                rules = _rules(page, raw)
                ruled = _ruled_tables(number, chars, rules)
                covered = [_bbox(table.cells) for table in ruled if table.cells]
                textual = _text_tables(number, chars, covered, len(ruled))
                tables.extend(ruled)
                tables.extend(textual)
            finally:
                page.close()
    finally:
        document.close()
    return tables


def text_layer_empty(path: Path) -> bool:
    """True when pdfium sees no characters. A scanned page looks like this."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return False
    try:
        document = pdfium.PdfDocument(str(path))
    except Exception:
        return False
    try:
        total = 0
        for page in document:
            textpage = page.get_textpage()
            total += textpage.count_chars()
            textpage.close()
            page.close()
            if total:
                return False
    finally:
        document.close()
    return True


def _characters(page) -> list[tuple[float, float, float, float, str]]:
    textpage = page.get_textpage()
    chars = []
    try:
        for index in range(textpage.count_chars()):
            left, bottom, right, top = textpage.get_charbox(index)
            if right < left:
                left, right = right, left
            if top < bottom:
                bottom, top = top, bottom
            glyph = textpage.get_text_range(index, 1)
            if not glyph or glyph.isspace():
                continue
            if right - left < 0.15 or top - bottom < 0.15:
                continue
            chars.append((left, bottom, right, top, glyph))
    finally:
        textpage.close()
    return chars


def _lines(chars: list[tuple[float, float, float, float, str]]):
    ordered = sorted(chars, key=lambda item: -((item[1] + item[3]) / 2))
    rows: list[list] = []
    for char in ordered:
        center = (char[1] + char[3]) / 2
        height = char[3] - char[1]
        if rows and abs(rows[-1][0] - center) <= max(TABLE_LINE_Y_TOLERANCE, rows[-1][2] * 0.7):
            rows[-1][1].append(char)
            count = len(rows[-1][1])
            rows[-1][0] = (rows[-1][0] * (count - 1) + center) / count
        else:
            rows.append([center, [char], height])
    for row in rows:
        row[1].sort(key=lambda item: item[0])
    return rows


def _column_gap(chars: list[tuple[float, float, float, float, str]]) -> float:
    gaps = []
    for _center, items, _height in _lines(chars):
        for left, right in zip(items, items[1:]):
            gap = right[0] - left[2]
            if 0.3 <= gap <= TABLE_WORD_GAP_CAP:
                gaps.append(gap)
    word = statistics.median(gaps) if gaps else 3.0
    return max(TABLE_MIN_COLUMN_GAP, word * TABLE_GAP_MULTIPLE)


def _tokens(items, cut: float):
    if not items:
        return []
    groups = [[items[0]]]
    for previous, char in zip(items, items[1:]):
        if char[0] - previous[2] > cut:
            groups.append([char])
        else:
            groups[-1].append(char)
    tokens = []
    for group in groups:
        left = min(item[0] for item in group)
        right = max(item[2] for item in group)
        bottom = min(item[1] for item in group)
        top = max(item[3] for item in group)
        text = "".join(item[4] for item in group).strip()
        if text:
            tokens.append((left, bottom, right, top, text))
    return tokens


def _text_tables(page: int, chars, covered, seen: int) -> list[DetectedTable]:
    if len(chars) < 6:
        return []
    rows = _lines(chars)
    cut = _column_gap(chars)
    token_rows = [_tokens(row[1], cut) for row in rows]
    columns = _column_anchors(token_rows)
    tables: list[DetectedTable] = []
    index = 0
    cursor = 0
    while cursor < len(rows):
        end = cursor
        while end < len(rows):
            hits = sum(1 for _anchor, lines in columns if end in lines)
            if hits < TABLE_MIN_COLUMNS:
                break
            end += 1
        run = list(range(cursor, end))
        if len(run) >= TABLE_MIN_ROWS:
            used = [
                (anchor, lines)
                for anchor, lines in columns
                if len(set(run) & lines) >= TABLE_MIN_ROWS
            ]
            used.sort()
            kept = [
                line
                for line in run
                if sum(1 for _anchor, lines in used if line in lines) >= TABLE_MIN_COLUMNS
            ]
            if len(used) >= TABLE_MIN_COLUMNS and len(kept) >= TABLE_MIN_ROWS:
                cells = _assign(token_rows, kept, used)
                if cells and not _inside(cells, covered):
                    caption = _caption(rows, kept[0], cut)
                    tables.append(
                        DetectedTable(
                            page=page,
                            table_id=f"p{page}t{seen + index + 1}",
                            caption=caption,
                            cells=tuple(cells),
                            empty=False,
                        )
                    )
                    index += 1
                cursor = end
                continue
        cursor += 1
    return tables


def _column_anchors(token_rows):
    edges = []
    for line, tokens in enumerate(token_rows):
        for left, _bottom, _right, _top, _text in tokens:
            edges.append((left, line))
    edges.sort()
    clusters = []
    for position, line in edges:
        if clusters and abs(clusters[-1][0] - position) <= TABLE_COLUMN_TOLERANCE:
            clusters[-1][1].append(position)
            clusters[-1][2].add(line)
            clusters[-1][0] = sum(clusters[-1][1]) / len(clusters[-1][1])
        else:
            clusters.append([position, [position], {line}])
    return [
        (cluster[0], cluster[2])
        for cluster in clusters
        if len(cluster[2]) >= TABLE_MIN_ROWS
    ]


def _assign(token_rows, kept, used) -> list[DetectedCell]:
    anchors = [anchor for anchor, _lines in used]
    cells: list[DetectedCell] = []
    for row_number, line in enumerate(kept, start=1):
        buckets: dict[int, list[tuple]] = {}
        for token in token_rows[line]:
            column = min(range(len(anchors)), key=lambda index: abs(token[0] - anchors[index]))
            if abs(token[0] - anchors[column]) > TABLE_COLUMN_TOLERANCE:
                continue
            buckets.setdefault(column, []).append(token)
        for column, tokens in sorted(buckets.items()):
            tokens.sort(key=lambda item: item[0])
            text = " ".join(token[4] for token in tokens).strip()
            if not text:
                continue
            left = min(token[0] for token in tokens)
            bottom = min(token[1] for token in tokens)
            right = max(token[2] for token in tokens)
            top = max(token[3] for token in tokens)
            cells.append(
                DetectedCell(row_number, column + 1, text, (left, bottom, right, top))
            )
    return cells


def _caption(rows, first_line: int, cut: float) -> str:
    if first_line == 0:
        return ""
    previous = rows[first_line - 1]
    gap = previous[0] - rows[first_line][0]
    if gap > TABLE_MAX_ROW_GAP:
        return ""
    tokens = _tokens(previous[1], cut)
    if len(tokens) != 1:
        return ""
    return tokens[0][4]


def _inside(cells: list[DetectedCell], covered: list[tuple[float, float, float, float]]) -> bool:
    box = _bbox(cells)
    if box is None:
        return False
    for other in covered:
        if other is None:
            continue
        if _overlap(box, other) > 0.6:
            return True
    return False


def _bbox(cells) -> tuple[float, float, float, float] | None:
    if not cells:
        return None
    return (
        min(cell.bbox[0] for cell in cells),
        min(cell.bbox[1] for cell in cells),
        max(cell.bbox[2] for cell in cells),
        max(cell.bbox[3] for cell in cells),
    )


def _overlap(left, right) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    if width <= 0 or height <= 0:
        return 0.0
    area = (left[2] - left[0]) * (left[3] - left[1])
    if area <= 0:
        return 0.0
    return (width * height) / area


def _rules(page, raw):
    horizontal = []
    vertical = []
    for obj in page.get_objects(max_depth=5):
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        try:
            left, bottom, right, top = obj.get_bounds()
        except Exception:
            continue
        if right < left:
            left, right = right, left
        if top < bottom:
            bottom, top = top, bottom
        width = right - left
        height = top - bottom
        if width >= TABLE_MIN_HLINE and height <= TABLE_MAX_RULE_THICKNESS:
            horizontal.append((left, right, (bottom + top) / 2))
        elif height >= TABLE_MIN_VLINE and width <= TABLE_MAX_RULE_THICKNESS:
            vertical.append(((left + right) / 2, bottom, top))
    return _merge_horizontal(horizontal), _merge_vertical(vertical)


def _merge_horizontal(lines):
    ordered = sorted(lines, key=lambda item: -item[2])
    merged = []
    for left, right, y_pos in ordered:
        if merged and abs(merged[-1][0] - y_pos) <= TABLE_RULE_TOLERANCE:
            merged[-1][1] = min(merged[-1][1], left)
            merged[-1][2] = max(merged[-1][2], right)
        else:
            merged.append([y_pos, left, right])
    return [(item[0], item[1], item[2]) for item in merged]


def _merge_vertical(lines):
    ordered = sorted(lines, key=lambda item: item[0])
    merged = []
    for x_pos, bottom, top in ordered:
        if merged and abs(merged[-1][0] - x_pos) <= TABLE_RULE_TOLERANCE:
            merged[-1][1] = min(merged[-1][1], bottom)
            merged[-1][2] = max(merged[-1][2], top)
        else:
            merged.append([x_pos, bottom, top])
    return [(item[0], item[1], item[2]) for item in merged]


def _ruled_tables(page: int, chars, rules) -> list[DetectedTable]:
    horizontal, vertical = rules
    tables: list[DetectedTable] = []
    cursor = 0
    while cursor < len(horizontal):
        group = [horizontal[cursor]]
        end = cursor + 1
        while end < len(horizontal):
            gap = group[-1][0] - horizontal[end][0]
            overlap = min(group[-1][2], horizontal[end][2]) - max(group[-1][1], horizontal[end][1])
            if TABLE_MIN_ROW_GAP <= gap <= TABLE_MAX_ROW_GAP and overlap >= TABLE_MIN_RULE_OVERLAP:
                group.append(horizontal[end])
                end += 1
                continue
            if 0 <= gap < TABLE_MIN_ROW_GAP:
                end += 1
                continue
            break
        if len(group) >= TABLE_MIN_ROWS:
            top = group[0][0]
            bottom = group[-1][0]
            left = min(item[1] for item in group)
            right = max(item[2] for item in group)
            crosses = [
                item
                for item in vertical
                if left - 2 <= item[0] <= right + 2
                and min(item[2], top) - max(item[1], bottom) >= 0.6 * (top - bottom)
            ]
            crosses.sort()
            if len(crosses) >= TABLE_MIN_COLUMNS:
                cells = _ruled_cells(chars, group, crosses)
                tables.append(
                    DetectedTable(
                        page=page,
                        table_id=f"p{page}r{len(tables) + 1}",
                        caption=_ruled_caption(chars, top),
                        cells=tuple(cells),
                        empty=not cells,
                    )
                )
            cursor = end
            continue
        cursor += 1
    return tables


def _ruled_cells(chars, group, crosses) -> list[DetectedCell]:
    cells: list[DetectedCell] = []
    for row_index, (upper, lower) in enumerate(zip(group, group[1:]), start=1):
        top = upper[0]
        bottom = lower[0]
        for column, (left_line, right_line) in enumerate(zip(crosses, crosses[1:]), start=1):
            left = left_line[0]
            right = right_line[0]
            if right - left < 8:
                continue
            glyphs = [
                char
                for char in chars
                if left + 1 <= (char[0] + char[2]) / 2 <= right - 1
                and bottom + 1 <= (char[1] + char[3]) / 2 <= top - 1
            ]
            glyphs.sort(key=lambda char: (-(char[1] + char[3]) / 2, char[0]))
            text = "".join(char[4] for char in glyphs).strip()
            if not text:
                continue
            cells.append(
                DetectedCell(
                    row_index,
                    column,
                    text,
                    (left, bottom, right, top),
                )
            )
    return cells


def _ruled_caption(chars, top: float) -> str:
    above = [
        char
        for char in chars
        if top < (char[1] + char[3]) / 2 <= top + TABLE_MAX_ROW_GAP
    ]
    if not above:
        return ""
    above.sort(key=lambda char: char[0])
    return "".join(char[4] for char in above).strip()
