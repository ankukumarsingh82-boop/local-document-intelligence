"""Read local text and PDF files into the index."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ldi.chunk import chunk_text
from ldi.config import (
    DOCX_SUFFIXES,
    IMAGE_SUFFIXES,
    MAX_FILE_BYTES,
    PDF_SUFFIXES,
    SKIP_DIR_NAMES,
    SUPPORTED_SUFFIXES,
    TEXT_SUFFIXES,
    XLSX_SUFFIXES,
    project_root,
)
from ldi.ocr import Scan, classify_page, image_count, recognize
from ldi.pdftable import extract_tables, text_layer_empty
from ldi.provisions import extract_provisions
from ldi.spans import build_spans
from ldi.store import Index, sha256_file, stored_file


@dataclass(frozen=True)
class FileCoverage:
    path: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class IngestReport:
    files: int = 0
    chunks: int = 0
    unchanged: int = 0
    skipped: int = 0
    coverage: tuple[FileCoverage, ...] = ()


def format_coverage(report: IngestReport) -> str:
    """One corpus file per line: indexed, ocr_pending, or failed(reason)."""
    lines = []
    for item in report.coverage:
        name = Path(item.path).name
        if item.status == "failed":
            reason = " ".join(item.reason.split())
            lines.append(f"{name}\tfailed({reason})")
        else:
            lines.append(f"{name}\t{item.status}")
    return "\n".join(lines)


def ingest(path: Path, index: Index, *, reset: bool = False, ocr: bool = True) -> IngestReport:
    if not path.exists():
        raise ValueError(f"Nothing is at {path}.")
    if path.is_file() and path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            "Index text, PDF, Word, Excel, and image files "
            "(.txt, .md, .markdown, .rst, .pdf, .docx, .xlsx, .png, .jpg, .jpeg, .webp)."
        )
    if path.resolve() == Path("/"):
        raise ValueError("Pick a documents folder. The whole disk is too large to index.")
    if reset:
        index.reset()

    report = IngestReport()
    root = project_root()
    folder = path.resolve()
    directory = folder.is_dir()
    seen: set[str] = set()
    deferred: list[tuple] = []
    coverage: list[FileCoverage] = []
    for file in _iter_files(path):
        suffix = file.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        stored = _stored_path(file, root)
        if directory:
            seen.add(stored)
        try:
            stat = file.stat()
        except OSError as exc:
            _skip(f"Skipped {file.name}: {exc}")
            _drop_text(index, stored, "skipped")
            coverage.append(FileCoverage(stored, "failed", str(exc)))
            report = _add(report, skipped=1)
            continue
        if stat.st_size > MAX_FILE_BYTES:
            reason = "larger than 32 MB"
            _skip(f"Skipped {file.name}: {reason}.")
            _drop_text(index, stored, "skipped")
            coverage.append(FileCoverage(stored, "failed", reason))
            report = _add(report, skipped=1)
            continue
        digest = sha256_file(file)
        if index.is_unchanged(stored, stat.st_mtime_ns, stat.st_size, digest):
            coverage.append(FileCoverage(stored, "indexed"))
            report = _add(report, unchanged=1)
            continue
        try:
            pages, extra_gaps, scans, merges = _read_document(file, suffix)
        except Exception as exc:
            _skip(f"Skipped {file.name}: {exc}")
            _drop_text(index, stored, "skipped")
            coverage.append(FileCoverage(stored, "failed", str(exc)))
            report = _add(report, skipped=1)
            continue
        item = (stored, stat, digest, suffix, pages, extra_gaps, scans, merges)
        if scans and not ocr:
            if pages:
                report = _add(report, files=1, chunks=_commit(index, item))
            else:
                _drop_text(index, stored, "ocr_pending")
                report = _add(report, skipped=1)
            coverage.append(FileCoverage(stored, "ocr_pending", f"{len(scans)} scanned pages"))
            continue
        if scans:
            deferred.append(item)
            continue
        report = _add(report, files=1, chunks=_commit(index, item))
        coverage.append(FileCoverage(stored, "indexed"))
    if deferred:
        added, deferred_coverage = _recognize_deferred(index, deferred)
        report = _add(report, **added)
        coverage.extend(deferred_coverage)
    if directory:
        index.purge_absent(seen, folder, root)
    index.refresh_labels()
    index.dedupe_passages()
    return IngestReport(
        report.files,
        report.chunks,
        report.unchanged,
        report.skipped,
        tuple(sorted(coverage, key=lambda item: Path(item.path).name)),
    )


def _add(report: IngestReport, **counts: int) -> IngestReport:
    return IngestReport(
        files=report.files + counts.get("files", 0),
        chunks=report.chunks + counts.get("chunks", 0),
        unchanged=report.unchanged + counts.get("unchanged", 0),
        skipped=report.skipped + counts.get("skipped", 0),
    )


def _skip(message: str) -> None:
    print(message, file=sys.stderr)


def _drop_text(index: Index, stored: str, reason: str) -> None:
    index.forget_text(stored)
    index.replace_gaps(stored, [(None, reason)])


def _pdf_file(stored: str) -> Path:
    return stored_file(stored, project_root())


def _commit(
    index: Index,
    item: tuple,
    confidences: list | None = None,
    page_sources: list[str] | None = None,
) -> int:
    *head, merges = item if len(item) == 8 else (*item, [])
    stored, stat, digest, suffix, pages, extra_gaps, _scans = head
    chunks: list[tuple[int | None, str]] = []
    gaps = list(extra_gaps)
    for page, text in pages:
        pieces = chunk_text(text)
        if pieces:
            chunks.extend((page, piece) for piece in pieces)
        elif suffix in PDF_SUFFIXES or suffix in IMAGE_SUFFIXES:
            gaps.append((page, "extraction_incomplete"))
    if not chunks and not _has_incomplete(gaps):
        gaps.append((None, "extraction_incomplete"))
    index.replace_file(stored, stat.st_mtime_ns, stat.st_size, chunks, digest)
    index.replace_gaps(stored, gaps)
    kind = "ocr" if suffix in IMAGE_SUFFIXES else "text"
    pdf = _pdf_file(stored) if suffix in PDF_SUFFIXES else None
    if pdf is not None and text_layer_empty(pdf):
        kind = "ocr"
    tables = [] if pdf is None else [table for table in extract_tables(pdf) if not table.empty]
    spans, passages = build_spans(
        pages,
        digest,
        kind,
        stored,
        merges,
        tables,
        confidences,
        page_sources,
    )
    from ldi.chunker import uncovered_lines

    for line in uncovered_lines(spans, passages):
        print(line, file=sys.stderr)
    index.replace_spans(stored, spans, passages)
    index.replace_provisions(stored, extract_provisions(stored, spans))
    return len(chunks)


def _recognize_deferred(index: Index, deferred: list[tuple]) -> tuple[dict[str, int], list[FileCoverage]]:
    scans = [scan for item in deferred for scan in item[6]]
    try:
        texts = recognize(scans)
        confidence = getattr(recognize, "confidence", None)
    except Exception as exc:
        covered = []
        for item in deferred:
            _skip(f"Skipped {Path(item[0]).name}: {exc}")
            _drop_text(index, item[0], "skipped")
            covered.append(FileCoverage(item[0], "failed", str(exc)))
        return {"skipped": len(deferred)}, covered
    cursor = 0
    files = 0
    chunks = 0
    covered = []
    for item in deferred:
        stored, stat, digest, suffix, pages, extra_gaps, item_scans = item[:7]
        page_text = list(pages)
        gaps = list(extra_gaps)
        sources = ["text"] * len(page_text)
        scores: list = [None] * len(page_text)
        for offset, (scan, text) in enumerate(zip(item_scans, texts[cursor : cursor + len(item_scans)])):
            if text:
                page_text.append((scan.page, text))
                sources.append("ocr")
                page_scores = None
                if confidence is not None and cursor + offset < len(confidence):
                    page_scores = confidence[cursor + offset]
                scores.append(page_scores)
            else:
                gaps.append((scan.page, "extraction_incomplete"))
        cursor += len(item_scans)
        chunks += _commit(
            index,
            (stored, stat, digest, suffix, page_text, gaps, []),
            scores,
            sources,
        )
        files += 1
        covered.append(FileCoverage(stored, "indexed"))
    return {"files": files, "chunks": chunks}, covered


def _has_incomplete(gaps: list[tuple[int | None, str]]) -> bool:
    return any(reason.startswith("extraction_incomplete") for _page, reason in gaps)


def _stored_path(file: Path, root: Path) -> str:
    resolved = file.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _iter_files(path: Path):
    if path.is_symlink():
        path = path.resolve()
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES and not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            file = Path(dirpath) / name
            if file.is_symlink():
                continue
            yield file


def read_file(
    file: Path,
) -> tuple[list[tuple[int | None, str]], list[tuple[int | None, str]], list[Scan]]:
    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Attach a .txt, .md, .pdf, or image file.")
    pages, gaps, scans, _merges = _read_document(file, suffix)
    return pages, gaps, scans


def text_from_upload(name: str, data: bytes) -> str:
    """Read one attached file. The bytes stay on this Mac."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("That file is larger than 32 MB.")
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        if b"\x00" in data[:8192]:
            raise ValueError("That file does not look like text.")
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            raise ValueError(f"{Path(name).name} has no text.")
        return text
    if suffix in PDF_SUFFIXES:
        return _pdf_text(data, Path(name).name)
    raise ValueError("Attach a .txt, .md, or .pdf file.")


def _pdf_text(data: bytes, name: str) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise ValueError(f"{name} could not be read as a PDF.") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"{name} is encrypted.") from exc
        if not unlocked:
            raise ValueError(f"{name} is encrypted.")
    parts: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            parts.append(f"Page {number}\n{text}")
    if not parts:
        raise ValueError(f"{name} has no text.")
    return "\n\n".join(parts)


def _read_document(
    file: Path, suffix: str
) -> tuple[
    list[tuple[int | None, str]],
    list[tuple[int | None, str]],
    list[Scan],
    list[tuple[str, int, int, int, int]],
]:
    if suffix in TEXT_SUFFIXES:
        raw = file.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ValueError("the file does not look like text.")
        return [(None, raw.decode("utf-8", errors="replace"))], [], [], []
    if suffix in PDF_SUFFIXES:
        pages, gaps, scans = _read_pdf(file)
        return pages, gaps, scans, []
    if suffix in DOCX_SUFFIXES:
        pages, gaps = _read_docx(file)
        return pages, gaps, [], []
    if suffix in XLSX_SUFFIXES:
        pages, gaps, merges = _read_xlsx(file)
        return pages, gaps, [], merges
    if suffix in IMAGE_SUFFIXES:
        return [], [], [Scan(file, None)], []
    raise ValueError("the file is not a document this tool indexes.")


def _read_docx(
    file: Path,
) -> tuple[list[tuple[int | None, str]], list[tuple[int | None, str]]]:
    from docx import Document

    document = Document(str(file))
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    if not parts:
        return [], [(None, "extraction_incomplete")]
    return [(None, "\n".join(parts))], []


def _read_xlsx(
    file: Path,
) -> tuple[
    list[tuple[int | None, str]],
    list[tuple[int | None, str]],
    list[tuple[str, int, int, int, int]],
]:
    from openpyxl import load_workbook

    formulas = load_workbook(file, data_only=False)
    cached = load_workbook(file, data_only=True)
    lines: list[str] = []
    gaps: list[tuple[int | None, str]] = []
    merges: list[tuple[str, int, int, int, int]] = []
    book = _BookValues(formulas, cached)
    try:
        for sheet in formulas.worksheets:
            for merged in sheet.merged_cells.ranges:
                merges.append(
                    (
                        sheet.title,
                        merged.min_row,
                        merged.min_col,
                        merged.max_row,
                        merged.max_col,
                    )
                )
            for row in sheet.iter_rows():
                for cell in row:
                    value = cell.value
                    if value is None or value == "":
                        continue
                    locator = f"{sheet.title}!{cell.coordinate}"
                    if isinstance(value, str) and value.startswith("="):
                        computed = book.number(sheet.title, cell.coordinate)
                        if computed is None:
                            gaps.append((None, f"extraction_incomplete {locator}"))
                            continue
                        lines.append(f"{locator} {_cell_text(computed)}")
                        continue
                    lines.append(f"{locator} {value}")
    finally:
        formulas.close()
        cached.close()
    if not lines:
        return [], gaps, merges
    return [(None, "\n".join(lines))], gaps, merges


class _BookValues:
    """Cached values win. A formula with no cached value is arithmetic on cells in the file."""

    def __init__(self, formulas, cached) -> None:
        self.formulas = {sheet.title: sheet for sheet in formulas.worksheets}
        self.cached = {sheet.title: sheet for sheet in cached.worksheets}
        self.memo: dict[tuple[str, str], Decimal | None] = {}
        self.stack: set[tuple[str, str]] = set()

    def number(self, sheet: str, coord: str) -> Decimal | None:
        key = (sheet, coord.upper())
        if key in self.memo:
            return self.memo[key]
        if key in self.stack:
            return None
        worksheet = self.formulas.get(sheet)
        if worksheet is None:
            return None
        raw = worksheet[coord].value
        stored = self.cached.get(sheet)
        cached_value = stored[coord].value if stored is not None else None
        if isinstance(raw, str) and raw.startswith("="):
            if cached_value not in (None, ""):
                value = _as_decimal(cached_value)
            else:
                self.stack.add(key)
                value = _calculated_value(raw, sheet, self)
                self.stack.discard(key)
        else:
            value = _as_decimal(raw)
        self.memo[key] = value
        return value


def _calculated_value(formula: str, sheet: str, book: _BookValues) -> Decimal | None:
    try:
        return _Formula(formula[1:].replace(" ", ""), sheet, book).parse()
    except (InvalidOperation, ZeroDivisionError):
        return None


class _Formula:
    def __init__(self, text: str, sheet: str, book: _BookValues) -> None:
        self.text = text
        self.sheet = sheet
        self.book = book
        self.index = 0

    def parse(self) -> Decimal | None:
        value = self._expr()
        if value is None or self.index != len(self.text):
            return None
        return value

    def _peek(self) -> str:
        if self.index >= len(self.text):
            return ""
        return self.text[self.index]

    def _expr(self) -> Decimal | None:
        value = self._term()
        if value is None:
            return None
        while self._peek() in {"+", "-"}:
            operator = self._peek()
            self.index += 1
            right = self._term()
            if right is None:
                return None
            value = value + right if operator == "+" else value - right
        return value

    def _term(self) -> Decimal | None:
        value = self._factor()
        if value is None:
            return None
        while self._peek() in {"*", "/"}:
            operator = self._peek()
            self.index += 1
            right = self._factor()
            if right is None or (operator == "/" and right == 0):
                return None
            value = value * right if operator == "*" else value / right
        return value

    def _factor(self) -> Decimal | None:
        if self._peek() == "(":
            self.index += 1
            value = self._expr()
            if self._peek() != ")":
                return None
            self.index += 1
            return value
        if self._peek() == "-":
            self.index += 1
            value = self._factor()
            return None if value is None else -value
        if self.text.startswith("SUM(", self.index):
            return self._sum()
        return self._atom()

    def _sum(self) -> Decimal | None:
        self.index += 4
        total = Decimal(0)
        found = False
        while self._peek() and self._peek() != ")":
            if self._peek() == ",":
                self.index += 1
                continue
            values = self._range()
            if values is None:
                return None
            total += sum(values, Decimal(0))
            found = True
        if self._peek() != ")" or not found:
            return None
        self.index += 1
        return total

    def _range(self) -> list[Decimal] | None:
        start = self._ref()
        if start is None:
            return None
        coords = [start]
        if self._peek() == ":":
            self.index += 1
            end = self._ref()
            if end is None:
                return None
            coords = _range_coords(start, end)
        values: list[Decimal] = []
        for sheet, coord in coords:
            value = self.book.number(sheet, coord)
            if value is None:
                return None
            values.append(value)
        return values

    def _atom(self) -> Decimal | None:
        number = re.match(r"\d+(?:\.\d+)?", self.text[self.index :])
        if number is not None and not re.match(r"[A-Z]", self.text[self.index :]):
            self.index += number.end()
            return Decimal(number.group(0))
        ref = self._ref()
        if ref is None:
            return None
        return self.book.number(ref[0], ref[1])

    def _ref(self) -> tuple[str, str] | None:
        match = re.match(
            r"(?:([A-Za-z][A-Za-z0-9]*)!)?\$?([A-Z]{1,3})\$?([0-9]+)",
            self.text[self.index :],
        )
        if match is None:
            return None
        self.index += match.end()
        sheet = match.group(1) or self.sheet
        return sheet, f"{match.group(2)}{match.group(3)}"


def _range_coords(start: tuple[str, str], end: tuple[str, str]) -> list[tuple[str, str]]:
    sheet = start[0]
    start_col, start_row = _split_coord(start[1])
    end_col, end_row = _split_coord(end[1])
    left, right = sorted((start_col, end_col))
    top, bottom = sorted((start_row, end_row))
    coords: list[tuple[str, str]] = []
    for column in range(left, right + 1):
        for row in range(top, bottom + 1):
            coords.append((sheet, f"{_column_letters(column)}{row}"))
    return coords


def _split_coord(coord: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)([0-9]+)", coord)
    if match is None:
        return 0, 0
    return _column_index(match.group(1)), int(match.group(2))


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


def _as_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.replace(",", ""))
        except InvalidOperation:
            return None
    return None


def _cell_text(value: Decimal) -> str:
    integral = value.to_integral()
    if value == integral:
        return str(integral)
    text = format(value, "f").rstrip("0").rstrip(".")
    return text or "0"


def _read_pdf(
    file: Path,
) -> tuple[list[tuple[int | None, str]], list[tuple[int | None, str]], list[Scan]]:
    from pypdf import PdfReader

    reader = PdfReader(str(file))
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError("the PDF is encrypted.") from exc
        if not unlocked:
            raise ValueError("the PDF is encrypted.")
    pages: list[tuple[int | None, str]] = []
    gaps: list[tuple[int | None, str]] = []
    scans: list[Scan] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        kind = classify_page(text, image_count(page))
        if kind == "text":
            pages.append((number, text.strip()))
        elif kind == "scan":
            scans.append(Scan(file, number))
        else:
            gaps.append((number, "extraction_incomplete"))
    return pages, gaps, scans
