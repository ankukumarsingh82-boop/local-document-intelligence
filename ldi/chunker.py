"""Pack span lines into passages. Full-text search does not read them."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from ldi.config import CHUNKER_VERSION
from ldi.spans import PassageRecord, SpanRecord, table_header_lines

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff\u2060"), None)
_HEADING = re.compile(
    r"^(?:(?P<num>\d+(?:\.\d+)+)|(?P<named>Annexure|Schedule)\s+(?:[IVXLCDM]+|\d+))\b",
    re.IGNORECASE,
)
_HYPHENATED = re.compile(r"[^\W\d_]+-[^\W\d_]+")
_ATOM = re.compile(
    r"\b(?:"
    r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r"|[A-Z]{2,}-\d{2,}"
    r"|\d{1,3}(?:,\d{2,3})+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?"
    r")\b"
)
_SENTENCE = re.compile(r"[.!?][\"')\]]*")
# A dotted token such as "i.e." or "Rs." is not a sentence end.
_ABBREVIATIONS = frozenset(
    "rs no ie eg etc mr mrs ms dr vs fig vol st".split()
)
_PDF_TABLE = re.compile(r"p\d+[tr]\d+")
# A line this short is a dash or an empty cell, not a garbled row.
LOW_QUALITY_MIN_CHARS = 4
LOW_QUALITY_RATIO = 0.50
_WORD_TABLE = re.compile(r"table-(\d+)")

_TOKENIZER = False
_INSTALLED = None


@dataclass(frozen=True)
class ChunkLimits:
    target: int
    hard_max: int
    overlap: int
    minimum: int
    max_tokens: int


@dataclass(frozen=True)
class ChunkLog:
    kind: str
    detail: str


@dataclass(frozen=True)
class _Piece:
    span_ids: tuple[str, ...]
    text: str
    blank_before: bool
    page: int | None
    heading: bool
    spans: tuple[SpanRecord, ...]
    join_log: ChunkLog | None = None


@dataclass(frozen=True)
class _Cut:
    end: int
    last_keep: int | None
    resume_index: int
    resume_skip: int
    hard_logs: tuple[ChunkLog, ...]
    kind: str


def default_limits() -> ChunkLimits:
    """Read the current config, so a sweep can retarget one process."""
    from ldi import config

    return ChunkLimits(
        target=config.PASSAGE_TARGET_CHARS,
        hard_max=config.PASSAGE_MAX_CHARS,
        overlap=config.PASSAGE_OVERLAP_CHARS,
        minimum=config.PASSAGE_MIN_CHARS,
        max_tokens=config.PASSAGE_MAX_TOKENS,
    )


def clean_line(text: str) -> str:
    """NFC, ligatures, and invisible characters. NFKC is not used."""
    cleaned = unicodedata.normalize("NFC", text)
    for source, target in _LIGATURES.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.translate(_INVISIBLE)
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ")
    return cleaned.strip()


def pack_passages(
    spans: list[SpanRecord],
    *,
    digest: str,
    path: str,
    limits: ChunkLimits | None = None,
    count_tokens=None,
) -> list[PassageRecord]:
    """Build passages from spans. Span text stays as the extractor stored it."""
    bounds = limits or default_limits()
    # Ingest does not import the model runtime. A sweep can install a counter.
    counter = count_tokens if count_tokens is not None else _INSTALLED
    dropped = _repeated_lines(spans)
    joins = _dehyphen(spans, dropped)
    consumed = {partner for _text, _log, partner in joins.values()}
    passages: list[PassageRecord] = []
    prose: list[_Piece] = []
    table: list[SpanRecord] = []
    table_id = ""
    caption: SpanRecord | None = None
    previous: SpanRecord | None = None
    stack: list[tuple[int, str]] = []
    seen_joins: set[tuple[str, ...]] = set()

    def flush_prose() -> None:
        nonlocal prose, stack
        if not prose:
            return
        built, stack = _pack_prose(
            prose,
            stack,
            bounds,
            counter,
            digest,
            path,
            seen_joins,
        )
        passages.extend(built)
        prose = []

    def flush_table() -> None:
        nonlocal table, table_id, caption
        if not table:
            return
        passages.extend(
            _pack_table(table, caption, stack, bounds, counter, digest, path)
        )
        table = []
        table_id = ""
        caption = None

    for span in spans:
        if span.table_id:
            flush_prose()
            if table and span.table_id != table_id:
                flush_table()
            if not table:
                caption = previous if previous and not previous.table_id else None
                table_id = span.table_id
            table.append(span)
            continue
        if id(span) in dropped or id(span) in consumed:
            continue
        if table:
            flush_table()
        joined = joins.get(id(span))
        if joined is not None:
            text, log, _partner = joined
            partner = _partner_span(spans, span)
            prose.append(
                _Piece(
                    span_ids=(span.span_id, partner.span_id),
                    text=text,
                    blank_before=span.blank_before,
                    page=span.page,
                    heading=_heading_level(text) is not None,
                    spans=(span, partner),
                    join_log=log,
                )
            )
            previous = partner
            continue
        else:
            text = clean_line(span.text)
            if not text:
                continue
            prose.append(
                _Piece(
                    span_ids=(span.span_id,),
                    text=text,
                    blank_before=span.blank_before,
                    page=span.page,
                    heading=_heading_level(text) is not None,
                    spans=(span,),
                )
            )
        previous = span
    flush_prose()
    flush_table()
    kept = _keep_passages(passages, bounds)
    kept = _attach_missing(kept, spans, dropped)
    kept = _dedupe_tables(kept)
    return _flag_low_quality(kept, spans)


def printable_ratio(text: str) -> float:
    """Share of the line that is a letter, a digit, or whitespace."""
    if not text:
        return 1.0
    good = sum(1 for char in text if char.isalnum() or char.isspace())
    return good / len(text)


def _flag_low_quality(passages: list[PassageRecord], spans: list[SpanRecord]) -> list[PassageRecord]:
    """Record a low-quality line. The stored text stays as it was read."""
    flagged: list[tuple[str, str]] = []
    for span in spans:
        raw = span.text.strip()
        if len(raw) < LOW_QUALITY_MIN_CHARS:
            continue
        ratio = printable_ratio(raw)
        if ratio < LOW_QUALITY_RATIO:
            flagged.append((span.span_id, f"{ratio:.2f} {span.text}"))
    if not flagged:
        return passages
    updated = []
    claimed: set[str] = set()
    for passage in passages:
        ids = set(passage.span_ids.split())
        extra = []
        for span_id, detail in flagged:
            if span_id in ids:
                extra.append(ChunkLog("low_quality", detail))
                claimed.add(span_id)
        if extra:
            passage = replace(passage, logs=passage.logs + tuple(extra))
        updated.append(passage)
    orphans = [detail for span_id, detail in flagged if span_id not in claimed]
    if orphans and updated:
        updated[0] = replace(
            updated[0],
            logs=updated[0].logs + tuple(ChunkLog("low_quality", detail) for detail in orphans),
        )
    return updated


def merge_overlapping(passages: list[PassageRecord]) -> list[PassageRecord]:
    """Keep the earlier passage and append lines the later one adds."""
    kept: list[PassageRecord] = []
    for passage in passages:
        ids = _ids(passage.span_ids)
        match = None
        for index, earlier in enumerate(kept):
            if ids & _ids(earlier.span_ids):
                match = index
                break
        if match is None:
            kept.append(passage)
            continue
        earlier = kept[match]
        have = earlier.text.splitlines()
        extra = [
            line
            for line in passage.text.splitlines()
            if line.strip() and line not in have
        ]
        text = earlier.text if not extra else earlier.text + "\n" + "\n".join(extra)
        merged = earlier.span_ids.split()
        seen = set(merged)
        for span_id in passage.span_ids.split():
            if span_id and span_id not in seen:
                merged.append(span_id)
                seen.add(span_id)
        kept[match] = replace(
            earlier,
            span_ids=" ".join(merged),
            text=text,
            passage_hash=_hash(text),
            page_start=_min_page(earlier.page_start, passage.page_start),
            page_end=_max_page(earlier.page_end, passage.page_end),
        )
    return kept


def install_token_counter(counter) -> None:
    """Use this counter for later packs. None clears it."""
    global _INSTALLED
    _INSTALLED = counter


def token_counter():
    """Count tokens with the Granite tokenizer. None when the files are absent."""
    global _TOKENIZER
    if _TOKENIZER is False:
        _TOKENIZER = _load_tokenizer()
    tokenizer = _TOKENIZER
    if tokenizer is None:
        return None

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def _load_tokenizer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        from mlx_lm.utils import load_tokenizer

        from ldi.config import CHAT_MODEL, MODELS
        from ldi.weights import local_snapshot

        local = local_snapshot(MODELS[CHAT_MODEL].repo)
        if local is None:
            return None
        return load_tokenizer(str(local))
    except Exception:
        return None


def _repeated_lines(spans: list[SpanRecord]) -> set[int]:
    pages = {span.page for span in spans if span.page is not None}
    if len(pages) < 2:
        return set()
    found: dict[str, set[int]] = {}
    owners: dict[str, list[int]] = {}
    for span in spans:
        if span.page is None or span.table_id:
            continue
        key = clean_line(span.text).casefold()
        if not key:
            continue
        found.setdefault(key, set()).add(span.page)
        owners.setdefault(key, []).append(id(span))
    half = len(pages) / 2
    dropped: set[int] = set()
    for key, seen in found.items():
        if len(seen) > half:
            dropped.update(owners[key])
    return dropped


def _dehyphen(spans: list[SpanRecord], dropped: set[int]) -> dict[int, tuple[str, ChunkLog, int]]:
    forms = set()
    for span in spans:
        for match in _HYPHENATED.finditer(clean_line(span.text)):
            forms.add(match.group(0).casefold())
    joins: dict[int, tuple[str, ChunkLog, int]] = {}
    consumed: set[int] = set()
    for index, span in enumerate(spans[:-1]):
        nxt = spans[index + 1]
        if id(span) in consumed or id(nxt) in consumed:
            continue
        if id(span) in dropped or id(nxt) in dropped:
            continue
        if span.table_id or nxt.table_id:
            continue
        if nxt.blank_before or span.page != nxt.page:
            continue
        joined = _try_join(clean_line(span.text), clean_line(nxt.text), forms)
        if joined is None:
            continue
        text, log = joined
        joins[id(span)] = (text, log, id(nxt))
        consumed.add(id(nxt))
    return joins


def _try_join(left: str, right: str, forms: set[str]) -> tuple[str, ChunkLog] | None:
    if len(left) < 2 or not left.endswith("-"):
        return None
    if not left[-2].isalpha() or not left[-2].islower():
        return None
    if not right or not right[0].islower():
        return None
    prefix = re.search(r"([^\W\d_]+)-$", left)
    suffix = re.match(r"([^\W\d_]+)", right)
    if prefix is None or suffix is None:
        return None
    if not prefix.group(1).islower() or not suffix.group(1).islower():
        return None
    form = f"{prefix.group(1)}-{suffix.group(1)}"
    if form.casefold() in forms:
        return None
    detail = (
        f"{prefix.group(1)}- + {suffix.group(1)} -> "
        f"{prefix.group(1)}{suffix.group(1)}"
    )
    return left[:-1] + right, ChunkLog("join", detail)


def _partner_span(spans: list[SpanRecord], span: SpanRecord) -> SpanRecord:
    for index, item in enumerate(spans[:-1]):
        if item is span:
            return spans[index + 1]
    return span


def _pack_prose(
    pieces: list[_Piece],
    stack: list[tuple[int, str]],
    limits: ChunkLimits,
    counter,
    digest: str,
    path: str,
    seen_joins: set[tuple[str, ...]],
) -> tuple[list[PassageRecord], list[tuple[int, str]]]:
    passages: list[PassageRecord] = []
    cursor = 0
    skip = 0
    steps = 0
    while cursor < len(pieces):
        steps += 1
        if steps > 100000:
            break
        cut = _pick(pieces, cursor, skip, limits, counter)
        text, ids, _logs, used, layout = _materialize(pieces, cursor, skip, cut)
        join_logs = []
        for piece in pieces[cursor:cut.end]:
            if piece.join_log is None or piece.span_ids in seen_joins:
                continue
            if piece.span_ids[0] not in ids:
                continue
            seen_joins.add(piece.span_ids)
            join_logs.append(piece.join_log)
        recorded, stack = _apply_headings(stack, pieces, cursor, cut.end)
        heading = " > ".join(title for _level, title in recorded)
        if text:
            passages.append(
                _record(
                    ids,
                    text,
                    used,
                    digest,
                    path,
                    heading,
                    tuple(join_logs) + cut.hard_logs,
                )
            )
        resume = (cut.resume_index, cut.resume_skip)
        from ldi import config

        if _finished_cut(cut, pieces) and not config.CARRY_AFTER_FINAL_CUT:
            carried = None
        else:
            carried = _carry(layout, text, pieces, (cursor, skip), resume, limits.overlap)
        nxt = carried if carried is not None else resume
        if nxt <= (cursor, skip):
            nxt = resume
        if nxt <= (cursor, skip):
            break
        cursor, skip = nxt
    return passages, stack


def _finished_cut(cut: _Cut, pieces: list[_Piece]) -> bool:
    """True when the cut already consumed every remaining piece."""
    return (
        cut.resume_index >= len(pieces)
        and cut.resume_skip == 0
        and cut.last_keep is None
    )


def _pick(
    pieces: list[_Piece],
    start: int,
    start_skip: int,
    limits: ChunkLimits,
    counter,
) -> _Cut:
    visible = _visible(pieces[start].text, start_skip, None)
    if not _within(visible, limits, counter):
        return _cut_inside(pieces, start, start_skip, limits, counter)
    end = start + 1
    while end < len(pieces):
        if _boundary_splits_atom(pieces[end - 1].text, pieces[end].text):
            end += 1
            continue
        text, _segments = _render(pieces, start, start_skip, end + 1, None)
        if not _within(text, limits, counter):
            break
        end += 1
    text, _segments = _render(pieces, start, start_skip, end, None)
    if end >= len(pieces) and len(text) <= limits.target:
        return _Cut(end, None, end, 0, (), "end")
    floor = max(limits.minimum, min(len(text), max(1, limits.target // 2)))
    best: tuple[int, int, int] | None = None
    for index in range(start + 1, end + 1):
        if index < len(pieces) and _boundary_splits_atom(
            pieces[index - 1].text, pieces[index].text
        ):
            continue
        left, _segments = _render(pieces, start, start_skip, index, None)
        if len(left) < floor and index < end:
            continue
        if index < len(pieces) and pieces[index].heading:
            rank = 0
        elif index < len(pieces) and pieces[index].blank_before:
            rank = 1
        else:
            rank = 2
        candidate = (rank, abs(len(left) - limits.target), index)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return _Cut(end, None, end, 0, (), "line")
    return _Cut(best[2], None, best[2], 0, (), ("heading", "blank", "line")[best[0]])


def _cut_inside(
    pieces: list[_Piece],
    start: int,
    start_skip: int,
    limits: ChunkLimits,
    counter,
) -> _Cut:
    text = pieces[start].text
    visible = text[start_skip:]
    budget = _budget(visible, limits, counter)
    window = visible[:budget]
    atoms = list(_ATOM.finditer(text))

    def legal(absolute: int) -> bool:
        if absolute <= start_skip or absolute >= len(text):
            return False
        if text[absolute - 1].isalnum() and text[absolute].isalnum():
            return False
        return not any(match.start() < absolute < match.end() for match in atoms)

    sentence_at = []
    for end in _sentence_ends(text):
        if end <= start_skip:
            continue
        right = end
        while right < len(text) and text[right].isspace():
            right += 1
        if right - start_skip > budget or right >= len(text):
            continue
        if legal(right):
            sentence_at.append(right)
    if sentence_at:
        right = min(sentence_at, key=lambda pos: (abs(pos - start_skip - limits.target), -pos))
        return _Cut(start + 1, right, start, right, (), "sentence")
    spaces = []
    for match in re.finditer(r"\s+", text):
        if match.end() <= start_skip or match.end() - start_skip > budget:
            continue
        if legal(match.end()):
            spaces.append(match.end())
    if spaces:
        right = min(spaces, key=lambda pos: (abs(pos - start_skip - limits.target), -pos))
        if right - start_skip < max(1, limits.target // 2):
            right = max(spaces)
        detail = f"space at {right - start_skip} of {len(visible)}"
        return _Cut(
            start + 1,
            right,
            start,
            right,
            (ChunkLog("hard_cut", detail),),
            "hard_cut",
        )
    word = re.search(r"\s", visible)
    if word is not None and legal(start_skip + word.start()):
        # A space inside an atom was skipped. Step past the first legal word.
        right = start_skip + word.end()
        while right < len(text) and text[right].isspace():
            right += 1
        if legal(right):
            detail = f"space at {right - start_skip} of {len(visible)}"
            return _Cut(
                start + 1,
                right,
                start,
                right,
                (ChunkLog("hard_cut", detail),),
                "hard_cut",
            )
    detail = f"kept {len(visible)} characters; no word boundary fits"
    return _Cut(start + 1, None, start + 1, 0, (ChunkLog("hard_cut", detail),), "hard_cut")


def _budget(text: str, limits: ChunkLimits, counter) -> int:
    if _within(text, limits, counter):
        return len(text)
    lo = 1
    hi = min(len(text), limits.hard_max)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _within(text[:mid], limits, counter):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _within(text: str, limits: ChunkLimits, counter) -> bool:
    if len(text) > limits.hard_max:
        return False
    if counter is None:
        return True
    try:
        tokens = counter(text)
    except Exception:
        return True
    if tokens is None:
        return True
    return tokens <= limits.max_tokens


def _boundary_splits_atom(left: str, right: str) -> bool:
    joined = f"{left} {right}"
    seam = len(left)
    for match in _ATOM.finditer(joined):
        if match.start() < seam < match.end():
            return True
    return False


def _visible(text: str, start_skip: int, last_keep: int | None) -> str:
    lo, hi, chunk = _slice(text, start_skip, last_keep)
    return chunk


def _slice(text: str, lo: int, hi: int | None) -> tuple[int, int, str]:
    stop = len(text) if hi is None else hi
    lo = max(0, min(lo, len(text)))
    stop = max(lo, min(stop, len(text)))
    while lo < stop and text[lo].isspace():
        lo += 1
    while stop > lo and text[stop - 1].isspace():
        stop -= 1
    return lo, stop, text[lo:stop]


def _render(
    pieces: list[_Piece],
    start: int,
    start_skip: int,
    end: int,
    last_keep: int | None,
) -> tuple[str, list[tuple[int, int, int, int]]]:
    parts: list[str] = []
    segments: list[tuple[int, int, int, int]] = []
    cursor = 0
    for index in range(start, end):
        piece = pieces[index]
        lo = start_skip if index == start else 0
        hi = last_keep if index == end - 1 and last_keep is not None else None
        abs_lo, _abs_hi, chunk = _slice(piece.text, lo, hi)
        if not chunk:
            continue
        blank = bool(parts) and piece.blank_before
        sep = "\n\n" if blank else ("\n" if parts else "")
        rend_start = cursor + len(sep)
        parts.append(sep + chunk)
        segments.append((rend_start, rend_start + len(chunk), index, abs_lo))
        cursor = rend_start + len(chunk)
    return "".join(parts), segments


def _materialize(pieces, start, start_skip, cut: _Cut):
    text, segments = _render(pieces, start, start_skip, cut.end, cut.last_keep)
    ids: list[str] = []
    used: list[SpanRecord] = []
    for index in range(start, cut.end):
        ids.extend(pieces[index].span_ids)
        used.extend(pieces[index].spans)
    return text, ids, [], used, (text, segments)


def _carry(layout, text, pieces, cursor, resume, overlap: int):
    if not text or overlap <= 0:
        return None
    _rendered, segments = layout
    start_at = _suffix_start(text, overlap)
    if start_at is None:
        return None
    point = _point_at(segments, start_at, pieces)
    if point is None or point <= cursor or point >= resume:
        return None
    index, skip = point
    if not _word_boundary(pieces[index].text, skip):
        return None
    return point


def _sentence_ends(text: str) -> list[int]:
    """Indexes just after a sentence mark and its closing quotes.

    With the boundary rule on, the next character has to be whitespace or the
    end of the text. "12.1" and "v2.0" stay one token. "Rs.", "No.", and
    "i.e." stay one token. A trailing "12.1." is still an end.
    """
    from ldi import config

    if not config.SENTENCE_NEEDS_BOUNDARY:
        return [match.end() for match in _SENTENCE.finditer(text)]
    ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index] not in ".!?":
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in "\"')]":
            end += 1
        if end < len(text) and not text[end].isspace():
            index += 1
            continue
        if text[index] == "." and _is_abbreviation(text, index):
            index += 1
            continue
        ends.append(end)
        index = end
    return ends


def _is_abbreviation(text: str, punct: int) -> bool:
    left = punct
    while left > 0 and (text[left - 1].isalnum() or text[left - 1] == "."):
        left -= 1
    token = text[left:punct].casefold().replace(".", "")
    return token in _ABBREVIATIONS


def _suffix_start(text: str, overlap: int) -> int | None:
    cut = None
    for end in _sentence_ends(text):
        nxt = end
        while nxt < len(text) and text[nxt].isspace():
            nxt += 1
        if nxt < len(text):
            cut = nxt
    if cut is not None and 0 < len(text) - cut <= overlap:
        return cut
    newline = text.rfind("\n")
    if newline == -1:
        return None
    line_start = newline + 1
    while line_start < len(text) and text[line_start] == "\n":
        line_start += 1
    if 0 < line_start < len(text) and len(text) - line_start <= overlap:
        return line_start
    return None


def _point_at(segments, pos: int, pieces: list[_Piece]):
    for rend_start, rend_end, index, abs_start in segments:
        if rend_start <= pos < rend_end:
            return index, abs_start + (pos - rend_start)
    for rend_start, _rend_end, index, abs_start in segments:
        if rend_start >= pos:
            return index, abs_start
    return None


def _word_boundary(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text):
        return True
    return not (text[index - 1].isalnum() and text[index].isalnum())


def _apply_headings(stack, pieces, start, end):
    current = list(stack)
    recorded = current
    leading = True
    for index in range(start, end):
        piece = pieces[index]
        if not piece.heading:
            leading = False
            continue
        current = _push_heading(current, piece.text)
        if leading:
            recorded = current
            leading = False
    return recorded, current


def _push_heading(stack: list[tuple[int, str]], text: str) -> list[tuple[int, str]]:
    found = _heading_level(text)
    if found is None:
        return stack
    level, title = found
    kept = [item for item in stack if item[0] < level]
    kept.append((level, title[:120]))
    return kept


def _heading_level(text: str) -> tuple[int, str] | None:
    match = _HEADING.match(text.strip())
    if match is None:
        return None
    if match.group("num"):
        return len(match.group("num").split(".")), text.strip()
    return 1, text.strip()


def _keep_passages(passages: list[PassageRecord], bounds: ChunkLimits) -> list[PassageRecord]:
    """Drop a short prose passage, or attach it when every line must be covered."""
    from ldi import config

    if not config.COVER_EVERY_LINE:
        return [
            passage
            for passage in passages
            if passage.source_type == "table" or len(passage.text) >= bounds.minimum
        ]
    kept: list[PassageRecord] = []
    pending: PassageRecord | None = None
    for passage in passages:
        short = passage.source_type != "table" and len(passage.text) < bounds.minimum
        if short:
            if kept and kept[-1].source_type != "table":
                kept[-1] = _merge_into(kept[-1], passage)
            else:
                pending = passage if pending is None else _merge_into(pending, passage)
            continue
        if pending is not None and passage.source_type != "table":
            passage = _merge_into(pending, passage)
            pending = None
        elif pending is not None:
            kept.append(pending)
            pending = None
        kept.append(passage)
    if pending is not None:
        kept.append(pending)
    return kept


def _merge_into(earlier: PassageRecord, later: PassageRecord) -> PassageRecord:
    text = f"{earlier.text}\n{later.text}".strip()
    ids = f"{earlier.span_ids} {later.span_ids}".strip()
    pages = [page for page in (earlier.page_start, later.page_start) if page is not None]
    ends = [page for page in (earlier.page_end, later.page_end) if page is not None]
    return replace(
        earlier,
        passage_id=_hash(f"{CHUNKER_VERSION}\n{ids}\n{text}"),
        span_ids=ids,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(ends) if ends else None,
        passage_hash=_hash(text),
        logs=earlier.logs + later.logs,
    )


def _dedupe_tables(passages: list[PassageRecord]) -> list[PassageRecord]:
    """Drop a repeated cell grid inside one document version and record where else it was."""
    from ldi import config

    if not config.SOURCE_EVIDENCE:
        return passages
    groups: list[list[PassageRecord]] = []
    current: list[PassageRecord] = []

    def flush() -> None:
        if current:
            groups.append(list(current))
            current.clear()

    for passage in passages:
        if passage.source_type != "table":
            flush()
            continue
        if current and (
            current[-1].heading_path != passage.heading_path or current[-1].sheet != passage.sheet
        ):
            flush()
        current.append(passage)
    flush()
    seen: dict[tuple[str, str], PassageRecord] = {}
    drop: set[int] = set()
    notes: dict[int, list[str]] = {}
    claimed: dict[int, list[str]] = {}
    for group in groups:
        version = ""
        for log in group[0].logs:
            if log.kind == "version" and log.detail:
                version = log.detail
                break
        grid = _hash("\n".join(passage.text for passage in group))
        key = (version, grid)
        where = _also_at(group[0])
        prior = seen.get(key)
        if prior is not None:
            for passage in group:
                drop.add(id(passage))
                claimed.setdefault(id(prior), []).extend(passage.span_ids.split())
            notes.setdefault(id(prior), []).append(where)
            continue
        seen[key] = group[0]
    if not drop and not notes:
        return passages
    kept = []
    for passage in passages:
        if id(passage) in drop:
            continue
        extra = notes.get(id(passage), [])
        extra_ids = claimed.get(id(passage), [])
        if extra or extra_ids:
            ids = passage.span_ids.split()
            for span_id in extra_ids:
                if span_id not in ids:
                    ids.append(span_id)
            joined = " ".join(ids)
            passage = replace(
                passage,
                span_ids=joined,
                passage_id=_hash(f"{CHUNKER_VERSION}\n{joined}\n{passage.text}"),
                logs=passage.logs + tuple(ChunkLog("also_at", detail) for detail in extra),
            )
        kept.append(passage)
    return kept


def _also_at(passage: PassageRecord) -> str:
    if passage.page_start is not None:
        return f"page {passage.page_start}"
    if passage.row_start is not None:
        return f"row {passage.row_start}"
    return passage.heading_path or "later in this document"


def _attach_missing(
    passages: list[PassageRecord],
    spans: list[SpanRecord],
    repeated: set[int],
) -> list[PassageRecord]:
    """Put a non-repeated line that no passage owns onto the nearest passage."""
    from ldi import config

    if not config.COVER_EVERY_LINE or not passages:
        return passages
    covered: set[str] = set()
    for passage in passages:
        covered.update(item for item in passage.span_ids.split() if item)
    missing = [
        span
        for span in spans
        if span.span_id not in covered and id(span) not in repeated and span.text.strip()
    ]
    if not missing:
        return passages
    updated = list(passages)
    for span in missing:
        host = min(
            range(len(updated)),
            key=lambda index: _line_distance(updated[index], span),
        )
        text = clean_line(span.text)
        if not text:
            continue
        ids = f"{updated[host].span_ids} {span.span_id}".strip()
        combined = f"{updated[host].text}\n{text}".strip()
        updated[host] = replace(
            updated[host],
            passage_id=_hash(f"{CHUNKER_VERSION}\n{ids}\n{combined}"),
            span_ids=ids,
            text=combined,
            passage_hash=_hash(combined),
        )
    return updated


def _line_distance(passage: PassageRecord, span: SpanRecord) -> tuple[int, int]:
    if passage.row_start is None:
        return (1, 0)
    return (0, abs(passage.row_start - span.line))


def uncovered_lines(spans: list[SpanRecord], passages: list[PassageRecord]) -> list[str]:
    """Source lines that are neither repeated nor stored on a passage."""
    repeated = _repeated_lines(spans)
    covered: set[str] = set()
    for passage in passages:
        covered.update(item for item in passage.span_ids.split() if item)
    missing = []
    for span in spans:
        if id(span) in repeated or not span.text.strip():
            continue
        if span.span_id not in covered:
            missing.append(span.text)
    return missing


def _pack_table(
    cells: list[SpanRecord],
    caption: SpanRecord | None,
    stack: list[tuple[int, str]],
    limits: ChunkLimits,
    counter,
    digest: str,
    path: str,
) -> list[PassageRecord]:
    """One passage per body row. Lookup strings stay in aliases, not in the text."""
    from ldi import config

    if not config.SOURCE_EVIDENCE:
        return _pack_table_legacy(cells, caption, stack, limits, counter, digest, path)
    return _pack_table_source(cells, caption, stack, digest, path)


def _pack_table_legacy(
    cells: list[SpanRecord],
    caption: SpanRecord | None,
    stack: list[tuple[int, str]],
    limits: ChunkLimits,
    counter,
    digest: str,
    path: str,
) -> list[PassageRecord]:
    """One passage per body row. Cell refs and the source line are aliases."""
    del caption, limits, counter
    by_line: dict[int, list[SpanRecord]] = {}
    for cell in cells:
        by_line.setdefault(cell.line, []).append(cell)
    header_at = _header_line(by_line)
    heading = _table_heading(cells[0].table_id, stack)
    sheet = _sheet_name(cells[0].table_id)
    label = sheet or "table"
    source = (
        "table"
        if any(cell.source_type == "table" for cell in cells)
        else ("ocr" if all(cell.source_type == "ocr" for cell in cells) else "text")
    )
    passages = []
    for line in sorted(by_line):
        if line <= header_at:
            continue
        row_cells = sorted(by_line[line], key=lambda cell: (cell.col or 0, cell.span_id))
        pairs: list[str] = []
        refs: list[str] = []
        aliases: list[str] = []
        values: list[str] = []
        for cell in row_cells:
            value = clean_line(cell.text)
            if not value:
                continue
            values.append(value)
            pairs.append(f"{_value_label(cell)}: {value}")
            if sheet and cell.col and cell.line:
                ref = f"{sheet}!{_column_letters(cell.col)}{cell.line}"
                refs.append(ref)
                aliases.append(ref)
                aliases.append(f"{ref} {value}")
        if not pairs:
            continue
        if not sheet and values:
            aliases.append(" | ".join(values))
        parts = [label]
        if refs:
            parts.append(" ".join(refs))
        parts.extend(pairs)
        passages.append(
            _record(
                [cell.span_id for cell in row_cells],
                " | ".join(parts),
                row_cells,
                digest,
                path,
                heading,
                (),
                sheet=sheet,
                row_start=line,
                row_end=line,
                source=source,
                aliases=tuple(dict.fromkeys(aliases)),
            )
        )
    return passages


def _pack_table_source(
    cells: list[SpanRecord],
    caption: SpanRecord | None,
    stack: list[tuple[int, str]],
    digest: str,
    path: str,
) -> list[PassageRecord]:
    """Quote the original cell text. Aliases are lookup strings only.

    The alias rule is mechanical:
    a spreadsheet cell adds ``Sheet!A1`` and ``Sheet!A1`` plus the cell text;
    a pipe row adds the original cell texts joined by `` | ``;
    a parenthetical acronym is stored by the index, not written into the text.
    """
    by_line: dict[int, list[SpanRecord]] = {}
    for cell in cells:
        by_line.setdefault(cell.line, []).append(cell)
    classified = []
    for line in sorted(by_line):
        ordered = sorted(by_line[line], key=lambda cell: (cell.col or 0, cell.span_id))
        classified.append(
            (line, [(clean_line(cell.text), cell.cell_type or "text") for cell in ordered])
        )
    titles, headers = table_header_lines(classified)
    skip = set(titles) | set(headers)
    heading = _table_heading(cells[0].table_id, stack)
    caption_text = clean_line(caption.text) if caption is not None else ""
    version = _version_token(f"{caption_text} {heading}")
    sheet = _sheet_name(cells[0].table_id)
    source = (
        "table"
        if any(cell.source_type == "table" for cell in cells)
        else ("ocr" if all(cell.source_type == "ocr" for cell in cells) else "text")
    )
    context: list[SpanRecord] = []
    for line in titles + headers:
        context.extend(sorted(by_line[line], key=lambda cell: (cell.col or 0, cell.span_id)))
    context_text = [clean_line(cell.text) for cell in context if clean_line(cell.text)]
    logs = (ChunkLog("version", version),) if version else ()
    body = [line for line in sorted(by_line) if line not in skip]
    passages = []

    def emit(used: list[SpanRecord], texts: list[str], aliases: list[str], line: int) -> None:
        if not texts:
            return
        passages.append(
            _record(
                [cell.span_id for cell in used if clean_line(cell.text)],
                "\n".join(texts),
                used,
                digest,
                path,
                heading,
                logs,
                sheet=sheet,
                row_start=line,
                row_end=line,
                source=source,
                aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
            )
        )

    if not body:
        emit(context, context_text, _row_aliases(sheet, context), headers[-1] if headers else 0)
        return passages
    for line in body:
        row = sorted(by_line[line], key=lambda cell: (cell.col or 0, cell.span_id))
        values = [clean_line(cell.text) for cell in row if clean_line(cell.text)]
        if not values:
            continue
        emit(
            context + row,
            context_text + values,
            _row_aliases(sheet, row),
            line,
        )
    return passages


def _row_aliases(sheet: str, cells: list[SpanRecord]) -> list[str]:
    """Lookup strings for one row. They are not evidence."""
    aliases: list[str] = []
    values: list[str] = []
    for cell in cells:
        value = clean_line(cell.text)
        if not value:
            continue
        values.append(value)
        if sheet and cell.col and cell.line:
            ref = f"{sheet}!{_column_letters(cell.col)}{cell.line}"
            aliases.append(ref)
            aliases.append(f"{ref} {value}")
    if not sheet and values:
        aliases.append(" | ".join(values))
    return aliases


def _version_token(text: str) -> str:
    folded = text.casefold()
    if re.search(r"\brevised\b", folded):
        return "revised"
    if re.search(r"\bexisting\b", folded):
        return "existing"
    match = re.search(r"\bversion\s+([a-z0-9.]+)", folded)
    if match:
        return f"version {match.group(1)}"
    return ""


def _header_line(by_line: dict[int, list[SpanRecord]]) -> int:
    from ldi import config

    if config.SOURCE_EVIDENCE:
        rows = []
        for line in sorted(by_line):
            ordered = sorted(by_line[line], key=lambda cell: (cell.col or 0, cell.span_id))
            rows.append(
                (line, [(clean_line(cell.text), cell.cell_type or "text") for cell in ordered])
            )
        _titles, headers = table_header_lines(rows)
        return headers[-1] if headers else -1
    for line in sorted(by_line):
        labels = [
            clean_line(cell.text)
            for cell in by_line[line]
            if cell.cell_type == "text" and clean_line(cell.text)
        ]
        # A title repeated across a merged row is not a header.
        if len(set(labels)) >= 2:
            return line
    return min(by_line)


def _column_letters(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _value_label(cell: SpanRecord) -> str:
    label = _header_name(cell) or "Value"
    if cell.cell_type not in {"num", "date", "null"}:
        return label
    extras = []
    if cell.period and cell.period.casefold() not in label.casefold():
        extras.append(cell.period)
    if cell.unit and cell.unit.casefold() not in label.casefold():
        extras.append(cell.unit)
    if extras:
        return f"{label} ({', '.join(extras)})"
    return label


def _header_name(cell: SpanRecord) -> str:
    prefix = f"{cell.table_id}/"
    if cell.header_path.startswith(prefix):
        return cell.header_path[len(prefix) :]
    return cell.header_path


def _sheet_name(table_id: str) -> str:
    if not table_id or _WORD_TABLE.fullmatch(table_id) or _PDF_TABLE.fullmatch(table_id):
        return ""
    return table_id


def _table_heading(table_id: str, stack: list[tuple[int, str]]) -> str:
    base = " > ".join(title for _level, title in stack)
    match = _WORD_TABLE.fullmatch(table_id)
    if match:
        extra = f"table {match.group(1)}"
    elif _PDF_TABLE.fullmatch(table_id):
        extra = f"table {table_id}"
    else:
        extra = ""
    if base and extra:
        return f"{base} > {extra}"
    return base or extra


def _record(
    ids: list[str],
    text: str,
    spans: list[SpanRecord],
    digest: str,
    path: str,
    heading: str,
    logs: tuple[ChunkLog, ...],
    sheet: str = "",
    row_start: int | None = None,
    row_end: int | None = None,
    source: str = "",
    aliases: tuple[str, ...] = (),
) -> PassageRecord:
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    joined = " ".join(ids)
    pages = [span.page for span in spans if span.page is not None]
    if not source:
        kinds = {span.source_type for span in spans}
        if kinds == {"ocr"}:
            source = "ocr"
        elif "table" in kinds and kinds <= {"table"}:
            source = "table"
        else:
            source = "text"
    return PassageRecord(
        passage_id=_hash(f"{CHUNKER_VERSION}\n{joined}\n{cleaned}"),
        span_ids=joined,
        text=cleaned,
        doc_sha=digest,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        sheet=sheet,
        row_start=row_start,
        row_end=row_end,
        heading_path=heading,
        source_type=source,
        passage_hash=_hash(cleaned),
        chunker_version=CHUNKER_VERSION,
        label=_label(path, heading),
        logs=logs,
        aliases="\n".join(alias for alias in aliases if alias),
    )


def _label(path: str, heading: str) -> str:
    name = Path(path).name if path else "Document"
    if heading:
        return f"{name} > {heading}:"
    return f"{name}:"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ids(joined: str) -> set[str]:
    return {item for item in joined.split() if item}


def _min_page(left: int | None, right: int | None) -> int | None:
    pages = [page for page in (left, right) if page is not None]
    return min(pages) if pages else None


def _max_page(left: int | None, right: int | None) -> int | None:
    pages = [page for page in (left, right) if page is not None]
    return max(pages) if pages else None
