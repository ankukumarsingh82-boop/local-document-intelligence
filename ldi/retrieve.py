"""Lexical retrieval and reciprocal rank fusion. Dense search stays off."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import ldi.config as config
from ldi.config import (
    AND_COVERAGE_RANK_BONUS,
    BM25_BODY_WEIGHT,
    BM25_HEADING_WEIGHT,
    DENSE_ENABLED,
    POINTER_LIMIT,
    RETRIEVAL_DEPTH,
    RRF_K,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_DENSE_TABLE,
    RRF_WEIGHT_LEXICAL,
)
from ldi.provisions import resolve
from ldi.query import (
    QueryFrame,
    fold_for_fts,
    idf,
    or_query,
    parse_question,
    phrase_query,
    slot_supported,
    tokenize,
)
from ldi.store import Hit, Index


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    pointer: list[Hit]
    before_hash: list[Hit]
    ranked: list[Hit]
    frame: QueryFrame
    log: dict
    resolutions: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _Row:
    passage_id: str
    path: str
    page: int | None
    text: str
    span_ids: str
    passage_hash: str
    heading_path: str
    superseded: bool
    span_id: str
    coverage: float
    full: bool
    lexical_rank: int
    exact: bool = False
    rrf: float = 0.0
    dense_only: bool = False


class DenseRetriever:
    """Disabled until a model is approved. It does not load weights."""

    def search(self, frame: QueryFrame, depth: int) -> list[str]:
        return []

    @property
    def available(self) -> bool:
        return False


def adjusted_rank(rank: int, full: bool) -> int:
    """Move a full-coverage hit up a few places. Rank 1 stays a score, not a bonus."""
    if not full or rank <= 1:
        return rank
    return max(2, rank - AND_COVERAGE_RANK_BONUS)


def rrf(
    lists: list[list[str]],
    *,
    k: int = RRF_K,
    weights: tuple[float, ...] | None = None,
    depth: int = RETRIEVAL_DEPTH,
    span_ids: dict[str, str] | None = None,
    lexical_ranks: dict[str, int] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists. A missing list contributes nothing for that passage."""
    labels = span_ids or {}
    lexical = lexical_ranks or {}
    if weights is None:
        weights = tuple(1.0 for _ in lists)
    scores: dict[str, float] = {}
    for weight, items in zip(weights, lists, strict=True):
        seen: set[str] = set()
        rank = 0
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            if rank > depth:
                break
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    ordered = sorted(
        scores,
        key=lambda item: (
            -scores[item],
            lexical.get(item, 10**9),
            labels.get(item, item),
        ),
    )
    return [(item, scores[item]) for item in ordered]


def retrieve(
    index: Index,
    question: str,
    limit: int,
    *,
    dense: DenseRetriever | None = None,
    today: date | None = None,
    depth: int = RETRIEVAL_DEPTH,
) -> Retrieval:
    started = time.perf_counter()
    frame = _frame(index, question, today)
    if not _searchable(frame):
        raise ValueError("That question has no words to search for.")
    if frame.hard_entities and _missing_entity(index, frame):
        return _empty(frame, started, dense_called=False)
    flags = index.file_flags()
    copies: dict[str, list[str]] = {}
    rows = _lexical_rows(index, frame, depth, flags, copies)
    retrieved = time.perf_counter()
    dense_ids: list[str] = []
    dense_called = False
    if dense is not None:
        dense_called = True
        dense_ids = _filter_dense(index, frame, dense.search(frame, depth), flags, copies)
    weights = _weights(frame)
    lexical_ids = [row.passage_id for row in rows]
    fused_scores = rrf(
        [lexical_ids, dense_ids],
        k=RRF_K,
        weights=weights,
        depth=depth,
        span_ids={row.passage_id: row.span_id for row in rows},
        lexical_ranks={row.passage_id: row.lexical_rank for row in rows},
    )
    fused_at = time.perf_counter()
    by_id = {row.passage_id: row for row in rows}
    lexical_set = set(lexical_ids)
    exact = _exact_ids(index, frame, depth)
    fused: list[_Row] = []
    for passage_id, score in fused_scores:
        row = by_id.get(passage_id)
        if row is None:
            row = _load(index, passage_id, flags, copies, frame)
            if row is None:
                continue
        fused.append(
            replace(
                row,
                rrf=score,
                exact=passage_id in exact,
                dense_only=passage_id not in lexical_set,
            )
        )
    fused.sort(key=lambda row: _order_key(row, frame.question))
    gated = [row for row in fused if _eligible(row, frame)]
    before = [_as_hit(row, copies, {}) for row in gated]
    ranked_rows, extras = _hash_dedupe(gated)
    ranked = [_as_hit(row, copies, extras) for row in ranked_rows]
    pointer_rows: list[_Row] = []
    for row in fused:
        if len(pointer_rows) >= POINTER_LIMIT:
            break
        if row.dense_only:
            if slot_supported(frame, row.text):
                pointer_rows.append(row)
            continue
        if _eligible(row, frame):
            pointer_rows.append(row)
    resolutions = _resolutions(index, frame, ranked_rows)
    finished = time.perf_counter()
    scores = [row.rrf for row in fused]
    log = {
        "route": "entity" if frame.entity_route else "table" if frame.table_numeric else "plain",
        "retrieve_s": round(retrieved - started, 6),
        "fuse_s": round(fused_at - retrieved, 6),
        "gate_s": round(finished - fused_at, 6),
        "margin": round(scores[0] - scores[1], 6) if len(scores) > 1 else None,
        "rrf": [round(score, 6) for score in scores],
        "coverage": {row.passage_id: round(row.coverage, 6) for row in fused},
        "dense_called": dense_called,
        "reduced": not DENSE_ENABLED,
    }
    return Retrieval(
        hits=ranked[:limit],
        pointer=[_as_hit(row, copies, extras) for row in pointer_rows],
        before_hash=before,
        ranked=ranked,
        frame=frame,
        log=log,
        resolutions=resolutions,
    )


def _frame(index: Index, question: str, today: date | None) -> QueryFrame:
    tokens = tuple(dict.fromkeys(token.norm for token in tokenize(question) if token.norm))
    return parse_question(
        question,
        today=today,
        frequencies=index.frequencies(tokens),
        passage_count=index.passage_count(),
        alias_rows=index.alias_rows(),
        indexed_names=index.indexed_names(),
    )


def _empty(frame: QueryFrame, started: float, *, dense_called: bool) -> Retrieval:
    now = time.perf_counter()
    return Retrieval(
        hits=[],
        pointer=[],
        before_hash=[],
        ranked=[],
        frame=frame,
        log={
            "route": "entity",
            "retrieve_s": round(now - started, 6),
            "fuse_s": 0.0,
            "gate_s": 0.0,
            "margin": None,
            "rrf": [],
            "coverage": {},
            "dense_called": dense_called,
            "reduced": not DENSE_ENABLED,
        },
        resolutions=(),
    )


def _searchable(frame: QueryFrame) -> bool:
    return bool(
        frame.required
        or frame.aliases
        or frame.ids
        or frame.names
        or frame.designators
        or frame.numbers
    )


def _missing_entity(index: Index, frame: QueryFrame) -> bool:
    for phrase in (*frame.ids, *frame.names):
        if not index.exact_ids(phrase_query(phrase), 1):
            return True
    return False


def _lexical_rows(
    index: Index,
    frame: QueryFrame,
    depth: int,
    flags: dict[str, tuple[bool, str | None]],
    copies: dict[str, list[str]],
) -> list[_Row]:
    terms = list(frame.required)
    for alias in frame.aliases:
        if alias not in terms:
            terms.append(alias)
    match = or_query(terms) if terms else ""
    found = index.lexical_ids(match, depth, BM25_HEADING_WEIGHT, BM25_BODY_WEIGHT)
    ordered = [passage_id for passage_id, _score in found]
    for passage_id in _exact_ids(index, frame, depth):
        if passage_id not in ordered:
            ordered.append(passage_id)
    rows: list[_Row] = []
    total = index.passage_count()
    frequencies = {token: index.document_frequency(token) for token in frame.required}
    for passage_id in ordered:
        row = _load(index, passage_id, flags, copies, frame, total, frequencies)
        if row is not None:
            rows.append(row)
    if frame.hard_entities:
        rows = [row for row in rows if slot_supported(frame, row.text)]
    ranked = _apply_bonus(rows)
    return [
        replace(row, lexical_rank=rank) for rank, row in enumerate(ranked, start=1)
    ]


def _apply_bonus(rows: list[_Row]) -> list[_Row]:
    adjusted = [
        (adjusted_rank(rank, row.full), rank, row)
        for rank, row in enumerate(rows, start=1)
    ]
    adjusted.sort(key=lambda item: (item[0], item[1], item[2].span_id, item[2].passage_id))
    return [item[2] for item in adjusted]


def _filter_dense(
    index: Index,
    frame: QueryFrame,
    passage_ids: list[str],
    flags: dict[str, tuple[bool, str | None]],
    copies: dict[str, list[str]],
) -> list[str]:
    if not frame.hard_entities:
        kept = []
        for passage_id in passage_ids:
            row = index.passage_row(passage_id)
            if row is None:
                continue
            _note_copy(row[0], flags, copies)
            if flags.get(row[0], (False, None))[1]:
                continue
            kept.append(passage_id)
        return kept
    kept = []
    for passage_id in passage_ids:
        row = index.passage_row(passage_id)
        if row is None:
            continue
        path, _span_ids, text, _page, _heading, _digest = row
        _note_copy(path, flags, copies)
        if flags.get(path, (False, None))[1]:
            continue
        if slot_supported(frame, text):
            kept.append(passage_id)
    return kept


def _exact_ids(index: Index, frame: QueryFrame, depth: int) -> set[str]:
    found: set[str] = set()
    phrases = (*frame.ids, *frame.names, *frame.designators, *frame.numbers)
    for phrase in phrases:
        found.update(index.exact_ids(phrase_query(phrase), max(depth, 1) * 5))
    return found


def _load(
    index: Index,
    passage_id: str,
    flags: dict[str, tuple[bool, str | None]],
    copies: dict[str, list[str]],
    frame: QueryFrame,
    total: int,
    frequencies: dict[str, int],
) -> _Row | None:
    stored = index.passage_row(passage_id)
    if stored is None:
        return None
    path, span_ids, text, page, heading, passage_hash = stored
    _note_copy(path, flags, copies)
    superseded, duplicate = flags.get(path, (False, None))
    if duplicate:
        return None
    words = set(fold_for_fts(text).split())
    weight = 0.0
    matched = 0.0
    for token in frame.required:
        token_weight = idf(frequencies.get(token, 0), total)
        weight += token_weight
        if token in words:
            matched += token_weight
    coverage = matched / weight if weight else 0.0
    full = bool(frame.required) and all(token in words for token in frame.required)
    span_list = str(span_ids).split()
    return _Row(
        passage_id=passage_id,
        path=str(path),
        page=int(page) if page else None,
        text=str(text),
        span_ids=str(span_ids),
        passage_hash=str(passage_hash),
        heading_path=str(heading),
        superseded=bool(superseded),
        span_id=span_list[0] if span_list else passage_id,
        coverage=coverage,
        full=full,
        lexical_rank=0,
    )


def _note_copy(
    path: str,
    flags: dict[str, tuple[bool, str | None]],
    copies: dict[str, list[str]],
) -> None:
    _superseded, duplicate = flags.get(path, (False, None))
    if not duplicate or path in copies.get(duplicate, []):
        return
    copies.setdefault(duplicate, []).append(path)


def _weights(frame: QueryFrame) -> tuple[float, float]:
    if frame.table_numeric:
        return (RRF_WEIGHT_LEXICAL, RRF_WEIGHT_DENSE_TABLE)
    return (RRF_WEIGHT_LEXICAL, RRF_WEIGHT_DENSE)


def _order_key(row: _Row, question: str) -> tuple:
    return (
        1 if row.superseded else 0,
        0 if row.exact else 1,
        -row.rrf,
        0 if _names_file(question, row.path) else 1,
        row.lexical_rank or 10**9,
        row.span_id,
        row.passage_id,
    )


def _names_file(question: str, path: str) -> bool:
    stem = Path(path).stem.casefold().replace("_", " ").replace("-", " ")
    if len(stem) < 3:
        return False
    folded = question.casefold().replace("_", " ").replace("-", " ")
    return stem in folded


def _eligible(row: _Row, frame: QueryFrame) -> bool:
    if row.dense_only:
        return False
    if frame.hard_entities and row.exact and slot_supported(frame, row.text):
        return True
    return row.coverage + 1e-12 >= config.LEXICAL_COVERAGE_FLOOR


def _hash_dedupe(rows: list[_Row]) -> tuple[list[_Row], dict[str, list[str]]]:
    kept: list[_Row] = []
    seen: dict[str, _Row] = {}
    extras: dict[str, list[str]] = {}
    for row in rows:
        if row.superseded or not row.passage_hash:
            kept.append(row)
            continue
        prior = seen.get(row.passage_hash)
        if prior is not None and not prior.superseded:
            bucket = extras.setdefault(prior.passage_id, [])
            if row.path not in bucket and row.path != prior.path:
                bucket.append(row.path)
            continue
        seen[row.passage_hash] = row
        kept.append(row)
    return kept, extras


def _as_hit(row: _Row, copies: dict[str, list[str]], extras: dict[str, list[str]]) -> Hit:
    found: list[str] = []
    for path in copies.get(row.path, []):
        if path not in found:
            found.append(path)
    for path in extras.get(row.passage_id, []):
        if path not in found and path != row.path:
            found.append(path)
    return Hit(
        path=row.path,
        page=row.page,
        text=row.text,
        superseded=row.superseded,
        also_found_in=tuple(found),
        passage_hash=row.passage_hash,
        span_id=row.span_id,
        span_ids=row.span_ids,
    )


def _resolutions(
    index: Index,
    frame: QueryFrame,
    rows: list[_Row],
) -> tuple[tuple[str, str, str], ...]:
    if not frame.as_of:
        return ()
    question = fold_for_fts(frame.question)
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row.path in seen:
            continue
        seen.add(row.path)
        stored = index.provisions_for(row.path)
        if not stored:
            continue
        keys = sorted({item.clause_key for item in stored if item.clause_key})
        chosen = [key for key in keys if key in question] or keys[:1]
        for key in chosen:
            found.append((row.path, key, resolve(stored, key, frame.as_of).status))
    return tuple(found)
