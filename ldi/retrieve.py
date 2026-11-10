"""Lexical retrieval and reciprocal rank fusion. Dense search stays off."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import ldi.config as config
from ldi.amounts import written_rupees
from ldi.config import (
    AND_COVERAGE_RANK_BONUS,
    BM25_BODY_WEIGHT,
    BM25_HEADING_WEIGHT,
    DENSE_ENABLED,
    NAME_BOOST,
    POINTER_LIMIT,
    RETRIEVAL_DEPTH,
    RRF_K,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_DENSE_TABLE,
    RRF_WEIGHT_LEXICAL,
)
from ldi.provisions import resolve
from ldi.query import (
    COMPARISON,
    QueryFrame,
    document_version,
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
    candidates: tuple[Hit, ...] = ()


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
    diagnostic: bool = False,
) -> Retrieval:
    started = time.perf_counter()
    frame = _frame(index, question, today)
    if not _searchable(frame):
        raise ValueError("That question has no words to search for.")
    flags = index.file_flags()
    named = source_names((path for path, (_, copy) in flags.items() if not copy), question)
    frame = _without_name_numbers(frame, named)
    sides = _comparison_sources(question, named, flags)
    if not sides and config.DOCUMENT_FIRST_RECONCILIATION:
        sides = _address_sources(index, question, flags, depth)
    missing = _missing_entities(index, frame)
    if missing:
        recovered = _recovery_frame(frame, missing)
        if recovered is None:
            return _empty(frame, started, dense_called=False)
        frame = recovered
    copies: dict[str, list[str]] = {}
    fields = _document_fields(question) if sides and config.DOCUMENT_FIRST_RECONCILIATION else ()
    rows = _lexical_rows(index, frame, depth, flags, copies, named, sides, fields)
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
        depth=depth + sum(row.path in sides for row in rows[depth:]),
        span_ids={row.passage_id: row.span_id for row in rows},
        lexical_ranks={row.passage_id: row.lexical_rank for row in rows},
    )
    fused_at = time.perf_counter()
    by_id = {row.passage_id: row for row in rows}
    lexical_set = set(lexical_ids)
    exact = _exact_ids(index, frame, depth)
    fused: list[_Row] = []
    dense_stats: tuple[int, dict[str, int]] | None = None
    for passage_id, score in fused_scores:
        row = by_id.get(passage_id)
        if row is None:
            if dense_stats is None:
                dense_stats = (
                    index.passage_count(),
                    {token: index.document_frequency(token) for token in frame.required},
                )
            row = _load(index, passage_id, flags, copies, frame, *dense_stats)
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
    gated = [
        row for row in fused
        if _eligible(row, frame) or (row.path in sides and _field_match(row.text, fields))
    ]
    if missing:
        scan_paths = {
            path for path in {row.path for row in gated}
            if any(span.source_type == "ocr" for span in index.spans_for(path))
        }
        gated = [row for row in gated if row.path in scan_paths]
    before = [_as_hit(row, copies, {}) for row in gated]
    ranked_rows, extras = _hash_dedupe(gated)
    if sides:
        ranked_rows = _named_first(ranked_rows, sides, question)
    ranked = [_as_hit(row, copies, extras) for row in ranked_rows]
    if missing:
        ranked = [replace(hit, required_entities=missing) for hit in ranked]
    if sides:
        ranked = [replace(hit, comparison_sources=sides) for hit in ranked]
    required_source = explicit_source(question, named)
    if required_source:
        ranked = [replace(hit, required_source=required_source) for hit in ranked]
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
        "named_files": sorted(named),
        "unverified_entities": list(missing),
        "comparison_sources": list(sides),
    }
    return Retrieval(
        hits=ranked[:limit],
        pointer=[_as_hit(row, copies, extras) for row in pointer_rows],
        before_hash=before,
        ranked=ranked,
        frame=frame,
        log=log,
        resolutions=resolutions,
        candidates=tuple(_as_hit(row, copies, {}) for row in fused) if diagnostic else (),
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


def _missing_entities(index: Index, frame: QueryFrame) -> tuple[str, ...]:
    return tuple(
        phrase for phrase in (*frame.ids, *frame.names)
        if not index.exact_ids(phrase_query(phrase), 1)
    )


def _recovery_frame(frame: QueryFrame, missing: tuple[str, ...]) -> QueryFrame | None:
    """Discover a scan from independent words while retaining unverified identity at release."""
    if not re.search(r"\b(?:invoice|proforma|receipt|quotation|bills?)\b", frame.question, re.I):
        return None
    if any(entity in frame.names and " " not in entity for entity in missing):
        return None
    required = tuple(term for term in frame.required if term not in missing)
    if sum(term.isalpha() and len(term) >= 3 for term in required) < 2:
        return None
    return replace(
        frame, required=required,
        ids=tuple(ident for ident in frame.ids if ident not in missing),
        names=tuple(name for name in frame.names if name not in missing),
    )


def _lexical_rows(
    index: Index,
    frame: QueryFrame,
    depth: int,
    flags: dict[str, tuple[bool, str | None]],
    copies: dict[str, list[str]],
    named: dict[str, tuple[float, frozenset[str]]] | None = None,
    sides: tuple[str, ...] = (),
    fields: tuple[str, ...] = (),
) -> list[_Row]:
    named = named or {}
    terms = list(frame.required)
    for alias in frame.aliases:
        if alias not in terms:
            terms.append(alias)
    match = or_query(terms) if terms else ""
    found = index.lexical_ids(match, depth, BM25_HEADING_WEIGHT, BM25_BODY_WEIGHT)
    reduced_frame = frame
    if sides:
        framing = {
            "both", "which", "versions", "version", "listed", "named", "first", "limb",
            "compare", "match", "same", "different", "does", "agree", "make",
        }
        terms = [term for term in terms if term not in framing]
        if terms:
            match = or_query(terms)
            found = index.lexical_ids(match, depth, BM25_HEADING_WEIGHT, BM25_BODY_WEIGHT)
            required = tuple(term for term in frame.required if term not in framing)
            reduced_frame = replace(frame, required=required or frame.required)
    elif named:
        found, reduced_frame = _rescore_named(index, frame, terms, found, named, depth)
    ordered = [passage_id for passage_id, _score in found]
    for path in sides:
        query = or_query(fields) if fields else match
        per_source = index.lexical_ids_in(query, path, depth, BM25_HEADING_WEIGHT, BM25_BODY_WEIGHT)
        for passage_id, _score in per_source:
            if passage_id not in ordered:
                ordered.append(passage_id)
    for passage_id in _exact_ids(index, frame, depth):
        if passage_id not in ordered:
            ordered.append(passage_id)
    rows: list[_Row] = []
    total = index.passage_count()
    frequencies = {token: index.document_frequency(token) for token in frame.required}
    for passage_id in ordered:
        row = _load(index, passage_id, flags, copies, reduced_frame if sides else frame, total, frequencies)
        if row is not None and row.path in named and not sides:
            row = _load(index, passage_id, flags, copies, reduced_frame, total, frequencies)
        if row is not None:
            rows.append(row)
    if frame.hard_entities:
        rows = [row for row in rows if slot_supported(frame, row.text)]
    ranked = _apply_bonus(rows)
    if sides:
        if fields:
            amount = _requested_amount(frame.question)
            ranked.sort(key=lambda row: (
                row.path not in sides,
                not (amount is not None and _mentions_amount(row.text, amount)),
                not _field_match(row.text, fields),
            ))
        ranked = _named_first(ranked, sides, frame.question)
    return [
        replace(row, lexical_rank=rank) for rank, row in enumerate(ranked, start=1)
    ]


def named_files(index: Index, question: str) -> dict[str, tuple[float, frozenset[str]]]:
    """Files the question names by a run of adjacent file-name words.

    The run needs two adjacent tokens, or two tokens that join into one name word
    ("Cal Poly" is calpoly), and one word that no other file name has. Scattered
    file-name words are content, not a reference to the file.
    """
    return source_names(
        (path for path, (_superseded, duplicate) in index.file_flags().items() if not duplicate),
        question,
    )


def source_names(paths: Iterable[str], question: str) -> dict[str, tuple[float, frozenset[str]]]:
    paths = sorted(paths)
    names = {path: _name_words(path) for path in paths}
    tokens = [_fold_name(token) for token in re.findall(r"[^\W_]+", _nfc(question))]
    found: dict[str, tuple[float, frozenset[str]]] = {}
    for path in paths:
        words = names[path]
        if not words:
            continue
        others = set().union(*(names[other] for other in paths if other != path))
        matched: set[str] = set()
        for run in _name_runs(tokens, words):
            if any(word not in others for word in run):
                matched.update(run)
        if matched:
            found[path] = (len(matched) / len(words), frozenset(matched))
    return found


def explicit_source(question: str, named: dict[str, tuple[float, frozenset[str]]]) -> str:
    if len(named) != 1 or COMPARISON.search(question):
        return ""
    path, (_, words) = next(iter(named.items()))
    return path if len(words) >= 3 or any(word.isdigit() for word in words) else ""


def _name_runs(tokens: list[str], words: set[str]) -> list[set[str]]:
    """Runs of at least two adjacent tokens covered by file-name words."""
    covered: list[set[str]] = [set() for _token in tokens]
    for position, token in enumerate(tokens):
        if token in words:
            covered[position].add(token)
        if position + 1 < len(tokens):
            joined = token + tokens[position + 1]
            if joined in words:
                covered[position].add(joined)
                covered[position + 1].add(joined)
    runs: list[set[str]] = []
    start = 0
    while start < len(tokens):
        if not covered[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(tokens) and covered[end + 1]:
            end += 1
        if end > start:
            runs.append(set().union(*covered[start : end + 1]))
        start = end + 1
    return runs


def _name_words(path: str) -> set[str]:
    return _name_tokens(Path(path).stem)


def _name_tokens(text: str) -> set[str]:
    return {_fold_name(word) for word in re.split(r"[\W_]+", _nfc(text)) if word}


def _fold_name(word: str) -> str:
    folded = word.casefold()
    if len(folded) > 3 and folded.endswith("s") and not folded.endswith("ss"):
        return folded[:-1]
    return folded


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _without_name_numbers(
    frame: QueryFrame, named: dict[str, tuple[float, frozenset[str]]]
) -> QueryFrame:
    """A year in a document's name is a reference to the document, not a value to find."""
    words = _all_name_words(named)
    if not words:
        return frame
    numbers = tuple(number for number in frame.numbers if _fold_name(number) not in words)
    return replace(frame, numbers=numbers)


def _all_name_words(named: dict[str, tuple[float, frozenset[str]]]) -> set[str]:
    return set().union(*(words for _share, words in named.values())) if named else set()


def _rescore_named(
    index: Index,
    frame: QueryFrame,
    terms: list[str],
    found: list[tuple[str, float]],
    named: dict[str, tuple[float, frozenset[str]]],
    depth: int,
) -> tuple[list[tuple[str, float]], QueryFrame]:
    """Score a named file's passages without its own name words, then boost them."""
    words = _all_name_words(named)
    reduced_terms = [term for term in terms if _fold_name(term) not in words]
    reduced_required = tuple(term for term in frame.required if _fold_name(term) not in words)
    reduced_frame = replace(frame, required=reduced_required) if reduced_required else frame
    match = or_query(reduced_terms) if reduced_terms else or_query(terms) if terms else ""
    scores = {passage_id: score for passage_id, score in found}
    order = {passage_id: rank for rank, (passage_id, _score) in enumerate(found)}
    owner: dict[str, str] = {}
    for passage_id in scores:
        stored = index.passage_row(passage_id)
        if stored is not None:
            owner[passage_id] = str(stored[0])
    best_share = max(share for share, _words in named.values())
    for path in sorted(named):
        share, _words = named[path]
        rescored = index.lexical_ids_in(match, path, depth, BM25_HEADING_WEIGHT, BM25_BODY_WEIGHT)
        if not rescored:
            continue
        for passage_id in [item for item, home in owner.items() if home == path]:
            scores.pop(passage_id, None)
        factor = 1.0 + NAME_BOOST * share / best_share
        for passage_id, score in rescored:
            scores[passage_id] = score * factor
            order.setdefault(passage_id, len(order))
    ranked = sorted(scores, key=lambda item: (scores[item], order[item], item))
    return [(passage_id, scores[passage_id]) for passage_id in ranked], reduced_frame


def _comparison_sources(question: str, named: dict, flags: dict) -> tuple[str, ...]:
    fields = _document_fields(question)
    active = {path for path, (old, copy) in flags.items() if not old and not copy}
    terms = _name_tokens(question)
    if config.DOCUMENT_FIRST_RECONCILIATION and "ledger" in terms and fields:
        periods = {term for term in terms if term.isdigit() and len(term) in (2, 4)}
        category = _ledger_category(question)
        ledgers = [path for path in active if "ledger" in _name_words(path)]
        supporting = [
            path for path in active if path not in ledgers and periods
            and periods <= _name_words(path) and category
            and _ledger_category(Path(path).stem) == category
        ]
        if len(ledgers) == len(supporting) == 1:
            return ledgers[0], supporting[0]
    if not (COMPARISON.search(question) or re.search(r"\bversions\b", question, re.I)):
        return ()
    selected = [path for path in named if path in active]
    if len(selected) >= 2:
        return tuple(selected)
    if len(selected) == 1 and re.search(r"\b(?:original|restated|amendment)\b", question, re.I):
        words = _name_words(selected[0]) - {
            "agreement", "amended", "amendment", "restated", "first", "original", "lease",
        }
        related = [
            path for path in active - set(selected)
            if len({word for word in words & _name_words(path) if word.isalpha()}) >= 2
        ]
        if len(related) == 1:
            return related[0], selected[0]
    if len(selected) == 1 and len(terms & {"handout", "manual", "form"}) >= 2:
        terms = terms - _name_words(selected[0])
        scores = [
            (len(terms & _name_words(path)), path) for path in active - set(selected)
        ]
        scores.sort(reverse=True)
        if scores and scores[0][0] >= 2 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            return selected[0], scores[0][1]
    versions: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for path in active:
        version = document_version(path)
        if version is not None:
            versions.setdefault(version[0], []).append((version[1], path))
    version_pairs = [
        tuple(path for _, path in sorted(rows)) for family, rows in versions.items()
        if len(rows) == 2 and (_name_tokens(family) <= terms or "version" in terms)
    ]
    if len(version_pairs) == 1:
        return version_pairs[0]
    for roles in (("invoice", "contract"), ("invoice", "payment"), ("quotation", "invoice")):
        if not set(roles) <= terms:
            continue
        matches = [sorted(path for path in active if role in _name_words(path)) for role in roles]
        if all(len(paths) == 1 for paths in matches) and matches[0] != matches[1]:
            return tuple(paths[0] for paths in matches)
    return ()


def _document_fields(question: str) -> tuple[str, ...]:
    terms = _name_tokens(question)
    if "address" in terms and COMPARISON.search(question):
        return "address", "place of business", "principal office", "mailing address", "registered office"
    if "ledger" in terms and terms & {"amount", "reconcile", "figure", "match"}:
        return "grand total", "total invoice value", "net amt", "payment advice", "rupees", "total"
    return ()


def _address_sources(index: Index, question: str, flags: dict, depth: int) -> tuple[str, ...]:
    if "address" not in _document_fields(question):
        return ()
    party = re.search(r"\b([A-Za-z]{3,})['’]s\s+address\b", question, re.I)
    if not party:
        return ()
    sources = set()
    for passage_id, _ in index.lexical_ids(
        phrase_query(f"the {party[1]}"), depth, BM25_HEADING_WEIGHT, BM25_BODY_WEIGHT
    ):
        row = index.passage_row(passage_id)
        if row is not None and flags.get(row[0]) == (False, None):
            sources.add(row[0])
    if len(sources) != 2:
        return ()
    one, other = sorted(sources)
    shared = (_name_words(one) & _name_words(other)) - {"agreement", "contract", "amendment"}
    return (one, other) if sum(word.isalpha() for word in shared) >= 2 else ()


def _requested_amount(question: str) -> Decimal | None:
    amount = written_rupees(question)
    if amount is not None:
        return amount
    match = re.search(r"(?:₹|\b(?:inr|rs\.?)\s*)\s*([0-9][0-9,]*(?:\.\d+)?)", question, re.I)
    return Decimal(match[1].replace(",", "")) if match else None


def _mentions_amount(text: str, amount: Decimal) -> bool:
    if written_rupees(text) == amount:
        return True
    folded = f" {fold_for_fts(text)} "
    canonical = format(amount.normalize(), "f")
    return re.search(rf"(?<![\d.]){re.escape(canonical)}(?:\.0+)?(?![\d.])", folded) is not None


def _ledger_category(text: str) -> str:
    match = re.search(r"(?<![A-Za-z])(?:e[- _]?journals?|e[- _]?books?|books?|delnet)(?![A-Za-z])", text, re.I)
    return match[0].casefold().replace("-", "").replace("_", "").replace(" ", "").rstrip("s") if match else ""


def _field_match(text: str, fields: tuple[str, ...]) -> bool:
    if not fields:
        return False
    folded = f" {fold_for_fts(text)} "
    return any(f" {fold_for_fts(field)} " in folded for field in fields)


def _named_first(rows: list[_Row], named: tuple[str, ...], question: str) -> list[_Row]:
    """Reserve the best current passage from each requested source."""
    subject = re.search(r"\b([a-z]+ leave)\b", question, re.I)
    if subject and re.search(r"\bsemester\b", question, re.I):
        rows = sorted(rows, key=lambda row: not (
            row.path in named and subject[1].casefold() in row.text.casefold()
            and re.search(r"\b\d+\s+(?:working\s+)?days?\s+(?:in|during|per)\s+(?:a\s+|the\s+)?semester\b",
                          row.text, re.I)
        ))
    leaders: list[_Row] = []
    for row in rows:
        if row.path in named and all(leader.path != row.path for leader in leaders):
            leaders.append(row)
    lead_ids = {row.passage_id for row in leaders}
    return leaders + [row for row in rows if row.passage_id not in lead_ids]


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
