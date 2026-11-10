"""Passage retrieval: exact ids, quoted operators, and a stable ranking."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from eval.traps import near_id_pair
from ldi.retrieve import adjusted_rank, retrieve
from ldi.spans import PassageRecord
from ldi.store import Index
from tests.fake_dense import FakeDenseRetriever


def test_the_bonus_cannot_pin_a_hit_to_the_top():
    assert adjusted_rank(1, True) == 1
    assert adjusted_rank(4, True) == 2
    assert adjusted_rank(5, False) == 5


def test_an_exact_id_beats_a_dense_neighbour_and_a_missing_id_is_not_found(tmp_path):
    left, right = near_id_pair(11)
    exact_id = "exact"
    near_id = "near"
    with _index(
        tmp_path,
        {
            "paid.md": (exact_id, f"The paid invoice number is {left} for the quarter."),
            "other.md": (near_id, f"The other invoice number is {right} for the quarter."),
        },
    ) as index:
        fake = FakeDenseRetriever([near_id, exact_id])
        found = retrieve(index, f"Where is invoice {left}?", 4, dense=fake)
        assert found.hits
        assert left in found.hits[0].text
        assert right not in found.hits[0].text
        assert fake.calls == 1
        absent = Index(tmp_path / "absent.sqlite")
        try:
            _add(absent, "other.md", near_id, f"The other invoice number is {right} for the quarter.")
            absent.refresh_labels()
            missed = FakeDenseRetriever([near_id])
            empty = retrieve(absent, f"Where is invoice {left}?", 4, dense=missed)
        finally:
            absent.close()
    assert empty.hits == []
    assert missed.calls == 0
    assert empty.log["dense_called"] is False


def test_numbers_designators_and_operator_words_are_searchable(tmp_path):
    with _index(
        tmp_path,
        {
            "fee.md": (
                "fee",
                "Annexure I sets Clause 4. The fee is INR 12,34,567 and the rate is 18.4.",
            ),
            "side.md": ("side", "Alpha sits near beta after the note."),
        },
    ) as index:
        annex = retrieve(index, "What does Annexure I say?", 4)
        clause = retrieve(index, "What is Clause 4?", 4)
        grouped = retrieve(index, "Show 12,34,567", 4)
        decimal = retrieve(index, "What is 18.4?", 4)
        operators = retrieve(index, "alpha and or near beta", 4)
    assert annex.hits and "Annexure I" in annex.hits[0].text
    assert clause.hits and "Clause 4" in clause.hits[0].text
    assert grouped.hits and "12,34,567" in grouped.hits[0].text
    assert decimal.hits and "18.4" in decimal.hits[0].text
    assert operators.hits
    assert operators.log["margin"] is None or isinstance(operators.log["margin"], float)


def test_the_same_question_ranks_the_same_way_twice(tmp_path):
    with _index(
        tmp_path,
        {
            "one.md": ("s1", "The alpha pump was serviced after the seal failed."),
            "two.md": ("s2", "The beta valve was replaced after the seal failed."),
        },
    ) as index:
        first = retrieve(index, "Which seal failed?", 4)
        second = retrieve(index, "Which seal failed?", 4)
    assert [hit.span_id for hit in first.hits] == [hit.span_id for hit in second.hits]
    assert first.log["rrf"] == second.log["rrf"]


def test_a_superseded_copy_is_kept_when_the_text_matches(tmp_path):
    text = "Leave is 21 days in the current staff handbook."
    with _index(
        tmp_path,
        {
            "current.md": ("cur", text),
            "old.md": ("old", text),
        },
    ) as index:
        index.db.execute("UPDATE files SET superseded = 1 WHERE path = ?", ("old.md",))
        index.db.commit()
        found = retrieve(index, "How many leave days?", 4)
    assert [hit.path for hit in found.ranked] == ["current.md", "old.md"]
    assert found.ranked[1].superseded is True


def test_the_ask_path_uses_the_retriever_for_source_evidence():
    import ldi.ask as ask

    source = inspect.getsource(ask)
    assert "retrieve(" in source
    retrieve_source = (Path(__file__).resolve().parents[1] / "ldi" / "retrieve.py").read_text()
    assert "mlx" not in retrieve_source


def _index(tmp_path: Path, rows: dict[str, tuple[str, str]]) -> Index:
    index = Index(tmp_path / "index.sqlite")
    for path, (span_id, text) in rows.items():
        _add(index, path, span_id, text)
    index.refresh_labels()
    return index


def _add(index: Index, path: str, span_id: str, text: str) -> None:
    digest = hashlib.sha256(path.encode()).hexdigest()
    passage = PassageRecord(
        passage_id=span_id,
        span_ids=span_id,
        text=text,
        passage_hash=hashlib.sha256(text.encode()).hexdigest(),
        page_start=None,
    )
    index.replace_file(path, 1, len(text), [(None, text)], digest)
    index.replace_spans(path, [], [passage])
