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


def test_a_year_in_a_named_guide_does_not_pin_its_cover_page(tmp_path):
    index = Index(tmp_path / "index.sqlite")
    try:
        _add_many(
            index,
            "guide_2026.pdf",
            [("cover", "Guide 2026"), ("body", "Maternity leave is 180 days for every employee.")],
        )
        _add_many(index, "other_2026.pdf", [("other", "The 2026 leave calendar lists every holiday.")])
        index.refresh_labels()
        found = retrieve(index, "Under the 2026 guide what maternity leave applies?", 4)
    finally:
        index.close()
    assert found.log["named_files"] == ["guide_2026.pdf"]
    assert found.hits[0].span_id == "body"
    assert found.frame.numbers == ()


def test_a_question_naming_two_files_gets_both(tmp_path):
    index = Index(tmp_path / "index.sqlite")
    try:
        _add_many(index, "alpha_tariff_2024.md", [("alpha", "Berth rates are 40 per metre.")])
        _add_many(index, "beta_tariff_2025.md", [("beta", "Berth rates are 55 per metre.")])
        for number in range(6):
            _add_many(
                index,
                f"note_{number}.md",
                [(f"note{number}", "Compare tariff rates: berth rates, tariff rates, and more rates.")],
            )
        index.refresh_labels()
        found = retrieve(
            index, "Compare berth rates in the alpha tariff and the beta tariff", 4
        )
    finally:
        index.close()
    top = [hit.path for hit in found.hits[:4]]
    assert "alpha_tariff_2024.md" in top
    assert "beta_tariff_2025.md" in top


def test_scattered_file_name_words_are_content_not_a_file_reference(tmp_path):
    from ldi.retrieve import named_files

    rows = {
        "dae_allowances_2024.pdf": ("dae", "Nursing allowance is 7200 a month for DAE staff."),
        "staff_notes.md": ("notes", "The nursing allowance was revised for DAE staff."),
    }
    with _index(tmp_path, rows) as index:
        assert named_files(index, "What is the DAE nursing allowance?") == {}
        found = retrieve(index, "What is the DAE nursing allowance?", 4)
    assert found.log["named_files"] == []


def test_a_dense_only_passage_loads_and_stays_out_of_hits(tmp_path):
    rows = {
        "pump.md": ("pump", "The alpha pump was serviced after the seal failed."),
        "gasket.md": ("gasket", "Gaskets are stored in the north cabinet."),
    }
    with _index(tmp_path, rows) as index:
        dense = FakeDenseRetriever(["gasket", "pump"])
        found = retrieve(index, "When was the alpha pump serviced?", 4, dense=dense)
    assert dense.calls == 1
    assert [hit.span_id for hit in found.hits] == ["pump"]
    assert "gasket" in found.log["coverage"]
    assert found.log["coverage"]["gasket"] == 0.0


def test_a_dense_only_pointer_needs_the_question_id(tmp_path):
    rows = {
        "paid.md": ("paid", "Invoice INV-4471 was paid in March."),
        "other.md": ("other", "Payments are posted at the end of each month."),
    }
    with _index(tmp_path, rows) as index:
        plain = retrieve(
            index, "When was the invoice paid?", 4, dense=FakeDenseRetriever(["other"])
        )
        with_id = retrieve(
            index, "When was invoice INV-4471 paid?", 4, dense=FakeDenseRetriever(["other"])
        )
    assert "other" in [hit.span_id for hit in plain.pointer]
    assert "other" not in [hit.span_id for hit in plain.hits]
    assert "other" not in [hit.span_id for hit in with_id.pointer]
    assert [hit.span_id for hit in with_id.hits] == ["paid"]


def _add_many(index: Index, path: str, passages: list[tuple[str, str]]) -> None:
    digest = hashlib.sha256(path.encode()).hexdigest()
    records = [
        PassageRecord(
            passage_id=span_id,
            span_ids=span_id,
            text=text,
            passage_hash=hashlib.sha256(f"{path}{text}".encode()).hexdigest(),
            page_start=None,
        )
        for span_id, text in passages
    ]
    body = "\n".join(text for _span_id, text in passages)
    index.replace_file(path, 1, len(body), [(None, body)], digest)
    index.replace_spans(path, [], records)


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
