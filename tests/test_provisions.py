"""The version table and resolver stay off the answer path."""

from pathlib import Path

from ldi.cli import main
from ldi.config import index_path
from ldi.provisions import (
    Resolution,
    extract_provisions,
    mentioned_pair,
    parse_dates,
    phrase_hits,
    resolve,
    supersession_date,
)
from ldi.store import Index


class _Span:
    def __init__(self, text, span_id, **extra):
        self.text = text
        self.span_id = span_id
        self.page = extra.get("page", 1)
        self.block = extra.get("block", 0)
        self.line = extra.get("line", 0)
        self.col = extra.get("col", 0)
        self.source_type = extra.get("source_type", "text")
        self.row_key = extra.get("row_key", "")
        self.table_id = extra.get("table_id", "")
        self.header_path = extra.get("header_path", "")


def test_the_resolver_is_not_imported_by_the_answer_path():
    root = Path(__file__).resolve().parents[1] / "ldi"
    for name in ("ask.py", "evidence.py", "verify.py"):
        assert "provisions" not in (root / name).read_text()


def test_dates_cover_numeric_long_and_fiscal_forms():
    assert parse_dates("on 01/04/2024") == ["2024-04-01"]
    assert parse_dates("1st April 2024") == ["2024-04-01"]
    assert parse_dates("w.e.f. 01.07.2023") == ["2023-07-01"]
    assert parse_dates("FY 2024-25") == ["2024-04-01"]
    assert parse_dates("Olst July,2023") == ["2023-07-01"]
    assert parse_dates("1St April 2024") == ["2024-04-01"]


def test_an_enhancement_links_the_old_rate_to_the_new_one():
    spans = [
        _Span(
            "Dearness allowance shall be enhanced from 42% to 46% of the Basic Pay "
            "with effect from 1st July, 2023.",
            "s1",
        )
    ]
    rows = extract_provisions("orders/a.pdf", spans)
    old = next(row for row in rows if row.value_text == "42%")
    new = next(row for row in rows if row.value_text == "46%")
    assert old.clause_key == "basic pay"
    assert old.status == "superseded"
    assert old.effective_to == "2023-07-01"
    assert old.superseded_by == new.provision_id
    assert new.status == "current"
    assert new.effective_from == "2023-07-01"
    assert new.value_span == "s1"
    assert resolve(rows, "basic pay", "2023-06-01").rows[0].value_text == "42%"
    assert resolve(rows, "basic pay", "2023-07-01").rows[0].value_text == "46%"
    assert [row.provision_id for row in extract_provisions("orders/a.pdf", spans)] == [
        row.provision_id for row in rows
    ]
    wef = extract_provisions(
        "orders/w.pdf",
        [
            _Span(
                "It shall be enhanced from the existing rate of 12% to 15% w.e.f. 01.04.2024.",
                "w",
            )
        ],
    )
    assert {row.value_text for row in wef} == {"12%", "15%"}
    assert next(row for row in wef if row.value_text == "15%").effective_from == "2024-04-01"


def test_an_impossible_day_uses_the_numeric_date_on_the_same_page():
    spans = [
        _Span(
            "It shall be enhanced from 10% to 12% of the Basic Pay with effect from 151 July,",
            "a",
            page=2,
        ),
        _Span("2024.", "b", page=2),
        _Span("see the note", "c", page=2),
        _Span("effective from 01.07.2024", "d", page=2),
        _Span("effective from 01.07.2020", "e", page=9),
    ]
    rows = extract_provisions("orders/split.pdf", spans)
    current = next(row for row in rows if row.value_text == "12%")
    assert current.effective_from == "2024-07-01"


def test_a_revision_header_pairs_the_two_amounts_and_keeps_the_chunk():
    spans = [
        _Span("Existing Rates Revised Rates w.e.f 01.01.2024", "h"),
        _Span("Rs 7200/- per month Rs 9000/- per month", "p"),
        _Span("1-2 Rs 720/- 1-2 Rs 900/-", "q"),
    ]
    rows = extract_provisions("orders/b.pdf", spans)
    monthly = [row for row in rows if row.value_text in {"7200", "9000"}]
    assert {row.value_text for row in monthly} == {"7200", "9000"}
    old = next(row for row in monthly if row.value_text == "7200")
    assert old.effective_to == "2024-01-01"
    assert old.superseded_by
    band = [row for row in rows if row.value_text in {"720", "900"}]
    assert {row.value_text for row in band} == {"720", "900"}


def test_two_cells_in_one_revised_row_become_a_pair():
    spans = [
        _Span("Existing Rates Revised Rates w.e.f. 01/01/2024", "h"),
        _Span("Rs 4100/-", "a", source_type="table", table_id="t", line=2, col=1, row_key="(PCA)"),
        _Span("Rs 5125/-", "b", source_type="table", table_id="t", line=2, col=2, row_key="(PCA)"),
    ]
    rows = extract_provisions("orders/c.pdf", spans)
    assert {row.value_text for row in rows} == {"4100", "5125"}
    old = next(row for row in rows if row.value_text == "4100")
    new = next(row for row in rows if row.value_text == "5125")
    assert old.superseded_by == new.provision_id
    assert old.value_span == "a"
    assert new.value_span == "b"
    labelled = extract_provisions(
        "orders/c.pdf",
        [
            _Span("Existing Rates Revised Rates w.e.f. 01/01/2024", "h"),
            _Span("Rs 1350/-", "a", source_type="table", table_id="t", line=3, col=1, row_key="PGA"),
            _Span(
                "Post Graduate Rs 1688/-",
                "b",
                source_type="table",
                table_id="t",
                line=3,
                col=2,
                row_key="PGA",
            ),
        ],
    )
    assert {row.value_text for row in labelled} == {"1350", "1688"}


def test_an_obligation_is_not_supersession_and_a_phrase_carries_its_date():
    assert phrase_hits("The employee must give 30 days' notice.") is False
    assert supersession_date("The employee must give 30 days' notice.") == ""
    assert supersession_date("This order supersedes the memorandum dated 01/04/2024.") == "2024-04-01"
    assert supersession_date("In partial modification of the order dated 15th August 2024.") == "2024-08-15"
    earlier = _Span(
        "The rate shall be enhanced from 10% to 12% of the Basic Pay with effect from 1st April 2024.",
        "old",
    )
    later = _Span(
        "In partial modification of the earlier order, the rate shall be enhanced from 12% to 14% "
        "of the Basic Pay with effect from 15th August 2024.",
        "new",
    )
    rows = extract_provisions("orders/d.pdf", [earlier, later])
    twelve = [row for row in rows if row.value_text == "12%" and row.effective_from == "2024-04-01"]
    assert twelve
    assert twelve[0].status == "superseded"
    assert twelve[0].superseded_by


def test_the_resolver_does_not_use_the_filename_and_reports_both_claimants():
    early = extract_provisions(
        "z-newest-name.pdf",
        [
            _Span(
                "It shall be enhanced from 10% to 12% of the Basic Pay with effect from 1st April 2020.",
                "e",
            )
        ],
    )
    late = extract_provisions(
        "a-old-name.pdf",
        [
            _Span(
                "It shall be enhanced from 12% to 14% of the Basic Pay with effect from 1st April 2024.",
                "l",
            )
        ],
    )
    found = resolve(early + late, "basic pay", "2025-01-01")
    assert found.status == "current"
    assert found.rows[0].value_text == "14%"
    assert found.rows[0].doc_id == "a-old-name.pdf"
    city = extract_provisions(
        "other.pdf",
        [
            _Span("Existing Rates Revised Rates w.e.f 01.01.2020", "h"),
            _Span("Rs 100/- per month Rs 200/- per month", "c"),
        ],
    )
    assert resolve(early + city, "basic pay", "2021-01-01").rows[0].value_text == "12%"
    assert resolve(early + city, city[0].clause_key, "2021-01-01").rows[0].value_text == "200"
    blank = [
        early[1].__class__(
            **{**early[1].__dict__, "provision_id": "one", "effective_from": "", "superseded_by": ""}
        ),
        early[1].__class__(
            **{
                **early[1].__dict__,
                "provision_id": "two",
                "value_text": "99%",
                "effective_from": "",
                "superseded_by": "",
            }
        ),
    ]
    conflict = resolve(blank, "basic pay", "2025-01-01")
    assert conflict.status == "conflicting"
    assert {row.provision_id for row in conflict.rows} == {"one", "two"}


def test_a_cycle_is_conflicting_and_not_a_silent_pick():
    rows = extract_provisions(
        "loop.pdf",
        [
            _Span(
                "It shall be enhanced from 10% to 12% of the Basic Pay with effect from 1st April 2024.",
                "s",
            )
        ],
    )
    first, second = rows
    looped = [
        first.__class__(**{**first.__dict__, "superseded_by": second.provision_id}),
        second.__class__(**{**second.__dict__, "superseded_by": first.provision_id}),
    ]
    found = resolve(looped, "basic pay", "2025-01-01")
    assert found.status == "conflicting"
    assert len(found.rows) == 2


def test_ingest_stores_provisions_and_leaves_the_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "note.txt"
    sentence = (
        "The allowance shall be enhanced from 10% to 12% of the Basic Pay "
        "with effect from 1st April 2024.\n"
    )
    note.write_text(sentence)
    assert main(["ingest", str(note)]) == 0
    with Index(index_path()) as index:
        body = index.db.execute("SELECT body FROM chunks").fetchone()[0]
        rows = index.db.execute(
            "SELECT value_text, status, superseded_by FROM provisions ORDER BY value_text"
        ).fetchall()
    assert "10%" in body
    assert "12%" in body
    assert [row[0] for row in rows] == ["10%", "12%"]
    assert rows[0][1] == "superseded"
    assert rows[0][2]
    assert mentioned_pair(sentence) == ("10%", "12%")
    assert isinstance(resolve([], "basic pay", "2024-04-01"), Resolution)
