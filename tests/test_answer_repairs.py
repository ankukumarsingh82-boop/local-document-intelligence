"""Regressions that exercise retrieval and release with indexed source data."""

from pathlib import Path

from openpyxl import Workbook

from ldi import config
from ldi.ask import settle_answer
from ldi.evidence import Evidence, attach_source, build_evidence, release_from_evidence
from ldi.ingest import ingest
from ldi.retrieve import retrieve
from ldi.spans import build_spans
from ldi.store import Hit, Index

_ABSTAIN = '{"status":"not_found","evidence":[],"answer":""}'


def test_formula_operands_do_not_veto_the_requested_cell(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Budget"
    sheet.append(["Department", "Budget", "Actual", "Variance"])
    sheet.append(["Operations", 120000, 50000, "=B2-C2"])
    workbook.save(tmp_path / "budget.xlsx")

    question = "Budget variance D2 formula B2-C2"
    with Index(tmp_path / "index.sqlite") as index:
        ingest(tmp_path / "budget.xlsx", index, ocr=False)
        found = retrieve(index, question, 4)
        hits = attach_source(question, found.hits, index.spans_for)
    assert found.frame.ids == ("d2",)
    _, answer = settle_answer(lambda _: _ABSTAIN, hits, question)
    assert answer.kind == "answer"
    assert 'cite: "Budget!D2 70000"' in answer.text


def test_invoice_comparison_reserves_a_contract_slot(tmp_path):
    question = "What did the Kestrel invoice bill as the annual fee, and does it match the contract?"
    (tmp_path / "vendor_invoice.md").write_text("Annual service fee billed: INR 2,60,00,000.")
    (tmp_path / "vendor_contract.md").write_text(
        "Clause 7.2 Fees: Annual fee is INR 2,40,00,000 payable quarterly."
    )
    (tmp_path / "board_note.md").write_text("Kestrel invoice billed an annual fee of INR 2,70,00,000.")
    with Index(tmp_path / "index.sqlite") as index:
        ingest(tmp_path, index, ocr=False)
        found = retrieve(index, question, 2)
        hits = attach_source(question, found.hits, index.spans_for)
    assert {Path(hit.path).stem for hit in hits} == {"vendor_invoice", "vendor_contract"}
    _, answer = settle_answer(lambda _: _ABSTAIN, hits, question)
    assert answer.kind == "conflict"
    assert "2,40,00,000" in answer.text and "2,60,00,000" in answer.text


def test_conflict_reply_cannot_release_an_unrelated_salary_advance():
    hits = [Hit("other.pdf", 2, "Salary advance requests need approval from the Managing Director.")]
    conflict = '{"status":"conflict","evidence":["E1"],"answer":"Salary advance requests"}'
    _, answer = settle_answer(lambda _: conflict, hits, "Who approved the manager's salary advance?")
    assert answer.kind == "not_found"


def test_missing_invoice_id_can_discover_a_scan_without_releasing_it(tmp_path):
    path = "supplier.pdf"
    text = "Pearson invoice 11269784\nGrand total INR 14000"
    spans, passages = build_spans([(1, text)], "digest", "ocr", path)
    with Index(tmp_path / "index.sqlite") as index:
        index.replace_file(path, 1, len(text), [(1, text)], "digest")
        index.replace_spans(path, spans, passages)
        question = "What is the Pearson invoice 11269783 grand total?"
        found = retrieve(index, question, 4)
        hits = attach_source(question, found.hits, index.spans_for)
    assert hits and found.log["unverified_entities"] == ["11269783"]
    _, answer = settle_answer(lambda _: _ABSTAIN, hits, question)
    assert answer.kind == "not_found"


def test_version_change_preserves_both_sources_in_order():
    items = [
        Evidence("E1", "license-1.2.pdf", "license-1.2.pdf, page 2", "Treaty article 272 applies."),
        Evidence("E2", "license-1.1.pdf", "license-1.1.pdf, page 2", "Treaty article 238 applies."),
    ]
    answer = release_from_evidence(
        '{"status":"answered","evidence":["E1"],"answer":"272"}',
        items, "How did the treaty article change from version 1.1 to 1.2?",
    )
    assert answer.kind == "answer"
    assert answer.text.startswith("Changed from 238 to 272.")
    assert answer.paths == ("license-1.1.pdf", "license-1.2.pdf")


def test_unpunctuated_scan_lines_keep_the_late_total():
    lines = ["Item and quantity " + str(i) for i in range(35)]
    lines.append("IGST 30,468.24")
    items = build_evidence("What IGST does the invoice charge?", [Hit("scan.pdf", 18, "\n".join(lines))])
    assert any(item.text == "IGST 30,468.24" for item in items)


def test_reconciliation_uses_both_amounts_and_keeps_the_written_quote(monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_FIRST_RECONCILIATION", True)
    question = "Does the 2017-18 DELNET ledger amount match?"
    items = [
        Evidence("E1", "library_ledger.xlsx", "library_ledger.xlsx", "Total INR 13,500"),
        Evidence("E2", "lib_2017-18_delnet.pdf", "lib_2017-18_delnet.pdf, page 3",
                 "Received Rupees thirteen thousand five hundred only."),
    ]
    result = release_from_evidence(_ABSTAIN, items, question)
    assert result.text.startswith("The amounts match.")
    assert "Rupees thirteen thousand five hundred only" in result.text
    assert release_from_evidence(_ABSTAIN, items[:1], question).kind == "retry"


def test_labeled_rate_columns_supply_the_revised_band_value():
    items = [
        Evidence("E1", "rates.pdf", "rates.pdf, page 2",
                 "Existing Rates Revised Rates w.e.f 01.01.2024"),
        Evidence("E2", "rates.pdf", "rates.pdf, page 2",
                 "Level Rate of NRPSA per month 3-5 Rs 990/- 3-5 Rs 1238/-"),
    ]
    question = "What is revised NRPSA for levels 3-5 effective January 2024?"
    answer = release_from_evidence(_ABSTAIN, items, question)
    assert answer.kind == "answer" and answer.text.splitlines()[0] == "1238"
    wrong_month = question.replace("January", "February")
    assert release_from_evidence(_ABSTAIN, items, wrong_month).kind == "retry"


def test_title_alone_is_not_a_leave_entitlement():
    hits = [Hit("guide.pdf", 1, "5.3. Maternity Leave/ Paternity Leave")]
    _, answer = settle_answer(lambda _: _ABSTAIN, hits, "What maternity leave is available?")
    assert answer.kind == "not_found"


def test_spreadsheet_list_number_is_not_the_requested_expenditure():
    hits = [Hit("ledger.xlsx", None, "Budget!A1 3. Shodhganga membership\nBudget!A2 Other service")]
    question = "What is the Shodhganga expenditure?"
    _, answer = settle_answer(lambda _: _ABSTAIN, hits, question)
    assert answer.kind == "not_found"


def test_extra_citation_does_not_hide_the_right_clause():
    items = [
        Evidence("E1", "rules.md", "rules.md", "No. 4 is the night watch."),
        Evidence("E2", "rules.md", "rules.md", "Clause 4.2 The draft limit is 3.5 metres."),
    ]
    result = release_from_evidence(
        '{"status":"answered","evidence":["E1","E2"],"answer":"3.5 metres"}',
        items, "What draft limit does clause 4.2 state?",
    )
    assert result.kind == "answer"
    assert "night watch" not in result.text


def test_employee_id_uses_the_exact_name_and_row(tmp_path):
    (tmp_path / "rows.md").write_text(
        "Name | ID\nAnita Das | EMP-6109\nAnita Dasgupta | EMP-6110\n"
    )
    question = "What is the employee ID for Anita Das?"
    with Index(tmp_path / "index.sqlite") as index:
        ingest(tmp_path, index, ocr=False)
        hits = attach_source(question, retrieve(index, question, 4).hits, index.spans_for)
    _, result = settle_answer(lambda _: _ABSTAIN, hits, question)
    assert result.kind == "answer"
    assert result.text.splitlines()[0] == "EMP-6109"
    assert "EMP-6110" not in result.text


def test_a_semester_conflict_uses_the_semester_value_on_each_side():
    sides = ("handout.pdf", "manual.pdf")
    items = [
        Evidence("E1", sides[0], sides[0], "Students get casual leave of 8 days during a semester.",
                 comparison_sources=sides),
        Evidence("E2", sides[1], sides[1], "Casual leave is 6 days during a semester and 4 days in summer.",
                 comparison_sources=sides),
    ]
    result = release_from_evidence(
        '{"status":"answered","evidence":["E1"],"answer":"8 days"}',
        items, "How much casual leave is there per semester, and do the documents differ?",
    )
    assert result.kind == "conflict"
    assert result.paths == sides
