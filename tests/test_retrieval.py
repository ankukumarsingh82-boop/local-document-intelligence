import os
import sqlite3

from ldi.ask import settle_answer, stale_paths
from ldi.chunk import chunk_text
from ldi.cli import main
from ldi.config import index_path
from ldi.prompt import SYSTEM
from ldi.store import Hit, Index, match_query
from tests.test_tool import _pdf_with_text


def test_line_break_hyphen_nbsp_and_accent_are_normalised():
    assert chunk_text("Travel is reim-\nbursed after approval.") == [
        "Travel is reimbursed after approval."
    ]
    assert chunk_text("The cafe\u0301\u00a0opens.") == ["The café opens."]


def test_hyphenated_word_is_searchable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "travel.md"
    note.write_text("Travel is reim-\nbursed after approval.\n")
    assert main(["ingest", str(note)]) == 0
    capsys.readouterr()
    assert main(["search", "reimbursed"]) == 0
    assert "reimbursed" in capsys.readouterr().out


def test_stopwords_do_not_pull_an_unrelated_note(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "seal.md").write_text(
        "The north pump was serviced. The replacement seal is part SEAL-441.\n"
    )
    (folder / "board.md").write_text("The schedule is posted. The board is updated.\n")
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "what is the replacement seal part?"]) == 0
    found = capsys.readouterr().out
    assert "SEAL-441" in found
    assert "board" not in found


def test_pay_each_year_retrieves_both_annual_fees(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "leave.md").write_text(
        "Employees accrue 21 days of paid annual leave per year.\n"
    )
    (folder / "contract.md").write_text("Annual fee is INR 2,40,00,000 payable quarterly.\n")
    (folder / "invoice.md").write_text("Kestrel Logistics annual fee: INR 2,60,00,000.\n")
    assert main(["ingest", str(folder)]) == 0
    question = "How much does Orvane pay Kestrel each year?"
    with Index(index_path()) as index:
        hits = index.search_question(question, limit=4)
    from ldi.evidence import build_evidence, conflicting

    pair = conflicting(question, build_evidence(question, hits))
    assert pair is not None
    quoted = pair[0].text + pair[1].text
    assert "2,40,00,000" in quoted
    assert "2,60,00,000" in quoted


def test_words_that_never_meet_fall_back_to_or(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "alpha.md").write_text("The alpha pump was serviced.\n")
    (folder / "beta.md").write_text("The beta valve was replaced.\n")
    assert main(["ingest", str(folder)]) == 0
    with Index(index_path()) as index:
        assert index.search(match_query("alpha beta"), limit=4) == []
        hits = index.search_question("alpha beta", limit=4)
    assert {hit.path for hit in hits} == {"documents/alpha.md", "documents/beta.md"}


def test_a_miss_uses_the_exact_sentence(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "alpha.md"
    note.write_text("The alpha pump is ready.\n")
    assert main(["ingest", str(note)]) == 0
    capsys.readouterr()
    assert main(["search", "warranty"]) == 0
    found = capsys.readouterr().out
    assert found.startswith("I could not find supporting evidence in the indexed text")
    assert "Coverage:" in found
    assert SYSTEM.endswith(
        "I could not find supporting evidence in the indexed text. Keep the answer short."
    )


def test_an_empty_pdf_page_is_named_in_the_coverage_note(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    pdf = tmp_path / "scanned_memo.pdf"
    pdf.write_bytes(_pdf_with_text(""))
    assert main(["ingest", str(pdf)]) == 0
    capsys.readouterr()
    assert main(["search", "seal"]) == 0
    found = capsys.readouterr().out
    assert "I could not find supporting evidence in the indexed text" in found
    assert "scanned_memo.pdf page 1 (extraction_incomplete)" in found


def test_same_size_and_mtime_still_reindexes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "amount.md"
    note.write_text("The amount is 45 units.\n")
    assert main(["ingest", str(note)]) == 0
    capsys.readouterr()
    stamp = note.stat().st_mtime_ns
    note.write_text("The amount is 90 units.\n")
    os.utime(note, ns=(stamp, stamp))
    assert note.stat().st_mtime_ns == stamp
    assert main(["ingest", str(note)]) == 0
    capsys.readouterr()
    assert main(["search", "amount"]) == 0
    found = capsys.readouterr().out
    assert "90" in found
    assert "45" not in found


def test_a_changed_file_is_stale_before_an_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "amount.md"
    note.write_text("The amount is 45 units.\n")
    assert main(["ingest", str(note)]) == 0
    note.write_text("The amount is 90 units.\n")
    capsys.readouterr()
    assert main(["ask", "amount"]) == 0
    found = capsys.readouterr().out
    assert found.startswith("Stale")
    assert "45" not in found
    assert "90" not in found
    with Index(index_path()) as index:
        hits = index.search_question("amount", limit=4)
    assert stale_paths(hits) == ["amount.md"]


def test_a_deleted_file_leaves_no_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "alpha.md").write_text("The alpha pump is ready.\n")
    beta = folder / "beta.md"
    beta.write_text("The beta valve is ready.\n")
    assert main(["ingest", str(folder)]) == 0
    beta.unlink()
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "beta"]) == 0
    assert "beta" not in capsys.readouterr().out
    assert main(["search", "alpha"]) == 0
    assert "alpha" in capsys.readouterr().out


def test_ingesting_one_file_keeps_the_other(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "alpha.md").write_text("The alpha pump is ready.\n")
    (folder / "beta.md").write_text("The beta valve is ready.\n")
    assert main(["ingest", str(folder)]) == 0
    assert main(["ingest", str(folder / "alpha.md")]) == 0
    capsys.readouterr()
    assert main(["search", "beta"]) == 0
    assert "beta" in capsys.readouterr().out


def test_a_superseded_file_is_labelled_and_ranked_after(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "leave_v1.md").write_text("Leave is 18 days.\nSUPERSEDED\n")
    (folder / "leave_v2.md").write_text(
        "Leave is 21 days.\nThis policy supersedes leave_v1.md\n"
    )
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "leave"]) == 0
    found = capsys.readouterr().out
    assert "1. documents/leave_v2.md\n" in found
    assert "2. documents/leave_v1.md [superseded]" in found


def test_different_integers_raise_a_conflict(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "budget_a.md").write_text("The budget is 2400000.\n")
    (folder / "budget_b.md").write_text("The budget is 2600000.\n")
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "budget"]) == 0
    found = capsys.readouterr().out
    assert found.startswith("Conflict")
    assert "2400000" in found
    assert "2600000" in found
    assert "The passages disagree. Show both values." in found


def test_a_named_copy_does_not_replace_the_original(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    text = "The agreement runs from 15 June 2026 to 14 June 2028.\n"
    (folder / "duplicate_vendor_contract_copy.md").write_text(text)
    (folder / "vendor_contract.md").write_text(text)
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "agreement"]) == 0
    found = capsys.readouterr().out
    assert "vendor_contract.md" in found
    assert "duplicate_vendor_contract_copy.md" not in found


def test_a_partial_match_is_kept_when_another_file_has_every_word(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "handbook.md").write_text(
        "The current leave policy has an effective date in the handbook.\n"
    )
    (folder / "policy.md").write_text(
        "Employees accrue 21 days of paid annual leave per year.\n"
    )
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "what is the effective date of the current leave policy?"]) == 0
    found = capsys.readouterr().out
    assert found.index("documents/policy.md") < found.index("documents/handbook.md")


def test_byte_identical_copies_take_one_slot(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    text = "The seal code is SEAL-441.\n"
    (folder / "a.md").write_text(text)
    (folder / "b.md").write_text(text)
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "SEAL-441"]) == 0
    found = capsys.readouterr().out
    assert "documents/a.md" in found
    assert "documents/b.md" not in found
    assert "\n  2. " not in found


def test_docx_heading_list_and_table_are_searchable(tmp_path, monkeypatch, capsys):
    from docx import Document

    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    path = folder / "employee_handbook.docx"
    document = Document()
    document.add_heading("Handbook", level=1)
    document.add_paragraph("The notice period is 30 days.", style="List Bullet")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Role"
    table.cell(0, 1).text = "Analyst"
    document.save(path)
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "Analyst"]) == 0
    found = capsys.readouterr().out
    assert "employee_handbook.docx" in found
    assert "Analyst" in found
    assert "Handbook" in found
    assert "notice period" in found


def test_a_png_scan_is_indexed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    seen = []

    def fake_recognize(scans):
        seen.append(scans)
        return ["Receipt total 1040"]

    monkeypatch.setattr("ldi.ingest.recognize", fake_recognize)
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "receipt.png").write_bytes(b"png")
    assert main(["ingest", str(folder)]) == 0
    assert len(seen) == 1
    assert seen[0][0].page is None
    capsys.readouterr()
    assert main(["search", "1040"]) == 0
    assert "1040" in capsys.readouterr().out


def test_xlsx_calculates_a_formula_and_names_the_cell(tmp_path, monkeypatch, capsys):
    from openpyxl import Workbook

    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    book = Workbook()
    sheet = book.active
    sheet.title = "Headcount"
    sheet["B3"] = 42
    sheet["C1"] = "=B3*2"
    sheet["E1"] = "=NOW()"
    book.save(folder / "headcount.xlsx")
    book.close()
    assert main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    assert main(["search", "84"]) == 0
    found = capsys.readouterr().out
    assert "Headcount!C1 84" in found
    assert "Headcount!B3" in found
    with Index(index_path()) as index:
        reasons = [reason for _path, _page, reason in index.list_gaps()]
    assert "extraction_incomplete Headcount!E1" in reasons


def test_an_old_index_gains_the_new_columns(tmp_path):
    db = tmp_path / "index.sqlite"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE files (
            path TEXT PRIMARY KEY,
            mtime_ns INTEGER NOT NULL,
            bytes INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE chunks USING fts5(
            path UNINDEXED,
            page UNINDEXED,
            body,
            tokenize = 'porter unicode61'
        );
        """
    )
    connection.execute("INSERT INTO files(path, mtime_ns, bytes) VALUES ('notes.md', 1, 2)")
    connection.commit()
    connection.close()
    with Index(db) as index:
        columns = {row[1] for row in index.db.execute("PRAGMA table_info(files)")}
        tables = {row[0] for row in index.db.execute("SELECT name FROM sqlite_master")}
    assert {"sha256", "superseded", "duplicate_of"} <= columns
    assert "gaps" in tables


def test_a_failed_citation_is_retried_once():
    hits = [Hit("notes.md", None, "The replacement seal is part SEAL-441.")]
    calls: list[str] = []

    def generate(reason: str) -> str:
        calls.append(reason)
        if reason == "":
            return '{"status":"answered","evidence":["E1"],"answer":"The total is 450000."}'
        return (
            '{"status":"answered","evidence":["E1"],'
            '"answer":"The replacement seal is part SEAL-441."}'
        )

    _reply, released = settle_answer(generate, hits, "What is the replacement seal?")
    assert calls[0] == ""
    assert calls[1]
    assert "450000" not in calls[1]
    assert released.kind == "answer"
    assert "450000" not in released.text
    assert '- cite: "The replacement seal is part SEAL-441."' in released.text


def test_a_supported_answer_is_not_retried():
    hits = [Hit("notes.md", None, "The replacement seal is part SEAL-441.")]
    calls: list[str] = []

    def generate(reason: str) -> str:
        calls.append(reason)
        return (
            '{"status":"answered","evidence":["E1"],'
            '"answer":"The replacement seal is part SEAL-441."}'
        )

    _reply, released = settle_answer(generate, hits, "What is the replacement seal?")
    assert calls == [""]
    assert released.kind == "answer"
