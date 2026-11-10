"""Passage packing leaves the full-text chunks on the old chunker."""

from ldi.chunker import ChunkLimits, merge_overlapping, pack_passages, uncovered_lines
from ldi.cli import main
from ldi.config import index_path
from ldi.spans import PassageRecord, build_spans
from ldi.store import Index

_DIGEST = "d" * 64
_SMALL = ChunkLimits(target=80, hard_max=120, overlap=40, minimum=30, max_tokens=10000)
_COUNTER = lambda _text: 1


def _pack(text, limits=_SMALL, pages=None, path="note.txt"):
    source = pages if pages is not None else [(None, text)]
    spans, _built = build_spans(source, _DIGEST, path=path)
    return pack_passages(
        spans,
        digest=_DIGEST,
        path=path,
        limits=limits,
        count_tokens=_COUNTER,
    )


def _rows(sql: str):
    with Index(index_path()) as index:
        return index.db.execute(sql).fetchall()


def test_no_passage_starts_or_ends_mid_word_and_hard_cuts_are_logged():
    passages = _pack(("alpha " * 400).strip())
    assert len(passages) > 1
    for passage in passages:
        assert passage.text.startswith("alpha")
        assert passage.text.endswith("alpha")
        assert set(passage.text.split()) == {"alpha"}
    for passage in passages[:-1]:
        assert any(log.kind == "hard_cut" for log in passage.logs)


def test_a_sentence_per_line_is_not_a_hard_cut():
    line = "The notice period is thirty days."
    passages = _pack("\n".join([line] * 40), ChunkLimits(200, 260, 40, 30, 10000))
    assert len(passages) > 1
    for passage in passages:
        assert passage.text.endswith(".")
        assert not any(log.kind == "hard_cut" for log in passage.logs)


def test_numbers_ids_and_dates_are_not_split():
    sentence = "The fee is INR 12,34,567 for part SEAL-441 dated 1 April 2026."
    text = ("The office keeps a short note. " * 12) + sentence + (" The file stays open. " * 12)
    passages = _pack(text)
    atoms = ("12,34,567", "SEAL-441", "1 April 2026")
    for atom in atoms:
        assert any(atom in passage.text for passage in passages)
        for passage in passages:
            if atom in passage.text:
                continue
            for size in range(2, len(atom)):
                assert not passage.text.endswith(atom[:size])
                assert not passage.text.startswith(atom[-size:])


def test_a_low_quality_line_is_flagged_and_left_unchanged():
    from ldi.chunker import LOW_QUALITY_RATIO

    raw = "The allowance is Rs 1800 per month.\n_..--,. \nThe next rate is Rs 2250 per month."
    passages = _pack(raw, ChunkLimits(800, 1000, 200, 30, 10000))
    assert any("_..--,." in passage.text for passage in passages)
    logs = [log for passage in passages for log in passage.logs if log.kind == "low_quality"]
    assert len(logs) == 1
    assert logs[0].detail.startswith("0.00 ")
    assert "_..--,." in logs[0].detail
    assert LOW_QUALITY_RATIO == 0.50


def test_every_table_passage_starts_with_its_header():
    text = "\n".join(
        [
            "Amounts in INR crore",
            "| City | Q2 FY26 |",
            "| --- | --- |",
            "| Alpha | 14.2 |",
            "| Total | 14.2 |",
            "| Subtotal | 1 |",
        ]
    )
    passages = _pack(text, ChunkLimits(800, 1000, 200, 30, 10000), path="cities.md")
    tables = [passage for passage in passages if passage.source_type == "table"]
    assert [passage.text for passage in tables] == [
        "City\nQ2 FY26\nAlpha\n14.2",
        "City\nQ2 FY26\nTotal\n14.2",
        "City\nQ2 FY26\nSubtotal\n1",
    ]
    assert all("City:" not in passage.text for passage in tables)
    assert tables[0].aliases == "Alpha | 14.2"
    assert all(passage.label not in passage.text for passage in tables)
    assert all("table " in passage.heading_path for passage in tables)


def test_an_excel_passage_stores_the_sheet_and_row_range(tmp_path, monkeypatch):
    from openpyxl import Workbook

    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    book = Workbook()
    sheet = book.active
    sheet.title = "Fees"
    sheet["A1"] = "Fee INR"
    sheet["B1"] = "Days"
    sheet["A2"] = "Alpha"
    sheet["B2"] = 5
    sheet["B3"] = 9
    sheet.merge_cells("A2:A3")
    book.save(tmp_path / "fees.xlsx")
    book.close()
    assert main(["ingest", str(tmp_path / "fees.xlsx")]) == 0
    rows = _rows(
        """
        SELECT text, sheet, row_start, row_end, label, aliases
        FROM passages WHERE source_type = 'table'
        ORDER BY row_start
        """
    )
    assert [row[0] for row in rows] == [
        "Fee INR\nDays\nAlpha\n5",
        "Fee INR\nDays\nAlpha\n9",
    ]
    assert all("Fee INR:" not in row[0] for row in rows)
    assert all(not row[0].strip().endswith("Fees!B2") for row in rows)
    for text, sheet_name, row_start, row_end, label, aliases in rows:
        assert sheet_name == "Fees"
        assert row_start == row_end
        assert label not in text
        assert f"Fees!B{row_start}" in aliases
        assert f"Fees!B{row_start} " in aliases
    bodies = [body for (body,) in _rows("SELECT body FROM chunks")]
    assert all(label not in body for _text, _sheet, _start, _end, label, _aliases in rows for body in bodies)


def test_reingest_writes_identical_passages(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "note.md"
    note.write_text(
        "1.2 Notice\nThe notice period is 30 days.\n\nSchedule II\nThe seal is SEAL-441.\n"
    )
    assert main(["ingest", str(note)]) == 0
    first = _rows(
        """
        SELECT passage_id, span_ids, text, doc_sha, page_start, page_end, sheet,
               row_start, row_end, heading_path, source_type, passage_hash,
               chunker_version, label
        FROM passages ORDER BY passage_id
        """
    )
    logs = _rows("SELECT passage_id, kind, detail FROM chunk_log ORDER BY passage_id, kind, detail")
    assert first
    assert all(row[12] for row in first)
    assert main(["ingest", str(note)]) == 0
    assert (
        _rows(
            """
            SELECT passage_id, span_ids, text, doc_sha, page_start, page_end, sheet,
                   row_start, row_end, heading_path, source_type, passage_hash,
                   chunker_version, label
            FROM passages ORDER BY passage_id
            """
        )
        == first
    )
    assert _rows(
        "SELECT passage_id, kind, detail FROM chunk_log ORDER BY passage_id, kind, detail"
    ) == logs
    chunks = _rows("SELECT body FROM chunks")
    for row in first:
        label = row[13]
        assert label not in row[2]
        assert all(label not in body for (body,) in chunks)


def test_a_lower_case_line_break_hyphen_is_joined_once_and_logged():
    passages = _pack(
        "Travel is reim-\nbursed after the finance office gives approval.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    assert any("reimbursed" in passage.text for passage in passages)
    assert any(log.kind == "join" for passage in passages for log in passage.logs)
    kept = _pack(
        "Read Annex-\nA before the finance office files the notice.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    assert "AnnexA" not in " ".join(passage.text for passage in kept)
    assert any("Annex-" in passage.text for passage in kept)
    blocked = _pack(
        "Travel is reim-\n\nbursed after the finance office gives approval.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    assert "reimbursed" not in " ".join(passage.text for passage in blocked)
    single = _pack(
        "The aa-\nbb-\ncc value is stored in the office register today.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    rendered = "\n".join(passage.text for passage in single)
    assert "aabbcc" not in rendered
    assert "aabb-" in rendered


def test_a_hyphenated_word_already_in_the_file_stays_split():
    passages = _pack(
        "The well-known rule is filed.\nThe well-\nknown copy is the one the office keeps.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    rendered = "\n".join(passage.text for passage in passages)
    assert "well-known" in rendered
    assert "wellknown" not in rendered


def test_ligatures_soft_hyphens_and_page_ranges():
    passages = _pack(
        "The of\ufb01ce opens after the finance team arrives today.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    assert "office" in passages[0].text
    soft = _pack(
        "The of\u00adfice opens after the finance team arrives today.",
        ChunkLimits(800, 1000, 200, 30, 10000),
    )
    assert "office" in soft[0].text
    across = _pack(
        "",
        ChunkLimits(800, 1000, 200, 30, 10000),
        pages=[
            (1, "The first page explains the notice period for staff."),
            (2, "The second page names the seal part SEAL-441 today."),
        ],
        path="memo.pdf",
    )
    assert across[0].page_start == 1
    assert across[0].page_end == 2
    split = _pack(
        "",
        ChunkLimits(800, 1000, 200, 30, 10000),
        pages=[
            (1, "Travel is reim-"),
            (2, "bursed after the finance office approves the request."),
        ],
    )
    assert "reimbursed" not in " ".join(passage.text for passage in split)


def test_a_running_header_on_most_pages_is_left_out():
    passages = _pack(
        "",
        ChunkLimits(800, 1000, 200, 30, 10000),
        pages=[
            (1, "CONFIDENTIAL\nThe fee on page one is 10 units today."),
            (2, "CONFIDENTIAL\nThe fee on page two is 20 units today."),
        ],
    )
    rendered = "\n".join(passage.text for passage in passages)
    assert "CONFIDENTIAL" not in rendered
    assert "10" in rendered and "20" in rendered


def test_overlap_keeps_the_higher_ranked_passage():
    earlier = PassageRecord(
        passage_id="one",
        span_ids="a b",
        text="alpha\nbeta",
        passage_hash="x",
    )
    later = PassageRecord(
        passage_id="two",
        span_ids="b c",
        text="beta\ngamma",
        passage_hash="y",
    )
    merged = merge_overlapping([earlier, later])
    assert len(merged) == 1
    assert merged[0].passage_id == "one"
    assert "gamma" in merged[0].text
    assert "c" in merged[0].span_ids.split()


def _span_sets(passages):
    return [set(passage.span_ids.split()) for passage in passages if passage.span_ids]


def test_the_final_cut_does_not_repeat_its_spans():
    lines = [
        f"Sentence {number} is a complete line for the harbour watch today."
        for number in range(8)
    ]
    limits = ChunkLimits(1200, 1400, 200, 30, 10000)
    passages = _pack("\n".join(lines), limits)
    assert len(passages) == 1
    longer = "\n".join(lines * 6)
    packed = _pack(longer, limits)
    sets = _span_sets(packed)
    assert len(sets) > 1
    for index, ids in enumerate(sets):
        neighbours = sets[max(0, index - 1) : index] + sets[index + 1 : index + 2]
        for other in neighbours:
            assert not ids <= other


def test_a_sentence_end_needs_whitespace_after_it():
    limits = ChunkLimits(90, 110, 20, 20, 10000)
    prefix = "The harbour office records this rule for the watch. "
    samples = [
        ("12.1", "Clause 12.1 stays in force for the whole season and the next one too."),
        ("v2.0", "Notice v2.0 was posted on the harbour gate before the evening watch began."),
        ("Rs. 480", "The posted fee is Rs. 480 for a single night at the inner berth today."),
        ("No. 4", "No. 4 is the night watch on the outer pier for this whole week."),
        ("i.e.", "A delay, i.e. a late tide, is logged once per watch by the harbour clerk."),
    ]
    for token, sentence in samples:
        passages = _pack(prefix + sentence, limits)
        assert any(token in passage.text for passage in passages), token


def test_a_trailing_clause_number_can_end_the_line():
    text = (
        "The harbour master reads the note for the watch today.\n"
        "Read clause 12.1.\n"
        "The harbour master signs the slip."
    )
    passages = _pack(text, ChunkLimits(70, 90, 10, 20, 10000))
    assert any("12.1." in passage.text for passage in passages)
    for left, right in zip(passages, passages[1:]):
        assert not (left.text.endswith("12") and right.text.startswith(".1"))
        assert not (left.text.endswith("12.") and right.text.startswith("1"))


def test_a_merged_title_row_is_not_the_column_header():
    text = "\n".join(
        [
            "| Harbour dues | Harbour dues |",
            "| Craft | Fee |",
            "| Skiff | 75 |",
        ]
    )
    spans, _built = build_spans([(None, text)], _DIGEST, path="dues.md")
    skiff = next(span for span in spans if span.text == "75")
    assert skiff.header_path.endswith("Fee")
    assert "Harbour dues" not in skiff.header_path
    passages = pack_passages(spans, digest=_DIGEST, path="dues.md")
    tables = [passage for passage in passages if passage.source_type == "table"]
    assert tables
    assert "Craft" in tables[0].text and "Skiff" in tables[0].text and "75" in tables[0].text
    assert "Harbour dues:" not in tables[0].text
    assert "Craft:" not in tables[0].text


def test_a_repeated_grid_is_kept_once_inside_one_version():
    grid = "| Lane | Boats |\n| North | 41 |\n| South | 8 |"
    same = f"Existing board\n{grid}\n\nExisting board\n{grid}\n"
    passages = _pack(same, ChunkLimits(800, 1000, 40, 20, 10000))
    kept = [passage for passage in passages if passage.source_type == "table" and "41" in passage.text]
    assert len(kept) == 1
    assert any(log.kind == "also_at" for log in kept[0].logs)
    both = f"Existing board\n{grid}\n\nRevised board\n{grid}\n"
    revised = _pack(both, ChunkLimits(800, 1000, 40, 20, 10000))
    copies = [passage for passage in revised if passage.source_type == "table" and "41" in passage.text]
    assert len(copies) == 2


def test_every_non_repeated_line_is_covered():
    text = "\n".join(
        [
            "The finance office keeps this notice for the whole season.",
            "Skiff",
            "The harbour master signs the slip after the watch ends.",
        ]
    )
    spans, _built = build_spans([(None, text)], _DIGEST, path="note.txt")
    passages = pack_passages(
        spans,
        digest=_DIGEST,
        path="note.txt",
        limits=ChunkLimits(800, 1000, 40, 30, 10000),
    )
    assert uncovered_lines(spans, passages) == []
    assert any("Skiff" in passage.text for passage in passages)
