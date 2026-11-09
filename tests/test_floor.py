"""The extra gold fills only missing ids, and the floor matches the saved answers."""

from __future__ import annotations

import json

from eval.config import GOLD_PATH
from eval.goldload import EXTRA_SOURCE, extra_rows, synthetic_cases
from eval.rescore import FLOOR_PATH, RAW_PATH, passing_ids, score_saved, pass1_shown


ADDED = {
    "CON-03",
    "DOCX-01",
    "DOCX-02",
    "DOCX-03",
    "DOCX-04",
    "DOCX-05",
    "DOCX-06",
    "DUP-01",
    "STALE-01",
    "STALE-03",
    "SUM-01",
    "SUM-02",
    "UNI-01",
    "VER-01",
    "VER-02",
    "XLSX-01",
    "XLSX-02",
    "XLSX-03",
    "XLSX-04",
}


def test_extra_gold_covers_the_nineteen_and_marks_its_source() -> None:
    rows = extra_rows()
    assert {row["id"] for row in rows} == ADDED
    assert {row["source"] for row in rows} == {EXTRA_SOURCE}
    original = {row["id"] for row in json.loads(GOLD_PATH.read_text())}
    assert original.isdisjoint(ADDED)


def test_an_existing_gold_row_is_not_replaced(tmp_path) -> None:
    cases = tmp_path / "cases.json"
    gold = tmp_path / "gold.json"
    extra = tmp_path / "extra.csv"
    cases.write_text(json.dumps([{"id": "DOCX-01", "category": "DOCX", "q": "leave"}]))
    gold.write_text(
        json.dumps(
            [{"id": "DOCX-01", "tag": "answerable", "anchors": ["from the original gold"]}]
        )
    )
    extra.write_text(
        "id,tag,mode,anchors,files,source\n"
        'DOCX-01,answerable,spans,"[""from the extra file""]","[]",added-step0.5\n'
    )
    loaded = synthetic_cases(cases, gold, extra)
    assert loaded[0].gold.anchors == ("from the original gold",)
    assert loaded[0].gold.mode == "spans"


def test_saved_answers_match_the_frozen_floor() -> None:
    rows = score_saved(pass1_shown(RAW_PATH))
    assert sum(1 for row in rows if row["fabrication"]) == 0
    assert passing_ids(rows) == json.loads(FLOOR_PATH.read_text())


def test_neighbour_traps_are_listed_for_later_steps() -> None:
    payload = json.loads((FLOOR_PATH.parent / "must_fix.json").read_text())
    assert payload["steps"] == [3, 6]
    kinds = {item["kind"]: item["ids"] for item in payload["traps"]}
    assert set(kinds) == {"adjacent-row", "near-name"}
    assert all(item["result"] == "neighbour" for item in payload["traps"])
    assert len(kinds["adjacent-row"]) == 2
    assert len(kinds["near-name"]) == 2
