"""The 72-case split is seeded and keeps sealed task ids out."""

from __future__ import annotations

from eval.goldload import synthetic_cases, task1_open_cases
from eval.splits import assign, held_count, read_splits


def test_held_count_keeps_one_id_on_each_side() -> None:
    assert held_count(1) == 0
    assert held_count(2) == 1
    assert held_count(5) == 2


def test_assign_is_stable() -> None:
    pairs = [("b", "lookup"), ("a", "lookup"), ("c", "table")]
    assert assign(pairs, seed=7) == assign(pairs, seed=7)


def test_committed_splits_match_the_suite() -> None:
    cases = synthetic_cases()
    pairs = [(case.id, case.category) for case in cases]
    assigned = assign(pairs)
    assert read_splits() == assigned
    assert set(assigned) == {case.id for case in cases}
    assert set(assigned.values()) == {"dev", "heldout"}


def test_open_task_rows_stop_before_the_seal(tmp_path) -> None:
    path = tmp_path / "gold.csv"
    path.write_text(
        "id,question,expected_answer,source_file,page_clause_cell,type,evidence_quote,verification\n"
        "PQA-001,What is the open fee?,40,open.md,page 1,lookup,The open fee is 40.,exact\n"
        "PQA-027,sealed question,sealed,sealed.md,page 1,lookup,sealed quote,exact\n"
    )
    rows = task1_open_cases(path)
    assert [row.id for row in rows] == ["PQA-001"]
    assert rows[0].gold.anchors == ("The open fee is 40.",)
