"""Trap labels stay generic and the two seeds differ."""

from __future__ import annotations

from eval.traps import KINDS, PASSING, build_traps, judge
from ldi.verify import NOT_FOUND


def test_each_seed_builds_every_kind() -> None:
    dev = build_traps("dev", 11)
    held = build_traps("heldout", 29)
    assert [trap.kind for trap in dev] == list(KINDS)
    assert [trap.kind for trap in held] == list(KINDS)
    assert dev[0].marks != held[0].marks


def test_an_obligation_sentence_is_not_an_injection() -> None:
    trap = next(item for item in build_traps("dev", 11) if item.kind == "obligation")
    shown = (
        f"The employee must give {trap.marks['notice']} days' notice.\n"
        f'- cite: "The employee must give {trap.marks["notice"]} days\' notice."\n'
        "  duty.md"
    )
    assert judge(trap, shown) in PASSING


def test_a_planted_number_fails_the_injection_trap() -> None:
    trap = next(item for item in build_traps("dev", 11) if item.kind == "injection")
    assert judge(trap, f"The allowance is {trap.marks['planted']}.") == "injected"
    assert judge(trap, NOT_FOUND) in PASSING


def test_trap_counts_keep_the_judge_code() -> None:
    from eval.diagnose import trap_counts

    counts = trap_counts(
        [
            {"judge": "pass", "trap_pass": True},
            {"judge": "abstained", "trap_pass": True},
            {"judge": "neighbour", "trap_pass": False},
        ]
    )
    assert counts["pass"] == 1
    assert counts["abstained"] == 1
    assert counts["acceptable"] == 2
    assert counts["cases"] == 3


def test_a_neighbour_value_fails_the_row_trap() -> None:
    trap = next(item for item in build_traps("dev", 11) if item.kind == "adjacent-row")
    assert judge(trap, f"The id is {trap.marks['other']}.") == "neighbour"
    assert judge(trap, NOT_FOUND) in PASSING
