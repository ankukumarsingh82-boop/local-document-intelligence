from pathlib import Path

from ldi.cards import (
    CARD_TOKEN_BUDGET,
    EXAMPLE_CODE,
    EXAMPLE_TOKEN_BUDGET,
    budget_problem,
    system_text,
)
from ldi.evalrun import exceeds_ceiling, user_text
from ldi.gold import CASES, FLOUR_PASSAGE, SEAL_PASSAGE
from ldi.grammar import suffix_ok
from ldi.score import fold_tool_lines, format_score, score_case


def _case(key: str):
    return next(item for item in CASES if item.key == key)


def test_gold_covers_the_real_notes_and_a_pdf_page():
    flour = Path("documents/flour-note.md").read_text()
    sample = Path("documents/sample.md").read_text()
    assert "FLOUR-12" in flour
    assert "no expiry date" in flour
    assert "SEAL-441" in sample
    assert "SEAL-441" in SEAL_PASSAGE
    assert "Page 1" in SEAL_PASSAGE
    assert EXAMPLE_CODE not in flour
    assert EXAMPLE_CODE not in sample
    assert {item.key for item in CASES} == {
        "flour-code",
        "seal-part",
        "rice-expiry",
        "warranty",
        "fee-percent",
    }
    assert "7,50,000" in _case("fee-percent").passage
    assert user_text(_case("flour-code")).startswith(FLOUR_PASSAGE.strip())


def test_card_budgets_and_condition_text():
    assert budget_problem(CARD_TOKEN_BUDGET, EXAMPLE_TOKEN_BUDGET) is None
    assert budget_problem(CARD_TOKEN_BUDGET + 1, 1) is not None
    base = "Stay."
    assert system_text("no-card", base) == base
    assert system_text("grammar", base) == system_text("card", base)
    assert EXAMPLE_CODE in system_text("card-example", base)
    assert "tool:" in system_text("tool", base)
    assert "cite:" in system_text("card", base)


def test_scorer_counts_without_a_model():
    flour = _case("flour-code")
    good = score_case(
        flour,
        'The flour bag code is FLOUR-12.\nChecks:\n- cite: "FLOUR-12"',
    )
    assert good.citation_pass == 1
    assert good.critical == 0
    assert good.format_failure == 0
    assert good.wrong_refusal == 0

    missing = score_case(flour, "The flour bag code is FLOUR-12.")
    assert missing.format_failure == 1
    assert missing.citation_insufficient == 1

    refused = score_case(flour, "The file does not contain the answer.")
    assert refused.wrong_refusal == 1

    copied = score_case(flour, f"The code is {EXAMPLE_CODE}.")
    assert copied.critical == 1

    bad = score_case(
        _case("seal-part"),
        'The seal is SEAL-999.\nChecks:\n- cite: "SEAL-999"',
    )
    assert bad.citation_fail == 1
    assert bad.critical == 1

    wrong_math = score_case(
        _case("fee-percent"),
        "Checks:\n- calc: 60% of 750000 = 400000",
    )
    assert wrong_math.critical == 1

    tool = score_case(flour, 'The flour bag code is FLOUR-12.\ntool: cite: "FLOUR-12"')
    assert tool.citation_pass == 1
    assert tool.format_failure == 0
    assert 'cite: "FLOUR-12"' in fold_tool_lines(
        'tool: cite: "FLOUR-12"'
    )


def test_checks_tail_stays_in_the_verifier_shape():
    assert suffix_ok("")
    assert suffix_ok('\n- cite: "FLOUR-12"')
    assert suffix_ok('\n- cite: "FLOUR-12"\n- calc: 2 * 13 = 26\n')
    assert suffix_ok("\n- c")
    assert not suffix_ok("\n- foo")
    assert not suffix_ok("\nThe passage does not say.")


def test_a_long_prompt_is_not_run():
    assert exceeds_ceiling(4096) is False
    assert exceeds_ceiling(4097) is True


def test_eval_command_does_not_download():
    from ldi.cli import _parser

    args = _parser().parse_args(["eval"])
    assert args.command == "eval"
    source = Path("ldi/evalrun.py").read_text()
    assert "snapshot_download" not in source
    assert "logits_processors" in Path("ldi/desk.py").read_text()
    assert "no-card" in format_score("no-card", score_case(_case("warranty"), "No warranty is written."))
