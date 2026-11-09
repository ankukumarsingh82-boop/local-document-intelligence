"""Gold anchors and the provenance checker, using made-up sentences."""

from __future__ import annotations

from eval.metrics import GoldView, metric_for
from eval.provenance import fabricated


def _shown(answer: str, quote: str, locator: str = "notes.md, page 1") -> str:
    return f'{answer}\n- cite: "{quote}"\n  {locator}'


def test_an_anchor_inside_the_citation_is_correct() -> None:
    shown = _shown("The fee is 40 units.", "The fee is 40 units.")
    gold = GoldView(abstain=False, anchors=("The fee is 40 units.",), scored=True)
    assert metric_for(gold, shown) == "correct"
    assert fabricated(shown) is False


def test_a_missing_anchor_is_a_false_not_found() -> None:
    gold = GoldView(abstain=False, anchors=("The fee is 40 units.",), scored=True)
    shown = "I could not find supporting evidence in the indexed text"
    assert metric_for(gold, shown) == "false_not_found"


def test_an_abstain_that_answers_is_a_false_accept() -> None:
    gold = GoldView(abstain=True, anchors=(), scored=True)
    shown = _shown("The fee is 40 units.", "The fee is 40 units.")
    assert metric_for(gold, shown) == "false_accept"


def test_a_summary_paraphrase_passes_when_the_citations_hold_the_spans() -> None:
    gold = GoldView(
        abstain=False,
        anchors=("47.85", "Net hires during Q2: 64 employees."),
        scored=True,
        mode="summary",
    )
    shown = (
        "Cloud revenue rose and the company hired.\n"
        '- cite: "Cloud 47.85"\n  report.pdf, page 1\n'
        '- cite: "Net hires during Q2: 64 employees."\n  report.pdf, page 3'
    )
    assert metric_for(gold, shown) == "correct"
    assert metric_for(gold, "I could not find supporting evidence in the indexed text") == "false_not_found"


def test_a_stale_release_passes_only_when_it_says_stale() -> None:
    gold = GoldView(abstain=False, anchors=(), scored=True, mode="stale")
    assert metric_for(gold, "Stale\nThe indexed copy does not match the file on disk: notes.txt") == "correct"
    assert metric_for(gold, "The notice period is 120 days.") == "wrong"


def test_an_unscored_row_stays_unscored() -> None:
    gold = GoldView(abstain=False, anchors=(), scored=False)
    assert metric_for(gold, "anything") == "unscored"


def test_a_number_absent_from_the_citation_is_fabrication() -> None:
    shown = _shown("The fee is 99 units.", "The fee is 40 units.")
    assert fabricated(shown) is True


def test_a_sum_of_cited_numbers_is_traced() -> None:
    shown = (
        "124\n"
        '- cite: "Opening 60"\n  book.xlsx, Headcount!B2\n'
        '- cite: "Hires 64"\n  book.xlsx, Headcount!B3'
    )
    assert fabricated(shown) is False


def test_a_signed_difference_of_cited_numbers_is_traced() -> None:
    shown = '-3\n- cite: "Opening 10. Closing 7."\n  book.md, page 1'
    assert fabricated(shown) is False


def test_evidence_in_hand_is_generated_even_when_the_wide_rank_is_deeper() -> None:
    from eval.diagnose import stage

    row = {
        "metric": "false_not_found",
        "timeout": False,
        "cited_has": False,
        "index_has": True,
        "evidence_has": True,
        "rank": 5,
    }
    assert stage(row) == "generated"


def test_a_name_missing_from_the_citation_is_fabrication() -> None:
    shown = _shown("The owner is Meera Iyengar.", "The owner is listed in the file.")
    assert fabricated(shown) is True
