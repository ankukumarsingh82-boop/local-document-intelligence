"""Fixed questions for the skill measurement. These strings are not downloaded."""

from __future__ import annotations

from dataclasses import dataclass


FLOUR_PASSAGE = (
    "The flour bin was refilled on 3 March. The new flour bag code is FLOUR-12.\n"
    "The sugar jar is still the old jar.\n"
    "There is no expiry date written for the rice.\n"
)

# The shape pypdf leaves after a text page: a page label, then the words.
SEAL_PASSAGE = (
    "Page 1\n"
    "The north pump was serviced on 12 March. The replacement seal is part SEAL-441.\n"
    "The south pump is still on the original seal.\n"
)

FEE_PASSAGE = "The fee base is 7,50,000. The rate is 60%.\n"


@dataclass(frozen=True)
class GoldCase:
    key: str
    passage: str
    question: str
    present_span: str | None
    wrong_values: tuple[str, ...]


CASES: tuple[GoldCase, ...] = (
    GoldCase(
        key="flour-code",
        passage=FLOUR_PASSAGE,
        question="What is the flour bag code?",
        present_span="FLOUR-12",
        wrong_values=("SEAL-441", "SEAL-999"),
    ),
    GoldCase(
        key="seal-part",
        passage=SEAL_PASSAGE,
        question="What is the replacement seal part?",
        present_span="SEAL-441",
        wrong_values=("SEAL-999", "FLOUR-12"),
    ),
    GoldCase(
        key="rice-expiry",
        passage=FLOUR_PASSAGE,
        question="What is the expiry date for the rice?",
        present_span="no expiry date",
        wrong_values=("1 January", "30 days"),
    ),
    GoldCase(
        key="warranty",
        passage=SEAL_PASSAGE,
        question="What is the warranty period for the north pump?",
        present_span=None,
        wrong_values=("30 days", "one year", "12 months"),
    ),
    GoldCase(
        key="fee-percent",
        passage=FEE_PASSAGE,
        question="What is 60 percent of the fee base?",
        present_span="60%",
        wrong_values=("500000", "5,00,000"),
    ),
)
