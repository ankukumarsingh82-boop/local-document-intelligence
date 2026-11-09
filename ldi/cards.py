"""Built-in task cards. A document cannot add or edit one."""

from __future__ import annotations

CARD_TOKEN_BUDGET = 192
EXAMPLE_TOKEN_BUDGET = 128
EXAMPLE_CODE = "WIDGET-77"

CARD = """Cited lookup.
Use only the passage in the question.
If the passage states the fact, answer with that fact and quote its words.
If the passage does not state it, say that the file does not contain the answer.
Do not guess. Do not calculate. Do not bring in a name or number that the passage does not contain.
End with a Checks block. Put the quote on a cite line.
Checks:
- cite: "words copied from the passage"
"""

EXAMPLE = """Example passage: The crate tag is WIDGET-77. Two crates arrived.
Example question: What is the crate tag?
Example answer: The crate tag is WIDGET-77.
Checks:
- cite: "WIDGET-77"
"""

TOOL_ADDENDUM = """You may add one tool line. Python runs that line. Do not calculate it yourself.
tool: cite: "words copied from the passage"
"""

CONDITIONS = ("no-card", "card", "card-example", "grammar", "tool")


def system_text(condition: str, base: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}.")
    if condition == "no-card":
        return base
    text = f"{base}\n\n{CARD.strip()}"
    if condition == "card-example":
        text = f"{text}\n\n{EXAMPLE.strip()}"
    elif condition == "tool":
        text = f"{text}\n\n{TOOL_ADDENDUM.strip()}"
    return text


def budget_problem(card_tokens: int, example_tokens: int) -> str | None:
    if card_tokens > CARD_TOKEN_BUDGET:
        return f"The lookup card is {card_tokens} tokens. The budget is {CARD_TOKEN_BUDGET}."
    if example_tokens > EXAMPLE_TOKEN_BUDGET:
        return (
            f"The lookup example is {example_tokens} tokens. "
            f"The budget is {EXAMPLE_TOKEN_BUDGET}."
        )
    return None
