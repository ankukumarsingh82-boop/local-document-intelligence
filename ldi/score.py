"""Score replies without asking the model."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ldi.cards import EXAMPLE_CODE
from ldi.gold import GoldCase
from ldi.verify import verify

_REFUSAL = re.compile(
    r"(do(?:es)? not contain|not in the (?:file|passage|document)|"
    r"cannot find|can't find|could not find)",
    re.IGNORECASE,
)


@dataclass
class Score:
    critical: int = 0
    citation_pass: int = 0
    citation_fail: int = 0
    citation_insufficient: int = 0
    wrong_refusal: int = 0
    format_failure: int = 0
    not_run: int = 0

    def add(self, other: Score) -> None:
        self.critical += other.critical
        self.citation_pass += other.citation_pass
        self.citation_fail += other.citation_fail
        self.citation_insufficient += other.citation_insufficient
        self.wrong_refusal += other.wrong_refusal
        self.format_failure += other.format_failure
        self.not_run += other.not_run


def score_case(case: GoldCase, reply: str) -> Score:
    score = Score()
    prepared = fold_tool_lines(reply)
    checks = verify(prepared, case.passage)
    if _format_failed(checks):
        score.format_failure = 1
    kind = _citation(checks)
    if kind == "Pass":
        score.citation_pass = 1
    elif kind == "Fail":
        score.citation_fail = 1
    else:
        score.citation_insufficient = 1
    if _critical(case, reply, checks):
        score.critical = 1
    if _wrong_refusal(case, reply):
        score.wrong_refusal = 1
    return score


def fold_tool_lines(reply: str) -> str:
    """Turn one tool line into a Checks line the verifier already understands."""
    extra: list[str] = []
    for raw in reply.splitlines():
        stripped = raw.strip()
        if not stripped.lower().startswith("tool:"):
            continue
        body = stripped.split(":", 1)[1].strip()
        if not body:
            continue
        extra.append(body if body.startswith("-") else f"- {body}")
    if not extra:
        return reply
    block = "\n".join(extra)
    if re.search(r"(?m)^[ \t]*Checks:[ \t]*$", reply):
        return f"{reply}\n{block}"
    return f"{reply}\nChecks:\n{block}"


def format_score(name: str, score: Score) -> str:
    return (
        f"{name}  critical {score.critical}  "
        f"citation pass {score.citation_pass} fail {score.citation_fail} "
        f"insufficient {score.citation_insufficient}  "
        f"wrong-refusal {score.wrong_refusal}  "
        f"format-failure {score.format_failure}  not-run {score.not_run}"
    )


def _format_failed(checks) -> bool:
    return (
        len(checks) == 1
        and checks[0].kind == "reply"
        and checks[0].status == "Insufficient"
    )


def _citation(checks) -> str:
    cites = [item for item in checks if item.kind == "cite"]
    if any(item.status == "Fail" for item in cites):
        return "Fail"
    if any(item.status == "Pass" for item in cites):
        return "Pass"
    return "Insufficient"


def _critical(case: GoldCase, reply: str, checks) -> bool:
    if EXAMPLE_CODE in reply:
        return True
    if any(value in reply for value in case.wrong_values):
        return True
    return any(item.kind == "calc" and item.status == "Fail" for item in checks)


def _wrong_refusal(case: GoldCase, reply: str) -> bool:
    if not case.present_span or case.present_span not in case.passage:
        return False
    return _REFUSAL.search(reply) is not None
