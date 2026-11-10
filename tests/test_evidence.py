"""Model-free checks for evidence ids, the JSON mask, and code-built citations."""

import re
from decimal import Decimal

from ldi.ask import retry_instruction, settle_answer
from ldi.evidence import (
    Evidence,
    _polarity_conflict,
    build_evidence,
    conflicting,
    evaluate_expression,
    fallback_release,
    format_evidence,
    integer_values,
    release_from_evidence,
)
from ldi.grammar import (
    choose_free,
    choose_structural,
    json_prefix_ok,
    parse_answer_json,
)
from ldi.store import Hit
from ldi.verify import NOT_FOUND, release_answer


def test_current_sentences_become_ids_and_cells_stay_whole():
    hits = [
        Hit("old.md", 1, "The notice period is 45 days.", superseded=True),
        Hit("book.xlsx", None, "Headcount!B3 42\nHeadcount!B4 84"),
        Hit("notes.md", 2, "The pump is ready. The seal is SEAL-441."),
    ]
    items = build_evidence("Where is the seal?", hits)
    assert [item.text for item in items][0] == "The seal is SEAL-441."
    assert all("45" not in item.text for item in items)
    shown = format_evidence(items)
    assert "45" not in shown
    assert "Headcount!B3 42" in shown
    assert "Headcount!B4" in {item.locator.rsplit(", ", 1)[-1] for item in items}
    assert all("\n" not in item.text for item in items)


def test_the_list_prefers_a_question_word_and_stops_at_twelve():
    sentences = " ".join(f"Word{number} is here." for number in range(20))
    items = build_evidence("Where is Word19?", [Hit("a.md", None, sentences)])
    assert len(items) == 12
    assert items[0].id == "E1"
    assert items[0].text == "Word19 is here."


def test_json_prefixes_accept_only_real_ids_and_the_calculator_form():
    allowed = {"E1", "E2"}
    assert json_prefix_ok("", allowed)
    assert not json_prefix_ok("   ", allowed)
    assert json_prefix_ok('{"status":"ans', allowed)
    assert not json_prefix_ok('{"status":"yes"', allowed)
    assert json_prefix_ok(
        '{"status":"answered","evidence":["E1"],"answer":"seal"}',
        allowed,
    )
    finished = '{"status":"not_found","evidence":[],"answer":""}'
    parsed = parse_answer_json(finished, allowed)
    assert parsed is not None
    assert parsed.status == "not_found"
    assert parsed.evidence == ()
    assert not json_prefix_ok(finished + " more", allowed)
    assert (
        parse_answer_json(
            '{"status":"answered","evidence":["E9"],"answer":"x"}',
            allowed,
        )
        is None
    )
    tool = (
        '{"status":"answered","evidence":["E1","E2"],"answer":"",'
        '"calc":{"tool":"calculator","args":{"expr":"E1 + E2"}}}'
    )
    parsed = parse_answer_json(tool, allowed)
    assert parsed is not None
    assert parsed.calc == "E1 + E2"
    assert json_prefix_ok(tool[:40], allowed)
    assert not json_prefix_ok(
        '{"status":"answered","evidence":["E1"],"answer":"","calc":"E1 + 4"}',
        allowed,
    )


def test_the_mask_prefers_a_legal_token():
    allowed = {"E1"}
    assert choose_structural("", [(10.0, "The", 1), (0.2, "{", 2)], allowed) == 2
    text = '{"status":"answered","evidence":["E1"],"answer":"'
    assert choose_free(text, [(9.0, '"oops', 4), (5.0, "seal", 3)], allowed) == 3


def test_the_quote_is_the_evidence_sentence():
    sentence = "The replacement seal is part SEAL-441."
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"The replacement seal is part SEAL-441."}',
        [Hit("notes.md", None, sentence)],
        "What is the replacement seal?",
    )
    assert released.kind == "answer"
    assert f'- cite: "{sentence}"' in released.text
    assert "notes.md" in released.text


def test_a_reversed_negation_is_not_released():
    source = "Unused parental leave is not encashable."
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"Unused parental leave is encashable."}',
        [Hit("policy.md", 1, source)],
        "Can unused parental leave be encashed?",
    )
    assert released.kind == "retry"
    assert released.text == ""
    assert "encashable" not in released.text


def test_a_planted_name_is_not_released():
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"The seal is in Kochi."}',
        [Hit("notes.md", None, "The replacement seal is part SEAL-441.")],
        "What is the replacement seal?",
    )
    assert released.kind == "retry"
    assert "Kochi" not in released.text


def test_another_benefit_in_the_same_policy_is_not_a_conflict():
    hits = [
        Hit("policy.md", 1, "Employees accrue 21 days of paid annual leave per year."),
        Hit("policy.md", 1, "A maximum of 5 unused days may be carried forward."),
        Hit("policy.md", 2, "Primary caregivers receive 26 weeks of paid leave."),
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"Employees accrue 21 days of paid annual leave per year."}',
        hits,
        "How many days of paid annual leave do employees accrue?",
    )
    assert released.kind == "answer"
    assert not released.text.startswith("Conflict")
    assert '- cite: "Employees accrue 21 days of paid annual leave per year."' in released.text


def test_conflict_shows_both_fees_whatever_the_model_picked():
    hits = [
        Hit("vendor_contract.pdf", 1, "Annual fee is INR 2,40,00,000 payable quarterly."),
        Hit("board_note.pdf", 1, "Kestrel Logistics annual fee: INR 2,60,00,000."),
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"The annual fee is 2400000."}',
        hits,
        "What is the annual fee payable to Kestrel Logistics?",
    )
    assert released.kind == "conflict"
    assert released.text.startswith("Conflict")
    assert "2,40,00,000" in released.text
    assert "2,60,00,000" in released.text
    assert "2400000" not in released.text


def test_the_calculator_cites_both_sentences_and_ignores_the_model_sum():
    hits = [
        Hit(
            "finance_report_q3.pdf",
            1,
            "At the start of Q2 the company had 1,180 employees. Net hires during Q2: 64 employees.",
        )
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E1","E2"],"answer":"999",'
        '"calc":{"tool":"calculator","args":{"expr":"E1 + E2"}}}',
        hits,
        "What was headcount at the end of Q2 if net hires are added to the opening headcount?",
    )
    assert released.kind == "answer"
    assert released.text.splitlines()[0] == "1244"
    assert "999" not in released.text
    assert '- cite: "At the start of Q2 the company had 1,180 employees."' in released.text
    assert '- cite: "Net hires during Q2: 64 employees."' in released.text
    assert evaluate_expression("E1 + E2", {"E1": Decimal("1180"), "E2": Decimal("64")}) == Decimal("1244")
    assert evaluate_expression("E1 + 4", {"E1": Decimal("1")}) is None
    assert evaluate_expression(
        "(E1 + E2) * E3",
        {"E1": Decimal(2), "E2": Decimal(3), "E3": Decimal(4)},
    ) == Decimal(20)


def test_a_raw_number_in_calc_is_rejected():
    hits = [Hit("book.xlsx", None, "Headcount!B3 42\nHeadcount!B4 84")]
    released = release_answer(
        '{"status":"answered","evidence":["E1","E2"],"answer":"","calc":"E1 + 4"}',
        hits,
        "What is the headcount if the two cells are added?",
    )
    assert released.kind == "answer"
    assert released.text.splitlines()[0] == "126"
    assert "E1 + 4" not in released.text


def test_an_added_question_sums_the_two_values_without_a_calc_string():
    hits = [Hit("book.xlsx", None, "Headcount!B2 1180\nHeadcount!B3 64")]
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"The headcount was 9999."}',
        hits,
        "What was headcount at the end of Q2 if net hires are added to the opening headcount?",
    )
    assert released.kind == "answer"
    assert released.text.splitlines()[0] == "1244"
    assert "9999" not in released.text
    assert '- cite: "Headcount!B2 1180"' in released.text
    assert '- cite: "Headcount!B3 64"' in released.text


def test_a_document_id_does_not_hide_two_fees():
    hits = [
        Hit("board.pdf", 1, "Board Note BN-2026-11 Kestrel Logistics annual fee: INR 2,60,00,000."),
        Hit("contract.pdf", 1, "Annual fee is INR 2,40,00,000 payable quarterly."),
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"The annual fee is 2600000."}',
        hits,
        "What is the annual fee payable to Kestrel Logistics?",
    )
    assert released.kind == "conflict"
    assert "2,60,00,000" in released.text
    assert "2,40,00,000" in released.text
    abstained = release_answer(
        '{"status":"not_found","evidence":[],"answer":""}',
        hits,
        "What is the annual fee payable to Kestrel Logistics?",
    )
    assert abstained.kind == "retry"
    assert abstained.text == ""

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, settled = settle_answer(
        generate,
        hits,
        "What is the annual fee payable to Kestrel Logistics?",
    )
    assert settled.kind == "conflict"
    assert "2,40,00,000" in settled.text
    assert "2,60,00,000" in settled.text


def test_a_q3_question_is_not_answered_from_q2():
    hits = [Hit("report.pdf", 1, "Cloud revenue in Q2 FY26 was 41.20.")]

    def generate(_reason: str) -> str:
        return '{"status":"answered","evidence":["E1"],"answer":"41.20"}'

    _reply, released = settle_answer(
        generate,
        hits,
        "What was Cloud segment revenue in Q3 FY26?",
    )
    assert released.kind == "not_found"
    assert released.text == NOT_FOUND
    assert "41.20" not in released.text


def test_a_second_failure_shows_one_sentence_or_the_exact_line():
    hits = [Hit("notes.md", None, "The replacement seal is part SEAL-441.")]

    def generate(_reason: str) -> str:
        return '{"status":"answered","evidence":["E1"],"answer":"The total is 450000."}'

    _reply, released = settle_answer(generate, hits, "What is the replacement seal?")
    assert released.kind == "answer"
    assert "450000" not in released.text
    assert "SEAL-441" in released.text
    assert "notes.md" in released.text

    def unrelated(_reason: str) -> str:
        return '{"status":"answered","evidence":["E1"],"answer":"The pump is ready."}'

    _reply, missed = settle_answer(
        unrelated,
        [Hit("notes.md", None, "The pump is ready.")],
        "Where is the warranty?",
    )
    assert missed.kind == "not_found"
    assert missed.text == NOT_FOUND


def test_a_penalty_question_does_not_reuse_the_fee_conflict():
    hits = [
        Hit("contract.pdf", 1, "Clause 7.2 Fees: Annual fee is INR 2,40,00,000 payable quarterly."),
        Hit("invoice.pdf", 1, "Annual service fee billed: INR 2,60,00,000."),
    ]

    def generate(_reason: str) -> str:
        return (
            '{"status":"answered","evidence":["E1"],'
            '"answer":"A penalty of 5% applies after 15 days."}'
        )

    _reply, released = settle_answer(
        generate,
        hits,
        "What penalty applies to late fee payment under the MSA?",
    )
    assert released.kind == "not_found"
    assert released.text == NOT_FOUND
    assert "2,40,00,000" not in released.text
    assert "2,60,00,000" not in released.text


def test_weeks_of_leave_are_not_a_conflict_with_days_of_notice():
    hits = [
        Hit("policy.pdf", 2, "Parental leave: Primary caregivers receive 26 weeks of paid leave."),
        Hit("handbook.docx", None, "Submit the request 45 days before parental leave starts."),
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"26"}',
        hits,
        "How many weeks of paid parental leave do primary caregivers receive?",
    )
    assert released.kind == "answer"
    assert released.text.splitlines()[0] == "26"
    assert "45" not in released.text
    assert "26 weeks of paid leave" in released.text


def test_a_decimal_crore_charge_is_not_a_second_integer():
    items = [
        Evidence("E1", "report.pdf", "report.pdf, page 1", "Table 1: Operating margin for Q2 FY26 was 14.2%."),
        Evidence(
            "E2",
            "report.pdf",
            "report.pdf, page 1",
            "Margin excludes a one-time restructuring charge of INR 3.40 crore.",
        ),
    ]
    assert integer_values(items[1].text) == set()
    assert integer_values("Kestrel Logistics annual fee: INR 2,60,00,000.") == {26000000}
    assert conflicting(
        "Does the Q2 FY26 operating margin include the restructuring charge?",
        items,
    ) is None


TABLE = (
    "Segment revenue (INR crore) Segment Q1 FY26 Q2 FY26 "
    "Cloud 41.20 47.85 Services 22.10 19.60 Hardware 9.75 10.05 Total 73.05 77.50"
)


def test_q2_cloud_revenue_rejects_the_q1_cell():
    hits = [Hit("finance.pdf", 1, TABLE)]
    question = "What was Cloud segment revenue in Q2 FY26?"
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"41.20"}',
        hits,
        question,
    )
    assert released.kind == "answer"
    assert released.text.splitlines()[0] == "47.85"
    assert "Cloud 41.20 47.85" in released.text

    def generate(_reason: str) -> str:
        return '{"status":"answered","evidence":[],"answer":"41.20"}'

    _reply, withheld = settle_answer(generate, hits, question)
    assert withheld.kind == "answer"
    assert withheld.text.splitlines()[0] == "47.85"
    assert "41.20" != withheld.text.splitlines()[0]


def test_a_services_change_is_the_later_column_minus_the_earlier_one():
    hits = [Hit("finance.pdf", 1, TABLE)]
    released = release_answer(
        '{"status":"not_found","evidence":[],"answer":"-22.10"}',
        hits,
        "By how much did Services revenue change from Q1 to Q2 FY26?",
    )
    assert released.kind == "answer"
    assert released.text.splitlines()[0] == "-2.5"
    assert "-22.10" not in released.text
    assert "Services 22.10 19.60" in released.text


def test_an_uncalculated_table_is_not_the_fallback():
    hits = [Hit("finance.pdf", 1, TABLE)]

    def generate(_reason: str) -> str:
        return '{"status":"answered","evidence":["E1"],"answer":"the whole table"}'

    _reply, released = settle_answer(
        generate,
        hits,
        "Which law governs the Kestrel MSA?",
    )
    assert released.kind == "not_found"
    assert released.text == NOT_FOUND
    assert "41.20" not in released.text


def test_c02_shows_both_annual_fees():
    hits = [
        Hit("contract.pdf", 1, "Annual fee is INR 2,40,00,000 payable quarterly."),
        Hit("invoice.pdf", 1, "Orvane pays Kestrel an annual service fee of INR 2,60,00,000."),
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E2"],"answer":"26000000"}',
        hits,
        "How much does Orvane pay Kestrel each year?",
    )
    assert released.kind == "conflict"
    assert "2,40,00,000" in released.text
    assert "2,60,00,000" in released.text


def test_notice_periods_from_different_fields_do_not_conflict():
    hits = [
        Hit("contract.pdf", 1, "Either party may terminate with 90 days written notice."),
        Hit("policy.pdf", 2, "Requests must be submitted 45 days before leave starts."),
    ]
    released = release_answer(
        '{"status":"answered","evidence":["E1"],"answer":"90 days"}',
        hits,
        "What is the termination notice period in the Kestrel MSA?",
    )
    assert released.kind == "answer"
    assert not released.text.startswith("Conflict")
    assert released.text.splitlines()[0] == "90 days"


def test_two_budget_columns_are_not_one_field():
    hits = [
        Hit(
            "book.xlsx",
            None,
            "Budget!A3 People Operations\nBudget!B2 Budget INR\nBudget!B3 1250000\n"
            "Budget!C2 Actual INR\nBudget!C3 1180000",
        )
    ]
    question = "What is the planned budget for People Operations?"
    assert conflicting(question, build_evidence(question, hits)) is None


def test_headcount_and_budget_cells_do_not_conflict():
    hits = [
        Hit(
            "book.xlsx",
            None,
            "Headcount!A3 Net hires Q2\nHeadcount!B1 Employees\nHeadcount!B3 64\n"
            "Budget!A3 Travel\nBudget!B1 Budget INR\nBudget!B3 1250000",
        )
    ]
    question = "How many net hires during Q2?"
    assert conflicting(question, build_evidence(question, hits)) is None


def test_a_kestrel_fee_is_not_the_governing_law_fallback():
    hits = [Hit("board.pdf", 1, "Kestrel Logistics annual fee: INR 2,60,00,000.")]

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, released = settle_answer(
        generate,
        hits,
        "Which law governs the Kestrel MSA?",
    )
    assert released.kind == "not_found"
    assert released.text == NOT_FOUND
    assert "2,60,00,000" not in released.text

    both = [
        Hit("board.pdf", 1, "Kestrel Logistics annual fee: INR 2,60,00,000."),
        Hit(
            "contract.pdf",
            1,
            "Clause 12.1 Governing law: This agreement is governed by the laws of India; "
            "courts at Pune have exclusive jurisdiction.",
        ),
    ]
    _reply, quoted = settle_answer(generate, both, "Which law governs the Kestrel MSA?")
    assert quoted.kind == "answer"
    assert "governed by the laws of India" in quoted.text
    assert "2,60,00,000" not in quoted.text


def test_a_calculated_cell_is_the_citation_for_its_formula():
    hits = [
        Hit(
            "book.xlsx",
            None,
            "Budget!A3 People Operations\nBudget!D2 Variance INR\nBudget!B3 1250000\n"
            "Budget!C3 1180000\nBudget!D3 70000\n"
            "Headcount!A4 Closing Q2\nHeadcount!B2 1180\nHeadcount!B3 64\nHeadcount!B4 1244",
        )
    ]
    variance = "Budget variance D3 formula B3-C3"
    items = build_evidence(variance, hits)
    cell = next(item for item in items if item.text.startswith("Budget!D3"))
    released = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"70000"}' % cell.id,
        items,
        variance,
    )
    assert released.kind == "answer"
    assert "Budget!D3" in released.text
    assert "70000" in released.text
    closing = "Headcount B4 total formula"
    closing_items = build_evidence(closing, hits)
    closing_cell = next(item for item in closing_items if item.text.startswith("Headcount!B4"))
    closed = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"1244"}' % closing_cell.id,
        closing_items,
        closing,
    )
    assert closed.kind == "answer"
    assert "Headcount!B4" in closed.text
    assert "1244" in closed.text


def test_a_quarter_label_is_not_added_to_the_audit_sentence():
    hits = [
        Hit("report.pdf", 1, "Orvane Systems - Q2 FY26 Quarterly Report (unaudited)"),
        Hit("book.xlsx", None, "Headcount!A5 Source quarter\nHeadcount!B5 Q2 FY26"),
    ]
    question = "Is the Q2 FY26 quarterly report audited?"
    items = build_evidence(question, hits)
    released = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"-"}'
        % '","'.join(item.id for item in items),
        items,
        question,
    )
    assert released.kind == "answer"
    assert "unaudited" in released.text
    assert "Headcount!B5" not in released.text


def test_a_role_is_evidence_for_who_holds_it():
    hits = [Hit("report.pdf", 4, "Chief Financial Officer: Arvind Raghunathan.")]
    question = "Who is the Chief Financial Officer?"

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, released = settle_answer(generate, hits, question)
    assert released.kind == "answer"
    assert "Arvind Raghunathan" in released.text


def test_parental_notice_does_not_require_every_question_word():
    hits = [
        Hit(
            "policy.pdf",
            1,
            "Section 3.2 Notice: Requests must be submitted 45 days before leave starts.",
        ),
        Hit("handbook.docx", None, "Submit the request 45 days before parental leave starts."),
        Hit("policy.pdf", 2, "Primary caregivers receive 26 weeks of paid parental leave."),
    ]
    question = "How many days of notice are required for parental leave requests under the current policy?"
    items = build_evidence(question, hits)
    handbook = next(item for item in items if "parental leave starts" in item.text)
    released = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"45 days"}' % handbook.id,
        items,
        question,
    )
    assert released.kind == "answer"
    assert "45" in released.text
    weeks = next(item for item in items if "26 weeks" in item.text)
    rejected = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"26"}' % weeks.id,
        items,
        question,
    )
    assert rejected.kind == "retry"
    assert "26" not in rejected.text


def test_an_explicit_cell_is_not_cited_from_a_pdf():
    hits = [
        Hit("report.pdf", 4, "Net hires during Q2: 64 employees."),
        Hit(
            "book.xlsx",
            None,
            "Headcount!A3 Net hires Q2\nHeadcount!B1 Employees\nHeadcount!B3 64",
        ),
    ]
    question = "How many net hires during Q2? (Headcount!B3)"
    items = build_evidence(question, hits)
    pdf = next(item for item in items if item.text.startswith("Net hires"))
    withheld = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"64"}' % pdf.id,
        items,
        question,
    )
    assert withheld.kind == "retry"
    assert "Net hires during Q2" not in withheld.text

    def generate(_reason: str) -> str:
        return '{"status":"answered","evidence":["%s"],"answer":"64"}' % pdf.id

    _reply, released = settle_answer(generate, hits, question)
    assert released.kind == "answer"
    assert "Headcount!B3" in released.text
    assert "64" in released.text
    assert "Net hires during Q2: 64" not in released.text


def test_the_handbook_passage_is_kept_and_the_title_is_not():
    hits = [
        Hit(
            "employee_handbook.docx",
            None,
            "Leave summary (current policy v2.0) Employees accrue 21 days of paid annual leave per year.",
        ),
        Hit(
            "employee_handbook.docx",
            None,
            "Orvane Systems - Employee Handbook Entirely fictional synthetic test data.",
        ),
    ]
    question = "How many days of annual leave in the employee handbook?"
    items = build_evidence(question, hits)
    leave = next(item for item in items if "21 days" in item.text)

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, released = settle_answer(generate, hits, question)
    assert released.kind == "answer"
    assert "21 days of paid annual leave" in released.text
    assert "Entirely fictional" not in released.text
    chosen = release_from_evidence(
        '{"status":"answered","evidence":["%s"],"answer":"21"}' % leave.id,
        items,
        question,
    )
    assert chosen.kind == "answer"
    assert "21 days of paid annual leave" in chosen.text


def test_an_audit_question_does_not_fall_back_to_a_quarter_cell():
    hits = [
        Hit(
            "finance.pdf",
            1,
            "Orvane Systems - Q2 FY26 Quarterly Report (unaudited) " + TABLE,
        ),
        Hit("book.xlsx", None, "Headcount!A5 Source quarter\nHeadcount!B5 Q2 FY26"),
    ]
    question = "Is the Q2 FY26 quarterly report audited?"

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, released = settle_answer(generate, hits, question)
    assert "Headcount!B5" not in released.text
    assert released.kind == "not_found"


def test_a_page_heading_covers_the_period_on_its_footnote():
    items = [
        Evidence(
            "E1",
            "report.pdf",
            "report.pdf, page 1",
            "Orvane Systems - Q2 FY26 Quarterly Report Operating margin for Q2 FY26 was 14.2%*.",
        ),
        Evidence(
            "E2",
            "report.pdf",
            "report.pdf, page 1",
            "^* Margin excludes a one-time restructuring charge of INR 3.40 crore.",
        ),
    ]
    released = release_from_evidence(
        '{"status":"answered","evidence":["E2"],"answer":"-"}',
        items,
        "Does the Q2 FY26 operating margin include the restructuring charge?",
    )
    assert released.kind == "answer"
    assert "excludes a one-time restructuring charge" in released.text


def test_a_name_and_a_service_are_not_refused():
    hits = [
        Hit(
            "unicode_names.pdf",
            1,
            "Résumé screening vendor: Zoë Ångström-Iyer Consulting LLP.",
        )
    ]

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, named = settle_answer(generate, hits, "Who is Zoë Ångström-Iyer?")
    assert named.kind == "answer"
    assert "Ångström-Iyer" in named.text
    _reply, firm = settle_answer(
        generate,
        hits,
        "Which firm provides résumé screening services?",
    )
    assert firm.kind == "answer"
    assert "Résumé screening vendor" in firm.text


def test_a_superseded_id_is_rejected_and_the_retry_has_no_example():
    from ldi.evidence import Evidence

    item = Evidence("E1", "old.md", "old.md, page 1", "The notice period is 45 days.", superseded=True)
    released = release_from_evidence(
        '{"status":"answered","evidence":["E1"],"answer":"The notice period is 45 days."}',
        [item],
        "What is the notice period?",
    )
    assert released.kind == "retry"
    assert "45" not in released.text
    reason = retry_instruction(released.reason)
    assert not re.search(r"\d", reason)
    assert "exact words from the file" not in reason


def test_a_headingless_document_stays_inside_the_line_cap():
    from types import SimpleNamespace

    from ldi.evidence import attach_source

    spans = [
        SimpleNamespace(
            span_id=f"s{number}",
            table_id="",
            block=0,
            line=number,
            col=0,
            text=f"Line {number} mentions harbour dues today.",
        )
        for number in range(200)
    ]
    hit = Hit(
        "note.md",
        None,
        spans[10].text,
        span_id=spans[10].span_id,
        span_ids=spans[10].span_id,
    )
    attached = attach_source("What are the harbour dues?", [hit], lambda _path: spans)
    items = build_evidence("What are the harbour dues?", attached, limit=12)
    assert len(items) <= 12
    assert any("Line 10" in item.text for item in items)
    assert all(item.text == spans[int(item.text.split()[1])].text for item in items)


def test_pinned_source_lines_survive_the_twelve_line_cap():
    lines = tuple(f"line {number}" for number in range(20))
    ids = tuple(f"s{number}" for number in range(20))
    hit = Hit(
        "board.md",
        None,
        "packed form that must not be quoted",
        span_ids=" ".join(ids),
        source_lines=lines,
        source_span_ids=ids,
        pinned_span_ids=("s17", "s18", "s19"),
    )
    items = build_evidence("berth", [hit], limit=12)
    quoted = [item.text for item in items]
    assert quoted[-3:] == ["line 17", "line 18", "line 19"] or {
        "line 17",
        "line 18",
        "line 19",
    } <= set(quoted)
    assert "packed form that must not be quoted" not in quoted
    assert all(item.text in lines for item in items)
    assert all("span s" in item.locator for item in items)
    assert all(":" not in item.text for item in items)


def _source_hit(path: str, lines: list[str], pinned: int = 0, prefix: str = "s") -> Hit:
    ids = tuple(f"{prefix}{number}" for number in range(len(lines)))
    return Hit(
        path,
        1,
        lines[0],
        span_ids=" ".join(ids),
        source_lines=tuple(lines),
        source_span_ids=ids,
        pinned_span_ids=ids[:pinned],
    )


def test_every_hit_shares_one_evidence_budget():
    from ldi.config import EVIDENCE_BUDGET

    hits = [
        _source_hit(f"doc{doc}.md", [f"Doc {doc} rule {number} applies." for number in range(20)], pinned=20)
        for doc in range(4)
    ]
    items = build_evidence("Which rule applies?", hits)
    assert len(items) <= EVIDENCE_BUDGET
    assert {item.path for item in items} == {f"doc{doc}.md" for doc in range(4)}
    assert all(item.text.endswith(".") for item in items)


def test_a_fallback_hit_keeps_its_best_sentence_beside_pinned_source_hits():
    fallback = Hit("memo.md", None, "The pump is ready. Harbour dues are 3-5 percent of cargo value.")
    crowded = [
        _source_hit(f"board{doc}.md", [f"Board {doc} line {number} is noted." for number in range(30)], pinned=30)
        for doc in range(3)
    ]
    items = build_evidence("What are the harbour dues?", [fallback, *crowded])
    assert any("Harbour dues are 3-5 percent" in item.text for item in items)


def test_a_digit_run_pins_its_line_and_a_word_inside_another_word_does_not():
    from types import SimpleNamespace

    from ldi.evidence import attach_source

    texts = [
        "1.2 Leave rules",
        "This leave was formerly called casual leave.",
        "A request needs 3-5 days of notice.",
        "Payroll closes on the last Friday.",
    ]
    spans = [
        SimpleNamespace(span_id=f"c{number}", table_id="", block=0, line=number, col=0, text=text)
        for number, text in enumerate(texts)
    ]
    hit = Hit("rules.md", None, texts[2], span_id="c2", span_ids="c2")
    numbers = attach_source("Is 3-5 days enough?", [hit], lambda _path: spans)[0]
    assert "c2" in numbers.pinned_span_ids
    form = attach_source("Which form applies?", [hit], lambda _path: spans)[0]
    assert "c1" not in form.pinned_span_ids


def test_a_cut_line_brings_its_continuation():
    lines = [
        "Unrelated opening sentence.",
        "Members may claim harbour dues for each vessel that berths after",
        "midnight on a working day.",
        "Closing sentence.",
    ]
    items = build_evidence("Can members claim harbour dues?", [_source_hit("dues.md", lines)], limit=2)
    assert [item.text for item in items] == lines[1:3]


_LEAD = "For MSc students in every department, absence from all academic activities for more than "


def _fragments() -> list[Evidence]:
    from dataclasses import replace

    from ldi.spans import build_spans

    place = "handbook.pdf, page 3, span "
    items = [
        Evidence("E1", "handbook.pdf", place + "a1", _LEAD + "20 working days (not necessarily"),
        Evidence("E2", "handbook.pdf", place + "a2", "contiguous) will lead to cancellation of registration."),
        Evidence("E3", "handbook.pdf", place + "a3", _LEAD + "30 working days contiguously in a"),
        Evidence("E4", "handbook.pdf", place + "a4", "semester will lead to termination of the programme."),
    ]
    spans, _ = build_spans([(3, "\n".join(item.text for item in items))], "digest")
    return [replace(item, span=span) for item, span in zip(items, spans, strict=True)]


def test_fragments_with_one_lead_in_are_not_a_conflict():
    items = _fragments()
    assert items[0].text[:80] == items[2].text[:80]
    question = "After how many working days of absence is the registration of an MSc student cancelled?"
    assert conflicting(question, items) is None
    released = release_from_evidence(
        '{"status":"answered","evidence":["E1","E2"],"answer":"20 working days"}',
        items,
        question,
    )
    assert released.kind == "answer"
    assert "20 working days" in released.text
    assert "30 working days" not in released.text


def test_a_question_verb_missing_from_the_chosen_line_is_released():
    items = [
        Evidence(
            "E1",
            "guide.pdf",
            "guide.pdf, page 2",
            "The hostel transfer forms need two signed warden letters.",
        )
    ]
    question = "Which hostel transfer forms include warden letters?"
    released = release_from_evidence(
        '{"status":"answered","evidence":["E1"],"answer":"two signed warden letters"}',
        items,
        question,
    )
    assert released.kind == "answer"
    assert '- cite: "The hostel transfer forms need two signed warden letters."' in released.text
    assert fallback_release(question, items).kind == "not_found"


def test_a_negated_word_conflicts_only_when_the_answer_says_it():
    contiguous = [
        Evidence(
            "E1",
            "handbook.pdf",
            "handbook.pdf, page 1",
            "Absence for more than 20 working days (not necessarily contiguous) cancels registration.",
        )
    ]
    assert not _polarity_conflict("Registration is cancelled after 20 working days of absence.", contiguous)
    covered = [Evidence("E1", "policy.md", "policy.md", "Sabbatical is not covered.")]
    assert _polarity_conflict("Sabbatical is covered.", covered)


def test_after_two_abstentions_a_derived_word_is_not_a_match():
    hits = [
        Hit("finance.pdf", 2, "Salary advance requests need approval from the Managing Director."),
    ]

    def generate(_reason: str) -> str:
        return '{"status":"not_found","evidence":[],"answer":""}'

    _reply, released = settle_answer(generate, hits, "Who approved the manager's salary advance?")
    assert released.kind == "not_found"
    assert released.text == NOT_FOUND
    governed = [
        Hit(
            "contract.pdf",
            1,
            "Clause 12.1 Governing law: This agreement is governed by the laws of India.",
        )
    ]
    _reply, quoted = settle_answer(generate, governed, "Which law governs the agreement?")
    assert quoted.kind == "answer"
    assert "governed by the laws of India" in quoted.text
