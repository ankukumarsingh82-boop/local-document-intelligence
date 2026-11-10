"""Question frames stay quoted, and intent words stay off the match list."""

from datetime import date

from ldi.query import fold_for_fts, or_query, parse_question, quote_token


def test_operator_words_are_quoted_literals():
    frame = parse_question("alpha and or near beta")
    query = or_query(frame.required)
    assert '"and"' in query
    assert '"or"' in query
    assert '"near"' in query
    assert "NEAR" not in query
    assert quote_token('a"b') == '"a""b"'


def test_one_character_tokens_that_carry_meaning_are_kept():
    frame = parse_question("Annexure I sets Clause 4 at 5%")
    assert "i" in frame.required
    assert "4" in frame.required
    assert "5" in frame.required
    assert "annexure i" in frame.designators
    assert "clause 4" in frame.designators
    assert "a" not in parse_question("a seal is ready").required


def test_numbers_fold_without_splitting_a_decimal():
    folded = fold_for_fts("INR 12,34,567 and the rate is 18.4 after approval.")
    parts = folded.split()
    assert "1234567" in parts
    assert "18.4" in parts
    assert "184" not in parts
    assert "inr" in parts
    assert "approval" in parts
    assert "approval." not in parts
    assert fold_for_fts("2 lakh or 2 lac, and 3 cr or ₹9") == "2 lakh or 2 lakh and 3 crore or inr 9"


def test_intent_words_and_qualifiers_leave_the_match_list():
    current = parse_question("What is the current leave policy?", today=date(2026, 11, 9))
    assert current.intent == "current"
    assert current.as_of == "2026-11-09"
    assert "current" not in current.required
    earlier = parse_question("What was the earlier rate?")
    assert earlier.intent == "historical"
    assert earlier.as_of == ""
    assert "earlier" not in earlier.required
    revised = parse_question("Which revised rate applies?", today=date(2026, 11, 9))
    assert revised.intent == "revised"
    assert revised.as_of == "2026-11-09"
    negated = parse_question("Is the fee not above the cap, subject to the note?")
    assert "not" in negated.negations
    assert "not" not in negated.required
    assert "subject to" in negated.qualifiers
    assert "subject" not in negated.required


def test_the_cap_keeps_rare_tokens_and_every_id():
    words = [f"term{chr(ord('a') + index)}" for index in range(25)]
    question = " ".join(words) + " AB-99999"
    frequencies = {word: index for index, word in enumerate(words)}
    frequencies["ab-99999"] = 1000
    frame = parse_question(question, frequencies=frequencies, passage_count=1000)
    assert "ab-99999" in frame.ids
    assert "ab-99999" in frame.required
    assert len(frame.required) == 20
    assert "terma" in frame.required
    assert "termy" not in frame.required


def test_aliases_expand_the_or_list_only():
    frame = parse_question(
        "How much do they pay each year under NPS rules?",
        alias_rows=(("nps", "national pension scheme"),),
    )
    assert "annual" in frame.aliases
    assert "fee" in frame.aliases
    assert "national" in frame.aliases
    assert "annual" not in frame.required
    assert "fee" not in frame.required
    assert "national" not in frame.required


def test_a_near_name_is_classified_and_not_rewritten():
    frame = parse_question("file of Ravi Kumari", indexed_names=("ravi kumar",))
    assert "ravi kumari" in frame.names
    assert "ravi kumar" not in frame.names
