"""Reciprocal rank fusion uses ranks, and equal scores break deterministically."""

from ldi.retrieve import rrf


def test_the_three_worked_examples_round_to_four_places():
    first = dict(
        rrf(
            [["A"], ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "A"]],
            k=60,
            weights=(1.0, 1.0),
            depth=40,
        )
    )
    second = dict(
        rrf(
            [["p1", "p2", "B"], ["q1", "q2", "B"]],
            k=60,
            weights=(1.0, 1.0),
            depth=40,
        )
    )
    missing = dict(rrf([["C"], []], k=60, weights=(1.0, 1.0), depth=40))
    assert round(first["A"], 4) == 0.0307
    assert round(second["B"], 4) == 0.0317
    assert round(missing["C"], 4) == 0.0164


def test_a_duplicate_inside_one_list_does_not_take_a_second_rank():
    scored = dict(rrf([["A", "A", "B"]], k=60, weights=(1.0,), depth=40))
    assert round(scored["A"], 4) == round(1 / 61, 4)
    assert round(scored["B"], 4) == round(1 / 62, 4)


def test_equal_scores_break_on_lexical_rank_then_span_id():
    by_rank = rrf(
        [["b", "a"], ["a", "b"]],
        k=60,
        weights=(1.0, 1.0),
        depth=40,
        lexical_ranks={"a": 2, "b": 1},
        span_ids={"a": "s0", "b": "s9"},
    )
    assert [item[0] for item in by_rank] == ["b", "a"]
    by_span = rrf(
        [["a", "b"], ["b", "a"]],
        k=60,
        weights=(1.0, 1.0),
        depth=40,
        lexical_ranks={"a": 1, "b": 1},
        span_ids={"a": "s2", "b": "s1"},
    )
    assert [item[0] for item in by_span] == ["b", "a"]
    assert by_span == rrf(
        [["a", "b"], ["b", "a"]],
        k=60,
        weights=(1.0, 1.0),
        depth=40,
        lexical_ranks={"a": 1, "b": 1},
        span_ids={"a": "s2", "b": "s1"},
    )
