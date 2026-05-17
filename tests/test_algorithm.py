"""
Unit tests for causalmath.algorithm — equivalence helpers and CoT parsing.

All tested functions are pure (no network calls, no LLM), so these tests
run offline and complete in milliseconds.
"""

import pytest
from causalmath.algorithm.equivalence import (
    _extract_boxed,
    _normalize,
    _fast_match,
    is_equivalent_reasoning_re,
)
from causalmath.algorithm.pns_cot import parse_nodes


# ── _extract_boxed ────────────────────────────────────────────────────────────

class TestExtractBoxed:
    def test_simple_integer(self):
        assert _extract_boxed(r"\boxed{42}") == "42"

    def test_simple_fraction(self):
        assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_nested_braces(self):
        # Nested braces like \boxed{(3, \frac{\pi}{2})} must not stop at first }
        result = _extract_boxed(r"\boxed{(3, \frac{\pi}{2})}")
        assert result == r"(3, \frac{\pi}{2})"

    def test_returns_last_boxed(self):
        # Multiple \boxed{} in a response — last one is the final answer
        text = r"So step 1 gives \boxed{3} and step 2 gives \boxed{72}."
        assert _extract_boxed(text) == "72"

    def test_no_boxed_returns_empty(self):
        assert _extract_boxed("The answer is 42.") == ""

    def test_unclosed_brace_returns_empty(self):
        assert _extract_boxed(r"\boxed{unclosed") == ""

    def test_whitespace_stripped(self):
        assert _extract_boxed(r"\boxed{  7  }") == "7"


# ── _normalize ────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_strips_dollar_signs(self):
        assert _normalize("$42$") == "42"

    def test_strips_degree_symbol(self):
        assert _normalize(r"90^\circ") == "90"
        assert _normalize("90°") == "90"

    def test_normalizes_dfrac_to_frac(self):
        result = _normalize(r"\dfrac{1}{2}")
        assert result == r"\frac{1}{2}"

    def test_strips_text_wrapper(self):
        assert _normalize(r"\text{Evelyn}") == "evelyn"

    def test_strips_left_right(self):
        result = _normalize(r"\left(3\right)")
        assert result == "(3)"

    def test_lowercases(self):
        assert _normalize("ABC") == "abc"

    def test_strips_whitespace(self):
        assert _normalize("  3  +  4  ") == "3+4"


# ── _fast_match ───────────────────────────────────────────────────────────────

class TestFastMatch:
    def test_both_boxed_equal(self):
        assert _fast_match(r"\boxed{42}", r"\boxed{42}") is True

    def test_both_boxed_not_equal(self):
        assert _fast_match(r"\boxed{42}", r"\boxed{7}") is False

    def test_boxed_vs_plain_equal(self):
        # Candidate has \boxed{}, reference is bare text
        assert _fast_match(r"\boxed{42}", "42") is True

    def test_boxed_vs_plain_not_equal(self):
        assert _fast_match(r"\boxed{42}", "7") is False

    def test_plain_vs_boxed_equal(self):
        assert _fast_match("42", r"\boxed{42}") is True

    def test_neither_boxed_equal(self):
        assert _fast_match("the answer is 72", "the answer is 72") is True

    def test_neither_boxed_not_equal(self):
        assert _fast_match("the answer is 72", "the answer is 10") is False

    def test_both_empty_returns_none(self):
        # Neither side has boxed or content to compare — must fall through to LLM
        assert _fast_match("", "") is None

    def test_dfrac_vs_frac_equal(self):
        # After normalisation \dfrac{1}{2} == \frac{1}{2}
        assert _fast_match(r"\boxed{\dfrac{1}{2}}", r"\boxed{\frac{1}{2}}") is True

    def test_degree_equivalence(self):
        assert _fast_match(r"\boxed{90^\circ}", r"\boxed{90}") is True


# ── is_equivalent_reasoning_re ────────────────────────────────────────────────

class TestIsEquivalentReasoningRe:
    def test_same_letter_match(self):
        assert is_equivalent_reasoning_re("The answer is B", "I think B is correct") is True

    def test_different_letters_no_match(self):
        assert is_equivalent_reasoning_re("Answer: A", "Answer: C") is False

    def test_no_letter_returns_false(self):
        assert is_equivalent_reasoning_re("no option here", "also none") is False

    def test_case_insensitive_not_required(self):
        # The regex uses \b([A-E])\b — uppercase only; lowercase letters should not match
        assert is_equivalent_reasoning_re("answer is b", "answer is b") is False

    def test_first_letter_wins(self):
        # re.search finds first match; both statements have A first → True
        assert is_equivalent_reasoning_re("A then B", "A or C") is True


# ── parse_nodes ───────────────────────────────────────────────────────────────

class TestParseNodes:
    def test_splits_on_double_newline(self):
        cot = "Step one.\n\nStep two.\n\nStep three."
        nodes = parse_nodes(cot)
        assert nodes == ["Step one.", "Step two.", "Step three."]

    def test_strips_whitespace_from_nodes(self):
        cot = "  Step one.  \n\n  Step two.  "
        nodes = parse_nodes(cot)
        assert nodes == ["Step one.", "Step two."]

    def test_empty_string_returns_empty_list(self):
        assert parse_nodes("") == []

    def test_no_double_newline_returns_single_node(self):
        assert parse_nodes("Only one step here.") == ["Only one step here."]

    def test_filters_blank_paragraphs(self):
        # Three consecutive newlines produce an empty paragraph — should be dropped
        cot = "Step one.\n\n\n\nStep two."
        nodes = parse_nodes(cot)
        assert nodes == ["Step one.", "Step two."]

    def test_single_newline_not_split(self):
        cot = "Line A.\nLine B."
        nodes = parse_nodes(cot)
        assert nodes == ["Line A.\nLine B."]
