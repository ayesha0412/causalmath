"""
Unit tests for causalmath.utils — prompts and reasoning-regex helpers.
"""

import pytest
from causalmath.utils.prompts import math_prompt, common_prompt


# ── Prompt content ────────────────────────────────────────────────────────────

class TestMathPrompt:
    def test_is_nonempty_string(self):
        assert isinstance(math_prompt, str) and len(math_prompt) > 0

    def test_instructs_final_answer(self):
        assert "final answer" in math_prompt.lower()

    def test_does_not_leak_template_variables(self):
        # Sanity check: no unfilled {placeholders} left in the prompt
        import re
        assert not re.search(r"\{[a-zA-Z_]+\}", math_prompt)


class TestCommonPrompt:
    def test_is_nonempty_string(self):
        assert isinstance(common_prompt, str) and len(common_prompt) > 0

    def test_contains_answer_format(self):
        # Must tell the model to output "Answer: A/B/C/D/E"
        assert "Answer:" in common_prompt

    def test_mentions_valid_options(self):
        assert all(letter in common_prompt for letter in ("A", "B", "C", "D", "E"))

    def test_does_not_leak_template_variables(self):
        import re
        assert not re.search(r"\{[a-zA-Z_]+\}", common_prompt)


# ── Cross-prompt sanity ───────────────────────────────────────────────────────

class TestPromptDistinctness:
    def test_math_and_common_are_different(self):
        assert math_prompt != common_prompt

    def test_both_are_strings(self):
        assert isinstance(math_prompt, str)
        assert isinstance(common_prompt, str)
