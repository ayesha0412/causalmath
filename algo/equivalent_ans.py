import re
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qwen_local import cerebras_query

def query_api(messages, model=None, temperature=0.7):
    return cerebras_query(messages, model=model, temperature=temperature, thinking=False)

llm_api = query_api


# ── Fast exact-match helpers ─────────────────────────────────────────────────

def _extract_boxed(text: str) -> str:
    """Pull the content of the last \\boxed{...} in text, handling nested braces."""
    # The simple regex [^}]* fails on nested braces like \boxed{(3, \frac{\pi}{2})}
    # because it stops at the first } (inside \frac). Use a brace-depth counter instead.
    results = []
    start = 0
    while True:
        idx = text.find(r'\boxed{', start)
        if idx == -1:
            break
        depth = 0
        for i in range(idx + 7, len(text)):  # 7 == len(r'\boxed{')
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    results.append(text[idx + 7:i])
                    start = i + 1
                    break
                depth -= 1
        else:
            break  # unclosed brace — skip
    return results[-1].strip() if results else ""


def _normalize(text: str) -> str:
    """Strip LaTeX wrappers and whitespace for lightweight comparison."""
    t = text.strip()
    t = re.sub(r"\\left|\\right|\\,|\\!", "", t)
    t = re.sub(r"\^\\circ|\\circ|\\degree|°", "", t)  # 90^\circ == 90
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)  # \text{Evelyn} → Evelyn
    t = re.sub(r"\\dfrac", r"\\frac", t)   # \dfrac and \frac are identical in value
    t = re.sub(r"\\tfrac", r"\\frac", t)   # same for \tfrac
    t = re.sub(r"\$+", "", t)
    t = re.sub(r"\s+", "", t)
    return t.lower()


def _fast_match(candidate: str, reference: str) -> bool | None:
    """
    Returns True/False if we can decide without an LLM call, else None.
    Tries boxed extraction first, then full-text normalisation.
    """
    c_box = _normalize(_extract_boxed(candidate))
    r_box = _normalize(_extract_boxed(reference))

    # Both have \boxed{} → compare extracted values
    if c_box and r_box:
        return c_box == r_box

    # One side has \boxed{}, other is plain text (common: model wraps answer,
    # ground truth is bare expression like \left(3,\frac{\pi}{2}\right)).
    # Compare the boxed content against the normalised plain text.
    if c_box and not r_box:
        r_norm = _normalize(reference)
        return c_box == r_norm

    if r_box and not c_box:
        c_norm = _normalize(candidate)
        return r_box == c_norm

    # Neither has \boxed{} → full-text normalisation
    c_norm = _normalize(candidate)
    r_norm = _normalize(reference)
    if c_norm and r_norm:
        return c_norm == r_norm

    return None   # can't decide; fall through to LLM


def _llm_yes_no(prompt: str) -> bool:
    answer = llm_api([{"role": "user", "content": prompt}])
    if re.search(r"\byes\b", answer, re.IGNORECASE):
        return True
    if re.search(r"\bno\b", answer, re.IGNORECASE):
        return False
    return False


# ── Public API ───────────────────────────────────────────────────────────────

def is_equivalent_answer(final_answer: str, ground_truth: str) -> bool:
    """Check if two final answers are equivalent (math datasets)."""
    print("ground_truth:", ground_truth, "| final_answer:", final_answer)

    fast = _fast_match(final_answer, ground_truth)
    if fast is not None:
        print("  [fast-match]", fast)
        return fast

    prompt = (
        "You are an intelligent assistant. Determine if the following two answers are equivalent:\n"
        f"Answer: {final_answer}\n"
        f"Reference: {ground_truth}\n"
        "Respond with 'yes' if they agree on the numerical value or expression, 'no' otherwise."
    )
    result = _llm_yes_no(prompt)
    print("  [llm-match]", result)
    return result


def is_equivalent_step(final_answer: str, ground_truth: str) -> bool:
    """Check if two reasoning steps are equivalent in meaning."""
    print("ground_truth:", ground_truth, "| final_answer:", final_answer)

    fast = _fast_match(final_answer, ground_truth)
    if fast is not None:
        return fast

    prompt = (
        "You are an intelligent assistant. Determine if the following two reasoning steps are equivalent in meaning:\n"
        f"Step 1: {final_answer}\n"
        f"Step 2: {ground_truth}\n"
        "Respond with 'yes' if they convey the same logic, 'no' otherwise."
    )
    return _llm_yes_no(prompt)


def is_equivalent_reasoning(statement1: str, statement2: str) -> bool:
    """Check if two commonsense statements contain the same option letter (LLM fallback)."""
    print("Statement 1:", statement1, "| Statement 2:", statement2)
    prompt = (
        "Determine whether the following two statements contain the same option letter:\n"
        f"Statement 1: {statement1}\n"
        f"Statement 2: {statement2}\n"
        "Respond with 'yes' if both contain the same option letter, 'no' otherwise."
    )
    return _llm_yes_no(prompt)


def is_equivalent_reasoning_re(statement1: str, statement2: str) -> bool:
    """Regex-only check for commonsense: compare extracted A-E option letters."""
    option_pattern = r"\b([A-E])\b"
    match1 = re.search(option_pattern, statement1)
    match2 = re.search(option_pattern, statement2)
    if match1 and match2:
        return match1.group(1) == match2.group(1)
    return False
