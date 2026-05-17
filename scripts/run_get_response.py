#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, List
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── CHANGE 1: replaced OpenAI proxy with local Qwen via Ollama ──
import sys, os
from causalmath.models.ollama_client import cerebras_query

def query_api(messages, model=None):
    """Use official Qwen3 thinking-mode params: temp=0.6, top_p=0.95, top_k=20."""
    return cerebras_query(messages, model=model)
# ────────────────────────────────────────────────────────────────


prompt_template = """## In-Context Learning Prompt (Sufficient and Necessary Reasoning)

### Instructions

When solving the following questions, your reasoning should:
- **Be Accurate:** Ensure your chain of thought leads to the correct answer without skipping any necessary logical steps.
- **Be Efficient:** Avoid unnecessary or redundant steps. Each step should be necessary to progress toward the solution.
- **Aim for Sufficient and Necessary Reasoning:** Only include steps that are both sufficient to reach the correct answer and necessary to avoid gaps or confusion. If a step can be removed without affecting correctness, remove it.
- **Notice the Pattern:** In the following examples, compare the original, verbose solution with the optimized solution. Learn to identify and eliminate redundant reasoning steps while preserving logical soundness.

---

### Example 1:

**Question:**  
Altitudes \\(\\overline{{AD}}\\) and \\(\\overline{{BE}}\\) of \\(\\triangle ABC\\) intersect at \\(H\\). If \\(\\angle BAC = 54^\\circ\\) and \\(\\angle ABC = 52^\\circ\\), what is \\(\\angle AHB\\)?

**Original, Verbose CoT:**  
To find \\(\\angle AHB\\), we can leverage properties involving altitudes in triangles and the supplementary nature of certain angle pairs.

First, calculate \\(\\angle ACB\\):
\\[
\\angle ACB = 180^\\circ - \\angle BAC - \\angle ABC = 180^\\circ - 54^\\circ - 52^\\circ = 74^\\circ
\\]

In \\(\\triangle ABC\\), the altitudes intersect at the orthocenter \\(H\\), and \\(\\angle AHD\\) and \\(\\angle BHE\\) are right angles.

Consider quadrilateral \\(AHBE\\), which is a cyclic quadrilateral because opposite angles sum to \\(180^\\circ\\).

Using this property:
\\[
\\angle AHB = 180^\\circ - \\angle AEB = 180^\\circ - \\angle ACB = 180^\\circ - 74^\\circ = 106^\\circ
\\]

Final Answer:
\\[
\\boxed{{106^\\circ}}
\\]

**Optimized, Sufficient and Necessary CoT:**  
\\[
\\angle ACB = 180^\\circ - 54^\\circ - 52^\\circ = 74^\\circ
\\]

\\[
\\angle AHB = 180^\\circ - \\angle ACB = 180^\\circ - 74^\\circ = 106^\\circ
\\]

---

### Example 2:

**Question:**  
Simplify: \\(\\frac{{\\sqrt{{2.5^2 - 0.7^2}}}}{{2.7 - 2.5}}\\).

**Original, Verbose CoT:**  
Calculate the denominator:
\\[
2.7 - 2.5 = 0.2
\\]

Simplify the numerator:
\\[
2.5^2 = 6.25
\\]

\\[
0.7^2 = 0.49
\\]

\\[
6.25 - 0.49 = 5.76
\\]

Take the square root:
\\[
\\sqrt{{5.76}} = 2.4
\\]

Substitute into the expression:
\\[
\\frac{{2.4}}{{0.2}} = 12
\\]

Final Answer:
\\[
\\boxed{{12}}
\\]

**Optimized, Sufficient and Necessary CoT:**  
\\[
2.5^2 - 0.7^2 = 6.25 - 0.49 = 5.76
\\]

\\[
\\sqrt{{5.76}} = 2.4
\\]

\\[
2.7 - 2.5 = 0.2
\\]

\\[
\\frac{{2.4}}{{0.2}} = 12
\\]

---

### Instructions Before Final Task

- **Notice** how the optimized solutions eliminate redundant reasoning steps while retaining the necessary calculations to reach the correct answer.
- **Learn** that your goal is to produce reasoning chains that are **sufficient and necessary**:  
  - Every step you include should **either contribute to the final answer or be necessary to ensure the logic is clear**.
  - If a step can **be skipped without compromising correctness or clarity, omit it**.
- **Output concise, accurate, and logically sound reasoning for the following problem directly, with no redundant analysis.**

---

### Now Solve This:

**Question:**  
{question}

**Your Simplified and Optimized Answer:**
"""


#####################################
# Worker function to get an answer  #
#####################################
def get_answer_for_line(record: dict) -> dict:
    """
    1) Receive a Python dict record.
    2) Extract the user query from either 'question' or 'problem' field.
    3) Ensure the output field for the question is 'question' and remove the 'problem' field if it exists.
    4) Call API to get an answer.
    5) Store the answer into the record under 'model_answer'.
    6) Return the updated record.
    """
    # Extract the user query: prefer 'question' if available, otherwise use 'problem'
    user_query = record.get("question") or record.get("problem") or ""

    # Force the output field for the question to be 'question'
    record["question"] = user_query

    # Remove the original 'problem' field to avoid duplicate query fields in the output
    if "problem" in record:
        del record["problem"]

    # Detect dataset type from question content to apply correct validation.
    # CSQA questions contain "Choices:" — all other datasets are math (expect \boxed{}).
    is_csqa = "Choices:" in user_query or "choices:" in user_query

    # ── RESEARCH QUALITY: Use full prompt_template with visible reasoning ──
    # This ensures consistent, high-quality CoT for PN/PS pruning evaluation
    messages = [
        {
            "role": "system",
            "content": (
                "You are Qwen3, a specialized reasoning model for mathematical problem solving.\n"
                "\n"
                "CRITICAL: Think step-by-step and output ALL reasoning steps explicitly.\n"
                "Do not hide reasoning. Make every step visible for evaluation.\n"
                "\n"
                "Quality Requirements:\n"
                "- Sufficient: Include all steps necessary to reach the correct answer\n"
                "- Necessary: Exclude redundant or unnecessary steps\n"
                "- Explicit: Write out all reasoning, do not compress internally\n"
            )
        },
        {
            "role": "user",
            "content": prompt_template.format(question=user_query)
        }
    ]
    # ────────────────────────────────────────────────────────────────────

    # Retry up to 3 times if the response looks invalid (hallucination or truncation).
    # Math datasets require \boxed{}; CSQA requires an Answer: A/B/C/D/E line.
    import re as _re
    answer = ""
    for attempt in range(3):
        answer = query_api(messages)
        if is_csqa:
            valid = bool(_re.search(r"Answer\s*:\s*[A-E]", answer, _re.IGNORECASE))
        else:
            valid = "\\boxed{" in answer
        if valid:
            break
        print(f"  ⚠ Invalid response (attempt {attempt + 1}/3), retrying...")

    # Store the answer into the record under 'model_answer'
    record["model_answer"] = answer

    # Return the modified record
    return record


#############################
# Main Script with Pool     #
#############################

_write_lock = threading.Lock()


def process_all_streaming(records_iter, output_file, workers=4):
    """
    Stream all records through a ThreadPoolExecutor.
    Writes each result as soon as it completes — no batch gap, GPU stays fed.
    """
    records = list(records_iter)
    total = len(records)

    with open(output_file, "a", encoding="utf-8") as f_out, \
         ThreadPoolExecutor(max_workers=workers) as executor:

        futures = {executor.submit(get_answer_for_line, rec): rec
                   for rec in records}
        done = 0
        for fut in as_completed(futures):
            try:
                item = fut.result()
                with _write_lock:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f_out.flush()
                done += 1
                print(f"  ✓ {done}/{total} done", end="\r")
            except Exception as e:
                done += 1
                print(f"\n  ✗ Failed: {e}")


def main():
    """
    Main function to process input file (JSONL or Parquet) and generate output JSONL.
    The input file, output file, and workers are passed as command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input JSONL/Parquet file and output a JSONL file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file (JSONL or Parquet format)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent threads for API calls. Set to 1-2 for deterministic, research-quality results (default: 1)."
    )
    args = parser.parse_args()

    input_file  = args.input_file
    output_file = args.output_file

    # Clear output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == ".jsonl":
        with open(input_file, "r", encoding="utf-8") as f_in:
            records = [json.loads(line.strip()) for line in f_in if line.strip()]
    elif file_ext == ".parquet":
        df = pd.read_parquet(input_file)
        records = df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported file format. Please use .jsonl or .parquet")

    print(f"Processing {len(records)} records with {args.workers} workers...")
    process_all_streaming(iter(records), output_file, workers=args.workers)
    print(f"\nDone! Outputs written to {output_file}")


if __name__ == "__main__":
    main()