#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, List
import openai
import json
import multiprocessing as mp
import pandas as pd
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from lightllm_api.llm_api import gpt_api_caller, gemini_api_caller, qwen_api_caller

openai.api_key = "sk-R7dHf9Dhzz3slsdPB44aBaE60d6e4e238677C31d9aD70e33"
openai.base_url = 'https://api.shubiaobiao.cn/v1/'

def query_api( 
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 1.0):
    try:
        print('query_api')
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        # print('get message')
        return response.choices[0].message.content.strip()

    except Exception as e:
        return str(e)


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
    4) Call GPT API to get an answer.
    5) Store the answer into the record under 'qwen_answer'.
    6) Return the updated record.
    """
    # Extract the user query: prefer 'question' if available, otherwise use 'problem'
    user_query = record.get("question") or record.get("problem") or ""
    
    # Force the output field for the question to be 'question'
    record["question"] = user_query
    
    # Remove the original 'problem' field to avoid duplicate query fields in the output
    if "problem" in record:
        del record["problem"]
    
    # Build the messages for the GPT model
    # messages = [
    #     {
    #         "role": "system",
    #         "content": (
    #             "You are a helpful assistant good at reasoning. Whenever doing multistep reasoning, "
    #             "please use two newline characters to split multiple steps (\\n\\n). For example:\n\n"
    #             "**Example:**\n"
    #             "User: Solve this problem step by step: A toy costs $10. If I buy 3 toys and get a $5 discount, "
    #             "how much do I pay in total?\n"
    #             "Assistant: First, calculate the total cost of the toys without the discount.\n\n"
    #             "3 toys * $10 per toy = $30\n\n"
    #             "Next, apply the $5 discount to the total cost.\n\n"
    #             "$30 - $5 = $25\n\n"
    #             "So, the total amount to pay is $25."
    #         )
    #     },
    #     {
    #         "role": "user",
    #         "content": user_query
    #     }
    # ]

    # For in-context learning, use the prompt template
    messages = [
        {   
            "role": "system",
            "content": (
                "You are a helpful assistant good at reasoning. Whenever doing multistep reasoning, "
                "please use two newline characters to split multiple steps (\\n\\n)."
            )
        },
        {
            "role": "user",
            # "content": prompt_template.format(question=user_query)   # Incontext learning
            "content": "Please solve the following promblem using concise reasoning, break down each step as simply as possible, dont't involve any unnecessary token or formula: \n" + user_query
        } 
    ]
    
    # Call the GPT-based API (update qwen_api_caller to your actual API function)
    answer = query_api(messages)
    
    # Store the answer into the record under 'qwen_answer'
    record["model_answer"] = answer
    
    # Return the modified record
    return record


#############################
# Main Script with Pool     #
#############################

def process_batch(batch, output_file):
    """
    Given a list of dict `batch`, calls the worker in parallel
    and writes results to output_file (in JSONL format).
    """
    # Adjust pool size to your needs
    with mp.Pool(32) as pool:
        results = pool.map(get_answer_for_line, batch)
    
    # Append the results as JSON lines
    with open(output_file, "a", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    """
    Main function to process input file (JSONL or Parquet) and generate output JSONL.
    The input file, output file, and batch size are passed as command-line arguments.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process an input JSONL/Parquet file and output a JSONL file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        # required=True,
        help="Path to the input file (JSONL or Parquet format)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        # required=True,
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing (default: 50)."
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    batch_size = args.batch_size

    # (Optional) Clear the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Determine file type (JSONL or Parquet) based on extension
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == ".jsonl":
        # For JSONL, parse each line into a dict
        def line_generator():
            with open(input_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    # Strip any leading/trailing whitespace and load the JSON object
                    yield json.loads(line.strip())
        records_iter = line_generator()

    elif file_ext == ".parquet":
        # For Parquet, read into a DataFrame, convert to list-of-dicts
        df = pd.read_parquet(input_file)
        records = df.to_dict(orient="records")
        records_iter = iter(records)

    else:
        raise ValueError("Unsupported file format. Please use .jsonl or .parquet")

    # Process records in batches
    batch = []
    for record in records_iter:
        batch.append(record)
        if len(batch) == batch_size:
            process_batch(batch, output_file)
            batch.clear()

    # Process any remaining records that don't fill an entire batch
    if batch:
        process_batch(batch, output_file)

    print(f"Done! Outputs written to {output_file}")

if __name__ == "__main__":
    main()