#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing as mp

import sys, os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lightllm_api.llm_api import gpt_api_caller, gemini_api_caller, qwen_api_caller




#####################################
# Worker function to get an answer  #
#####################################
def get_answer_for_line(json_line: str) -> dict:
    """
    1) Parse the JSON line.
    2) Extract the user query.
    3) Call GPT API to get an answer.
    4) Store the answer into the record (e.g., under "assistant_answer").
    5) Return the updated record as a dict.
    """
    data = json.loads(json_line.strip())
    
    # Extract the user query
    user_query = data.get("problem", [])
    messages = []
    # Build the messages for the GPT model
    messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )
    
    # Call the GPT-based API
    answer = qwen_api_caller(messages)
    
    # Store the answer into the record
    data["qwq_answer"] = answer
    
    # Return the modified record
    return data

#############################
# Main Script with Pool     #
#############################
def main():
    """
    1. Open an input JSONL file (which has user queries).
    2. For each line, use GPT to answer the user's query.
    3. Write lines (with the new 'assistant_answer') to an output JSONL file.
    4. Multiprocessing is used for faster parallel processing.
    """

    # Input and output file paths
    input_file = "data/MATH-500/test.jsonl"
    output_file = "data/MATH-500/test_qwq_answered.jsonl"

    batch_size = 50
    lines_processed = 0  # Starting point for resuming

    # Open input file for reading and output file for appending
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:

        # Skip already processed lines
        for _ in range(lines_processed):
            f_in.readline()

        while True:
            batch_lines = []
            for _ in range(batch_size):
                line = f_in.readline()
                if not line:
                    break
                batch_lines.append(line)

            if not batch_lines:
                break  # No more lines to process

            with mp.Pool(mp.cpu_count()*8) as pool:
                results = pool.map(get_answer_for_line, batch_lines)

            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

            lines_processed += len(batch_lines)
            print(f"Processed {lines_processed} lines...")

    print(f"Done! outputs written to {output_file}")


if __name__ == "__main__":
    main()