#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing as mp
import sys
import os

# Ensure we can import the calculate_ps_pn function (adjust the path as needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# If your code is in a module, e.g. "my_chain_metrics.py":
# from my_chain_metrics import calculate_ps_pn

# Otherwise, if the code with calculate_ps_pn(...) is in the same directory, import directly.
from final.pnps_cot import calculate_ps_pn  # <-- EDIT to match your actual import

##########################################
# Worker function to compute metrics     #
##########################################
def get_metrics_for_line(json_line: str) -> dict:
    """
    1) Parse the JSON line.
    2) Extract 'question' (query), 'qwq' (the chain-of-thought response),
       and 'solution' (ground truth).
    3) Call calculate_ps_pn(...) to compute metrics.
    4) Append the metrics to the record and return it as a dict.
    """
    data = json.loads(json_line.strip())

    # Extract fields from the JSON record
    question = data.get("question", "")
    model_cot = data.get("qwq_answer", "")       # chain-of-thought / response
    ground_truth = data.get("answer", "")

    # Compute metrics (adjust the parameters as you like)
    results = calculate_ps_pn(
        query=question,
        response=model_cot,
        ground_truth=ground_truth,
        threshold=0.3,         # PN threshold
        reasoning_attempts=3,  # forward-passes for evaluation
        do_type=0,             # direct rollouts or prompt intervention
        alter_attempts=3       # how many times to try altering a step
    )

    # Append the metrics to the JSON record
    # e.g., store them under "metrics"
    data["metrics"] = results

    return data


########################################
# Main Script with Pool for batching   #
########################################
def main():
    """
    Reads lines from an input JSONL, computes chain-of-thought metrics,
    and writes the updated lines to an output JSONL file.
    """

    input_file = "data/gsm8k/test_qwq_answered.jsonl"           # <-- EDIT as needed
    output_file = "data/gsm8k/test_qwq_with_algo_results.jsonl"  # <-- EDIT as needed

    batch_size = 50
    lines_processed = 0  # Example: set to 0 to start from the beginning

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:

        # Skip already processed lines if resuming
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

            # Parallel processing of lines in the batch
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(get_metrics_for_line, batch_lines)

            # Write the updated JSON objects to output
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

            lines_processed += len(batch_lines)
            print(f"Processed {lines_processed} lines...")

    print(f"Done! Outputs written to {output_file}")


if __name__ == "__main__":
    main()
