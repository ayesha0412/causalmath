#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
import sys
from functools import partial

# Ensure we can import the calculate_ps_pn function (adjust the path as needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from final.pnps_cot import calculate_ps_pn  # Adjust this import if needed


def extract_field(data: dict, possible_keys: list) -> str:
    """
    Attempts to extract a field from the JSON record using a list of possible keys.
    
    1. First, try to find an exact match.
    2. If not found, search keys for a substring match (case insensitive).
    3. Return an empty string if nothing is found.
    """
    # Try exact match
    for key in possible_keys:
        if key in data and data[key]:
            return data[key]
    
    # Fallback: substring matching (case insensitive)
    for key, value in data.items():
        for candidate in possible_keys:
            if candidate.lower() in key.lower() and value:
                return value

    return ""


def get_metrics_for_line(json_line: str, prompt_based: bool) -> dict:
    """
    Parse a JSON line, extract relevant fields (using a list of candidate keys),
    compute metrics with calculate_ps_pn, and append the results to the record.
    
    Parameters:
        json_line (str): A single line from a JSONL file.
        prompt_based (bool): If True, use prompt-based intervention; otherwise, use direct rollouts.
    
    Returns:
        dict: The original record updated with a "metrics" field.
    """
    data = json.loads(json_line.strip())

    # Try different possible keys for each field.
    question = extract_field(data, ["question", "problem"])
    model_cot = extract_field(data, ["gpt4o_answer", "qwq_answer", "qwen_answer"])
    ground_truth = extract_field(data, ["answer", "solution"])

    # Pass the boolean flag to calculate_ps_pn by converting it into the expected type
    results = calculate_ps_pn(
        query=question,
        response=model_cot,
        ground_truth=ground_truth,
        threshold=0.3,           # PN threshold
        reasoning_attempts=3,    # forward-passes for evaluation
        do_type=1 if prompt_based else 0,  # Use prompt-based (1) or direct rollout (0)
        alter_attempts=3         # How many times to try altering a step
    )

    # Append the computed metrics to the data record.
    data["metrics"] = results
    return data


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute chain-of-thought metrics from a JSONL file."
    )
    parser.add_argument(
        "--input_file", "-i",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        required=True,
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=20,
        help="Batch size for processing lines. Default is 20."
    )
    parser.add_argument(
        "--lines_processed", "-l",
        type=int,
        default=0,
        help="Number of lines to skip (already processed). Default is 0."
    )
    parser.add_argument(
        "--prompt_based",
        action="store_true",
        help="If specified, use prompt-based method instead of direct rollouts."
    )
    return parser.parse_args()


def main():
    """
    Main script function.
    
    Reads lines from an input JSONL file, computes chain-of-thought metrics for each
    record using multiprocessing, and writes the updated records to an output JSONL file.
    """
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    batch_size = args.batch_size
    lines_processed = args.lines_processed
    prompt_based = args.prompt_based

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:

        # Skip lines if resuming from a previous run.
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

            # Process the batch in parallel.
            with mp.Pool(mp.cpu_count()) as pool:
                func = partial(get_metrics_for_line, prompt_based=prompt_based)
                results = pool.map(func, batch_lines)

            # Write each updated JSON record as a new line.
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

            lines_processed += len(batch_lines)
            print(f"Processed {lines_processed} lines...")

    print(f"Done! Outputs written to {output_file}")


if __name__ == "__main__":
    main()
