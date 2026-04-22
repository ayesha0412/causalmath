#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from tqdm import tqdm
from colorama import init, Fore, Style
import time

init(autoreset=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# CHANGE 1: Use calculate_ps_pn instead of test_original_metrics
# Original repo used test_original_metrics which only measures PN without pruning.
# calculate_ps_pn does both measurement AND pruning, producing full Table 1 metrics.
from algo.pnps_cot import calculate_ps_pn as test_original_metrics


def extract_field(data: dict, possible_keys: list) -> str:
    for key in possible_keys:
        if key in data and data[key]:
            return data[key]
    for key, value in data.items():
        for candidate in possible_keys:
            if candidate.lower() in key.lower() and value:
                return value
    return ""


def get_metrics_for_line(json_line: str, prompt_based: bool, threshold: float, rollouts: int) -> dict:
    data = json.loads(json_line.strip())

    question = extract_field(data, ["question"])
    model_cot = extract_field(data, ["model_answer"])
    ground_truth = extract_field(data, ["answer"])

    # CHANGE 2: Unescape \\n\\n to real \n\n so parse_nodes splits correctly
    # Without this, the entire CoT is treated as one step
    model_cot = model_cot.replace("\\n\\n", "\n\n")

    results = test_original_metrics(
        query=question,
        response=model_cot,
        ground_truth=ground_truth,
        threshold=threshold,
        reasoning_attempts=rollouts,
        do_type=1 if prompt_based else 0,
        alter_attempts=3
    )

    data["metrics"] = results
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute chain-of-thought metrics from a JSONL file."
    )
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_file", "-o", type=str, required=True,
                        help="Path to the output JSONL file.")
    parser.add_argument("--batch_size", "-b", type=int, default=20,
                        help="Batch size for processing lines. Default is 20.")
    parser.add_argument("--lines_processed", "-l", type=int, default=0,
                        help="Number of lines to skip (already processed). Default is 0.")
    parser.add_argument("--prompt_based", action="store_true",
                        help="Use prompt-based method instead of direct rollouts.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="PN threshold below which a step is pruned. Default 0.5.")
    parser.add_argument("--rollouts", "-k", type=int, default=5,
                        help="Monte-Carlo rollouts per step for PN estimation. Default 5.")
    parser.add_argument("--append", action="store_true", default=True,
                        help="Append results to the output file instead of overwriting.")
    return parser.parse_args()


def main():
    args = parse_args()

    # CHANGE 3: Auto-switch prompt based on dataset
    # math_prompt for GSM-8K, MATH-500, AIME
    # common_prompt for CommonsenseQA
    import algo.pnps_cot as pnps_module
    from algo.prompts import common_prompt, math_prompt
    is_csqa = "commonsenseqa" in args.input_file or "csqa" in args.input_file
    if is_csqa:
        pnps_module.total_prompt = common_prompt
        pnps_module.is_commonsense = True
    else:
        pnps_module.total_prompt = math_prompt
        pnps_module.is_commonsense = False
    print(f"Using prompt: {'common (commonsense)' if is_csqa else 'math'}")

    input_file = args.input_file
    output_file = args.output_file
    lines_processed = args.lines_processed
    prompt_based = args.prompt_based
    threshold = args.threshold
    rollouts = args.rollouts
    append_mode = args.append
    batch_size = args.batch_size if args.batch_size > 0 else 1

    mode = "a" if append_mode else "w"
    error_file = output_file.replace(".jsonl", "_errors.jsonl")
    log_file = output_file.replace(".jsonl", "_log.txt")

    # CHANGE 4: English log messages
    log = open(log_file, "a", encoding="utf-8")
    log.write(f"\n====== Run started {time.ctime()} ======\n")
    log.write(f"Input: {input_file}\nOutput: {output_file} (mode={mode})\nErrors: {error_file}\n\n")

    print(f"{Fore.BLUE}📂 Input file: {input_file}")
    print(f"{Fore.BLUE}📄 Output file: {output_file} ({'append' if append_mode else 'overwrite'})")
    print(f"{Fore.BLUE}⚙️ Starting, skipping {lines_processed} lines, batch size {batch_size}\n")

    start_time = time.time()

    with open(input_file, "r", encoding="utf-8") as f_in, \
        open(output_file, mode, encoding="utf-8") as f_out, \
        open(error_file, "a", encoding="utf-8") as f_err:

        for _ in range(lines_processed):
            f_in.readline()

        line_num = lines_processed
        eof = False

        pbar = tqdm(desc="🚀 Processing", unit="lines", ncols=80)

        while not eof:
            batch = []
            for _ in range(batch_size):
                line = f_in.readline()
                if not line:
                    eof = True
                    break
                batch.append(line)

            for idx, line in enumerate(batch):
                line_num += 1
                try:
                    result = get_metrics_for_line(line, prompt_based, threshold, rollouts)
                    print("Writing content: ", result)
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"{Fore.GREEN}✅ Successfully wrote line {line_num}{Style.RESET_ALL}")
                    log.write(f"[OK] Line {line_num} written.\n")
                    log.flush()
                except Exception as e:
                    f_err.write(line.strip() + "\n")
                    f_err.flush()
                    print(f"{Fore.RED}❌ Line {line_num} failed: {e}{Style.RESET_ALL}")
                    log.write(f"[ERR] Line {line_num} failed: {e}\n")
                    log.flush()

                pbar.update(1)

        pbar.close()

    elapsed = time.time() - start_time
    print(f"\n{Fore.GREEN}🎉 Done! {line_num - lines_processed} lines processed in {elapsed:.2f}s{Style.RESET_ALL}")
    log.write(f"\n✅ All done. {line_num - lines_processed} lines in {elapsed:.2f}s\n")
    log.write(f"====== Run ended {time.ctime()} ======\n")
    log.close()


if __name__ == "__main__":
    main()