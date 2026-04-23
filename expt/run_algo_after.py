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

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from algo.pnps_cot import test_original_metrics


def extract_field(data: dict, possible_keys: list) -> str:
    for key in possible_keys:
        if key in data and data[key]:
            return data[key]
    for key, value in data.items():
        for candidate in possible_keys:
            if candidate.lower() in key.lower() and value:
                return value
    return ""


def get_metrics_for_line(json_line: str, prompt_based: bool) -> dict:
    data = json.loads(json_line.strip())
    metric_data = data["metrics"]
    question = extract_field(data, ["question"])
    model_cot = extract_field(metric_data, ["final_chain"])
    # print("model_cot:", model_cot)
    # print("model_cot type:", type(model_cot))
    ground_truth = extract_field(data, ["answer"])
    model_cot = "\n\n".join(model_cot)
    
    # print("model_cot:", model_cot)

    results = test_original_metrics(
        query=question,
        response=model_cot,
        ground_truth=ground_truth,
        reasoning_attempts=3,
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
    parser.add_argument("--append", action="store_true", default=True,
                        help="Append results to the output file instead of overwriting.")
    return parser.parse_args()

def main():
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    lines_processed = args.lines_processed
    prompt_based = args.prompt_based
    append_mode = args.append
    batch_size = args.batch_size if args.batch_size > 0 else 1

    mode = "a" if append_mode else "w"
    error_file = output_file.replace(".jsonl", "_errors.jsonl")
    log_file = output_file.replace(".jsonl", "_log.txt")

    # Initialize log file
    log = open(log_file, "a", encoding="utf-8")
    log.write(f"\n====== Processing started {time.ctime()} ======\n")
    log.write(f"Reading from: {input_file}\nOutput to: {output_file} (mode={mode})\nError log: {error_file}\n\n")

    print(f"{Fore.BLUE}📂 Input file: {input_file}")
    print(f"{Fore.BLUE}📄 Output file: {output_file} ({'append' if append_mode else 'overwrite'})")
    print(f"{Fore.BLUE}⚙️ Starting processing, skipping first {lines_processed} lines, processing {batch_size} lines per batch\n")

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
                    result = get_metrics_for_line(line, prompt_based)
                    print("Writing content: ", result)
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()  # Flush immediately
                    print(f"{Fore.GREEN}✅ Successfully wrote line {line_num}{Style.RESET_ALL}")
                    log.write(f"[OK] Line {line_num} written.\n")
                except Exception as e:
                    f_err.write(line.strip() + "\n")
                    f_err.flush()  # Flush error file immediately if needed
                    print(f"{Fore.RED}❌ Line {line_num} processing failed: {e}{Style.RESET_ALL}")
                    log.write(f"[ERR] Line {line_num} failed: {e}\n")
                
                pbar.update(1)

        pbar.close()

    elapsed = time.time() - start_time
    print(f"\n{Fore.GREEN}🎉 All processing completed! Total {line_num - lines_processed} lines processed, took {elapsed:.2f} seconds.{Style.RESET_ALL}")
    log.write(f"\n✅ All complete, total {line_num - lines_processed} lines, took {elapsed:.2f} seconds.\n")
    log.write(f"====== Processing ended {time.ctime()} ======\n")
    log.close()

if __name__ == "__main__":
    main()