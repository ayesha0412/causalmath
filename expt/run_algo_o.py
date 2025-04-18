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

    question = extract_field(data, ["question"])
    model_cot = extract_field(data, ["model_answer"])
    ground_truth = extract_field(data, ["answer"])
    model_cot = "\n\n".join(model_cot)

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

    # 日志文件初始化
    log = open(log_file, "a", encoding="utf-8")
    log.write(f"\n====== 处理开始 {time.ctime()} ======\n")
    log.write(f"读取自：{input_file}\n输出至：{output_file}（mode={mode}）\n错误记录：{error_file}\n\n")

    print(f"{Fore.BLUE}📂 输入文件：{input_file}")
    print(f"{Fore.BLUE}📄 输出文件：{output_file}（{'追加' if append_mode else '覆盖'}）")
    print(f"{Fore.BLUE}⚙️ 开始处理，跳过前 {lines_processed} 行，每批处理 {batch_size} 行\n")

    start_time = time.time()

    with open(input_file, "r", encoding="utf-8") as f_in, \
        open(output_file, mode, encoding="utf-8") as f_out, \
        open(error_file, "a", encoding="utf-8") as f_err:

        for _ in range(lines_processed):
            f_in.readline()

        line_num = lines_processed
        eof = False

        pbar = tqdm(desc="🚀 正在处理", unit="行", ncols=80)

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
                    print("写入内容为：", result)
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()  # 添加这一行立即刷新
                    print(f"{Fore.GREEN}✅ 成功写入第 {line_num} 行{Style.RESET_ALL}")
                    log.write(f"[OK] Line {line_num} written.\n")
                except Exception as e:
                    f_err.write(line.strip() + "\n")
                    f_err.flush()  # 如果需要错误文件也立即刷新，可以加这里
                    print(f"{Fore.RED}❌ 第 {line_num} 行处理失败：{e}{Style.RESET_ALL}")
                    log.write(f"[ERR] Line {line_num} failed: {e}\n")
                
                pbar.update(1)

        pbar.close()

    elapsed = time.time() - start_time
    print(f"\n{Fore.GREEN}🎉 全部处理完成！共处理 {line_num - lines_processed} 行，耗时 {elapsed:.2f} 秒。{Style.RESET_ALL}")
    log.write(f"\n✅ 全部完成，共 {line_num - lines_processed} 行，耗时 {elapsed:.2f} 秒。\n")
    log.write(f"====== 处理结束 {time.ctime()} ======\n")
    log.close()

if __name__ == "__main__":
    main()