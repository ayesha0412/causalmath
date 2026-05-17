#!/usr/bin/env python3
"""
RQ02 – ICL Experiment Runner (Table 2: ICL on Non-Reasoning Models).

Hardware target: RTX 5080 (16 GB VRAM).
Supported local models via Ollama:
  qwen3-14b   → 9.3 GB  (primary)
  qwen3-8b    → 5.2 GB
  llama3.1-8b → 4.9 GB

Each experiment produces two files:
  {dataset}_{method}_{model}.jsonl   — machine-readable results (one JSON per line)
  {dataset}_{method}_{model}.log     — human-readable detailed log for transparency

Usage:
  python expt/run_icl.py \\
      --input   data/MATH-500/test.jsonl \\
      --output  data/results_icl/math500_ours_icl_qwen3-14b.jsonl \\
      --method  ours_icl --dataset math500 --model qwen3-14b
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional


from causalmath.models.ollama_client import cerebras_query as call_model  # TODO: create proper model_client wrapper
# TODO: move icl_prompts.py to src/causalmath/utils/
from causalmath.utils.icl_prompts import (  # noqa: F401 — create icl_prompts.py in utils
    build_prompt,
    load_ours_icl_examples,
)
from causalmath.algorithm.equivalence import is_equivalent_answer

_write_lock = threading.Lock()

# Default PNS-optimized trace files used as Ours-ICL exemplars.
# These are the actual PNS-pruned results from RQ1.
DEFAULT_PNS = {
    "math500": r"data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl",
    "gsm8k":   r"data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl",
}
MAX_TOKENS_DEFAULT = {"math500": 8192, "gsm8k": 1536}


# ── Answer extraction ─────────────────────────────────────────────────────────

def _extract_boxed(text):
    start = text.rfind("\\boxed{")
    if start == -1:
        return ""
    depth, i = 0, start + len("\\boxed{")
    while i < len(text):
        if text[i] == "{":   depth += 1
        elif text[i] == "}":
            if depth == 0:   return text[start + len("\\boxed{"):i]
            depth -= 1
        i += 1
    return ""

def _extract_answer(text, dataset_type):
    if "gsm" in dataset_type.lower():
        m = re.search(r"####\s*([\d,.\-]+)", text)
        return m.group(1).replace(",", "").strip() if m else ""
    return _extract_boxed(text)

def _check_accuracy(response, ground_truth, dataset_type):
    candidate = _extract_answer(response, dataset_type)
    if not candidate:
        return 0, candidate
    return int(is_equivalent_answer(candidate, ground_truth)), candidate

def _count_steps(text):
    return len([p for p in re.split(r"\n\n+", text) if p.strip()])


# ── Logging helpers ───────────────────────────────────────────────────────────

def _write_log_header(log_fh, args, max_tokens, n_examples, total, pns_file=None):
    sep = "=" * 72
    log_fh.write(f"{sep}\n")
    log_fh.write(f"RQ02 ICL EXPERIMENT LOG\n")
    log_fh.write(f"{sep}\n")
    log_fh.write(f"  Method       : {args.method}\n")
    log_fh.write(f"  Model        : {args.model}\n")
    log_fh.write(f"  Dataset      : {args.dataset}\n")
    log_fh.write(f"  Input file   : {args.input}\n")
    log_fh.write(f"  Output file  : {args.output}\n")
    log_fh.write(f"  Max tokens   : {max_tokens}\n")
    log_fh.write(f"  Temperature  : {args.temperature}\n")
    log_fh.write(f"  Total records: {total}\n")
    if args.method == "ours_icl":
        log_fh.write(f"  ICL examples : {n_examples} (static top-reduction from cross-dataset PNS pool)\n")
        log_fh.write(f"  PNS file     : {pns_file}\n")
    log_fh.write(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fh.write(f"{sep}\n\n")
    log_fh.flush()


def _write_log_record(log_fh, idx, total, record, result, examples_used):
    sep = "-" * 72
    correct_str = "YES" if result["is_correct"] else "NO"
    trunc_str   = "  [TRUNCATED - hit token limit]" if result["token_count"] >= MAX_TOKENS_DEFAULT.get(result["dataset"], 8192) else ""

    log_fh.write(f"[{idx:04d}/{total}]  {datetime.now().strftime('%H:%M:%S')}\n")
    log_fh.write(f"Question     : {record.get('question', record.get('problem',''))[:200]}\n")
    log_fh.write(f"Ground truth : {result['ground_truth']}\n")
    log_fh.write(f"Extracted    : {result.get('extracted_answer', 'N/A')}\n")
    log_fh.write(f"Correct      : {correct_str}{trunc_str}\n")
    log_fh.write(f"Tokens       : {result['token_count']}  |  Steps: {result['step_count']}\n")

    if examples_used:
        log_fh.write(f"ICL examples used:\n")
        for i, ex in enumerate(examples_used, 1):
            log_fh.write(f"  [{i}] {ex['question'][:100]}\n")

    # Write model answer (truncated to 600 chars for readability)
    answer_preview = result["model_answer"].replace("\n", " ")[:600]
    log_fh.write(f"Model answer : {answer_preview}{'...' if len(result['model_answer']) > 600 else ''}\n")
    log_fh.write(f"{sep}\n\n")
    log_fh.flush()


def _write_log_summary(log_fh, records_done, correct, avg_tok, avg_step, truncated, start_time):
    sep = "=" * 72
    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    log_fh.write(f"\n{sep}\n")
    log_fh.write(f"SUMMARY\n")
    log_fh.write(f"{sep}\n")
    log_fh.write(f"  Total processed : {records_done}\n")
    log_fh.write(f"  Correct         : {correct}/{records_done} = {correct/max(records_done,1)*100:.2f}%\n")
    log_fh.write(f"  Avg tokens      : {avg_tok:.1f}\n")
    log_fh.write(f"  Avg steps       : {avg_step:.1f}\n")
    log_fh.write(f"  Truncated       : {truncated} responses hit token limit\n")
    log_fh.write(f"  Duration        : {h}h {m}m {s}s\n")
    log_fh.write(f"  Finished        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fh.write(f"{sep}\n")
    log_fh.flush()


# ── Per-record worker ─────────────────────────────────────────────────────────

def process_record(rec, method, dataset_type, model, api_base, api_key,
                   static_examples, n_examples,
                   temperature, max_tokens):
    question     = rec.get("question") or rec.get("problem", "")
    ground_truth = str(rec.get("answer", rec.get("ground_truth", "")))

    examples = static_examples if method == "ours_icl" else None

    messages = build_prompt(question=question, method=method,
                             dataset_type=dataset_type, ours_icl_examples=examples)

    response, api_tokens = call_model(messages=messages, model=model,
                                       api_base=api_base, api_key=api_key,
                                       temperature=temperature, max_tokens=max_tokens,
                                       thinking=False)

    token_count = api_tokens if api_tokens is not None else len(response.split())
    step_count  = _count_steps(response)
    is_correct, extracted = _check_accuracy(response, ground_truth, dataset_type)

    result = {
        "question":         question,
        "ground_truth":     ground_truth,
        "extracted_answer": extracted,
        "model_answer":     response,
        "method":           method,
        "model":            model,
        "dataset":          dataset_type,
        "token_count":      token_count,
        "step_count":       step_count,
        "is_correct":       is_correct,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result, examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RQ02 ICL experiment runner with detailed logging")
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--method",      required=True,
                        choices=["standard_cot","fast_solve","reduction","cod","ours_icl"])
    parser.add_argument("--dataset",     required=True, choices=["gsm8k","math500"])
    parser.add_argument("--model",       default="qwen3-8b")
    parser.add_argument("--api_base",    default=None)
    parser.add_argument("--api_key",     default=None)
    parser.add_argument("--pns_file",    default=None)
    parser.add_argument("--n_examples",  type=int,   default=3)
    parser.add_argument("--workers",     type=int,   default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens",  type=int,   default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    max_tokens = args.max_tokens or MAX_TOKENS_DEFAULT.get(args.dataset, 8192)
    pns_file   = args.pns_file or DEFAULT_PNS.get(args.dataset)
    log_path   = args.output.replace(".jsonl", ".log")

    # Pre-load PNS-optimized examples if a file is given; otherwise
    # build_prompt falls back to hardcoded PNS-style examples per dataset.
    static_examples = None
    if args.method == "ours_icl" and pns_file:
        static_examples = load_ours_icl_examples(pns_file, n=args.n_examples)

    # Resume: skip already-answered questions
    done_questions = set()
    if os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            if line.strip():
                try: done_questions.add(json.loads(line)["question"])
                except: pass

    with open(args.input, encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]
    records = [r for r in all_records
               if (r.get("question") or r.get("problem","")) not in done_questions]
    total = len(all_records)

    print(f"\nMethod: {args.method} | Model: {args.model} | Dataset: {args.dataset} | max_tokens: {max_tokens}")
    print(f"Records: {len(records)} to process ({len(done_questions)} already done)")
    print(f"Log: {log_path}\n")

    start_time    = datetime.now()
    done_count    = 0
    correct_total = 0
    tok_total     = 0
    step_total    = 0
    trunc_total   = 0

    with open(args.output, "a", encoding="utf-8") as f_out, \
         open(log_path, "a", encoding="utf-8") as log_fh, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:

        # Write log header once (only if this is a fresh start)
        if not done_questions:
            _write_log_header(log_fh, args, max_tokens, args.n_examples, total, pns_file)

        futures = {
            pool.submit(process_record,
                        rec, args.method, args.dataset, args.model,
                        args.api_base, args.api_key,
                        static_examples, args.n_examples,
                        args.temperature, max_tokens): rec
            for rec in records
        }

        for fut in as_completed(futures):
            done_count += 1
            try:
                result, examples_used = fut.result()
                correct_total += result["is_correct"]
                tok_total     += result["token_count"]
                step_total    += result["step_count"]
                truncated      = result["token_count"] >= max_tokens
                if truncated:
                    trunc_total += 1

                idx = len(done_questions) + done_count

                with _write_lock:
                    # Write JSONL result
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    # Write detailed log entry
                    _write_log_record(log_fh, idx, total, futures[fut], result, examples_used)

                mark    = "[OK]" if result["is_correct"] else "[X] "
                trunc_s = " [TRUNC]" if truncated else ""
                acc_pct = correct_total / done_count * 100
                print(f"  {mark} {len(done_questions)+done_count:4d}/{total} | "
                      f"acc={acc_pct:.1f}% | tok={result['token_count']} | "
                      f"steps={result['step_count']}{trunc_s}")

            except Exception as e:
                print(f"  [FAIL] Record failed: {e}")
                with _write_lock:
                    log_fh.write(f"[FAIL] {datetime.now().strftime('%H:%M:%S')} Error: {e}\n\n")

    # Summary
    avg_tok  = tok_total  / max(done_count, 1)
    avg_step = step_total / max(done_count, 1)
    final_acc = correct_total / max(done_count, 1) * 100

    with open(log_path, "a", encoding="utf-8") as log_fh:
        _write_log_summary(log_fh, done_count, correct_total, avg_tok, avg_step, trunc_total, start_time)

    print(f"\nFinal: {correct_total}/{done_count} correct  acc={final_acc:.1f}%")
    print(f"Avg tokens={avg_tok:.0f}  Avg steps={avg_step:.1f}  Truncated={trunc_total}")
    print(f"Results -> {args.output}")
    print(f"Log     -> {log_path}")


if __name__ == "__main__":
    main()
