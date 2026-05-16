#!/usr/bin/env python3
"""
RQ02 ICL — Windows runner. Runs all 5 methods x chosen models x 2 datasets.

Run from the repo root:
    python expt/run_rq02_icl_all.py
    python expt/run_rq02_icl_all.py --models qwen3-14b          # just one model
    python expt/run_rq02_icl_all.py --datasets math500           # just one dataset
    python expt/run_rq02_icl_all.py --methods ours_icl standard_cot  # subset
"""

import argparse
import subprocess
import sys
import os
import json
import glob
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON = sys.executable   # uses the same python that launched this script

ALL_MODELS  = ["qwen3-14b", "qwen3-8b", "llama3.1-8b"]
ALL_METHODS = ["standard_cot", "fast_solve", "reduction", "cod", "ours_icl"]
ALL_DATASETS = ["math500", "gsm8k"]

# Input test files (raw questions, no model answers)
INPUTS = {
    "math500": "data/MATH-500/test.jsonl",
    "gsm8k":   "data/gsm8k/test.jsonl",
}

# Cross-dataset PNS files: examples drawn from the OPPOSITE dataset to prevent
# data leakage (test questions must not appear in the ICL example pool).
PNS_FILES = {
    "math500": "data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl",
    "gsm8k":   "data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl",
}

OUTPUT_DIR = "data/results_icl"
LOG_FILE        = "expt/rq02_icl_master_log.txt"   # high-level run log
PER_EXPT_LOGS   = "data/results_icl"              # detailed per-experiment .log files
# ─────────────────────────────────────────────────────────────────────────────


def run_one(model, dataset, method, log_fh, workers=1):
    output   = os.path.join(OUTPUT_DIR, f"{dataset}_{method}_{model}.jsonl")
    log_file = output.replace(".jsonl", ".log")
    cmd = [
        PYTHON, "-u", "expt/run_icl.py",
        "--input",   INPUTS[dataset],
        "--output",  output,
        "--method",  method,
        "--dataset", dataset,
        "--model",   model,
        "--workers", str(workers),
    ]
    if method == "ours_icl":
        cmd += ["--n_examples", "3"]

    t_start = datetime.now()
    header  = (f"\n{'='*60}\n"
               f"  Model={model}  Dataset={dataset}  Method={method}\n"
               f"  Output : {output}\n"
               f"  Log    : {log_file}\n"
               f"  Started: {t_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"{'='*60}")
    print(header)
    log_fh.write(header + "\n")
    log_fh.flush()

    result  = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = datetime.now() - t_start
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    status  = "DONE" if result.returncode == 0 else f"FAILED (code={result.returncode})"
    footer  = f"  [{status}]  Duration: {h}h {m}m {s}s  -> see {log_file}\n"
    print(footer)
    log_fh.write(footer + "\n")
    log_fh.flush()


def run_eval(model, dataset, log_fh):
    cmd = [
        PYTHON, "expt/eval_icl.py",
        "--results_dir", OUTPUT_DIR,
        "--model",       model,
        "--dataset",     dataset,
        "--csv",         os.path.join(OUTPUT_DIR, f"{dataset}_{model}_table.csv"),
    ]
    print(f"\n  >> Evaluation table for Model={model} Dataset={dataset}")
    log_fh.write(f"\n[EVAL] {model} {dataset}\n")
    subprocess.run(cmd, text=True)
    log_fh.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Which models to run (default: all 3)")
    parser.add_argument("--methods",  nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Which ICL methods to run (default: all 5)")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS,
                        help="Which datasets to run (default: both)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers per experiment (default: 1, try 2-3 for 8b)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("expt", exist_ok=True)

    total = len(args.models) * len(args.datasets) * len(args.methods)
    print(f"\nRQ02 ICL — {total} experiments")
    print(f"  Models:   {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Methods:  {args.methods}")
    print(f"  Results -> {OUTPUT_DIR}/")
    print(f"  Log     -> {LOG_FILE}")

    with open(LOG_FILE, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n[START] {datetime.now()}\n")
        done = 0
        for model in args.models:
            for dataset in args.datasets:
                for method in args.methods:
                    done += 1
                    print(f"\n[{done}/{total}]", end="")
                    run_one(model, dataset, method, log_fh, workers=args.workers)
                # Print comparison table after all methods for this model+dataset
                run_eval(model, dataset, log_fh)

        # ── Master summary across all completed experiments ──────────────
        log_fh.write(f"\n{'='*60}\nMASTER SUMMARY\n{'='*60}\n")
        result_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl")))
        for rpath in result_files:
            if "pilot" in rpath:
                continue
            try:
                recs    = [json.loads(l) for l in open(rpath, encoding="utf-8") if l.strip()]
                acc     = sum(r["is_correct"] for r in recs) / max(len(recs), 1) * 100
                avg_tok = sum(r["token_count"] for r in recs) / max(len(recs), 1)
                avg_stp = sum(r["step_count"]  for r in recs) / max(len(recs), 1)
                trunc   = sum(1 for r in recs if r["token_count"] >= 8192)
                log_fh.write(
                    f"  {os.path.basename(rpath):50s}  "
                    f"n={len(recs):4d}  acc={acc:.1f}%  "
                    f"tok={avg_tok:.0f}  steps={avg_stp:.1f}  trunc={trunc}\n"
                )
            except Exception as e:
                log_fh.write(f"  {os.path.basename(rpath)}: could not read ({e})\n")
        log_fh.write(f"\n[DONE] {datetime.now()}\n")

    print(f"\nAll done.")
    print(f"Master log -> {LOG_FILE}")
    print(f"Per-experiment logs -> {OUTPUT_DIR}/*.log")
    print(f"Tables -> {OUTPUT_DIR}/*_table.csv")


if __name__ == "__main__":
    main()
