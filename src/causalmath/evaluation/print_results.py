#!/usr/bin/env python3
"""
Print results table for the fast SFT pipeline.
Reads files named: qwen3_{original,causal,noncausal}_{gsm8k,math500}.jsonl

Usage:
  python sft/print_results_fast.py --results_dir data/results_sft/fast
"""

import argparse
import json
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/results_sft/fast")
    return p.parse_args()


def load_results(path):
    if not os.path.exists(path):
        return None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass
    if not records:
        return None
    correct   = sum(r.get("is_correct", 0) for r in records)
    toks      = [r.get("token_count", 0) for r in records]
    steps     = [r.get("step_count",  0) for r in records]
    return {
        "n":        len(records),
        "acc":      correct / len(records) * 100,
        "avg_tok":  sum(toks)   / len(records),
        "avg_step": sum(steps)  / len(records),
    }


def fmt(r, baseline=None):
    """Format a result cell with change vs baseline."""
    if r is None:
        return f"{'MISSING':>8}   {'--':>6}   {'--':>6}"
    acc_str  = f"{r['acc']:.1f}%"
    tok_str  = f"{r['avg_tok']:.0f}"
    step_str = f"{r['avg_step']:.1f}"
    if baseline:
        da  = r['acc']      - baseline['acc']
        dt  = r['avg_tok']  - baseline['avg_tok']
        ds  = r['avg_step'] - baseline['avg_step']
        acc_str  += f" ({da:+.1f}%)"
        tok_str  += f" ({dt:+.0f})"
        step_str += f" ({ds:+.1f})"
    return f"{tok_str:>16}  {step_str:>8}  {acc_str:>14}"


def main():
    args = get_args()
    d = args.results_dir

    conditions = ["original", "noncausal", "causal"]
    datasets   = [("gsm8k", "GSM8K (1319 standard test)"), ("math500", "MATH-500 (500 standard test)")]

    # Load all results
    results = {}
    for cond in conditions:
        results[cond] = {}
        for ds, _ in datasets:
            path = os.path.join(d, f"qwen3_{cond}_{ds}.jsonl")
            results[cond][ds] = load_results(path)

    print()
    print("=" * 90)
    print("  FAST SFT PIPELINE RESULTS  |  Qwen3-1.7B  |  Causal vs Noncausal")
    print("=" * 90)
    print(f"  Results dir: {d}")
    print("=" * 90)

    for ds, ds_label in datasets:
        base = results["original"][ds]
        print(f"\n  [{ds_label}]")
        print(f"  {'Condition':<12}  {'Tokens (delta)':>16}  {'Steps (delta)':>8}  {'Accuracy (delta)':>14}")
        print(f"  {'-'*12}  {'-'*16}  {'-'*8}  {'-'*14}")

        print(f"  {'Original':<12}  {fmt(base)}")
        for cond in ["noncausal", "causal"]:
            r = results[cond][ds]
            print(f"  {cond:<12}  {fmt(r, base)}")

    print()
    print("=" * 90)
    print("  KEY METRICS (paper alignment: Causal should have fewer tokens/steps)")
    print("=" * 90)

    for ds, ds_label in datasets:
        base  = results["original"][ds]
        causal = results["causal"][ds]
        noncausal = results["noncausal"][ds]
        if not (base and causal and noncausal):
            continue
        print(f"\n  {ds_label}:")
        print(f"    Causal vs Original:    {causal['acc'] - base['acc']:+.1f}% acc,  "
              f"{causal['avg_tok']  - base['avg_tok']:+.0f} tokens,  "
              f"{causal['avg_step'] - base['avg_step']:+.1f} steps")
        print(f"    Causal vs Noncausal:   {causal['acc'] - noncausal['acc']:+.1f}% acc,  "
              f"{causal['avg_tok']  - noncausal['avg_tok']:+.0f} tokens,  "
              f"{causal['avg_step'] - noncausal['avg_step']:+.1f} steps")

    print()
    print("=" * 90)
    print("  FILES USED")
    print("=" * 90)
    for cond in conditions:
        for ds, _ in datasets:
            path = os.path.join(d, f"qwen3_{cond}_{ds}.jsonl")
            exists = "OK" if os.path.exists(path) else "MISSING"
            n = results[cond][ds]["n"] if results[cond][ds] else 0
            print(f"  [{exists:7s}]  {path}  ({n} records)")
    print()


if __name__ == "__main__":
    main()
