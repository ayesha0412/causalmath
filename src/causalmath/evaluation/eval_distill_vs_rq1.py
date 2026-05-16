#!/usr/bin/env python3
"""
Evaluate self-distillation iterations and compare with RQ1 baselines.

RQ1 asks: Does causal CoT (PNS-pruned) achieve competitive accuracy
with fewer tokens/steps vs standard CoT?

This script compares:
  - RQ1 baselines: causal SFT 1.7B, noncausal SFT 1.7B, original 1.7B
  - Self-distillation: 14B iter0 (base), 14B iter1, 14B iter2 ...

Metrics: accuracy, avg_tokens, step_count, token compression ratio

Usage:
  python sft/eval_distill_vs_rq1.py
  python sft/eval_distill_vs_rq1.py --extra models/distill/qwen3_14b_iter2
"""
import os, sys, json, argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--extra", nargs="*", default=[],
                   help="Additional result files to include in comparison")
    return p.parse_args()


# ── Known result files ────────────────────────────────────────────────────────
BASELINES = {
    "Original 1.7B (base)":     "data/results_sft/fast/qwen3_original_gsm8k.jsonl",
    "Noncausal SFT 1.7B":       "data/results_sft/fast/qwen3_noncausal_gsm8k.jsonl",
    "Causal SFT 1.7B  [RQ1]":  "data/results_sft/fast/qwen3_causal_gsm8k.jsonl",
    "Causal DPO 1.7B":          "data/results_sft/fast/qwen3_causal_dpo_v4_gsm8k.jsonl",
}

DISTILL = {
    "14B Distill Iter1":  "data/results_sft/fast/qwen3_14b_iter1_gsm8k.jsonl",
    "14B Distill Iter2":  "data/results_sft/fast/qwen3_14b_iter2_gsm8k.jsonl",
    "14B Distill Iter3":  "data/results_sft/fast/qwen3_14b_iter3_gsm8k.jsonl",
}


def load_results(path):
    if not os.path.exists(path):
        return None
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not records:
        return None

    total   = len(records)
    correct = sum(1 for r in records if r.get("is_correct", 0))
    acc     = correct / total * 100

    tok_field = next((k for k in ["token_count", "tokens", "length"] if k in records[0]), None)
    avg_tok = (sum(r.get(tok_field, 0) for r in records) / total) if tok_field else None

    step_field = next((k for k in ["step_count", "steps", "num_steps"] if k in records[0]), None)
    avg_steps = (sum(r.get(step_field, 0) for r in records) / total) if step_field else None

    return {
        "total":     total,
        "correct":   correct,
        "acc":       acc,
        "avg_tok":   avg_tok,
        "avg_steps": avg_steps,
    }


def fmt(val, fmt_str, fallback="  —  "):
    if val is None:
        return fallback
    return format(val, fmt_str)


def print_table(sections):
    col_w = [38, 8, 10, 10]
    header = f"{'Model':<{col_w[0]}} {'Acc %':>{col_w[1]}} {'Avg Tok':>{col_w[2]}} {'Avg Steps':>{col_w[3]}}"
    sep    = "-" * sum(col_w)
    print(f"\n{'='*sum(col_w)}")
    print("  Self-Distillation vs RQ1 Comparison (GSM8K test, n=1319)")
    print(f"{'='*sum(col_w)}")
    print(header)
    print(sep)

    for section_name, rows in sections:
        print(f"\n  {section_name}")
        for label, res in rows:
            if res is None:
                print(f"  {label:<{col_w[0]-2}} {'(no file)':>{col_w[1]+col_w[2]+col_w[3]+2}}")
                continue
            acc_s   = fmt(res["acc"],       ".2f")
            tok_s   = fmt(res["avg_tok"],   ".0f")
            step_s  = fmt(res["avg_steps"], ".1f")
            print(f"  {label:<{col_w[0]-2}} {acc_s:>{col_w[1]}} {tok_s:>{col_w[2]}} {step_s:>{col_w[3]}}")

    print(f"\n{'='*sum(col_w)}")
    print("  RQ1 Claim: Causal SFT achieves competitive accuracy with fewer tokens.")
    print(f"{'='*sum(col_w)}\n")


def compression_analysis(sections):
    """Print token compression ratios relative to noncausal SFT."""
    noncausal_tok = None
    for _, rows in sections:
        for label, res in rows:
            if res and "Noncausal" in label:
                noncausal_tok = res["avg_tok"]

    if not noncausal_tok:
        return

    print("  Token Compression vs Noncausal SFT baseline:")
    print(f"  {'Model':<38} {'Ratio':>8} {'Saved':>8}")
    print("  " + "-" * 56)
    for _, rows in sections:
        for label, res in rows:
            if res and res["avg_tok"]:
                ratio = res["avg_tok"] / noncausal_tok
                saved = (1 - ratio) * 100
                print(f"  {label:<38} {ratio:>8.3f} {saved:>7.1f}%")
    print()


def main():
    args = get_args()

    baseline_rows = [(label, load_results(path)) for label, path in BASELINES.items()]
    distill_rows  = [(label, load_results(path)) for label, path in DISTILL.items()
                     if load_results(path) is not None]

    for path in args.extra:
        label = os.path.basename(path).replace(".jsonl", "")
        distill_rows.append((label, load_results(path)))

    sections = [
        ("RQ1 Baselines", baseline_rows),
    ]
    if distill_rows:
        sections.append(("Self-Distillation 14B", distill_rows))

    print_table(sections)
    compression_analysis(sections)


if __name__ == "__main__":
    main()
