#!/usr/bin/env python3
"""
One-shot summary of ALL experiment results.
Run this after all evals complete to get the full comparison table.

Usage:
  python sft/summarize_all_results.py
"""
import json, os, sys

RESULTS = {
    "RQ1 Baselines (Qwen3-1.7B, GSM8K 1319q)": [
        ("Original 1.7B",       "data/results_sft/fast/qwen3_original_gsm8k.jsonl"),
        ("Noncausal SFT 1.7B",  "data/results_sft/fast/qwen3_noncausal_gsm8k.jsonl"),
        ("Causal SFT 1.7B",     "data/results_sft/fast/qwen3_causal_gsm8k.jsonl"),
        ("Causal DPO v4 1.7B",  "data/results_sft/fast/qwen3_causal_dpo_v4_gsm8k.jsonl"),
        ("Causal SC@5 1.7B",    "data/results_sft/fast/qwen3_causal_sc5_gsm8k.jsonl"),
    ],
    "Self-Distillation (Qwen3-8B, GSM8K 200q)": [
        ("Original 8B (base)",  "data/results_sft/distill/original_gsm8k.jsonl"),
        ("Noncausal 8B SFT",    "data/results_sft/distill/noncausal_gsm8k.jsonl"),
        ("Causal 8B Iter1",     "data/results_sft/distill/causal_gsm8k.jsonl"),
    ],
    "MATH-500 (Qwen3-1.7B)": [
        ("Noncausal SFT 1.7B",  "data/results_sft/fast/qwen3_noncausal_math500.jsonl"),
        ("Causal SFT 1.7B",     "data/results_sft/fast/qwen3_causal_math500.jsonl"),
    ],
}


def load(path):
    if not os.path.exists(path):
        return None, "NOT FOUND"
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not data:
        return None, "EMPTY (in progress?)"

    total   = len(data)
    # Correctness field detection
    corr_field = next((k for k in ["is_correct_sc","is_correct","correct"] if k in data[0]), None)
    correct = sum(1 for r in data if r.get(corr_field, 0)) if corr_field else 0
    acc     = correct / total * 100

    tok_field  = next((k for k in ["token_count","tokens","length"] if k in data[0]), None)
    avg_tok    = int(sum(r.get(tok_field, 0) for r in data) / total) if tok_field else None

    step_field = next((k for k in ["step_count","steps","num_steps"] if k in data[0]), None)
    avg_steps  = round(sum(r.get(step_field, 0) for r in data) / total, 1) if step_field else None

    return {
        "total": total, "correct": correct, "acc": acc,
        "avg_tok": avg_tok, "avg_steps": avg_steps,
    }, "OK"


def main():
    W = [42, 7, 10, 10, 10]
    header = f"{'Model':<{W[0]}} {'n':>{W[1]}} {'Acc %':>{W[2]}} {'AvgTok':>{W[3]}} {'AvgSteps':>{W[4]}}"
    sep    = "-" * sum(W)

    print(f"\n{'='*sum(W)}")
    print("  Complete Results Summary -- PNS Causal CoT Project")
    print(f"{'='*sum(W)}")
    print(header)

    for section, rows in RESULTS.items():
        print(f"\n  {section}")
        print(f"  {sep}")
        for label, path in rows:
            res, status = load(path)
            if res is None:
                print(f"  {label:<{W[0]-2}} {'':>{W[1]}} {status:>{W[2]+W[3]+W[4]+2}}")
            else:
                n_s     = str(res["total"])
                acc_s   = f"{res['acc']:.1f}%"
                tok_s   = str(res["avg_tok"]) if res["avg_tok"] else "—"
                step_s  = str(res["avg_steps"]) if res["avg_steps"] else "—"
                print(f"  {label:<{W[0]-2}} {n_s:>{W[1]}} {acc_s:>{W[2]}} {tok_s:>{W[3]}} {step_s:>{W[4]}}")

    print(f"\n{'='*sum(W)}")

    # Token compression analysis — use GSM8K Noncausal SFT 1.7B as baseline
    noncausal_tok = None
    res_nc, _ = load("data/results_sft/fast/qwen3_noncausal_gsm8k.jsonl")
    if res_nc and res_nc["avg_tok"]:
        noncausal_tok = res_nc["avg_tok"]

    if noncausal_tok:
        print(f"\n  Token Compression (vs Noncausal SFT 1.7B baseline = {noncausal_tok} tok):")
        print(f"  {'Model':<40} {'Ratio':>8} {'Saved':>8}")
        print("  " + "-" * 58)
        for _, rows in RESULTS.items():
            for label, path in rows:
                res, _ = load(path)
                if res and res["avg_tok"]:
                    ratio = res["avg_tok"] / noncausal_tok
                    saved = (1 - ratio) * 100
                    bar = "#" * max(0, int((1 - ratio) * 20))
                    print(f"  {label:<40} {ratio:>8.3f} {saved:>7.1f}%  {bar}")

    print()


if __name__ == "__main__":
    main()
