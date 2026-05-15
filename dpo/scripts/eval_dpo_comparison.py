#!/usr/bin/env python3
"""
Generate SFT vs DPO comparison table.

Reads evaluation results from both SFT and DPO runs, computes accuracy metrics,
and produces a comparison table showing DPO improvements over SFT baseline.
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


def load_results(results_dir, prefix, dataset):
    """Load evaluation results matching pattern: {prefix}_{condition}_{dataset}.jsonl"""
    files = list(Path(results_dir).glob(f"{prefix}_*_{dataset}.jsonl"))
    if not files:
        return None

    results = defaultdict(lambda: {"correct": 0, "total": 0})
    for file in files:
        # Parse filename: prefix_model_condition_dataset.jsonl
        parts = file.stem.split("_")
        if len(parts) < 2:
            continue
        condition = parts[-2] if len(parts) >= 2 else "unknown"

        with open(file) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        results[condition]["total"] += 1
                        results[condition]["correct"] += rec.get("is_correct", 0)
                    except:
                        pass

    return results if results else None


def compute_metrics(results, condition):
    """Compute accuracy from results dict."""
    if condition not in results or results[condition]["total"] == 0:
        return None
    r = results[condition]
    acc = 100.0 * r["correct"] / r["total"]
    return {"accuracy": acc, "correct": r["correct"], "total": r["total"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_results_dir", default="data/results_sft",
                        help="Directory with SFT evaluation results")
    parser.add_argument("--dpo_results_dir", default="data/results_dpo",
                        help="Directory with DPO evaluation results")
    parser.add_argument("--output", default="data/comparison_table.txt",
                        help="Output file for comparison table")
    parser.add_argument("--dataset", default="math500",
                        help="Dataset to compare (math500, gsm8k, etc.)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # List available models/conditions from SFT results
    sft_files = list(Path(args.sft_results_dir).glob(f"*_{args.dataset}.jsonl"))
    if not sft_files:
        print(f"ERROR: No SFT results found in {args.sft_results_dir} for {args.dataset}")
        return 1

    # Extract unique model keys
    models = set()
    for file in sft_files:
        # Filename: model_condition_dataset.jsonl
        parts = file.stem.split("_")
        if len(parts) >= 2:
            model = parts[0]
            models.add(model)

    models = sorted(models)
    print(f"\nFound models: {models}")

    comparison = []
    comparison.append(f"{'Model':<15} {'Condition':<12} {'SFT':<10} {'DPO':<10} {'Improvement':<12}")
    comparison.append("=" * 60)

    for model in models:
        # Load SFT and DPO results
        sft_pattern = f"{model}_"
        dpo_pattern = f"{model}_dpo_"

        sft_results = load_results(args.sft_results_dir, sft_pattern.rstrip("_"), args.dataset)
        dpo_results = load_results(args.dpo_results_dir, dpo_pattern.rstrip("_"), args.dataset)

        if not sft_results and not dpo_results:
            continue

        conditions = set()
        if sft_results:
            conditions.update(sft_results.keys())
        if dpo_results:
            conditions.update(dpo_results.keys())

        for condition in sorted(conditions):
            sft_m = compute_metrics(sft_results or {}, condition)
            dpo_m = compute_metrics(dpo_results or {}, condition)

            sft_str = f"{sft_m['accuracy']:.1f}%" if sft_m else "N/A"
            dpo_str = f"{dpo_m['accuracy']:.1f}%" if dpo_m else "N/A"

            if sft_m and dpo_m:
                improvement = dpo_m["accuracy"] - sft_m["accuracy"]
                imp_str = f"{improvement:+.1f}%" if improvement != 0 else "—"
            else:
                imp_str = "N/A"

            line = f"{model:<15} {condition:<12} {sft_str:<10} {dpo_str:<10} {imp_str:<12}"
            comparison.append(line)

    # Print and save
    table = "\n".join(comparison)
    print("\n" + table)

    with open(args.output, "w") as f:
        f.write(table + "\n")
    print(f"\nComparison table saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
