#!/usr/bin/env python3
"""
Prepare DPO training data from PNS-optimized and original reasoning.

DPO requires preference pairs: (prompt, chosen, rejected).
We use:
  - chosen: PNS-pruned CoT (shorter, essential steps only)
  - rejected: original verbose CoT (longer, redundant reasoning)

This creates automatic preference signals without human annotation.
"""

import argparse
import json
import os
from pathlib import Path


def load_jsonl(path):
    """Load JSONL file as list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def find_pns_file(original_path, pns_data_dir="data/MATH-500"):
    """Infer PNS file name from original data path."""
    basename = Path(original_path).stem
    # Try exact match first
    pns_file = os.path.join(pns_data_dir, f"{basename}_pns_full.jsonl")
    if os.path.exists(pns_file):
        return pns_file
    # Try with suffix variations
    for suffix in ["_pns_full", "_pns_200", "_pns"]:
        pns_file = os.path.join(pns_data_dir, f"{basename}{suffix}.jsonl")
        if os.path.exists(pns_file):
            return pns_file
    return None


def prepare_dpo_pairs(original_records, pns_records, condition="unknown"):
    """
    Create DPO preference pairs from original and PNS-pruned reasoning.

    Args:
        original_records: List of dicts with reasoning from SFT data
        pns_records: List of dicts with PNS-pruned reasoning
        condition: Condition label (original, noncausal, causal)

    Returns:
        List of dicts with keys: prompt, chosen, rejected, question, answer
    """
    pairs = []

    # Index PNS records by question for fast lookup
    pns_by_q = {}
    for rec in pns_records:
        q = rec.get("question") or rec.get("problem", "")
        pns_by_q[q] = rec

    for orig_rec in original_records:
        question = orig_rec.get("question") or orig_rec.get("problem", "")
        answer = orig_rec.get("answer") or orig_rec.get("ground_truth", "")

        if question not in pns_by_q:
            # No PNS equivalent — skip this pair
            continue

        pns_rec = pns_by_q[question]

        # Extract reasoning from messages format (SFT training data)
        if "messages" in orig_rec:
            # Original has messages format
            for msg in orig_rec["messages"]:
                if msg.get("role") == "assistant":
                    original_reasoning = msg.get("content", "")
                    break
            else:
                original_reasoning = ""
        else:
            # Fallback: use full response
            original_reasoning = orig_rec.get("model_answer", "")

        # PNS reasoning (pruned)
        if "messages" in pns_rec:
            for msg in pns_rec["messages"]:
                if msg.get("role") == "assistant":
                    pns_reasoning = msg.get("content", "")
                    break
            else:
                pns_reasoning = ""
        else:
            pns_reasoning = pns_rec.get("model_answer", "")

        if not original_reasoning or not pns_reasoning:
            # Skip if either is missing
            continue

        # Prefer shorter, more efficient reasoning (PNS) over verbose (original)
        # Both should end with \boxed{answer}
        prompt = question

        pair = {
            "prompt": prompt,
            "chosen": pns_reasoning,  # PNS-pruned: efficient
            "rejected": original_reasoning,  # Original: verbose
            "question": question,
            "answer": str(answer),
            "condition": condition,
        }
        pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data", required=True,
                        help="Original SFT training JSONL (with reasoning)")
    parser.add_argument("--pns_data", default=None,
                        help="PNS-optimized JSONL (auto-inferred if not set)")
    parser.add_argument("--output_dir", default="data/sft",
                        help="Output directory for DPO training data")
    parser.add_argument("--condition", default="unknown",
                        choices=["original", "noncausal", "causal"],
                        help="Condition label")
    parser.add_argument("--reverse_pairs", action="store_true",
                        help="Swap chosen/rejected (use if PNS is worse than original)")
    args = parser.parse_args()

    print(f"Loading original data: {args.original_data}")
    original = load_jsonl(args.original_data)
    print(f"  {len(original)} records")

    # Infer PNS file if not provided
    if args.pns_data is None:
        args.pns_data = find_pns_file(args.original_data)
        if args.pns_data is None:
            print("ERROR: Could not find PNS data file. Specify with --pns_data")
            return 1

    print(f"Loading PNS data: {args.pns_data}")
    if not os.path.exists(args.pns_data):
        print(f"ERROR: PNS file not found: {args.pns_data}")
        return 1

    pns = load_jsonl(args.pns_data)
    print(f"  {len(pns)} records")

    print("\nPreparing DPO pairs...")
    pairs = prepare_dpo_pairs(original, pns, condition=args.condition)
    print(f"  Created {len(pairs)} preference pairs")

    # Reverse chosen/rejected if needed (e.g., if PNS is worse than original)
    if args.reverse_pairs:
        print(f"  Reversing chosen/rejected pairs (PNS as rejected, original as chosen)...")
        for pair in pairs:
            pair["chosen"], pair["rejected"] = pair["rejected"], pair["chosen"]

    if not pairs:
        print("ERROR: No pairs created. Check that question keys match between datasets.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output filename
    basename = os.path.splitext(os.path.basename(args.original_data))[0]
    # Replace "sft_" with "dpo_" if present, else prepend "dpo_"
    if basename.startswith("sft_"):
        output_name = basename.replace("sft_", "dpo_") + ".jsonl"
    else:
        output_name = f"dpo_{basename}.jsonl"

    output_path = os.path.join(args.output_dir, output_name)

    print(f"\nSaving DPO data to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"✓ DPO data prepared. {len(pairs)} pairs saved.")
    print(f"\nNext step: python sft/train_dpo.py --data {output_path} --output_dir models/sft/deepseek_dpo_{args.condition}")
    return 0


if __name__ == "__main__":
    exit(main())
