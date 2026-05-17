#!/usr/bin/env python3
"""
Build causal DPO preference pairs.

chosen:   PNS final_chain from gsm8k_pns file (causally necessary steps, correct)
rejected: Causal SFT model's wrong generation (same style, wrong answer)

Why this works:
  - Both chosen and rejected are short, causal-style chains
  - DPO teaches the model: "generate correct causal reasoning, not wrong causal reasoning"
  - Directly fixes the accuracy drop from 83.1% -> 77.3% in Causal SFT
  - Does NOT re-introduce verbosity (both sides are compact)

Usage:
  python dpo/prepare_dpo_pairs_causal.py
"""
import json
import os
import argparse

PNS_FILE     = "data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl"
REJECTED_FILE = "data/dpo/rejected_chains_gsm8k.jsonl"
OUTPUT        = "data/dpo/dpo_pairs_causal.jsonl"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pns",      default=PNS_FILE,      help="PNS output file (chosen source)")
    p.add_argument("--rejected", default=REJECTED_FILE,  help="SFT model wrong outputs")
    p.add_argument("--output",   default=OUTPUT,         help="Output DPO pairs file")
    p.add_argument("--min_steps", type=int, default=2,   help="Min final_chain steps to include")
    p.add_argument("--min_pn",    type=float, default=0.3, help="Min avg_PN to include")
    return p.parse_args()


def load_pns_chosen(path, min_steps, min_pn):
    """
    Load PNS file and index usable chosen chains by question.
    Filters to records where:
      - PS(chain) == 1  (correct answer)
      - step_length < original_step_length  (pruning actually happened)
      - step_length >= min_steps  (has real reasoning, not just boxed answer)
      - avg_PN >= min_pn  (meaningful causal signal)
    """
    index = {}
    total = skipped_ps = skipped_no_prune = skipped_short = skipped_pn = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            m = r.get("metrics", {})

            if m.get("PS(chain)", 0) != 1:
                skipped_ps += 1
                continue
            if m.get("step_length", 0) >= m.get("original_step_length", 0):
                skipped_no_prune += 1
                continue
            if m.get("step_length", 0) < min_steps:
                skipped_short += 1
                continue
            avg_pn = m.get("avg_PN(steps)") or 0
            if avg_pn < min_pn:
                skipped_pn += 1
                continue

            final_chain = m.get("final_chain", [])
            chosen_text = "\n\n".join(final_chain) if isinstance(final_chain, list) else str(final_chain)
            index[r["question"].strip()] = chosen_text

    print(f"  PNS file: {total} total")
    print(f"  Skipped PS=0: {skipped_ps}")
    print(f"  Skipped no pruning: {skipped_no_prune}")
    print(f"  Skipped chosen too short (<{min_steps} steps): {skipped_short}")
    print(f"  Skipped low PN (<{min_pn}): {skipped_pn}")
    print(f"  Usable chosen chains: {len(index)}")
    return index


def load_rejected(path):
    """Load rejected chains indexed by question."""
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            index[r["question"].strip()] = r["rejected_chain"]
    print(f"  Rejected chains: {len(index)}")
    return index


def main():
    args = get_args()
    print("Building causal DPO pairs...")
    print(f"  chosen  = PNS final_chain (correct + causally pruned)")
    print(f"  rejected = SFT-causal model's wrong generation\n")

    print("[1] Loading chosen chains from PNS file...")
    chosen_index = load_pns_chosen(args.pns, args.min_steps, args.min_pn)

    print("\n[2] Loading rejected chains from SFT model...")
    rejected_index = load_rejected(args.rejected)

    print("\n[3] Building pairs...")
    pairs = []
    skipped_no_chosen = skipped_no_rejected = skipped_identical = 0

    all_questions = set(chosen_index.keys()) | set(rejected_index.keys())
    for q in all_questions:
        chosen = chosen_index.get(q)
        rejected = rejected_index.get(q)

        if not chosen:
            skipped_no_chosen += 1
            continue
        if not rejected:
            skipped_no_rejected += 1
            continue
        if chosen.strip() == rejected.strip():
            skipped_identical += 1
            continue

        pairs.append({
            "prompt":   q,
            "chosen":   chosen,
            "rejected": rejected,
        })

    print(f"  Skipped (no PNS chosen):  {skipped_no_chosen}")
    print(f"  Skipped (no rejected):    {skipped_no_rejected}")
    print(f"  Skipped (identical pair): {skipped_identical}")
    print(f"  Final DPO pairs:          {len(pairs)}")

    if pairs:
        chosen_words  = sum(len(p["chosen"].split())   for p in pairs) / len(pairs)
        rejected_words = sum(len(p["rejected"].split()) for p in pairs) / len(pairs)
        print(f"\n  Avg words — chosen (causal): {chosen_words:.0f}  |  rejected (wrong): {rejected_words:.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pairs)} pairs → {args.output}")

    if pairs:
        s = pairs[0]
        print(f"\n[SAMPLE PAIR]")
        print(f"  prompt:   {s['prompt'][:80]}...")
        print(f"  chosen:   {s['chosen'][:150]}...")
        print(f"  rejected: {s['rejected'][:150]}...")


if __name__ == "__main__":
    main()
