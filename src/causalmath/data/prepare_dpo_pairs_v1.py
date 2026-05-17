#!/usr/bin/env python3
"""
Stage 3: Build DPO preference pairs.

chosen:   NON-CAUSAL COT chain (correct + verbose, full intermediate steps)
rejected: SFT-causal model's wrong generation

Why COT as chosen?
  The causal SFT model drops accuracy because PNS chains are too compressed
  (~58 words avg). The 1.7B model learns compressed style but skips steps on
  hard problems. DPO with COT-chosen teaches the model to prefer correct,
  step-by-step reasoning over its own wrong compressed outputs.

  Paper framing: "Causal SFT compresses reasoning; DPO with COT preference
  learning recovers accuracy while remaining more efficient than the original."

Output format (standard DPO):
  {"prompt": ..., "chosen": ..., "rejected": ...}

Usage:
  python dpo/prepare_dpo_pairs.py
"""
import json, os, argparse

NONCAUSAL_TRAIN = "data/sft/fast/noncausal_train.jsonl"   # COT chains (correct + verbose)
REJECTED_FILE   = "data/dpo/rejected_chains.jsonl"
OUTPUT          = "data/dpo/dpo_pairs.jsonl"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chosen",   default=NONCAUSAL_TRAIN, help="Chosen chains (COT)")
    p.add_argument("--rejected", default=REJECTED_FILE,   help="Rejected chains file")
    p.add_argument("--output",   default=OUTPUT,          help="Output DPO pairs file")
    return p.parse_args()


def load_chosen_by_question(path):
    """Index noncausal (COT) training data by question text."""
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                msgs = r.get("messages", [])
                if len(msgs) >= 2:
                    q = msgs[0]["content"].strip()
                    index[q] = msgs[1]["content"]  # assistant content (chosen)
    return index


def main():
    args = get_args()
    print("Building DPO preference pairs...")
    print("  chosen  = noncausal COT chain (correct + verbose)")
    print("  rejected = SFT-causal model's wrong generation")

    # Load chosen chains (noncausal COT - correct + has intermediate steps)
    chosen_index = load_chosen_by_question(args.chosen)
    print(f"  Chosen chains (COT): {len(chosen_index)}")

    # Load rejected chains (SFT-causal model's wrong answers)
    rejected = []
    with open(args.rejected, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rejected.append(json.loads(line))
    print(f"  Rejected chains (wrong SFT-causal): {len(rejected)}")

    # Build pairs: for each wrong SFT output, find the COT chain for same question
    pairs = []
    skipped_no_chosen = 0
    skipped_same = 0
    for rec in rejected:
        q = rec["question"].strip()
        chosen = chosen_index.get(q)
        if not chosen:
            skipped_no_chosen += 1
            continue

        rejected_chain = rec["rejected_chain"]

        # Sanity check: chosen and rejected should be different
        if chosen.strip() == rejected_chain.strip():
            skipped_same += 1
            continue

        pairs.append({
            "prompt":   q,
            "chosen":   chosen,          # COT chain (correct + step-by-step)
            "rejected": rejected_chain,  # SFT-causal model's wrong chain
        })

    print(f"  Skipped (no COT match):    {skipped_no_chosen}")
    print(f"  Skipped (identical pair):  {skipped_same}")
    print(f"  Final DPO pairs:           {len(pairs)}")

    # Stats: chosen chain length vs rejected
    if pairs:
        chosen_words  = sum(len(p["chosen"].split())   for p in pairs) / len(pairs)
        rejected_words = sum(len(p["rejected"].split()) for p in pairs) / len(pairs)
        print(f"\n  Avg words - chosen (COT): {chosen_words:.0f}  |  rejected (wrong): {rejected_words:.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved: {args.output}")

    # Show sample
    if pairs:
        sample = pairs[0]
        print(f"\n[SAMPLE PAIR]")
        print(f"  prompt:   {sample['prompt'][:80]}...")
        print(f"  chosen:   {sample['chosen'][:120]}...")
        print(f"  rejected: {sample['rejected'][:120]}...")


if __name__ == "__main__":
    main()
