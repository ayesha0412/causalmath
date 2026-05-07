#!/usr/bin/env python3
"""
Prepare SFT training data - OPTION B (fast iteration).

Creates two matched datasets:
  - causal:    1100 PNS GSM8K (PS=1, final_chain)
  - noncausal: 115 new Qwen3-14B + 1100 COT GSM8K

Both use 2-message format:
  [{"role": "user", "content": question},
   {"role": "assistant", "content": "<think>\n{chain}\n</think>\n\n{final_answer}"}]

Usage:
  python sft/prepare_sft_fast.py
"""

import argparse
import json
import os
import random
import re


# ── Paths ─────────────────────────────────────────────────────────────────────

# GSM8K
PNS_GSM8K          = "data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl"
COT_GSM8K          = "data/gsm8k/gsm8k_cot_qwen3-14b-thinking-v2.jsonl"
NEW_QWEN3_GSM8K    = "data/gsm8k/gsm8k_cot_train_detailed.jsonl"

# MATH-500 (safe split: only 400 train questions, 100 held-out for testing)
PNS_MATH           = "data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl"
COT_MATH           = "data/MATH-500/test_math500_answered_qwen3-14B-thinking-v2.jsonl"
TRAIN_MATH_SPLIT   = "data/MATH-500/train_math500_split.jsonl"  # Use this for indexing

OUT_DIR = "data/sft/fast"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path, limit=None, skip=0):
    """Load JSONL with optional limit and skip."""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def load_jsonl_by_question(path):
    """Index a JSONL by question text for fast lookup."""
    index = {}
    for item in load_jsonl(path):
        q = item.get("question") or item.get("problem", "")
        index[q.strip()] = item
    return index


def chain_to_think(chain_text, answer):
    """
    Build assistant content with <think> block from chain text.
    """
    chain = chain_text.strip() if isinstance(chain_text, str) else ""
    final_ans = f"$\\boxed{{{answer}}}$"
    return f"<think>\n{chain}\n</think>\n\n{final_ans}"


def make_message(question, assistant_content):
    """Create 2-message format for training."""
    return {
        "messages": [
            {"role": "user",      "content": question.strip()},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── Build datasets ─────────────────────────────────────────────────────────────

def build_causal_from_pns(pns_path, max_n=1100, train_split_path=None):
    """
    Build causal dataset from PNS file.
    Uses PS=1 entries (single correct chain) and their final_chain field.

    If train_split_path provided, only use questions from that split (for MATH).
    """
    print(f"\nLoading PNS from {pns_path}...")
    pns_items = load_jsonl(pns_path)
    print(f"  Total items: {len(pns_items)}")

    # If we have a train split, filter to only use those questions
    allowed_qs = None
    if train_split_path:
        allowed_qs = set()
        with open(train_split_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    allowed_qs.add(rec.get('problem', '').strip())
        print(f"  Filtering to {len(allowed_qs)} training split questions")

    # Filter PS=1 only
    ps1 = [x for x in pns_items
           if x.get("metrics", {}).get("PS(chain)", 0) == 1
           and x.get("metrics", {}).get("final_chain")
           and (not allowed_qs or x.get("question", "").strip() in allowed_qs)]
    print(f"  PS=1 items: {len(ps1)}")

    if max_n and len(ps1) > max_n:
        ps1 = ps1[:max_n]
        print(f"  Using first {max_n}")

    causal_recs = []
    for item in ps1:
        q = item["question"].strip()
        answer = item["answer"]
        chain = item["metrics"]["final_chain"]

        # chain is a list of strings; join with newlines
        if isinstance(chain, list):
            chain_text = "\n".join(s.strip() for s in chain if s.strip())
        else:
            chain_text = str(chain).strip()

        content = chain_to_think(chain_text, answer)
        causal_recs.append(make_message(q, content))

    print(f"  Created {len(causal_recs)} causal records")
    return causal_recs


def build_noncausal(new_path, cot_path, max_new=115, max_cot=1100, train_split_path=None):
    """
    Build noncausal dataset from:
    1. New Qwen3-14B generation (optional) - use full_response
    2. COT file - use model_answer

    If train_split_path provided, only use questions from that split (for MATH).
    """
    noncausal_recs = []

    # If we have a train split, filter to only use those questions
    allowed_qs = None
    if train_split_path:
        allowed_qs = set()
        with open(train_split_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    allowed_qs.add(rec.get('problem', '').strip())
        print(f"  Filtering to {len(allowed_qs)} training split questions")

    # Part 1: New Qwen3-14B (optional, only for GSM8K)
    if new_path:
        print(f"\n  Loading NEW from {new_path}...")
        new_items = load_jsonl(new_path, limit=max_new)
        print(f"    Loaded: {len(new_items)}")

        for item in new_items:
            q = item["question"].strip()
            answer = item["answer"]
            full_resp = item.get("full_response", "").strip()

            if full_resp:
                # full_response already has <think>...</think>, just wrap it
                content = full_resp if "</think>" in full_resp else f"<think>\n{full_resp}\n</think>\n\n$\\boxed{{{answer}}}$"
                noncausal_recs.append(make_message(q, content))

        print(f"    Created {len(noncausal_recs)} records from NEW")

    # Part 2: COT
    print(f"  Loading COT from {cot_path}...")
    cot_items = load_jsonl(cot_path, limit=max_cot)
    print(f"    Loaded: {len(cot_items)}")

    new_count = len(noncausal_recs)
    for item in cot_items:
        q = item["question"].strip()
        # Skip if we're filtering and question not in training split
        if allowed_qs and q not in allowed_qs:
            continue

        answer = item["answer"]
        model_answer = item.get("model_answer", "").strip()

        if model_answer:
            # Wrap in <think> if not already
            if "<think>" in model_answer and "</think>" in model_answer:
                content = model_answer
            else:
                content = chain_to_think(model_answer, answer)
            noncausal_recs.append(make_message(q, content))

    print(f"    Created {len(noncausal_recs) - new_count} records from COT")
    print(f"  Total: {len(noncausal_recs)} records")
    return noncausal_recs


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_gsm8k_causal",  type=int, default=1100, help="GSM8K PNS PS=1 for causal")
    p.add_argument("--n_math_causal",   type=int, default=500,  help="MATH PNS PS=1 for causal")
    p.add_argument("--n_gsm8k_new",     type=int, default=115,  help="New Qwen3-14B examples")
    p.add_argument("--n_gsm8k_cot",     type=int, default=1100, help="GSM8K COT v2 for noncausal")
    p.add_argument("--n_math_cot",      type=int, default=500,  help="MATH COT for noncausal")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--out_dir",         default=OUT_DIR)
    return p.parse_args()


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):4d} records -> {path}")


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("PREPARING SFT DATA - OPTION B (Fast Iteration with MATH-500)")
    print("=" * 70)

    all_causal = []
    all_noncausal = []

    # ── GSM8K ────────────────────────────────────────────────────────────────
    print("\n[GSM8K]")
    causal_gsm8k = build_causal_from_pns(PNS_GSM8K, max_n=args.n_gsm8k_causal)
    noncausal_gsm8k = build_noncausal(NEW_QWEN3_GSM8K, COT_GSM8K,
                                      max_new=args.n_gsm8k_new, max_cot=args.n_gsm8k_cot)
    all_causal.extend(causal_gsm8k)
    all_noncausal.extend(noncausal_gsm8k)
    print(f"  GSM8K causal: {len(causal_gsm8k)}, noncausal: {len(noncausal_gsm8k)}")

    # ── MATH-500 (safe split: only use 400 training examples) ──────────────
    print("\n[MATH-500] (using 400 training split, 100 held-out for testing)")
    causal_math = build_causal_from_pns(PNS_MATH, max_n=args.n_math_causal,
                                        train_split_path=TRAIN_MATH_SPLIT)
    noncausal_math = build_noncausal(None, COT_MATH,
                                     max_new=0, max_cot=args.n_math_cot,
                                     train_split_path=TRAIN_MATH_SPLIT)
    all_causal.extend(causal_math)
    all_noncausal.extend(noncausal_math)
    print(f"  MATH causal: {len(causal_math)}, noncausal: {len(noncausal_math)}")

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(all_causal)
    rng.shuffle(all_noncausal)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Causal records    : {len(all_causal)}")
    print(f"  GSM8K: {len(causal_gsm8k)} (PNS PS=1)")
    print(f"  MATH:  {len(causal_math)} (PNS PS=1)")
    print(f"Noncausal records : {len(all_noncausal)}")
    print(f"  GSM8K: {len(noncausal_gsm8k)} (115 new + {args.n_gsm8k_cot} COT)")
    print(f"  MATH:  {len(noncausal_math)} ({args.n_math_cot} COT)")
    print("=" * 70)

    # Verify format
    if all_causal:
        sample = all_causal[0]
        print(f"\n[CAUSAL SAMPLE]")
        print(f"  user:      {sample['messages'][0]['content'][:60]!r}...")
        asst = sample['messages'][1]['content']
        print(f"  has_think: {'<think>' in asst}")
        print(f"  has_close: {'</think>' in asst}")
        print(f"  len:       {len(asst)} chars")

    if all_noncausal:
        sample = all_noncausal[0]
        print(f"\n[NONCAUSAL SAMPLE]")
        print(f"  user:      {sample['messages'][0]['content'][:60]!r}...")
        asst = sample['messages'][1]['content']
        print(f"  has_think: {'<think>' in asst}")
        print(f"  has_close: {'</think>' in asst}")
        print(f"  len:       {len(asst)} chars")

    # Save
    print("\nSaving...")
    save_jsonl(all_causal, os.path.join(args.out_dir, "causal_train.jsonl"))
    save_jsonl(all_noncausal, os.path.join(args.out_dir, "noncausal_train.jsonl"))

    print("\nDone! Ready for training with train_correct.py")


if __name__ == "__main__":
    main()
