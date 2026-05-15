#!/usr/bin/env python3
"""
Self-Distillation Loop — Step 3: SFT Data Preparation.

Converts PNS output (from run_algo_o.py) into causal SFT training format
(same format as data/gsm8k/train_causal_fast.jsonl).

Filters: PS(chain)=1, step_length >= min_steps, avg_PN >= min_pn

Usage:
  python sft/prepare_distill_sft.py \
    --pns data/gsm8k/gsm8k_pns_distill_iter1.jsonl \
    --output data/gsm8k/train_causal_distill_iter1.jsonl
"""
import json, os, argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pns",       required=True, help="PNS output from run_algo_o.py")
    p.add_argument("--output",    required=True)
    p.add_argument("--min_steps", type=int,   default=1,
                   help="Min PNS steps to keep (relaxed from 2 → 1 for self-distillation)")
    p.add_argument("--min_pn",    type=float, default=0.1,
                   help="Min avg PN score to keep (relaxed from 0.3 → 0.1)")
    p.add_argument("--iter",      type=int,   default=1, help="Distillation iteration number")
    return p.parse_args()


def format_chain(final_chain, answer):
    """Format PNS causal chain as cot_response (matches train_causal_fast.jsonl format).

    PNS often retains the boxed-answer line as the last step. Strip it before
    appending a clean one to avoid duplication in training data.
    """
    import re
    steps = list(final_chain) if isinstance(final_chain, list) else [str(final_chain)]
    # Remove trailing step if it is just the boxed answer (avoid duplication)
    if steps and re.search(r"\\boxed\{", steps[-1]) and "$$" in steps[-1]:
        steps = steps[:-1]
    body = "\n\n".join(steps)
    return f"{body}\n\n$$\n\\boxed{{{answer}}}\n$$"


def main():
    args = get_args()

    records = []
    skipped = total = 0

    with open(args.pns, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            m = r.get("metrics", {})

            if m.get("PS(chain)", 0) != 1:
                skipped += 1; continue
            if m.get("step_length", 0) < args.min_steps:
                skipped += 1; continue
            if (m.get("avg_PN(steps)") or 0) < args.min_pn:
                skipped += 1; continue

            fc = m.get("final_chain", [])
            if not fc:
                skipped += 1; continue

            answer = str(r.get("answer", "")).strip()
            if not answer:
                skipped += 1; continue

            cot = format_chain(fc, answer)
            records.append({
                "question":        r["question"],
                "answer":          answer,
                "cot_response":    cot,
                "thinking_enabled": False,
                "source":          f"self-distill-iter{args.iter}",
                "pns_steps":       m.get("step_length"),
                "avg_pn":          round(m.get("avg_PN(steps)") or 0, 3),
            })

    print(f"PNS records  : {total}")
    print(f"Passed filter: {len(records)}  ({len(records)/total*100:.1f}%)")
    print(f"Skipped      : {skipped}")
    if records:
        avg_steps = sum(r["pns_steps"] for r in records) / len(records)
        avg_pn    = sum(r["avg_pn"]    for r in records) / len(records)
        print(f"Avg steps    : {avg_steps:.1f}  |  Avg PN: {avg_pn:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} training examples -> {args.output}")


if __name__ == "__main__":
    main()
