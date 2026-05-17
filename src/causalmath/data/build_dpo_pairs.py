#!/usr/bin/env python3
"""
Build DPO v3 pairs with correct format.

chosen:   PNS final_chain steps + \\boxed{correct_answer} + </think> + $\\boxed{}$
          (matches EXACTLY what eval_correct.py expects after <think> prefix)
rejected: Causal SFT wrong outputs stripped of leading <think>\n
          (same compact format, wrong answer)

Both sides have identical format -> no format mismatch.
"""
import json, os, re, argparse

PNS_FILE      = "data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl"
SFT_EVAL_FILE = "data/results_sft/fast/qwen3_causal_gsm8k.jsonl"
OUTPUT        = "data/dpo/dpo_pairs_v3.jsonl"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pns",       default=PNS_FILE)
    p.add_argument("--sft_eval",  default=SFT_EVAL_FILE)
    p.add_argument("--output",    default=OUTPUT)
    p.add_argument("--min_steps", type=int,   default=2)
    p.add_argument("--min_pn",    type=float, default=0.3)
    return p.parse_args()


def strip_leading_think(text):
    """Remove <think>\n prefix if present — prompt already supplies it."""
    t = text.strip()
    if t.startswith("<think>"):
        t = t[len("<think>"):].lstrip("\n")
    return t


def format_chosen(final_chain, answer):
    """Format PNS steps as chosen completion (after <think>\n prompt)."""
    steps = "\n\n".join(final_chain) if isinstance(final_chain, list) else str(final_chain)
    # Match causal SFT output style exactly
    return f"{steps}\n$$\n\\boxed{{{answer}}}\n$$\n</think>\n\n$\\boxed{{{answer}}}$"


def load_pns(path, min_steps, min_pn):
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("metrics", {})
            if m.get("PS(chain)", 0) != 1:
                continue
            if m.get("step_length", 0) < min_steps:
                continue
            if (m.get("avg_PN(steps)") or 0) < min_pn:
                continue
            fc = m.get("final_chain", [])
            if not fc:
                continue
            answer = str(r.get("answer", "")).strip()
            if not answer:
                continue
            index[r["question"].strip()] = (fc, answer)
    print(f"PNS chosen: {len(index)} usable")
    return index


def load_sft_wrong(path):
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("is_correct", 1) == 1:
                continue  # skip correct ones
            q = r.get("question", "").strip()
            ma = r.get("model_answer", "")
            rejected = strip_leading_think(ma)
            if rejected:
                index[q] = rejected
    print(f"SFT wrong (rejected): {len(index)}")
    return index


def main():
    args = get_args()
    print("Building DPO v3 pairs (correct format)...")
    pns = load_pns(args.pns, args.min_steps, args.min_pn)
    wrong = load_sft_wrong(args.sft_eval)

    pairs = []
    for q, rejected in wrong.items():
        if q not in pns:
            continue
        fc, answer = pns[q]
        chosen = format_chosen(fc, answer)
        if chosen.strip() == rejected.strip():
            continue
        pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected})

    print(f"Final DPO v3 pairs: {len(pairs)}")
    if pairs:
        avg_c = sum(len(p["chosen"].split()) for p in pairs) / len(pairs)
        avg_r = sum(len(p["rejected"].split()) for p in pairs) / len(pairs)
        print(f"Avg words — chosen: {avg_c:.0f} | rejected: {avg_r:.0f}")
        print(f"\nSample chosen:\n{pairs[0]['chosen'][:300]}")
        print(f"\nSample rejected:\n{pairs[0]['rejected'][:300]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
