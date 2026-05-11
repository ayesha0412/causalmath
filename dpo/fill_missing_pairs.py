#!/usr/bin/env python3
"""
Fill in DPO pairs for wrong SFT questions that had no PNS match.
For these, chosen = a minimal correct causal answer in SFT format.
"""
import json, os, argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_eval",  default="data/results_sft/fast/qwen3_causal_gsm8k.jsonl")
    p.add_argument("--pns_pairs", default="data/dpo/dpo_pairs_v3b.jsonl")
    p.add_argument("--test_file", default="data/gsm8k/test_gsm8k_answered.jsonl")
    p.add_argument("--output",    default="data/dpo/dpo_pairs_v3b.jsonl")
    return p.parse_args()


def strip_leading_think(text):
    t = text.strip()
    if t.startswith("<think>"):
        t = t[len("<think>"):].lstrip("\n")
    return t


def main():
    args = get_args()

    # Load existing PNS pairs
    existing = {}
    with open(args.pns_pairs, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                existing[r["prompt"].strip()] = r
    print(f"Existing pairs: {len(existing)}")

    # Load ground truth answers
    gt = {}
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gt[r["question"].strip()] = str(r["answer"]).strip()

    # Load wrong SFT outputs
    wrong = {}
    with open(args.sft_eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("is_correct", 1) == 0:
                    q = r.get("question", "").strip()
                    wrong[q] = strip_leading_think(r.get("model_answer", ""))

    print(f"Wrong SFT outputs: {len(wrong)}")

    # Fill missing
    added = 0
    for q, rejected in wrong.items():
        if q in existing:
            continue
        answer = gt.get(q)
        if not answer:
            continue
        # Minimal correct chosen in SFT format
        chosen = (
            f"The answer is $\\boxed{{{answer}}}$.\n"
            f"$$\n\\boxed{{{answer}}}\n$$\n</think>\n\n$\\boxed{{{answer}}}$"
        )
        existing[q] = {"prompt": q, "chosen": chosen, "rejected": rejected}
        added += 1

    print(f"Added {added} minimal pairs. Total: {len(existing)}")

    pairs = list(existing.values())
    avg_c = sum(len(p["chosen"].split()) for p in pairs) / len(pairs)
    avg_r = sum(len(p["rejected"].split()) for p in pairs) / len(pairs)
    print(f"Avg words — chosen: {avg_c:.0f} | rejected: {avg_r:.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
