#!/usr/bin/env python3
"""
Build noncausal baseline from the SAME questions as causal iter1,
using raw (unfiltered) 14B chains. This ensures the only difference
between causal and noncausal is PNS filtering — no format confound.

Usage:
  python sft/prepare_noncausal.py \
    --causal  data/self_distill/iter2/train.jsonl \
    --raw_cot data/gsm8k/gsm8k_cot_qwen3-14b-thinking-v2.jsonl \
    --output  data/self_distill/noncausal_messages.jsonl
"""
import json, argparse, os, re

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--causal",  required=True, help="PNS-filtered causal training set")
    p.add_argument("--raw_cot", required=True, help="Raw 14B CoT (same source, no PNS)")
    p.add_argument("--output",  required=True)
    return p.parse_args()

def wrap_thinking(chain, answer):
    chain = chain.strip()
    chain = re.sub(r'\n*\$\$\s*\\boxed\{[^}]*\}\s*\$\$\s*$', '', chain).strip()
    return f"<think>\n{chain}\n</think>\n\n$$\n\\boxed{{{answer}}}\n$$"

def main():
    args = get_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Get the exact questions in the causal set
    causal = [json.loads(l) for l in open(args.causal, encoding="utf-8") if l.strip()]
    causal_qs = {r["question"].strip()[:120] for r in causal}
    print(f"Causal set   : {len(causal)} questions")

    # Build lookup from raw CoT file
    raw = [json.loads(l) for l in open(args.raw_cot, encoding="utf-8") if l.strip()]
    raw_lookup = {r["question"].strip()[:120]: r for r in raw}
    print(f"Raw CoT file : {len(raw)} records")

    out = []
    missing = 0
    for r in causal:
        key = r["question"].strip()[:120]
        raw_r = raw_lookup.get(key)
        if not raw_r:
            missing += 1
            continue
        chain = raw_r.get("model_answer", "").strip()
        ans   = str(r.get("answer", "")).strip()
        if not chain or not ans:
            missing += 1
            continue
        out.append({"messages": [
            {"role": "user",      "content": r["question"].strip()},
            {"role": "assistant", "content": wrap_thinking(chain, ans)},
        ]})

    with open(args.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Output       : {len(out)} records -> {args.output}")
    print(f"Missing      : {missing}")
    print(f"Causal/Noncausal use exact same {len(out)} questions — only difference is PNS filtering")

if __name__ == "__main__":
    main()
