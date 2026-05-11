#!/usr/bin/env python3
"""
Build combined DPO pairs from GSM8K + MATH-500.

GSM8K chosen:   PNS final_chain + \\boxed{answer} (relaxed filters)
                Fallback: minimal correct answer in SFT format
MATH-500 chosen: minimal correct answer in SFT format (same compact style)
rejected:        Causal SFT wrong outputs (both datasets)

More diverse training data -> better reasoning generalisation.
"""
import json, os, argparse

GSM8K_PNS    = "data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl"
GSM8K_SFT    = "data/results_sft/fast/qwen3_causal_gsm8k.jsonl"
GSM8K_TEST   = "data/gsm8k/test_gsm8k_answered.jsonl"
MATH_SFT     = "data/results_sft/fast/qwen3_causal_math500.jsonl"
MATH_TEST    = "data/MATH-500/test_math500_answered.jsonl"
MATH_PNS     = "data/MATH-500/math500_pns_qwen3-thinking-v2.jsonl"
OUTPUT       = "data/dpo/dpo_pairs_combined.jsonl"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",    default=OUTPUT)
    p.add_argument("--min_steps", type=int,   default=1)
    p.add_argument("--min_pn",    type=float, default=0.1)
    return p.parse_args()


def strip_think(text):
    t = text.strip()
    if t.startswith("<think>"):
        t = t[len("<think>"):].lstrip("\n")
    return t


def minimal_chosen(answer):
    """Minimal correct chosen in exact SFT output format."""
    return (
        f"$$\n\\boxed{{{answer}}}\n$$\n</think>\n\n$\\boxed{{{answer}}}$"
    )


def pns_chosen(final_chain, answer):
    """PNS causal chain + boxed answer in SFT output format."""
    steps = "\n\n".join(final_chain) if isinstance(final_chain, list) else str(final_chain)
    return f"{steps}\n$$\n\\boxed{{{answer}}}\n$$\n</think>\n\n$\\boxed{{{answer}}}$"


def load_gt(path, q_field="question", a_field="answer"):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                idx[r[q_field].strip()] = str(r.get(a_field, "")).strip()
    return idx


def load_wrong_sft(path):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("is_correct", 1) == 0:
                q = r.get("question", "").strip()
                idx[q] = strip_think(r.get("model_answer", ""))
    return idx


def load_pns_index(path, min_steps, min_pn):
    idx = {}
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
            ans = str(r.get("answer", "")).strip()
            if ans:
                idx[r["question"].strip()] = (fc, ans)
    return idx


def build_pairs(wrong_sft, gt, pns_idx, dataset_tag):
    pairs = []
    pns_used = minimal_used = 0
    for q, rejected in wrong_sft.items():
        answer = gt.get(q, "")
        if not answer or not rejected:
            continue
        if q in pns_idx:
            fc, ans = pns_idx[q]
            chosen = pns_chosen(fc, ans)
            pns_used += 1
        else:
            chosen = minimal_chosen(answer)
            minimal_used += 1
        if chosen.strip() == rejected.strip():
            continue
        pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected, "dataset": dataset_tag})
    print(f"  {dataset_tag}: {len(pairs)} pairs (PNS: {pns_used}, minimal: {minimal_used})")
    return pairs


def main():
    args = get_args()
    print("Building combined DPO pairs (GSM8K + MATH-500)...\n")

    # --- GSM8K ---
    print("[GSM8K]")
    gsm_gt    = load_gt(GSM8K_TEST)
    gsm_wrong = load_wrong_sft(GSM8K_SFT)
    gsm_pns   = load_pns_index(GSM8K_PNS, args.min_steps, args.min_pn)
    print(f"  Wrong: {len(gsm_wrong)} | PNS available: {len(gsm_pns)}")
    gsm_pairs = build_pairs(gsm_wrong, gsm_gt, gsm_pns, "gsm8k")

    # --- MATH-500 ---
    print("\n[MATH-500]")
    math_gt    = load_gt(MATH_TEST)
    math_wrong = load_wrong_sft(MATH_SFT)
    math_pns   = load_pns_index(MATH_PNS, args.min_steps, args.min_pn)
    print(f"  Wrong: {len(math_wrong)} | PNS available: {len(math_pns)}")
    math_pairs = build_pairs(math_wrong, math_gt, math_pns, "math500")

    # --- Combine ---
    all_pairs = gsm_pairs + math_pairs
    print(f"\nTotal: {len(gsm_pairs)} GSM8K + {len(math_pairs)} MATH-500 = {len(all_pairs)} pairs")

    if all_pairs:
        avg_c = sum(len(p["chosen"].split())   for p in all_pairs) / len(all_pairs)
        avg_r = sum(len(p["rejected"].split()) for p in all_pairs) / len(all_pairs)
        print(f"Avg words — chosen: {avg_c:.0f} | rejected: {avg_r:.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
