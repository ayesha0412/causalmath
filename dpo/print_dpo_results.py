#!/usr/bin/env python3
"""
Print final comparison: Original vs SFT-Causal vs DPO-Causal
"""
import argparse, json, os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/results_sft/fast")
    return p.parse_args()


def load(path):
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        return None
    correct = sum(r.get("is_correct", 0) for r in recs)
    return {
        "n":     len(recs),
        "acc":   correct / len(recs) * 100,
        "tok":   sum(r.get("token_count", 0) for r in recs) / len(recs),
        "steps": sum(r.get("step_count",  0) for r in recs) / len(recs),
    }


def row(label, r, base=None):
    if r is None:
        return f"  {label:<20}  MISSING"
    acc_s  = f"{r['acc']:.1f}%"
    tok_s  = f"{r['tok']:.0f}"
    step_s = f"{r['steps']:.1f}"
    if base:
        acc_s  += f" ({r['acc']  - base['acc']:+.1f}%)"
        tok_s  += f" ({r['tok']  - base['tok']:+.0f})"
        step_s += f" ({r['steps']- base['steps']:+.1f})"
    return f"  {label:<20}  Acc: {acc_s:<18}  Tok: {tok_s:<14}  Steps: {step_s}"


def main():
    args = get_args()
    d = args.results_dir

    datasets = [
        ("gsm8k",   "GSM8K (1319)"),
        ("math500", "MATH-500 (500)"),
    ]

    conditions = [
        ("qwen3_original",    "Original (base)"),
        ("qwen3_causal",      "SFT-Causal"),
        ("qwen3_causal_dpo",  "DPO-Causal (Ours)"),
        ("qwen3_noncausal",   "SFT-Noncausal"),
    ]

    print()
    print("=" * 80)
    print("  COMBINED SFT + DPO RESULTS  |  Qwen3-1.7B")
    print("  DPO: chosen=COT (correct+verbose)  rejected=SFT-causal wrong chains")
    print("=" * 80)

    for ds_key, ds_label in datasets:
        print(f"\n  [{ds_label}]")
        print(f"  {'Method':<20}  {'Accuracy':<18}  {'Tokens':<14}  Steps")
        print(f"  {'-'*20}  {'-'*18}  {'-'*14}  {'-'*10}")

        base = load(os.path.join(d, f"qwen3_original_{ds_key}.jsonl"))
        for key, label in conditions:
            r = load(os.path.join(d, f"{key}_{ds_key}.jsonl"))
            b = base if key != "qwen3_original" else None
            print(row(label, r, b))

    print()
    print("=" * 80)
    print("  KEY CLAIMS (paper alignment)")
    print("=" * 80)

    for ds_key, ds_label in datasets:
        orig  = load(os.path.join(d, f"qwen3_original_{ds_key}.jsonl"))
        sft   = load(os.path.join(d, f"qwen3_causal_{ds_key}.jsonl"))
        dpo   = load(os.path.join(d, f"qwen3_causal_dpo_{ds_key}.jsonl"))
        nc    = load(os.path.join(d, f"qwen3_noncausal_{ds_key}.jsonl"))

        if not all([orig, sft, dpo]):
            continue

        print(f"\n  {ds_label}:")
        print(f"    SFT-Causal  vs Original:   acc {sft['acc']-orig['acc']:+.1f}%  "
              f"tok {sft['tok']-orig['tok']:+.0f}  steps {sft['steps']-orig['steps']:+.1f}")
        print(f"    DPO-Causal  vs SFT-Causal: acc {dpo['acc']-sft['acc']:+.1f}%  "
              f"tok {dpo['tok']-sft['tok']:+.0f}  steps {dpo['steps']-sft['steps']:+.1f}")
        if nc:
            print(f"    DPO-Causal  vs Noncausal:  acc {dpo['acc']-nc['acc']:+.1f}%  "
                  f"tok {dpo['tok']-nc['tok']:+.0f}  steps {dpo['steps']-nc['steps']:+.1f}")
    print()


if __name__ == "__main__":
    main()
