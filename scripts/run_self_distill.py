#!/usr/bin/env python3
"""
Self-Distillation Loop — one iteration at a time.

Data flywheel: Qwen3-14B (via Ollama) seeds iteration 0. From iteration 1
onward, the fine-tuned HF adapter generates its own CoT on new questions.
PNS pruning (always Ollama) filters to necessary steps. The model trains on
its own self-curated chains — no teacher model needed.

Pipeline per iteration:
  1. Sample N new questions from train.jsonl (not used in prior iters)
  2. Generate CoT
       iter 0  → Ollama (expt/get_response.py, Qwen3-14B)
       iter 1+ → HF adapter from previous iteration
  3. PNS pruning (expt/run_algo_o.py, Ollama rollouts)
  4. Filter PS=1, convert to <think> SFT format
  5. Build cumulative training set (all iterations combined)
  6. Train Qwen/Qwen3-1.7B with LoRA from scratch on cumulative data
  7. Evaluate on GSM8K test set

State is persisted to data/self_distill/state.json so the loop is resumable.

Usage:
  # Run from project root with Ollama on port 11436 (separate terminal)
  python sft/self_distill_loop.py --iter 0 --n_questions 500
  python sft/self_distill_loop.py --iter 1 --n_questions 500
  python sft/self_distill_loop.py --iter 2 --n_questions 500

  # Skip individual stages (e.g. CoT already done):
  python sft/self_distill_loop.py --iter 1 --skip_cot --skip_pns
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON      = "C:/Users/ayesha/anaconda3/envs/dpo_env/python.exe"
TRAIN_JSONL = "data/gsm8k/train.jsonl"
TEST_JSONL  = "data/gsm8k/test.jsonl"
BASE_DIR    = "data/self_distill"
MODELS_DIR  = "models/self_distill"
RESULTS_DIR = "data/results_sft/self_distill"
STATE_FILE  = "data/self_distill/state.json"


# ── State helpers ─────────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"iterations_complete": [], "used_questions": []}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    if desc:
        print(f"  {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    subprocess.run([str(c) for c in cmd], check=True)


# ── PNS → SFT format ─────────────────────────────────────────────────────────

def pns_to_sft(pns_record):
    """
    Convert one PNS-pruned record to the 2-message SFT training format.
    Returns None when PS(chain) != 1 or the pruned chain is empty.
    """
    metrics = pns_record.get("metrics", {})
    if metrics.get("PS(chain)", 0) != 1:
        return None
    chain = metrics.get("final_chain", [])
    if not chain:
        return None

    question = pns_record["question"].strip()
    answer   = str(pns_record.get("answer", "")).strip()

    chain_text = "\n\n".join(s.strip() for s in chain if s.strip())

    # Wrap answer in \boxed{} if not already
    if "\\boxed" not in answer:
        final_ans = f"$\\boxed{{{answer}}}$"
    else:
        final_ans = answer

    assistant_content = f"<think>\n{chain_text}\n</think>\n\n{final_ans}"

    return {
        "messages": [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── Pipeline stages ───────────────────────────────────────────────────────────

def step_generate_ollama(questions_jsonl, cot_jsonl):
    """Iter 0: generate CoT via Ollama (Qwen3-14B)."""
    run_cmd(
        [PYTHON, "expt/get_response.py",
         questions_jsonl, cot_jsonl, "--workers", "2"],
        desc="[CoT] Generating via Ollama / Qwen3-14B",
    )


def step_generate_hf(questions_jsonl, cot_jsonl, adapter_dir):
    """Iter 1+: generate CoT via fine-tuned HF adapter."""
    run_cmd(
        [PYTHON, "sft/generate_cot_distill.py",
         "--adapter",   adapter_dir,
         "--input",     questions_jsonl,
         "--output",    cot_jsonl,
         "--batch_size", "2"],
        desc=f"[CoT] Generating via HF adapter ({adapter_dir})",
    )


def step_pns(cot_jsonl, pns_jsonl):
    """Run PNS pruning. Always uses Ollama for reliable Monte-Carlo rollouts."""
    # Count existing lines so the script can resume mid-file
    existing = 0
    if os.path.exists(pns_jsonl):
        with open(pns_jsonl, encoding="utf-8") as f:
            existing = sum(1 for l in f if l.strip())

    run_cmd(
        [PYTHON, "expt/run_algo_o.py",
         "--input_file",     cot_jsonl,
         "--output_file",    pns_jsonl,
         "--batch_size",     "5",
         "--lines_processed", str(existing),
         "--rollouts",       "3",
         "--threshold",      "0.5",
         "--workers",        "2",
         "--append"],
        desc="[PNS] Running PNS pruning via Ollama",
    )


def step_train(cumul_jsonl, model_dir, iter_n, base_model, lr):
    run_cmd(
        [PYTHON, "sft/train_correct.py",
         "--model",       base_model,
         "--data",        cumul_jsonl,
         "--output_dir",  model_dir,
         "--run_name",    f"self_distill_iter{iter_n}",
         "--condition",   "causal",
         "--load_in_4bit",
         "--epochs",      "2",
         "--batch_size",  "1",
         "--grad_accum",  "8",
         "--lr",          str(lr),
         "--lora_r",      "16",
         "--lora_alpha",  "32",
         "--max_seq_len", "2048"],
        desc=f"[Train] SFT iter {iter_n} on {cumul_jsonl}",
    )


def step_eval(model_dir, iter_n, base_model):
    out = os.path.join(RESULTS_DIR, f"iter{iter_n}_gsm8k.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_cmd(
        [PYTHON, "sft/eval_correct.py",
         "--adapter",    model_dir,
         "--input",      TEST_JSONL,
         "--output",     out,
         "--dataset",    "gsm8k",
         "--load_in_4bit",
         "--batch_size", "4"],
        desc=f"[Eval] Iter {iter_n} on GSM8K test",
    )
    # Print accuracy inline
    if os.path.exists(out):
        results = load_jsonl(out)
        correct = sum(1 for r in results if r.get("is_correct", False))
        total   = len(results)
        acc     = 100 * correct / total if total else 0
        print(f"\n  [ITER {iter_n}] GSM8K accuracy: {correct}/{total} = {acc:.1f}%")
    return out


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iter",        type=int, required=True,
                   help="Iteration index (0 = seed from Ollama, 1+ = self-generated CoT)")
    p.add_argument("--n_questions", type=int, default=500,
                   help="New questions to sample this iteration")
    p.add_argument("--base_model",  default="Qwen/Qwen3-1.7B",
                   help="HF base model to fine-tune")
    p.add_argument("--seed",        type=int, default=42)
    # LR schedule across iterations — can be overridden
    p.add_argument("--lr",          type=float, default=None,
                   help="Learning rate (default: 2e-4 iter0, 1e-4 iter1, 5e-5 iter2+)")
    # Skip flags for resuming partial runs
    p.add_argument("--skip_cot",   action="store_true", help="Skip CoT generation")
    p.add_argument("--skip_pns",   action="store_true", help="Skip PNS pruning")
    p.add_argument("--skip_train", action="store_true", help="Skip SFT training")
    p.add_argument("--skip_eval",  action="store_true", help="Skip evaluation")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    n = args.iter

    # Default LR schedule: start high, decay across iterations
    if args.lr is not None:
        lr = args.lr
    else:
        lr = {0: 2e-4, 1: 1e-4}.get(n, 5e-5)

    print(f"\n{'#'*60}")
    print(f"  SELF-DISTILLATION — ITERATION {n}")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  base_model={args.base_model}  n_questions={args.n_questions}  lr={lr}")
    print(f"{'#'*60}\n")

    # ── Paths ────────────────────────────────────────────────────────
    iter_dir        = os.path.join(BASE_DIR, f"iter_{n}")
    questions_jsonl = os.path.join(iter_dir, "questions.jsonl")
    cot_jsonl       = os.path.join(iter_dir, "cot.jsonl")
    pns_jsonl       = os.path.join(iter_dir, "pns.jsonl")
    sft_jsonl       = os.path.join(iter_dir, "train.jsonl")
    cumul_jsonl     = os.path.join(BASE_DIR,  "cumulative_train.jsonl")
    model_dir       = os.path.join(MODELS_DIR, f"iter_{n}")

    os.makedirs(iter_dir,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load state ───────────────────────────────────────────────────
    state   = load_state()
    used_qs = set(state["used_questions"])

    # ── Step 1: Sample questions ─────────────────────────────────────
    print(f"[1/7] Sampling {args.n_questions} new questions from {TRAIN_JSONL}")
    all_train = load_jsonl(TRAIN_JSONL)
    available = [r for r in all_train if r["question"].strip() not in used_qs]
    rng       = random.Random(args.seed + n * 1000)
    sampled   = rng.sample(available, min(args.n_questions, len(available)))
    print(f"  Available: {len(available):,}  Sampled: {len(sampled)}")
    save_jsonl(sampled, questions_jsonl)

    # ── Step 2: Generate CoT ─────────────────────────────────────────
    print(f"\n[2/7] CoT generation")
    if not args.skip_cot:
        if n == 0:
            step_generate_ollama(questions_jsonl, cot_jsonl)
        else:
            prev_adapter = os.path.join(MODELS_DIR, f"iter_{n - 1}")
            if os.path.exists(os.path.join(prev_adapter, "adapter_config.json")):
                step_generate_hf(questions_jsonl, cot_jsonl, prev_adapter)
            else:
                print(f"  WARNING: adapter not found at {prev_adapter}, falling back to Ollama")
                step_generate_ollama(questions_jsonl, cot_jsonl)
    else:
        print(f"  [SKIP] using existing {cot_jsonl}")

    # ── Step 3: PNS pruning ──────────────────────────────────────────
    print(f"\n[3/7] PNS pruning")
    if not args.skip_pns:
        step_pns(cot_jsonl, pns_jsonl)
    else:
        print(f"  [SKIP] using existing {pns_jsonl}")

    # ── Step 4: Convert PNS → SFT format ────────────────────────────
    print(f"\n[4/7] Converting PNS output → SFT format")
    pns_records = load_jsonl(pns_jsonl)
    sft_records, n_ps1 = [], 0
    for rec in pns_records:
        r = pns_to_sft(rec)
        if r:
            sft_records.append(r)
            n_ps1 += 1
    print(f"  PNS total: {len(pns_records)}  PS=1 kept: {n_ps1}  "
          f"yield: {100*n_ps1/max(len(pns_records),1):.1f}%")
    save_jsonl(sft_records, sft_jsonl)

    # ── Step 5: Build cumulative training set ────────────────────────
    print(f"\n[5/7] Building cumulative training set (all {n+1} iteration(s))")
    all_sft = []
    for i in range(n + 1):
        path = os.path.join(BASE_DIR, f"iter_{i}", "train.jsonl")
        if os.path.exists(path):
            recs = load_jsonl(path)
            all_sft.extend(recs)
            print(f"  iter_{i}: {len(recs)} records")
        else:
            print(f"  iter_{i}: MISSING ({path})")
    rng.shuffle(all_sft)
    save_jsonl(all_sft, cumul_jsonl)
    print(f"  Total cumulative: {len(all_sft):,} records")

    # ── Step 6: Train ────────────────────────────────────────────────
    print(f"\n[6/7] Training")
    if not args.skip_train:
        step_train(cumul_jsonl, model_dir, n, args.base_model, lr)
    else:
        print(f"  [SKIP]")

    # ── Step 7: Evaluate ─────────────────────────────────────────────
    print(f"\n[7/7] Evaluation")
    if not args.skip_eval:
        step_eval(model_dir, n, args.base_model)
    else:
        print(f"  [SKIP]")

    # ── Persist state ────────────────────────────────────────────────
    new_qs = [r["question"].strip() for r in sampled]
    state["used_questions"] = list(used_qs | set(new_qs))
    state["iterations_complete"].append({
        "iter":         n,
        "n_sampled":    len(sampled),
        "n_ps1":        n_ps1,
        "n_cumulative": len(all_sft),
        "lr":           lr,
        "timestamp":    datetime.now().isoformat(),
    })
    save_state(state)

    print(f"\n{'#'*60}")
    print(f"  ITERATION {n} COMPLETE")
    print(f"  Adapter : {model_dir}")
    print(f"  Data    : {cumul_jsonl}  ({len(all_sft):,} records)")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
