#!/usr/bin/env python3
"""
Self-Distillation Loop — Single End-to-End Script.

Run this file once; it executes all iterations sequentially and compares
the final results against the RQ1 baselines.

Data flywheel:
  Iter 1  seed  — PNS-prune top-100 hardest detailed_cot_for_dpo chains
                  (threshold=0.3, relaxed), fine-tune Qwen3-14B, eval on GSM8K test.
  Iter 2  boost — Use pre-computed gsm8k_pns_qwen3-14b-thinking-v2.jsonl
                  (1320 records, no generation or PNS needed), combine with
                  iter 1 data for cumulative retraining, eval.

Comparison: accuracy + avg tokens + compression ratio vs RQ1 baselines.

Prereq: Ollama running on port 11436 with qwen3:14b (for iter 1 PNS rollouts).
        Run: start_ollama_gpu.bat (sets OLLAMA_NUM_PARALLEL=4 on port 11436)

Usage:
  # Full 2-iteration loop:
  python run_self_distill_detailed.py

  # Run a specific iteration only (skips completed ones automatically):
  python run_self_distill_detailed.py --start_iter 2

  # Skip individual sub-steps (when resuming a crashed run):
  python run_self_distill_detailed.py --start_iter 1 --skip_pns --skip_train

  # Dry run — print what would happen without executing:
  python run_self_distill_detailed.py --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# Force UTF-8 output on Windows terminals (avoids cp1252 encoding errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ─────────────────────────────────────────────────────────────

PYTHON      = r"C:/Users/ayesha/anaconda3/envs/dpo_env/python.exe"
BASE_MODEL  = "Qwen/Qwen3-14B"           # model to fine-tune
MAX_ITER    = 2                          # total self-distillation iterations

# Iter 1 seed — top 100 hardest records (18.7 avg steps) from detailed_cot_for_dpo.jsonl
# Selected to maximise PNS signal: harder questions → more causally necessary steps
SEED_COT    = "data/self_distill/seed_hard100.jsonl"

# Iter 2 uses pre-computed PNS output — skip generation and PNS entirely.
# gsm8k_pns_qwen3-14b-thinking-v2.jsonl has 1320 records (1133 pass PS=1+PN>=0.1).
EXISTING_ITER_PNS = {
    2: "data/self_distill/iter2/pns.jsonl",  # pre-populated from gsm8k_pns_qwen3-14b-thinking-v2.jsonl
}

# Questions for future self-generation iterations (not used in 2-iter plan)
TRAIN_JSONL = "data/gsm8k/train.jsonl"
N_GEN       = 100                        # new questions to generate per iteration (if needed)

# Evaluation target — 500 questions for speed (~20-30 min on 14B vs 4 hrs for full 1319)
TEST_JSONL  = "data/gsm8k/test_500.jsonl"

# Output dirs
DISTILL_DIR  = "data/self_distill"
MODELS_DIR   = "models/self_distill"
RESULTS_DIR  = "data/results_sft/self_distill"

# PNS settings — relaxed to preserve detail
PNS_THRESHOLD = "0.3"   # default 0.5 is too aggressive; 0.3 keeps more steps
PNS_ROLLOUTS  = "3"     # 3 rollouts is faster, still reliable for GSM8K
PNS_WORKERS   = "4"

# SFT filter settings — also relaxed
MIN_PN    = "0.1"
MIN_STEPS = "1"

# Training hyperparams for 14B QLoRA
EPOCHS      = "2"
BATCH_SIZE  = "1"
GRAD_ACCUM  = "8"
LR          = "2e-4"
LORA_R      = "16"
LORA_ALPHA  = "32"
MAX_SEQ_LEN = "2048"    # detailed chains are long — must be higher than default 1024

STATE_FILE  = os.path.join(DISTILL_DIR, "state.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg, char="="):
    w = 68
    print(f"\n{char*w}")
    print(f"  {msg}")
    print(f"{char*w}")


def section(msg):
    print(f"\n{'-'*60}")
    print(f"  {msg}")
    print(f"{'-'*60}")


def run(cmd, dry=False):
    """Run a subprocess command, printing it first."""
    display = " ".join(str(c) for c in cmd)
    print(f"\n  $ {display}")
    if dry:
        print("  [DRY RUN — skipped]")
        return
    subprocess.run([str(c) for c in cmd], check=True)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"done_iters": [], "used_questions": []}


def save_state(state):
    os.makedirs(DISTILL_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Per-iteration paths ───────────────────────────────────────────────────────

def paths(n):
    """Return all file paths for iteration n."""
    d = os.path.join(DISTILL_DIR, f"iter{n}")
    return {
        "dir":         d,
        "cot":         os.path.join(d, "cot.jsonl"),        # CoT before PNS
        "pns":         os.path.join(d, "pns.jsonl"),        # PNS output
        "train":       os.path.join(d, "train.jsonl"),      # SFT format for this iter
        "cumul":       os.path.join(DISTILL_DIR, f"cumulative_iter{n}.jsonl"),
        "adapter":     os.path.join(MODELS_DIR, f"iter{n}"),
        "eval_result": os.path.join(RESULTS_DIR, f"iter{n}_gsm8k.jsonl"),
    }


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_pns(input_jsonl, pns_out, dry=False):
    """Run PNS pruning with relaxed threshold."""
    section(f"PNS pruning  (threshold={PNS_THRESHOLD}, rollouts={PNS_ROLLOUTS})")
    existing = count_lines(pns_out)
    run([
        PYTHON, "expt/run_algo_o.py",
        "--input_file",      input_jsonl,
        "--output_file",     pns_out,
        "--threshold",       PNS_THRESHOLD,
        "--rollouts",        PNS_ROLLOUTS,
        "--workers",         PNS_WORKERS,
        "--lines_processed", str(existing),
        "--append",
    ], dry=dry)


def stage_prepare(pns_jsonl, train_out, iter_n, dry=False):
    """Convert PNS output → SFT cot_response format."""
    section("Preparing SFT training data")
    run([
        PYTHON, "sft/prepare_distill_sft.py",
        "--pns",       pns_jsonl,
        "--output",    train_out,
        "--min_pn",    MIN_PN,
        "--min_steps", MIN_STEPS,
        "--iter",      str(iter_n),
    ], dry=dry)


def stage_cumulative(iter_n, cumul_out, dry=False):
    """Combine all iterations' SFT data into one training file."""
    section(f"Building cumulative training set (iters 1–{iter_n})")
    if dry:
        print(f"  [DRY RUN] would merge iter 1..{iter_n} → {cumul_out}")
        return
    all_recs = []
    for i in range(1, iter_n + 1):
        p = paths(i)["train"]
        if os.path.exists(p):
            recs = load_jsonl(p)
            all_recs.extend(recs)
            print(f"    iter{i}: {len(recs)} records")
        else:
            print(f"    iter{i}: MISSING {p}")
    import random
    random.Random(42).shuffle(all_recs)
    save_jsonl(all_recs, cumul_out)
    print(f"  Total: {len(all_recs)} records → {cumul_out}")


def stage_train(data_jsonl, adapter_out, iter_n, dry=False):
    """Fine-tune Qwen3-14B with QLoRA."""
    section(f"Training Qwen3-14B  (iter {iter_n})")
    run([
        PYTHON, "sft/train_14b_sft.py",
        "--data",        data_jsonl,
        "--output_dir",  adapter_out,
        "--base_model",  BASE_MODEL,
        "--epochs",      EPOCHS,
        "--batch_size",  BATCH_SIZE,
        "--grad_accum",  GRAD_ACCUM,
        "--lr",          LR,
        "--lora_r",      LORA_R,
        "--lora_alpha",  LORA_ALPHA,
        "--max_seq_len", MAX_SEQ_LEN,
    ], dry=dry)


def stage_eval(adapter_dir, eval_out, dry=False):
    """Evaluate fine-tuned adapter on GSM8K test set.

    Uses --load_in_4bit for 14B (16GB VRAM), and --no_thinking because
    train_14b_sft.py trains with enable_thinking=True whose output is the
    cot_response — the model must still answer with thinking enabled at eval.
    """
    section("Evaluating on GSM8K test set")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run([
        PYTHON, "sft/eval_correct.py",
        "--adapter",     adapter_dir,
        "--input",       TEST_JSONL,
        "--output",      eval_out,
        "--dataset",     "gsm8k",
        "--batch_size",  "1",      # 14B: batch=1 is safest on 16GB with 4-bit
        "--load_in_4bit",          # required for 14B on 16GB VRAM
    ], dry=dry)


def stage_generate(adapter_dir, questions_jsonl, cot_out, dry=False):
    """Generate detailed CoT from the fine-tuned model on new questions."""
    section("Generating detailed CoT from fine-tuned model")
    run([
        PYTHON, "sft/gen_self_distill.py",
        "--adapter",        adapter_dir,
        "--input",          questions_jsonl,
        "--output",         cot_out,
        "--max_new_tokens", "2048",
        "--load_in_4bit",
        "--detailed",       # use markdown section prompt matching seed data format
    ], dry=dry)


def stage_compare(dry=False):
    """Print comparison table against RQ1 baselines."""
    section("Comparison vs RQ1 baselines")
    if dry:
        print("  [DRY RUN] would print comparison table")
        return
    run([PYTHON, "sft/eval_distill_vs_rq1.py"])


# ── Question sampling (for iter 2+) ──────────────────────────────────────────

def sample_new_questions(iter_n, n, state, dry=False):
    """
    Sample N questions from train.jsonl not used in prior iterations.
    Saves to data/self_distill/iter{n}/questions.jsonl.
    Returns path to the questions file.
    """
    import random
    p = paths(iter_n)
    questions_out = os.path.join(p["dir"], "questions.jsonl")

    if dry:
        print(f"  [DRY RUN] would sample {n} new questions → {questions_out}")
        return questions_out

    used = set(state.get("used_questions", []))
    all_q = load_jsonl(TRAIN_JSONL)
    available = [r for r in all_q if r["question"].strip() not in used]
    rng = random.Random(42 + iter_n * 100)
    sampled = rng.sample(available, min(n, len(available)))
    print(f"  Available: {len(available):,}  Sampling: {len(sampled)}")

    os.makedirs(p["dir"], exist_ok=True)
    save_jsonl(sampled, questions_out)
    print(f"  Saved → {questions_out}")
    return questions_out


# ── Accuracy printer ─────────────────────────────────────────────────────────

def print_accuracy(eval_jsonl, label):
    if not os.path.exists(eval_jsonl):
        print(f"  {label}: (no results file yet)")
        return
    recs    = load_jsonl(eval_jsonl)
    correct = sum(1 for r in recs if r.get("is_correct", False))
    total   = len(recs)
    avg_tok = sum(r.get("token_count", 0) for r in recs) / max(total, 1)
    print(f"  {label:<40} {correct}/{total} = {100*correct/max(total,1):.1f}%  "
          f"avg_tok={avg_tok:.0f}")


# ── Main iteration runner ────────────────────────────────────────────────────

def run_iteration(n, args, state):
    dry = args.dry_run
    p   = paths(n)
    os.makedirs(p["dir"], exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    iter_label = {
        1: "seed (top-100 hardest, threshold=0.3 PNS)",
        2: "pre-computed PNS boost (gsm8k_pns_qwen3-14b-thinking-v2)",
    }.get(n, f"self-generated CoT iter {n}")
    banner(f"ITERATION {n}  —  {iter_label}", char="#")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ── 1. Obtain CoT / PNS input ─────────────────────────────────────────────
    if n == 1:
        # Seed: use top-100 hardest detailed CoT directly — no generation needed
        cot_src = SEED_COT
        section("Iter 1: using seed_hard100.jsonl (top-100 hardest questions)")
        print(f"  Records in seed file: {count_lines(SEED_COT)}")
    elif n in EXISTING_ITER_PNS:
        # Use pre-computed PNS output — skip both generation and PNS
        existing_pns = EXISTING_ITER_PNS[n]
        section(f"Iter {n}: using pre-computed PNS data (no generation/PNS needed)")
        print(f"  Source: {existing_pns}  ({count_lines(existing_pns)} records)")
        # Jump directly to prepare step using pre-computed file
        stage_prepare(existing_pns, p["train"], n, dry=dry)
        stage_cumulative(n, p["cumul"], dry=dry)
        if not args.skip_train:
            stage_train(p["cumul"], p["adapter"], n, dry=dry)
        else:
            section("Skipping training (--skip_train)")
        if not args.skip_eval:
            stage_eval(p["adapter"], p["eval_result"], dry=dry)
            print_accuracy(p["eval_result"], f"Iter {n} (self-distill)")
        else:
            section("Skipping eval (--skip_eval)")
        if not dry:
            state.setdefault("done_iters", [])
            if n not in state["done_iters"]:
                state["done_iters"].append(n)
            state[f"iter{n}"] = {
                "timestamp":    datetime.now().isoformat(),
                "pns_records":  count_lines(existing_pns),
                "train_records": count_lines(p["train"]),
                "cumul_records": count_lines(p["cumul"]),
            }
            save_state(state)
        banner(f"ITERATION {n} COMPLETE", char="#")
        return
    else:
        # Future iters: generate from previous iteration's fine-tuned model
        cot_src = p["cot"]
        if not args.skip_gen:
            prev_adapter = paths(n - 1)["adapter"]
            questions_f  = sample_new_questions(n, N_GEN, state, dry=dry)
            stage_generate(prev_adapter, questions_f, cot_src, dry=dry)
        else:
            section("Skipping CoT generation (--skip_gen)")

    # ── 2. PNS pruning ────────────────────────────────────────────────────────
    if not args.skip_pns:
        stage_pns(cot_src, p["pns"], dry=dry)
    else:
        section("Skipping PNS (--skip_pns)")
        print(f"  Using existing: {p['pns']}  ({count_lines(p['pns'])} records)")

    # ── 3. Prepare SFT data for this iteration ───────────────────────────────
    stage_prepare(p["pns"], p["train"], n, dry=dry)

    # ── 4. Build cumulative training set ─────────────────────────────────────
    stage_cumulative(n, p["cumul"], dry=dry)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    if not args.skip_train:
        stage_train(p["cumul"], p["adapter"], n, dry=dry)
    else:
        section("Skipping training (--skip_train)")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    if not args.skip_eval:
        stage_eval(p["adapter"], p["eval_result"], dry=dry)
        print_accuracy(p["eval_result"], f"Iter {n} (self-distill)")
    else:
        section("Skipping eval (--skip_eval)")

    # ── Update state ──────────────────────────────────────────────────────────
    if not dry:
        # Track questions used so iter 2+ don't repeat them
        if n > 1:
            q_file = os.path.join(p["dir"], "questions.jsonl")
            if os.path.exists(q_file):
                new_qs = [r["question"].strip() for r in load_jsonl(q_file)]
                state["used_questions"] = list(
                    set(state.get("used_questions", [])) | set(new_qs)
                )

        state.setdefault("done_iters", [])
        if n not in state["done_iters"]:
            state["done_iters"].append(n)
        state[f"iter{n}"] = {
            "timestamp": datetime.now().isoformat(),
            "pns_records": count_lines(p["pns"]),
            "train_records": count_lines(p["train"]),
            "cumul_records": count_lines(p["cumul"]),
        }
        save_state(state)

    banner(f"ITERATION {n} COMPLETE", char="#")


# ── Final comparison table ────────────────────────────────────────────────────

def print_final_table():
    banner("FINAL RESULTS — Self-Distillation vs RQ1 Baselines")

    # RQ1 baseline result files
    baselines = {
        "Original 1.7B (base)":    "data/results_sft/fast/qwen3_original_gsm8k.jsonl",
        "Noncausal SFT 1.7B":      "data/results_sft/fast/qwen3_noncausal_gsm8k.jsonl",
        "Causal SFT 1.7B  [RQ1]":  "data/results_sft/fast/qwen3_causal_gsm8k.jsonl",
        "Causal DPO 1.7B":         "data/results_sft/fast/qwen3_causal_dpo_v4_gsm8k.jsonl",
    }

    # Collect all result rows
    rows = []
    for label, path in baselines.items():
        rows.append(("RQ1", label, path))
    for n in range(1, MAX_ITER + 1):
        p = paths(n)
        rows.append(("Distill", f"14B Self-Distill Iter {n}", p["eval_result"]))

    # Print table
    print(f"\n  {'Condition':<42} {'Acc %':>7} {'Avg Tok':>9} {'Steps':>7}")
    print(f"  {'─'*42} {'─'*7} {'─'*9} {'─'*7}")

    noncausal_tok = None
    for group, label, path in rows:
        if group == "Distill":
            print()  # blank line between groups

        if not os.path.exists(path):
            print(f"  {label:<42} {'(missing)':>25}")
            continue

        recs    = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        correct = sum(1 for r in recs if r.get("is_correct", False))
        total   = len(recs)
        acc     = 100 * correct / max(total, 1)

        tok_key  = next((k for k in ["token_count", "tokens"] if k in recs[0]), None)
        step_key = next((k for k in ["step_count",  "steps"]  if k in recs[0]), None)

        avg_tok  = sum(r.get(tok_key, 0) for r in recs) / max(total, 1) if tok_key else None
        avg_step = sum(r.get(step_key, 0) for r in recs) / max(total, 1) if step_key else None

        if "Noncausal" in label and avg_tok:
            noncausal_tok = avg_tok

        tok_s  = f"{avg_tok:.0f}"  if avg_tok  else "  —  "
        step_s = f"{avg_step:.1f}" if avg_step else "  —  "
        print(f"  {label:<42} {acc:>7.2f} {tok_s:>9} {step_s:>7}")

    # Token compression analysis
    if noncausal_tok:
        print(f"\n  Token compression vs Noncausal SFT baseline ({noncausal_tok:.0f} tok):")
        for group, label, path in rows:
            if not os.path.exists(path):
                continue
            recs    = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            tok_key = next((k for k in ["token_count", "tokens"] if k in recs[0]), None)
            if not tok_key:
                continue
            avg_tok = sum(r.get(tok_key, 0) for r in recs) / max(len(recs), 1)
            ratio   = avg_tok / noncausal_tok
            saved   = (1 - ratio) * 100
            print(f"    {label:<42} ×{ratio:.3f}  ({saved:+.1f}%)")

    print(f"\n  RQ1 Claim: Causal SFT achieves competitive accuracy with fewer tokens.")
    print(f"  Self-Distillation tests if PNS filtering holds without a teacher model.")
    print()


# ── Arg parsing ───────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Self-distillation loop: seed from detailed CoT, iterate, compare vs RQ1."
    )
    p.add_argument("--start_iter", type=int, default=1,
                   help="First iteration to run (default: 1). Skips already-done iters automatically.")
    p.add_argument("--end_iter",   type=int, default=MAX_ITER,
                   help=f"Last iteration to run (default: {MAX_ITER})")
    # Skip flags for resuming partial runs
    p.add_argument("--skip_gen",   action="store_true", help="Skip CoT generation (iter 2+)")
    p.add_argument("--skip_pns",   action="store_true", help="Skip PNS pruning")
    p.add_argument("--skip_train", action="store_true", help="Skip SFT training")
    p.add_argument("--skip_eval",  action="store_true", help="Skip evaluation")
    p.add_argument("--dry_run",    action="store_true", help="Print commands without running")
    p.add_argument("--results_only", action="store_true",
                   help="Print comparison table only (no training)")
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args  = get_args()
    state = load_state()

    if args.results_only:
        print_final_table()
        return

    banner(
        f"SELF-DISTILLATION LOOP  —  iters {args.start_iter}–{args.end_iter}\n"
        f"  Iter 1 seed : {SEED_COT}  (100 records, avg 18.7 steps)\n"
        f"  Iter 2 boost: pre-computed PNS (1320 records, no PNS needed)\n"
        f"  Model : {BASE_MODEL}  (4-bit QLoRA)\n"
        f"  PNS threshold={PNS_THRESHOLD}  rollouts={PNS_ROLLOUTS}  [iter1 only]\n"
        f"  Eval  : {TEST_JSONL}  (500 questions)\n"
        f"  Ollama must be running on port 11436 (qwen3:14b) for iter 1"
    )

    for n in range(args.start_iter, args.end_iter + 1):
        # Skip iterations that were already completed (resumable)
        if n in state.get("done_iters", []) and not any([
            args.skip_gen, args.skip_pns, args.skip_train, args.skip_eval
        ]):
            print(f"\n  [SKIP] iter {n} already in state.json — use --start_iter to rerun")
            continue
        run_iteration(n, args, state)

    print_final_table()
    stage_compare(dry=args.dry_run)


if __name__ == "__main__":
    main()
