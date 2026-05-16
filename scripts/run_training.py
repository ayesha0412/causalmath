#!/usr/bin/env python3
"""
PNS-Guided Causal CoT — Training Entry Point
=============================================
Fine-tunes Qwen3-1.7B with LoRA under Causal or Noncausal conditions
on GSM-8k and MATH-500 training data.

Causal condition  : trains on PNS-filtered chains (causally necessary steps only)
Noncausal condition: trains on raw Qwen3-14B chains (no PNS filtering)

Usage
-----
# Causal SFT (paper Table VI — Ours Causal):
python train.py --condition causal --model Qwen/Qwen3-1.7B \
    --data data/sft/causal_train.jsonl \
    --output_dir models/sft/qwen3_causal

# Noncausal SFT (paper Table VI — Ours Noncausal):
python train.py --condition noncausal --model Qwen/Qwen3-1.7B \
    --data data/sft/noncausal_train.jsonl \
    --output_dir models/sft/qwen3_noncausal

Checkpoints
-----------
The models that produced the paper results are saved at:
  Causal    : models/sft/qwen3_fast_causal/   (checkpoint-543, May 7)
  Noncausal : models/sft/qwen3_fast_noncausal/ (checkpoint-603, May 7)
"""

import argparse
import sys
import os

from causalmath.training.train_sft import main as _train_main


def get_args():
    p = argparse.ArgumentParser(description="Train Qwen3-1.7B SFT (Causal or Noncausal)")
    p.add_argument("--condition",   required=True, choices=["causal", "noncausal"],
                   help="Training condition")
    p.add_argument("--model",       default="Qwen/Qwen3-1.7B",
                   help="HuggingFace base model ID")
    p.add_argument("--data",        required=True,
                   help="JSONL training file with {messages: [{role, content}]}")
    p.add_argument("--output_dir",  default="models/sft/run",
                   help="Directory to save LoRA adapter")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--max_seq_len", type=int,   default=1024)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"\n{'='*60}")
    print(f"  PNS Causal CoT — SFT Training")
    print(f"  Condition  : {args.condition}")
    print(f"  Model      : {args.model}")
    print(f"  Data       : {args.data}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}  LR: {args.lr}  LoRA r={args.lora_r}")
    print(f"{'='*60}\n")

    # Build argv for the underlying training script
    sys.argv = [
        "train_sft.py",
        "--model",       args.model,
        "--data",        args.data,
        "--output_dir",  args.output_dir,
        "--condition",   args.condition,
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--grad_accum",  str(args.grad_accum),
        "--lr",          str(args.lr),
        "--lora_r",      str(args.lora_r),
        "--lora_alpha",  str(args.lora_alpha),
        "--max_seq_len", str(args.max_seq_len),
        "--seed",        str(args.seed),
    ]
    _train_main()
