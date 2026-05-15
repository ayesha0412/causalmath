#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training — fine-tune with preference pairs.

DPO trains a model to prefer one response over another without a separate
reward model. Given (prompt, chosen, rejected), it optimizes:
  max E[log(σ(β(y_c - y_r)))]

Works with any base model, including pre-trained SFT checkpoints.

Usage:
  python sft/train_dpo.py \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --sft_adapter models/sft/deepseek_causal \
      --data data/sft/dpo_causal_train.jsonl \
      --output_dir models/sft/deepseek_dpo_causal \
      --run_name deepseek_dpo_causal
"""

import argparse
import json
import os
import sys
import platform
from datetime import datetime

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU
os.environ["FLASH_ATTENTION_SKIP_KERNEL_LAUNCH"] = "1"
os.environ["FORCE_CPU_ATTENTION"] = "1"  # Disable all attention kernels
os.environ["XFORMERS_IGNORE_UNAVAILABLE"] = "1"  # Disable xformers


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Base model ID (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    p.add_argument("--sft_adapter", default=None,
                   help="Path to SFT adapter to use as starting point (optional)")
    p.add_argument("--data", required=True,
                   help="DPO training JSONL (prompt, chosen, rejected format)")
    p.add_argument("--output_dir", default="models/sft/dpo_output")
    p.add_argument("--run_name", default="dpo_run")
    p.add_argument("--condition", default="unknown",
                   choices=["original", "noncausal", "causal", "unknown"])
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # DPO-specific
    p.add_argument("--dpo_beta", type=float, default=0.1,
                   help="Temperature for DPO (higher = more aggressive preference distinction)")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    # Training
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    return p.parse_args()


def write_log(log_fh, msg, also_print=True):
    log_fh.write(msg + "\n")
    log_fh.flush()
    if also_print:
        print(msg)


def main():
    args = get_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("pip install transformers peft trl accelerate datasets")
        sys.exit(1)

    # Import DPO trainer with fallback for version incompatibilities
    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        if "llm_blender" in str(e) or "TRANSFORMERS_CACHE" in str(e):
            print(f"WARNING: Optional dependency issue: {e}")
            print("Attempting workaround: downgrade transformers")
            print("  Run: pip install transformers==4.35.0")
            sys.exit(1)
        else:
            raise

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.log")

    with open(log_path, "w", encoding="utf-8") as log_fh:

        # ── Log header ───────────────────────────────────────────────────────
        sep = "=" * 72
        write_log(log_fh, sep)
        write_log(log_fh, "DPO TRAINING LOG")
        write_log(log_fh, sep)
        write_log(log_fh, f"  Run name      : {args.run_name}")
        write_log(log_fh, f"  Condition     : {args.condition}")
        write_log(log_fh, f"  Model         : {args.model}")
        write_log(log_fh, f"  SFT adapter   : {args.sft_adapter or '(base model only)'}")
        write_log(log_fh, f"  Training data : {args.data}")
        write_log(log_fh, f"  Output dir    : {args.output_dir}")
        write_log(log_fh, f"  Started       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_log(log_fh, sep)
        write_log(log_fh, "HYPERPARAMETERS")
        write_log(log_fh, f"  DPO beta      : {args.dpo_beta}")
        write_log(log_fh, f"  Label smooth  : {args.label_smoothing}")
        write_log(log_fh, f"  LoRA r        : {args.lora_r}")
        write_log(log_fh, f"  LoRA alpha    : {args.lora_alpha}")
        write_log(log_fh, f"  LoRA dropout  : {args.lora_dropout}")
        write_log(log_fh, f"  Max seq len   : {args.max_seq_len}")
        write_log(log_fh, f"  Epochs        : {args.epochs}")
        write_log(log_fh, f"  Batch size    : {args.batch_size} x grad_accum {args.grad_accum} = effective {args.batch_size * args.grad_accum}")
        write_log(log_fh, f"  Learning rate : {args.lr}")
        write_log(log_fh, f"  Quantization  : none (bfloat16, full-precision LoRA)")
        write_log(log_fh, f"  LR scheduler  : cosine with 5% warmup")
        write_log(log_fh, f"  Seed          : {args.seed}")
        write_log(log_fh, sep)
        write_log(log_fh, "ENVIRONMENT")
        write_log(log_fh, f"  Python        : {sys.version.split()[0]}")
        write_log(log_fh, f"  Platform      : {platform.platform()}")
        write_log(log_fh, f"  PyTorch       : {torch.__version__}")
        write_log(log_fh, f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            write_log(log_fh, f"  GPU           : {torch.cuda.get_device_name(0)}")
            write_log(log_fh, f"  VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        write_log(log_fh, sep)

        # ── Load data ────────────────────────────────────────────────────────
        write_log(log_fh, f"Loading DPO data: {args.data}")
        records = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        write_log(log_fh, f"  Total training pairs: {len(records)}")

        dataset = Dataset.from_list(records)

        # ── Load tokenizer ───────────────────────────────────────────────────
        write_log(log_fh, f"Loading tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        write_log(log_fh, f"  Vocab size: {tokenizer.vocab_size}")

        # ── Load base model ──────────────────────────────────────────────────
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        write_log(log_fh, f"Loading model in {dtype}: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Disable flash attention
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        vram_used = torch.cuda.memory_allocated() / 1024**3
        write_log(log_fh, f"  Model loaded. VRAM used: {vram_used:.2f} GB")

        # ── Load SFT adapter if provided ──────────────────────────────────────
        if args.sft_adapter:
            write_log(log_fh, f"Loading SFT adapter: {args.sft_adapter}")
            model = PeftModel.from_pretrained(model, args.sft_adapter)
            model = model.merge_and_unload()  # Merge for training stability
            write_log(log_fh, f"  SFT adapter merged with base model")

        # ── Apply LoRA for DPO training ──────────────────────────────────────
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        write_log(log_fh, f"  LoRA applied for DPO training.")
        write_log(log_fh, f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        write_log(log_fh, sep)

        # ── DPO Training ─────────────────────────────────────────────────────
        use_bf16 = torch.cuda.is_bf16_supported()

        steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
        warmup_steps = max(1, int(0.05 * steps_per_epoch * args.epochs))

        training_args = DPOConfig(
            output_dir=args.output_dir,
            run_name=args.run_name,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=args.seed,
            save_strategy="steps",
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            logging_dir=os.path.join(args.output_dir, "logs"),
            report_to="none",
            dataloader_num_workers=0,
            beta=args.dpo_beta,
            label_smoothing=args.label_smoothing,
            max_length=2048,  # TRL 1.3.0 uses max_length instead of max_prompt_length
            remove_unused_columns=False,  # Keep all columns
        )

        write_log(log_fh, "Creating DPOTrainer...")
        sys.stdout.flush()
        sys.stderr.flush()
        
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        
        write_log(log_fh, "DPOTrainer created successfully. Starting training...")
        sys.stdout.flush()
        sys.stderr.flush()

        write_log(log_fh, "DPO TRAINING STARTED")
        t_start = datetime.now()

        train_result = trainer.train()

        elapsed = datetime.now() - t_start
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m, s = divmod(rem, 60)

        # ── Log training results ─────────────────────────────────────────────
        write_log(log_fh, sep)
        write_log(log_fh, "DPO TRAINING COMPLETE")
        write_log(log_fh, f"  Duration          : {h}h {m}m {s}s")
        write_log(log_fh, f"  Train loss        : {train_result.training_loss:.4f}")
        write_log(log_fh, f"  Total steps       : {train_result.global_step}")

        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        write_log(log_fh, f"  Peak VRAM used    : {vram_peak:.2f} GB")

        # Log loss history
        if hasattr(trainer.state, "log_history"):
            write_log(log_fh, "\nLOSS HISTORY (step, loss):")
            for entry in trainer.state.log_history:
                if "loss" in entry:
                    write_log(log_fh,
                        f"  step={entry.get('step',0):5d}  loss={entry['loss']:.4f}  "
                        f"lr={entry.get('learning_rate', 0):.2e}")

        write_log(log_fh, sep)
        write_log(log_fh, f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_log(log_fh, sep)

        # ── Save ─────────────────────────────────────────────────────────────
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Save config summary as JSON
        config_summary = {
            "run_name": args.run_name,
            "condition": args.condition,
            "model": args.model,
            "sft_adapter": args.sft_adapter,
            "data": args.data,
            "n_train": len(records),
            "dpo_beta": args.dpo_beta,
            "label_smoothing": args.label_smoothing,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
            "train_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "duration_s": int(elapsed.total_seconds()),
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as cf:
            json.dump(config_summary, cf, indent=2)

        write_log(log_fh, f"Adapter saved    -> {args.output_dir}")
        write_log(log_fh, f"Training log     -> {log_path}")
        write_log(log_fh, f"Run config JSON  -> {args.output_dir}/run_config.json")


if __name__ == "__main__":
    main()
