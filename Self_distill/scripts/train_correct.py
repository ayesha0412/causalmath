#!/usr/bin/env python3
"""
Correct SFT training for Qwen3 / DeepSeek-R1-family reasoning models.

Fixes all known issues:
  1. <think> preservation — applies chat template ONLY to the user turn,
     then appends the full assistant content (which starts with <think>)
     verbatim. Avoids tokenizer silently stripping <think> blocks.
  2. Label masking — only the assistant response tokens are trained on;
     prompt tokens are masked to -100. This is critical for small datasets
     (800 examples) where training on prompt tokens adds noise.
  3. Correct message indexing — works with 2-message [user, assistant] format
     produced by prepare_sft_correct.py.
  4. bfloat16 + LoRA — no Unsloth required, works with RTX 5080 + CUDA 12.8.

Supported models:
  Qwen/Qwen3-1.7B  (primary)
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  microsoft/Phi-4-mini-reasoning

Usage:
  python sft/train_correct.py \\
      --model Qwen/Qwen3-1.7B \\
      --data  data/sft/v2/causal_train.jsonl \\
      --output_dir models/sft/qwen3_causal \\
      --condition causal

  # Noncausal baseline:
  python sft/train_correct.py \\
      --model Qwen/Qwen3-1.7B \\
      --data  data/sft/v2/noncausal_train.jsonl \\
      --output_dir models/sft/qwen3_noncausal \\
      --condition noncausal
"""

import os
import sys

# CRITICAL: Set these BEFORE any torch/transformers imports
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Force UTF-8 stdout so log messages with special chars don't crash on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import platform
from datetime import datetime


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--data",         required=True,
                   help="Training JSONL from prepare_sft_correct.py")
    p.add_argument("--output_dir",   default="models/sft/output")
    p.add_argument("--run_name",     default="sft_run")
    p.add_argument("--condition",    default="unknown",
                   choices=["original", "noncausal", "causal", "unknown"])
    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    p.add_argument("--max_seq_len",  type=int,   default=2048)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--logging_steps",type=int,   default=10)
    p.add_argument("--save_steps",   type=int,   default=200)
    p.add_argument("--load_in_4bit", action="store_true",
                   help="QLoRA 4-bit (required for 7B+ on 16GB GPU)")
    return p.parse_args()


def log(fh, msg, print_too=True):
    fh.write(msg + "\n")
    fh.flush()
    if print_too:
        print(msg)


def build_training_text(tokenizer, user_content, assistant_content):
    """
    Build full training text with correct <think> handling.

    Strategy:
      1. Apply chat template to user-only messages → gets the BOS + role prefix
         ending with the generation prompt (which for DeepSeek ends in <think>\\n)
      2. Strip the <think>\\n suffix so we don't duplicate it
      3. Append the full assistant_content (which starts with <think>)
      4. Append EOS token

    This preserves the full <think>...</think>...answer structure in the
    training target while using the tokenizer's canonical prompt format.
    """
    user_msgs = [{"role": "user", "content": user_content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    # Qwen3 requires enable_thinking=True to emit <think> prefix
    try:
        prompt = tokenizer.apply_chat_template(user_msgs, enable_thinking=True, **kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(user_msgs, **kwargs)
    # Remove the <think>\\n generation prefix — our content already starts with it
    for suffix in ("<think>\n", "<think>"):
        if prompt.endswith(suffix):
            prompt = prompt[: -len(suffix)]
            break

    return prompt, prompt + assistant_content + tokenizer.eos_token


def tokenize_dataset(records, tokenizer, max_seq_len):
    """
    Tokenize all records with proper label masking.

    Returns a list of dicts with input_ids and labels where:
      - prompt tokens → label = -100  (not trained on)
      - response tokens → label = token_id (trained on)

    Examples where the prompt alone exceeds max_seq_len are skipped.
    """
    result = []
    skipped = 0

    for rec in records:
        msgs = rec["messages"]
        if len(msgs) != 2:
            skipped += 1
            continue

        user_content      = msgs[0]["content"]
        assistant_content = msgs[1]["content"]

        if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            skipped += 1
            continue

        prompt_text, full_text = build_training_text(
            tokenizer, user_content, assistant_content
        )

        # Tokenize prompt and full sequence separately to determine mask boundary
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)

        if len(prompt_ids) >= max_seq_len:
            skipped += 1
            continue

        # Truncate to max_seq_len
        full_ids = full_ids[:max_seq_len]

        # Labels: -100 for prompt, actual ids for response
        n_prompt = len(prompt_ids)
        labels   = [-100] * min(n_prompt, len(full_ids)) + full_ids[n_prompt:]
        labels   = labels[:max_seq_len]

        assert len(full_ids) == len(labels), \
            f"Length mismatch: {len(full_ids)} vs {len(labels)}"

        # Require at least 4 trainable tokens
        n_trainable = sum(1 for l in labels if l != -100)
        if n_trainable < 4:
            skipped += 1
            continue

        result.append({"input_ids": full_ids, "labels": labels})

    return result, skipped


class LabelMaskingCollator:
    """
    Pads input_ids (right-pad with pad_token_id) and
    labels (right-pad with -100) to the longest sequence in the batch.
    """
    def __init__(self, pad_token_id):
        self.pad_id = pad_token_id

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels    = []
        attn_mask = []
        for f in features:
            ids  = f["input_ids"]
            lbls = f["labels"]
            pad  = max_len - len(ids)
            input_ids.append(ids  + [self.pad_id] * pad)
            labels.append(   lbls + [-100]        * pad)
            attn_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
        }


def main():
    args = get_args()

    try:
        import torch
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                   TrainingArguments, Trainer, BitsAndBytesConfig)
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing: {e}")
        print("pip install transformers peft accelerate datasets torch")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.log")

    with open(log_path, "w", encoding="utf-8") as lf:
        sep = "=" * 72

        log(lf, sep)
        log(lf, "SFT TRAINING LOG (train_correct.py)")
        log(lf, sep)
        log(lf, f"  Run name   : {args.run_name}")
        log(lf, f"  Condition  : {args.condition}")
        log(lf, f"  Model      : {args.model}")
        log(lf, f"  Data       : {args.data}")
        log(lf, f"  Output     : {args.output_dir}")
        log(lf, f"  Started    : {datetime.now():%Y-%m-%d %H:%M:%S}")
        log(lf, sep)
        log(lf, "HYPERPARAMETERS")
        log(lf, f"  LoRA r/alpha    : {args.lora_r}/{args.lora_alpha}")
        log(lf, f"  Max seq len     : {args.max_seq_len}")
        log(lf, f"  Epochs          : {args.epochs}")
        log(lf, f"  Batch (eff)     : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
        log(lf, f"  LR / scheduler  : {args.lr} / cosine")
        log(lf, f"  Label masking   : YES (prompt tokens -> -100)")
        log(lf, sep)

        # ── Tokenizer ────────────────────────────────────────────────────────
        log(lf, f"Loading tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        log(lf, f"  Vocab: {tokenizer.vocab_size}  Pad: {tokenizer.pad_token!r}")

        # ── Data ─────────────────────────────────────────────────────────────
        log(lf, f"Loading data: {args.data}")
        raw_records = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))
        log(lf, f"  Raw records: {len(raw_records)}")

        log(lf, "Tokenizing with label masking...")
        tokenized, skipped = tokenize_dataset(raw_records, tokenizer, args.max_seq_len)
        log(lf, f"  Tokenized: {len(tokenized)}  Skipped: {skipped}")

        if len(tokenized) < 10:
            log(lf, "ERROR: Too few training examples after tokenization. Check data format.")
            sys.exit(1)

        # Stats
        lens = [len(t["input_ids"]) for t in tokenized]
        n_trainable = [sum(1 for l in t["labels"] if l != -100) for t in tokenized]
        log(lf, f"  Seq len  — mean: {sum(lens)/len(lens):.0f}  max: {max(lens)}")
        log(lf, f"  Trainable tokens — mean: {sum(n_trainable)/len(n_trainable):.0f}  min: {min(n_trainable)}")

        # Show a decoded sample to verify format
        sample_ids    = tokenized[0]["input_ids"]
        sample_labels = tokenized[0]["labels"]
        sample_text   = tokenizer.decode(sample_ids, skip_special_tokens=False)
        response_ids  = [i for i, l in zip(sample_ids, sample_labels) if l != -100]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
        log(lf, f"\n  [SAMPLE full text (first 400 chars)]:\n  {sample_text[:400]!r}")
        log(lf, f"  [SAMPLE response target (first 300 chars)]:\n  {response_text[:300]!r}")
        log(lf, "")

        # Sanity check: response should contain <think>
        if "<think>" not in response_text:
            log(lf, "WARNING: <think> not found in response target. Check data format.")
        if "</think>" not in response_text:
            log(lf, "WARNING: </think> not found in response target. Reasoning chain may be truncated.")

        dataset = Dataset.from_list(tokenized)

        # ── Model ─────────────────────────────────────────────────────────────
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs = dict(device_map="auto", trust_remote_code=True, attn_implementation="eager")
        if args.load_in_4bit:
            log(lf, f"Loading model (4-bit QLoRA): {args.model}")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            log(lf, f"Loading model ({dtype}): {args.model}")
            load_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.config.use_cache = False
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        vram_used = torch.cuda.memory_allocated() / 1024**3
        log(lf, f"  VRAM after model load: {vram_used:.2f} GB")

        # ── LoRA ─────────────────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            r              = args.lora_r,
            lora_alpha     = args.lora_alpha,
            lora_dropout   = args.lora_dropout,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        log(lf, f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        log(lf, sep)

        # ── Training args ─────────────────────────────────────────────────────
        steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
        warmup_steps    = max(1, int(0.05 * steps_per_epoch * args.epochs))
        use_bf16        = torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir                  = args.output_dir,
            run_name                    = args.run_name,
            num_train_epochs            = args.epochs,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            learning_rate               = args.lr,
            lr_scheduler_type           = "cosine",
            warmup_steps                = warmup_steps,
            weight_decay                = 0.01,
            max_grad_norm               = 1.0,
            bf16                        = use_bf16,
            fp16                        = not use_bf16,
            seed                        = args.seed,
            save_strategy               = "steps",
            save_steps                  = args.save_steps,
            logging_steps               = args.logging_steps,
            logging_dir                 = os.path.join(args.output_dir, "logs"),
            report_to                   = "none",
            dataloader_num_workers      = 0,
            remove_unused_columns       = False,  # keep input_ids, labels, attention_mask
        )

        collator = LabelMaskingCollator(pad_token_id=tokenizer.pad_token_id)

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = dataset,
            data_collator = collator,
        )

        log(lf, "TRAINING STARTED")
        t_start = datetime.now()
        result  = trainer.train()
        elapsed = datetime.now() - t_start
        h, rem  = divmod(int(elapsed.total_seconds()), 3600)
        m, s    = divmod(rem, 60)

        log(lf, sep)
        log(lf, "TRAINING COMPLETE")
        log(lf, f"  Duration     : {h}h {m}m {s}s")
        log(lf, f"  Train loss   : {result.training_loss:.4f}")
        log(lf, f"  Total steps  : {result.global_step}")
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        log(lf, f"  Peak VRAM    : {vram_peak:.2f} GB")

        if hasattr(trainer.state, "log_history"):
            log(lf, "\nLOSS HISTORY:")
            for entry in trainer.state.log_history:
                if "loss" in entry:
                    log(lf, f"  step={entry.get('step',0):5d}  "
                            f"loss={entry['loss']:.4f}  "
                            f"lr={entry.get('learning_rate',0):.2e}")
        log(lf, sep)

        # ── Save ─────────────────────────────────────────────────────────────
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        config = {
            "run_name":    args.run_name,
            "condition":   args.condition,
            "model":       args.model,
            "data":        args.data,
            "n_train":     len(tokenized),
            "lora_r":      args.lora_r,
            "lora_alpha":  args.lora_alpha,
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "grad_accum":  args.grad_accum,
            "lr":          args.lr,
            "max_seq_len": args.max_seq_len,
            "train_loss":  result.training_loss,
            "total_steps": result.global_step,
            "duration_s":  int(elapsed.total_seconds()),
            "label_masking": True,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as cf:
            json.dump(config, cf, indent=2)

        log(lf, f"Adapter   -> {args.output_dir}")
        log(lf, f"Log       -> {log_path}")
        log(lf, f"Config    -> {args.output_dir}/run_config.json")


if __name__ == "__main__":
    main()
