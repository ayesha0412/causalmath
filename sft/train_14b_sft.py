#!/usr/bin/env python3
"""
Fine-tune Qwen3-14B on PNS-curated causal chains (Self-Distillation Loop).

Loads 14B in 4-bit (QLoRA) — requires ~10-12GB VRAM.
Training data format: same as train_causal_fast.jsonl
  fields: question, answer, cot_response, thinking_enabled

Usage — Iteration 1 (uses existing 14B PNS data):
  python sft/prepare_distill_sft.py \
    --pns data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl \
    --output data/gsm8k/train_causal_14b_iter1.jsonl --iter 1

  python sft/train_14b_sft.py \
    --data data/gsm8k/train_causal_14b_iter1.jsonl \
    --output_dir models/distill/qwen3_14b_iter1

Usage — Iteration 2 (generate from fine-tuned 14B, then PNS, then retrain):
  python sft/gen_self_distill.py \
    --adapter models/distill/qwen3_14b_iter1 \
    --load_in_4bit \
    --output data/gsm8k/gsm8k_cot_14b_iter2.jsonl

  python expt/run_algo_o.py \
    --input  data/gsm8k/gsm8k_cot_14b_iter2.jsonl \
    --output data/gsm8k/gsm8k_pns_14b_iter2.jsonl --workers 4

  python sft/prepare_distill_sft.py \
    --pns data/gsm8k/gsm8k_pns_14b_iter2.jsonl \
    --output data/gsm8k/train_causal_14b_iter2.jsonl --iter 2

  python sft/train_14b_sft.py \
    --data data/gsm8k/train_causal_14b_iter2.jsonl \
    --output_dir models/distill/qwen3_14b_iter2
"""
import os, sys, json, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, ".")

BASE_MODEL = "Qwen/Qwen3-14B"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True,
                   help="SFT training data (from prepare_distill_sft.py)")
    p.add_argument("--output_dir",  required=True,
                   help="Where to save the LoRA adapter")
    p.add_argument("--base_model",  default=BASE_MODEL)
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--max_seq_len", type=int,   default=512)
    return p.parse_args()


def build_prompt_parts(tokenizer, question, cot_response):
    """Return (full_text, prompt_only_text) for label masking.

    Uses enable_thinking=True so training format matches eval_correct.py exactly.
    The cot_response becomes the 'thinking' content the model learns to generate.
    """
    msgs = [{"role": "user", "content": question.strip()}]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt + cot_response, prompt


def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    # ── Load data ────────────────────────────────────────────────────────────
    records = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    print(f"Training examples: {len(records)} from {args.data}")

    # ── Load 14B in 4-bit ────────────────────────────────────────────────────
    print(f"Loading {args.base_model} in 4-bit QLoRA...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # ── Prepare for QLoRA (required before adding LoRA to a 4-bit model) ────
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Build dataset with label masking ──────────────────────────────────────
    # Tokenize full text and prompt separately; mask prompt tokens to -100
    # so the model only learns to predict the cot_response (not the question).
    full_texts, prompt_texts = zip(*[
        build_prompt_parts(tokenizer, r["question"], r["cot_response"])
        for r in records
    ])

    def tokenize_with_labels(batch):
        full  = tokenizer(list(batch["full_text"]),  truncation=True,
                          max_length=args.max_seq_len, padding="max_length")
        # Tokenize prompt-only (no padding) to find the boundary
        prompts = tokenizer(list(batch["prompt_text"]), truncation=True,
                            max_length=args.max_seq_len, padding=False,
                            add_special_tokens=False)
        labels = []
        for input_ids, prompt_ids in zip(full["input_ids"], prompts["input_ids"]):
            lbl = list(input_ids)
            # Mask everything up to (and including) the last prompt token
            prompt_len = len(prompt_ids)
            for k in range(min(prompt_len, len(lbl))):
                lbl[k] = -100
            # Also mask padding tokens
            for k in range(len(lbl)):
                if input_ids[k] == tokenizer.pad_token_id:
                    lbl[k] = -100
            labels.append(lbl)
        full["labels"] = labels
        return full

    dataset = Dataset.from_dict({"full_text": list(full_texts), "prompt_text": list(prompt_texts)})
    dataset = dataset.map(tokenize_with_labels, batched=True,
                          remove_columns=["full_text", "prompt_text"])

    # ── Training ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\n{'='*70}")
    print(f"Self-Distillation: Fine-tuning Qwen3-14B on causal chains")
    print(f"  Data     : {args.data}  ({len(records)} examples)")
    print(f"  Epochs   : {args.epochs}  LR: {args.lr}  LoRA r={args.lora_r}")
    print(f"  Output   : {args.output_dir}")
    print(f"{'='*70}\n")

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved → {args.output_dir}")


if __name__ == "__main__":
    main()
