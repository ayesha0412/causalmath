#!/usr/bin/env python3
"""
GRPO training on causal SFT model.
Reward = correctness (1 if answer matches GT, 0 otherwise).
No pairs needed — just questions + ground-truth answers.

TRL 1.3.0 GRPOTrainer with Qwen3-1.7B causal SFT adapter as starting point.

Usage:
  python grpo/train_grpo.py \
    --sft_adapter models/sft/qwen3_fast_causal \
    --output_dir models/grpo/qwen3_causal_grpo \
    --dataset data/gsm8k/train.jsonl
"""
import os, sys, json, re, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
sys.path.insert(0, ".")

from algo.equivalent_ans import _extract_boxed, _normalize, _fast_match


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_adapter",  default="models/sft/qwen3_fast_causal")
    p.add_argument("--output_dir",   default="models/grpo/qwen3_causal_grpo")
    p.add_argument("--dataset",      default="data/gsm8k/train.jsonl")
    p.add_argument("--lr",           type=float, default=1e-6)
    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--batch_size",   type=int,   default=2,  help="Per-device train batch")
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--num_gen",      type=int,   default=4,  help="Completions per prompt (G)")
    p.add_argument("--max_prompt",   type=int,   default=256)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_samples",  type=int,   default=3000, help="Cap training set size")
    return p.parse_args()


def extract_answer(text):
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""


def is_correct(pred, gt):
    if not pred:
        return False
    fast = _fast_match(pred, gt)
    if fast is not None:
        return bool(fast)
    return _normalize(pred) == _normalize(gt)


def correctness_reward(completions, ground_truth, **kwargs):
    """
    Reward function for GRPOTrainer.
    completions: list of strings (generated text after the prompt)
    ground_truth: list of strings (GT answers, one per prompt)
    Returns list of floats.
    """
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        ans_part = comp.split("</think>", 1)[1].strip() if "</think>" in comp else comp
        pred = extract_answer(ans_part) or extract_answer(comp)
        rewards.append(1.0 if is_correct(pred, gt) else 0.0)
    return rewards


def main():
    args = get_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, get_peft_model, LoraConfig
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset

    # ── Load SFT model ──────────────────────────────────────────────────────
    print(f"Loading SFT adapter: {args.sft_adapter}")
    with open(f"{args.sft_adapter}/adapter_config.json") as f:
        cfg = json.load(f)
    base_name = cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    # Merge SFT adapter into base so GRPO trains on top of SFT weights
    sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    model = sft_model.merge_and_unload()
    print("SFT adapter merged into base model.")

    # Add fresh GRPO LoRA on top of merged SFT model
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    records = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_samples and len(records) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(records)
        records = records[:args.max_samples]
    print(f"Training on {len(records)} examples.")

    def make_prompt(q):
        msgs = [{"role": "user", "content": q.strip()}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    dataset = Dataset.from_list([
        {"prompt": make_prompt(r["question"]), "ground_truth": str(r["answer"])}
        for r in records
    ])

    # ── GRPO config ─────────────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        num_generations=args.num_gen,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_new_tokens,
        temperature=0.8,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=correctness_reward,
        args=grpo_cfg,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    # ── Save merged model ────────────────────────────────────────────────────
    print(f"\nMerging GRPO LoRA and saving full model to {args.output_dir} ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
