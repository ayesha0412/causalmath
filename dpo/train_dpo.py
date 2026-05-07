#!/usr/bin/env python3
"""
Stage 4: DPO training on top of SFT-causal checkpoint.

chosen:   PNS causal chain (correct + concise)
rejected: SFT model's wrong generation

Starts from SFT adapter, not base model — standard SFT→DPO pipeline.

Usage:
  python dpo/train_dpo.py \
      --sft_adapter models/sft/qwen3_fast_causal \
      --data        data/dpo/dpo_pairs.jsonl \
      --output_dir  models/dpo/qwen3_causal_dpo
"""
import os, sys, json, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_adapter", required=True,
                   help="SFT adapter to start DPO from")
    p.add_argument("--data",        default="data/dpo/dpo_pairs.jsonl")
    p.add_argument("--output_dir",  default="models/dpo/qwen3_causal_dpo")
    p.add_argument("--beta",        type=float, default=0.1,
                   help="DPO beta — controls KL penalty strength")
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--max_seq_len", type=int,   default=2048)
    return p.parse_args()


def build_prompt(tokenizer, question):
    msgs = [{"role": "user", "content": question.strip()}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig

    # Load base from SFT adapter config
    with open(f"{args.sft_adapter}/adapter_config.json") as f:
        cfg = json.load(f)
    base_name = cfg["base_model_name_or_path"]

    print(f"Loading SFT model: {args.sft_adapter} (base: {base_name})")
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")

    # Load SFT adapter as starting point
    model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    model = model.merge_and_unload()  # merge SFT weights, then add DPO LoRA

    # Add fresh LoRA for DPO
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load DPO pairs
    pairs = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"DPO pairs loaded: {len(pairs)}")

    # Build dataset with prompt formatted
    dataset = Dataset.from_dict({
        "prompt":   [build_prompt(tokenizer, p["prompt"])   for p in pairs],
        "chosen":   [p["chosen"]                             for p in pairs],
        "rejected": [p["rejected"]                           for p in pairs],
    })

    os.makedirs(args.output_dir, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir             = args.output_dir,
        num_train_epochs       = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate          = args.lr,
        beta                   = args.beta,
        max_length             = args.max_seq_len,
        max_prompt_length      = 512,
        logging_steps          = 10,
        save_steps             = 100,
        save_total_limit       = 2,
        bf16                   = True,
        remove_unused_columns  = False,
    )

    trainer = DPOTrainer(
        model     = model,
        args      = dpo_config,
        train_dataset = dataset,
        tokenizer = tokenizer,
    )

    print(f"\n{'='*70}")
    print(f"DPO Training: SFT-Causal -> DPO-Causal")
    print(f"  Beta: {args.beta}  Epochs: {args.epochs}  Pairs: {len(pairs)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDPO training complete! Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
