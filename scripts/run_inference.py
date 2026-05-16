#!/usr/bin/env python3
"""
PNS-Guided Causal CoT — Inference Script
=========================================
Runs math problems through a trained Qwen3-1.7B LoRA adapter and returns
the answer with its reasoning chain.

Paper models (Qwen3-1.7B, LoRA r=16)
--------------------------------------
  Causal SFT    : models/sft/qwen3_fast_causal/    GSM-8k 77.26% | MATH-500 57.6%
  Noncausal SFT : models/sft/qwen3_fast_noncausal/ GSM-8k 83.09% | MATH-500 66.6%

Usage
-----
  # Single question:
  python inference.py --adapter models/sft/qwen3_fast_causal \
      --question "If 3 workers finish a job in 6 days, how many days for 9 workers?"

  # Interactive mode:
  python inference.py --adapter models/sft/qwen3_fast_causal

  # Compare causal vs noncausal:
  python inference.py --compare \
      --question "Janet has 24 apples, gives away 1/3, eats 2. How many remain?"
"""

import argparse
import json
import os
import sys
import time

os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve problems step by step. "
    "Each step should be on its own paragraph. "
    "End with your final answer as \\boxed{answer}."
)

ADAPTERS = {
    "causal":    "models/sft/qwen3_fast_causal",
    "noncausal": "models/sft/qwen3_fast_noncausal",
}


def load_model(adapter_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"ERROR: adapter_config.json not found at {adapter_path!r}\n"
                 f"Make sure the adapter path is correct.")

    with open(cfg_path, encoding="utf-8") as f:
        base_id = json.load(f)["base_model_name_or_path"]

    print(f"Base model : {base_id}")
    print(f"Adapter    : {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if torch.cuda.is_available():
        print(f"VRAM used  : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("Ready.\n")
    return model, tokenizer


def run_inference(model, tokenizer, question: str, max_new_tokens: int = 512) -> dict:
    import torch

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, enable_thinking=False, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0

    gen_ids  = out[0][prompt_len:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    thinking, visible = "", raw_text
    if "<think>" in raw_text and "</think>" in raw_text:
        thinking = raw_text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        visible  = raw_text.split("</think>", 1)[1].strip()
    elif "<think>" in raw_text:
        thinking = raw_text.split("<think>", 1)[1].strip()
        visible  = ""

    steps = [s.strip() for s in thinking.split("\n\n") if s.strip()] if thinking else \
            [s.strip() for s in visible.split("\n\n") if s.strip()]

    return {
        "question":    question,
        "reasoning":   thinking or visible,
        "answer":      visible,
        "token_count": len(gen_ids),
        "step_count":  len(steps),
        "elapsed_s":   round(elapsed, 2),
    }


def print_result(result: dict, label: str = ""):
    sep = "=" * 65
    if label:
        print(f"\n{sep}\n  [{label}]\n{sep}")
    print(f"Question : {result['question']}\n")
    steps = [s for s in result["reasoning"].split("\n\n") if s.strip()]
    if steps:
        print("Reasoning:")
        for i, s in enumerate(steps, 1):
            print(f"  Step {i}: {s.strip()}")
    print(f"\nAnswer   : {result['answer']}")
    print(f"Tokens   : {result['token_count']}  |  Steps: {result['step_count']}  |  {result['elapsed_s']}s")
    print(sep)


def get_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--adapter", help="Path to LoRA adapter (or: causal / noncausal)")
    grp.add_argument("--compare", action="store_true",
                     help="Compare causal vs noncausal side by side")
    p.add_argument("--question",       default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    return p.parse_args()


def main():
    args = get_args()
    adapter = args.adapter
    if adapter in ADAPTERS:
        adapter = ADAPTERS[adapter]

    question = args.question

    if args.compare:
        import torch
        if not question:
            question = input("Question: ").strip()
        for name, path in ADAPTERS.items():
            if not os.path.exists(path):
                print(f"[SKIP] {name}: not found at {path}")
                continue
            model, tok = load_model(path)
            result = run_inference(model, tok, question, args.max_new_tokens)
            print_result(result, label=name.upper())
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    if not adapter:
        print("Available adapters:")
        for k, v in ADAPTERS.items():
            status = "exists" if os.path.exists(v) else "NOT FOUND"
            print(f"  {k:12s}  {v}  [{status}]")
        adapter = input("\nEnter adapter path or name: ").strip()
        if adapter in ADAPTERS:
            adapter = ADAPTERS[adapter]

    model, tokenizer = load_model(adapter)

    if question:
        print_result(run_inference(model, tokenizer, question, args.max_new_tokens))
    else:
        print("Interactive — type a question and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                q = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if q:
                print_result(run_inference(model, tokenizer, q, args.max_new_tokens))


if __name__ == "__main__":
    main()
