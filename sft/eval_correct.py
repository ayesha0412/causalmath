#!/usr/bin/env python3
"""
Evaluate a fine-tuned adapter (or base model) on test datasets.

Prompt format matches train_correct.py exactly:
  apply_chat_template([user_msg], enable_thinking=True, add_generation_prompt=True)
  → ends with <think>\\n  (for Qwen3 / DeepSeek)
  → model completes from here

Usage:
  # Base model (Original baseline):
  python sft/eval_correct.py \\
      --base_model Qwen/Qwen3-1.7B \\
      --input  data/MATH-500/test.jsonl \\
      --output data/results_sft/v2/qwen3_original_math500.jsonl \\
      --dataset math500

  # Fine-tuned adapter (Causal / Noncausal):
  python sft/eval_correct.py \\
      --adapter models/sft/qwen3_causal \\
      --input   data/gsm8k/test.jsonl \\
      --output  data/results_sft/v2/qwen3_causal_gsm8k.jsonl \\
      --dataset gsm8k
"""

import os
import sys

# CRITICAL: Set BEFORE any torch/transformers imports
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

import argparse
import json
import re
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))
from algo.equivalent_ans import _fast_match, _extract_boxed, _normalize


def is_correct_local(extracted: str, ground_truth: str) -> bool:
    """Local-only equivalence check — no LLM fallback needed for SFT eval."""
    if not extracted:
        return False
    fast = _fast_match(extracted, ground_truth)
    if fast is not None:
        return fast
    # _fast_match returned None (ambiguous) — fall back to normalised string match
    return _normalize(extracted) == _normalize(ground_truth)


DATASET_MAX_TOKENS = {
    "math500":       8192,
    "gsm8k":         2048,
    "commonsenseqa": 2048,
    "aime":          16384,
}


def get_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter",    help="Path to LoRA adapter dir")
    grp.add_argument("--base_model", help="HF model ID for base-model eval")
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--dataset",  required=True,
                   choices=["math500", "gsm8k", "commonsenseqa", "aime"])
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--batch_size",     type=int, default=4)
    return p.parse_args()


def extract_answer(text, dataset):
    """
    Extract final answer from model output.
    Searches the full response (including inside <think>) using rfind so we
    get the LAST \\boxed{} — the final answer even when </think> never closed.
    """
    if dataset in ("math500", "gsm8k", "aime"):
        boxed = _extract_boxed(text)
        if boxed:
            return boxed
    if dataset == "gsm8k":
        m = re.search(r"####\s*([\d,.\-]+)", text)
        if m: return m.group(1).replace(",", "").strip()
    if dataset == "commonsenseqa":
        m = re.search(r"\*\*([A-E])\*\*", text.upper())
        if m: return m.group(1)
        m = re.search(r"answer\s+is\s+\**([A-E])\**", text, re.IGNORECASE)
        if m: return m.group(1).upper()
        matches = re.findall(r"\b([A-E])\b", text.upper())
        return matches[-1] if matches else ""
    if dataset == "aime":
        m = re.search(r"\b(\d{1,3})\b", text)
        if m: return m.group(1)
    return ""


def strip_think(text):
    """Return post-</think> content. If never closed, return full text so
    extract_answer can still find \\boxed{} inside the reasoning chain."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def count_steps(text):
    """Count double-newline-separated paragraphs in the <think> block."""
    if "<think>" in text and "</think>" in text:
        think = text.split("<think>", 1)[1].split("</think>", 1)[0]
        return len([p for p in re.split(r"\n\n+", think) if p.strip()])
    return len([p for p in re.split(r"\n\n+", text) if p.strip()])


def build_prompt(tokenizer, question):
    """
    Build the inference prompt matching train_correct.py exactly.
    Qwen3 needs enable_thinking=True to emit <think>\\n as generation prefix.
    Falls back gracefully for other model families.
    """
    msgs   = [{"role": "user", "content": question.strip()}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        prompt = tokenizer.apply_chat_template(msgs, enable_thinking=True, **kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(msgs, **kwargs)
    return prompt


def main():
    args = get_args()
    max_new_tokens = args.max_new_tokens or DATASET_MAX_TOKENS[args.dataset]

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        print(f"Missing: {e}\npip install transformers peft accelerate")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.adapter:
        cfg_path = os.path.join(args.adapter, "adapter_config.json")
        with open(cfg_path) as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg["base_model_name_or_path"]
        print(f"Loading adapter: {args.adapter}  (base: {base_name})")
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        base_name = args.base_model
        print(f"Loading base model: {base_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batch generation

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype          = dtype,
        device_map           = "auto",
        trust_remote_code    = True,
        attn_implementation  = "eager",
    )

    if args.adapter:
        model = PeftModel.from_pretrained(base_model, args.adapter)
        print(f"  Adapter loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    else:
        model = base_model

    model.eval()

    # Load test data
    print(f"Loading: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(all_records)} test records")

    # Resume support
    done = set()
    if os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            if line.strip():
                try: done.add(json.loads(line)["question"])
                except: pass

    records = [r for r in all_records
               if (r.get("question") or r.get("problem", "")) not in done]
    print(f"  {len(records)} remaining  ({len(done)} already done)")

    correct = tok_sum = step_sum = done_n = 0

    with open(args.output, "a", encoding="utf-8") as fout:
        for i in range(0, len(records), args.batch_size):
            batch     = records[i : i + args.batch_size]
            questions = [r.get("question") or r.get("problem", "") for r in batch]
            gts       = [str(r.get("answer", r.get("answerKey", ""))) for r in batch]

            prompts = [build_prompt(tokenizer, q) for q in questions]

            # Show first 3 prompts to verify format
            if done_n == 0 and i == 0:
                print(f"\n[DEBUG prompt (first 300 chars)]:\n{prompts[0][:300]!r}\n")

            enc = tokenizer(
                prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=1024
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens = max_new_tokens,
                    do_sample      = False,
                    temperature    = 1.0,
                    top_p          = 1.0,
                    pad_token_id   = tokenizer.eos_token_id,
                )

            in_len = enc["input_ids"].shape[1]
            for j, (rec, gt) in enumerate(zip(batch, gts)):
                q         = questions[j]
                new_toks  = out[j][in_len:]
                response  = tokenizer.decode(new_toks, skip_special_tokens=True)
                ans_part  = strip_think(response)
                extracted = extract_answer(response, args.dataset)
                is_correct= int(is_correct_local(extracted, gt))
                tok_cnt   = int(len(new_toks))
                step_cnt  = count_steps(response)

                correct  += is_correct
                tok_sum  += tok_cnt
                step_sum += step_cnt
                done_n   += 1

                # Debug: print first 3 responses
                if done_n <= 3:
                    print(f"\n[DEBUG response #{done_n}] ({tok_cnt} toks):")
                    print(f"  response[:300]: {response[:300]!r}")
                    print(f"  extracted: {extracted!r}  gt: {gt!r}  correct: {is_correct}")

                fout.write(json.dumps({
                    "question":         q,
                    "ground_truth":     gt,
                    "extracted_answer": extracted,
                    "model_answer":     response,
                    "is_correct":       is_correct,
                    "token_count":      tok_cnt,
                    "step_count":       step_cnt,
                    "dataset":          args.dataset,
                    "condition":        args.adapter or "original",
                    "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, ensure_ascii=False) + "\n")
                fout.flush()

            total_done = len(done) + done_n
            print(f"  [{total_done:4d}/{len(all_records)}]  "
                  f"acc={correct/done_n*100:.1f}%  "
                  f"tok={tok_sum/done_n:.0f}  steps={step_sum/done_n:.1f}",
                  flush=True)

    acc   = correct / max(done_n, 1) * 100
    tok   = tok_sum  / max(done_n, 1)
    steps = step_sum / max(done_n, 1)
    print(f"\nFinal: acc={acc:.1f}%  avg_tokens={tok:.0f}  avg_steps={steps:.1f}")
    print(f"Results → {args.output}")


if __name__ == "__main__":
    main()
