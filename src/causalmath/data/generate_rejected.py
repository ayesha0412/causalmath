#!/usr/bin/env python3
"""
Stage 2: Generate rejected chains from the SFT-causal model.
Runs questions through the model at temperature=0.8, keeps wrong answers.

Usage:
  python dpo/gen_rejected.py \
      --adapter models/sft/qwen3_fast_causal \
      --input   data/sft/fast/causal_train.jsonl \
      --output  data/dpo/rejected_chains.jsonl \
      --n_samples 3
"""
import os, sys, json, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

from causalmath.algorithm.equivalence import _extract_boxed, _normalize


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",   required=True)
    p.add_argument("--input",     default="data/sft/fast/causal_train.jsonl")
    p.add_argument("--output",    default="data/dpo/rejected_chains.jsonl")
    p.add_argument("--n_samples", type=int, default=3,
                   help="Attempts per question to find a wrong answer")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--max_questions",  type=int, default=None,
                   help="Cap number of questions (for speed)")
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
    from peft import PeftModel

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load SFT model
    print(f"Loading adapter: {args.adapter}")
    with open(f"{args.adapter}/adapter_config.json") as f:
        cfg = json.load(f)
    base_name = cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # Load training questions + ground truths
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                msgs = r.get("messages", [])
                if len(msgs) >= 2:
                    question = msgs[0]["content"]
                    # extract ground truth answer from assistant message
                    asst = msgs[1]["content"]
                    boxed = _extract_boxed(asst)
                    if boxed:
                        records.append({"question": question, "answer": boxed})

    if args.max_questions:
        records = records[:args.max_questions]

    print(f"Attempting to generate rejected chains for {len(records)} questions")
    print(f"  n_samples={args.n_samples}, temperature={args.temperature}")

    # Resume support
    done_qs = set()
    if os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            if line.strip():
                try:
                    done_qs.add(json.loads(line)["question"])
                except:
                    pass
    records = [r for r in records if r["question"] not in done_qs]
    print(f"  {len(records)} remaining ({len(done_qs)} already done)")

    found = 0
    with open(args.output, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            q  = rec["question"]
            gt = str(rec["answer"])
            prompt = build_prompt(tokenizer, q)
            enc = tokenizer(prompt, return_tensors="pt").to("cuda")

            wrong_chain = None
            for _ in range(args.n_samples):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens = args.max_new_tokens,
                        do_sample      = True,
                        temperature    = args.temperature,
                        top_p          = 0.9,
                        pad_token_id   = tokenizer.eos_token_id,
                    )
                new_toks = out[0][enc["input_ids"].shape[1]:]
                response = tokenizer.decode(new_toks, skip_special_tokens=True)
                extracted = _extract_boxed(response)

                # Keep as rejected if answer is WRONG
                if extracted and _normalize(extracted) != _normalize(gt):
                    wrong_chain = response
                    break

            if wrong_chain:
                found += 1
                fout.write(json.dumps({
                    "question":      q,
                    "answer":        gt,
                    "rejected_chain": wrong_chain,
                }, ensure_ascii=False) + "\n")
                fout.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(records)}]  wrong chains found: {found}")

    print(f"\nDone. Found {found}/{len(records)} wrong chains.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
