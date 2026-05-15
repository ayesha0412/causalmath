#!/usr/bin/env python3
"""
Self-Distillation Loop — Step 1: CoT Generation.

Generates reasoning chains from the current best 1.7B model on GSM8K train
questions. Output goes directly into run_algo_o.py (PNS pipeline) which
expects: question, answer, model_answer fields.

Loop:
  Iter 1: models/sft/qwen3_fast_causal         → generate → PNS → retrain
  Iter 2: models/distill/qwen3_distill_iter1    → generate → PNS → retrain
  Iter 3: models/distill/qwen3_distill_iter2    → generate → PNS → retrain

Usage:
  python sft/gen_self_distill.py \
    --adapter models/sft/qwen3_fast_causal \
    --output data/gsm8k/gsm8k_cot_distill_iter1.jsonl

  python sft/gen_self_distill.py \
    --base_model models/distill/qwen3_distill_iter1 \
    --output data/gsm8k/gsm8k_cot_distill_iter2.jsonl
"""
import os, sys, json, re, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
sys.path.insert(0, ".")

from datetime import datetime
from algo.equivalent_ans import _extract_boxed, _normalize, _fast_match


def get_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter",    help="SFT LoRA adapter path")
    grp.add_argument("--base_model", help="Full merged model path (HuggingFace ID or local)")
    p.add_argument("--input",      default="data/gsm8k/train.jsonl")
    p.add_argument("--output",     required=True)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_samples",    type=int, default=0, help="0 = all")
    p.add_argument("--only_correct",   action="store_true",
                   help="Only save records where model answer is correct")
    p.add_argument("--load_in_4bit",   action="store_true",
                   help="4-bit quantization (required for 14B on 16GB VRAM)")
    p.add_argument("--detailed",       action="store_true",
                   help="Use detailed markdown prompt matching detailed_cot_for_dpo.jsonl format")
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


SYSTEM_PROMPT = (
    "You are a precise math problem solver. "
    "Show every intermediate step explicitly and in full detail. "
    "Structure your solution as numbered steps, where each step performs exactly one operation. "
    "Show all arithmetic calculations, unit conversions, and logical deductions separately. "
    "End your solution with the final answer in \\boxed{}."
)

# Matches the detailed_cot_for_dpo.jsonl format — markdown sections with bullet points
# and LaTeX equations, separated by \n\n for clean PNS node splitting.
DETAILED_SYSTEM_PROMPT = (
    "You are a precise math problem solver.\n"
    "Structure your solution using named ### sections for each logical part of the problem.\n"
    "Under each section:\n"
    "  - Use bullet points (- ) for every sub-step\n"
    "  - Show every arithmetic operation explicitly (e.g. 210 ÷ 70 = 3)\n"
    "  - Display all equations in $$ LaTeX $$ blocks\n"
    "  - State the result of each section in bold at the end\n"
    "Separate each section with a blank line so steps are clearly delimited.\n"
    "End with a ### Final Answer section containing $$\\boxed{answer}$$.\n"
    "Never skip intermediate steps or compress multiple operations into one line."
)


def build_prompt(tokenizer, question, detailed=False):
    msgs = [
        {"role": "system", "content": DETAILED_SYSTEM_PROMPT if detailed else SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def extract_chain(full_response):
    """Pull out the reasoning chain (between <think> and </think> if present)."""
    if "<think>" in full_response and "</think>" in full_response:
        chain = full_response.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    elif "</think>" in full_response:
        chain = full_response.split("</think>", 1)[0].strip()
    else:
        chain = full_response.strip()
    return chain


def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # ── Load model ──────────────────────────────────────────────────────────
    from transformers import BitsAndBytesConfig
    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def load_base(name):
        kw = dict(device_map="auto", trust_remote_code=True, attn_implementation="eager")
        if quant_cfg:
            kw["quantization_config"] = quant_cfg
        else:
            kw["torch_dtype"] = torch.bfloat16
        return AutoModelForCausalLM.from_pretrained(name, **kw)

    if args.adapter:
        with open(f"{args.adapter}/adapter_config.json") as f:
            cfg = json.load(f)
        base_name = cfg["base_model_name_or_path"]
        print(f"Loading adapter: {args.adapter}  (base: {base_name})"
              + ("  [4-bit]" if args.load_in_4bit else ""))
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        base = load_base(base_name)
        model = PeftModel.from_pretrained(base, args.adapter)
        label = args.adapter.split("/")[-1]
    else:
        print(f"Loading full model: {args.base_model}"
              + ("  [4-bit]" if args.load_in_4bit else ""))
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = load_base(args.base_model)
        label = args.base_model.split("/")[-1]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    # ── Load dataset ─────────────────────────────────────────────────────────
    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.max_samples and len(records) > args.max_samples:
        import random; random.seed(42); random.shuffle(records)
        records = records[:args.max_samples]

    # Resume: skip already done questions
    done = set()
    if os.path.exists(args.output):
        for l in open(args.output, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["question"].strip()[:100])
                except:
                    pass
    records = [r for r in records if r["question"].strip()[:100] not in done]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Model   : {label}")
    print(f"Input   : {args.input}  ({len(records)} remaining, {len(done)} done)")
    print(f"Output  : {args.output}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    correct = saved = total = 0

    with open(args.output, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            question = rec["question"].strip()
            gt = str(rec["answer"]).strip()

            prompt = build_prompt(tokenizer, question, detailed=args.detailed)
            enc = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_toks = out[0][enc["input_ids"].shape[1]:]
            full_response = tokenizer.decode(new_toks, skip_special_tokens=True)

            # Extract answer from post-</think> section
            ans_part = full_response.split("</think>", 1)[1].strip() if "</think>" in full_response else full_response
            pred = extract_answer(ans_part) or extract_answer(full_response)
            correct_flag = is_correct(pred, gt)

            correct += int(correct_flag)
            total += 1

            # model_answer = chain only (PNS splits this on \n\n into steps)
            # Keep full response so PNS can see the whole reasoning
            chain = extract_chain(full_response)

            entry = {
                "question":     question,
                "answer":       gt,
                "model_answer": chain,         # PNS reads this field
                "is_correct":   int(correct_flag),
                "token_count":  int(len(new_toks)),
                "source_model": label,
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            if not args.only_correct or correct_flag:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()
                saved += 1

            if total % 50 == 0 or total == 1:
                print(f"  [{total+len(done):5d}/{len(done)+len(records)}]  "
                      f"acc={correct/total*100:.1f}%  saved={saved}",
                      flush=True)

    print(f"\n{'='*60}")
    print(f"Done. Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Saved {saved} records → {args.output}")
    print(f"  (use only_correct records for PNS: {correct} chains)")


if __name__ == "__main__":
    main()
