#!/usr/bin/env python3
"""
Self-Consistency evaluation (SC@N).
Generate N answers per question with temperature sampling, take majority vote.
Wang et al. 2022 — consistently lifts math accuracy 3-5%.

Usage:
  python sft/eval_sc.py --adapter models/sft/qwen3_fast_causal --n 5
  python sft/eval_sc.py --base_model models/dpo/qwen3_causal_dpo_v3 --n 5
"""
import os, sys, json, re, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
from datetime import datetime
from collections import Counter
from causalmath.algorithm.equivalence import _extract_boxed, _normalize, _fast_match


def get_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter",    help="LoRA adapter path")
    grp.add_argument("--base_model", help="Full model path")
    p.add_argument("--input",    default="data/gsm8k/test_gsm8k_answered.jsonl")
    p.add_argument("--output",   required=True)
    p.add_argument("--dataset",  default="gsm8k", choices=["gsm8k","math500","commonsenseqa"])
    p.add_argument("--n",        type=int, default=5,   help="Samples per question")
    p.add_argument("--temp",     type=float, default=0.7)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    return p.parse_args()


def extract_answer(text, dataset):
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    if dataset == "gsm8k":
        m = re.search(r"####\s*([\d,.\-]+)", text)
        if m:
            return m.group(1).replace(",", "").strip()
    return ""


def is_correct(extracted, gt):
    if not extracted:
        return False
    fast = _fast_match(extracted, gt)
    if fast is not None:
        return bool(fast)
    return _normalize(extracted) == _normalize(gt)


def majority_vote(answers):
    """Return the most common non-empty answer."""
    valid = [a for a in answers if a]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def build_prompt(tokenizer, question):
    msgs = [{"role": "user", "content": question.strip()}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
               add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
               add_generation_prompt=True)


def make_log_path(output_path):
    stem = os.path.splitext(os.path.basename(output_path))[0]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return os.path.join("logs", f"sc_{stem}_{ts}.txt")


def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if args.adapter:
        with open(f"{args.adapter}/adapter_config.json") as f:
            cfg = json.load(f)
        base_name = cfg["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        dtype = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype,
            device_map="auto", trust_remote_code=True, attn_implementation="eager")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        label = args.adapter.split("/")[-1]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype,
            device_map="auto", trust_remote_code=True, attn_implementation="eager")
        label = args.base_model.split("/")[-1]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    log_path = make_log_path(args.output)
    os.makedirs("logs", exist_ok=True)

    header = (
        f"SC Eval Log\n"
        f"  Model   : {label}\n"
        f"  Input   : {args.input}\n"
        f"  Output  : {args.output}\n"
        f"  Dataset : {args.dataset}\n"
        f"  SC@N    : {args.n}  temp={args.temp}\n"
        f"  Total Q : {len(records)}\n"
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*60}\n"
    )
    print(header, end="")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(header)

    print(f"Loaded {len(records)} questions | SC@{args.n} | temp={args.temp}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    correct_sc = correct_greedy = total = 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        for i, rec in enumerate(records):
            question = rec.get("question", rec.get("problem", ""))
            gt = str(rec.get("answer", rec.get("ground_truth", "")))
            prompt = build_prompt(tokenizer, question)
            enc = tokenizer(prompt, return_tensors="pt").to("cuda")

            all_answers = []
            all_responses = []

            # Generate N samples
            for _ in range(args.n):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temp,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_toks = out[0][enc["input_ids"].shape[1]:]
                response = tokenizer.decode(new_toks, skip_special_tokens=True)
                ans_part = response.split("</think>", 1)[1].strip() if "</think>" in response else response
                extracted = extract_answer(ans_part, args.dataset) or extract_answer(response, args.dataset)
                all_answers.append(extracted)
                all_responses.append(response)

            # Majority vote
            voted = majority_vote(all_answers)
            greedy_ans = all_answers[0]  # first sample as proxy for greedy

            ok_sc = int(is_correct(voted, gt))
            ok_greedy = int(is_correct(greedy_ans, gt))
            correct_sc += ok_sc
            correct_greedy += ok_greedy
            total += 1

            f_out.write(json.dumps({
                "question":    question,
                "ground_truth": gt,
                "voted_answer": voted,
                "all_answers": all_answers,
                "is_correct_sc": ok_sc,
                "is_correct_greedy": ok_greedy,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False) + "\n")
            f_out.flush()

            if (i + 1) % 20 == 0 or i == 0:
                line = (f"  [{i+1:4d}/{len(records)}]  "
                        f"SC@{args.n}={correct_sc/total*100:.1f}%  "
                        f"greedy={correct_greedy/total*100:.1f}%")
                print(line)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(line + "\n")

    summary = (
        f"\n{'='*60}\n"
        f"FINAL  SC@{args.n}: {correct_sc/total*100:.2f}%   "
        f"greedy: {correct_greedy/total*100:.2f}%\n"
        f"Total questions: {total}\n"
        f"Correct SC@{args.n}: {correct_sc}  |  Correct greedy: {correct_greedy}\n"
        f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*60}\n"
        f"Results  -> {args.output}\n"
        f"Log      -> {log_path}\n"
    )
    print(summary)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(summary)


if __name__ == "__main__":
    main()
