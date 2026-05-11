#!/usr/bin/env python3
"""
Generate detailed COT chains from Qwen3-14B with batched GPU inference.

Forces explicit arithmetic at every step so PNS rollouts identify
causally necessary steps that still retain full working.
Outputs model_answer field so expt/run_algo_o.py can run PNS directly.

Usage:
  python dpo/gen_detailed_for_dpo.py --batch_size 4
"""
import os, sys, json, argparse
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
sys.path.insert(0, ".")

from algo.equivalent_ans import _extract_boxed, _normalize

# IMPORTANT: Steps must be separated by blank lines (\n\n) because parse_nodes() in
# algo/pnps_cot.py splits on '\n\n'. Each paragraph = one reasoning step.
SYSTEM_PROMPT = (
    "You are a precise math solver. "
    "Structure your solution as numbered paragraphs, with a BLANK LINE between every step. "
    "Each step must show explicit arithmetic (e.g. '48 / 2 = 24'). "
    "Never skip or combine steps. "
    "End with a final paragraph that states the answer in \\boxed{}."
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="Qwen/Qwen3-14B")
    p.add_argument("--input",      default="data/sft/fast/causal_train.jsonl")
    p.add_argument("--output",     default="data/dpo/detailed_cot_for_dpo.jsonl")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Questions per GPU batch. 4 works for 14B 4-bit on 24GB VRAM, "
                        "try 2 if OOM.")
    p.add_argument("--max_new_tokens", type=int, default=2048,
                   help="Max tokens per response. 2048 is enough for GSM8K.")
    return p.parse_args()


def build_prompt(tok, question):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    # enable_thinking=False: run Qwen3 in standard (non-thinking) mode.
    # Thinking mode ignores temperature/top_p/top_k, wraps output in <think> tags,
    # and produces one dense block that parse_nodes() cannot split into steps.
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def load_records(path):
    records = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                q   = r["messages"][0]["content"].strip()
                asst = r["messages"][1]["content"] if len(r["messages"]) > 1 else ""
                ans = _extract_boxed(asst) or ""
            else:
                q   = r["question"].strip()
                ans = str(r.get("answer", "")).strip()
            if q not in seen and ans:
                seen.add(q)
                records.append({"question": q, "answer": ans})
    return records


def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    records = load_records(args.input)
    print(f"Input questions: {len(records)}")

    # Resume support
    done = set()
    if os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            if line.strip():
                try:
                    done.add(json.loads(line)["question"].strip())
                except:
                    pass
    records = [r for r in records if r["question"].strip() not in done]
    print(f"  {len(records)} remaining  ({len(done)} already done)")

    if not records:
        print("All done!")
        return

    print(f"\nLoading {args.model} (4-bit, batch_size={args.batch_size})...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant,
        device_map="auto", trust_remote_code=True, attn_implementation="eager")
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM after load: {vram:.1f} GB")
    print(f"  Processing {len(records)} questions in batches of {args.batch_size}\n")

    correct = total = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as fout:
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start : batch_start + args.batch_size]

            prompts = [build_prompt(tok, r["question"]) for r in batch]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,   # 768 was too short; GSM8K + system prompt ~400-600 tokens
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens = args.max_new_tokens,
                    do_sample      = False,
                    pad_token_id   = tok.eos_token_id,
                )

            in_len = enc["input_ids"].shape[1]

            for j, rec in enumerate(batch):
                new_toks  = out[j][in_len:]
                response  = tok.decode(new_toks, skip_special_tokens=True).strip()

                # With enable_thinking=False, Qwen3 outputs clean step-by-step text
                # with no <think> tags. Still handle residual tags defensively.
                if "</think>" in response:
                    chain_for_pns = response.split("</think>", 1)[1].strip()
                else:
                    chain_for_pns = response

                # Verify the chain has multiple \n\n-separated steps (parse_nodes check)
                step_count = len([s for s in chain_for_pns.split("\n\n") if s.strip()])
                if step_count < 2:
                    print(f"  ⚠  Only {step_count} step(s) detected for question — "
                          "CoT may not be properly structured for PNS")

                extracted  = _extract_boxed(response)
                gt         = str(rec["answer"]).strip()
                is_correct = bool(extracted and _normalize(extracted) == _normalize(gt))

                correct += int(is_correct)
                total   += 1

                fout.write(json.dumps({
                    "question":      rec["question"],
                    "answer":        gt,
                    "full_response": response,       # full output with <think> tags
                    "model_answer":  chain_for_pns,  # clean chain — what PNS parses
                    "extracted":     extracted,
                    "is_correct":    is_correct,
                    "token_count":   int((new_toks != tok.pad_token_id).sum().item()),
                }, ensure_ascii=False) + "\n")
            fout.flush()

            done_so_far = len(done) + total
            total_qs    = len(done) + len(records)
            print(f"  [{done_so_far:4d}/{total_qs}]  "
                  f"acc={correct/total*100:.1f}%  "
                  f"batch_tok={out.shape[1]-in_len}",
                  flush=True)

    print(f"\nDone. Correct: {correct}/{total} = {correct/total*100:.1f}%")

    lines = [json.loads(l) for l in open(args.output, encoding="utf-8") if l.strip()]
    toks  = [r["token_count"] for r in lines]
    print(f"Avg tokens: {sum(toks)/len(toks):.0f}  Min: {min(toks)}  Max: {max(toks)}")


if __name__ == "__main__":
    main()
