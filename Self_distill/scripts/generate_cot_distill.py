#!/usr/bin/env python3
"""
Generate CoT using a HuggingFace model (base or LoRA adapter) for self-distillation.

For iterations 1+, the fine-tuned model generates its own CoT on new questions.
The <think> content is extracted as model_answer so run_algo_o.py can PNS-prune it.

Output schema: {"question": ..., "model_answer": ..., "answer": ...}
Same schema as expt/get_response.py — compatible with expt/run_algo_o.py.

Usage:
  # Iter 1+: use fine-tuned adapter from previous iteration
  python sft/generate_cot_distill.py \
      --adapter   models/self_distill/iter_0 \
      --input     data/self_distill/iter_1/questions.jsonl \
      --output    data/self_distill/iter_1/cot.jsonl

  # Debugging with base model
  python sft/generate_cot_distill.py \
      --base_model Qwen/Qwen3-1.7B \
      --input     data/self_distill/iter_0/questions.jsonl \
      --output    data/self_distill/iter_0/cot_hf.jsonl
"""

import os
import sys
import json
import argparse

os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSM8K_SYSTEM = (
    "You are a math reasoning assistant. "
    "Solve problems step by step. "
    "Each step should be on its own paragraph. "
    "End with your final answer as \\boxed{answer}."
)

GSM8K_USER = "{question}"


def get_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter",    help="Path to LoRA adapter dir (from previous iter)")
    grp.add_argument("--base_model", help="HF model ID for base-model generation")
    p.add_argument("--input",          required=True, help="JSONL with {question, answer}")
    p.add_argument("--output",         required=True, help="Output JSONL with model_answer")
    p.add_argument("--max_new_tokens", type=int,   default=2048)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--load_in_4bit",   action="store_true")
    return p.parse_args()


def extract_model_answer(text):
    """
    Extract the reasoning chain for PNS analysis.

    The fine-tuned model puts reasoning inside <think>...</think> and outputs
    a brief boxed answer after. We use the <think> content as model_answer
    because that's what parse_nodes in pnps_cot.py needs to split into steps.

    If no <think> block exists (e.g. base model without think tokens), use
    the full output — it should already be a visible CoT.
    """
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if "<think>" in text:
        # No closing tag — take everything after the opening tag
        return text.split("<think>", 1)[1].strip()
    # No think block at all — use full visible output
    return text.strip()


def load_done(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if d.get("model_answer"):
                            done.add(d["question"].strip())
                    except Exception:
                        pass
    return done


def main():
    args = get_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Resolve base model ID
    if args.adapter:
        cfg_path = os.path.join(args.adapter, "adapter_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        base_id = cfg["base_model_name_or_path"]
        print(f"Base model : {base_id}")
        print(f"Adapter    : {args.adapter}")
    else:
        base_id = args.base_model
        print(f"Base model : {base_id}  (no adapter)")

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batch generation

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = dict(device_map="auto", trust_remote_code=True, attn_implementation="eager")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        print("Loading model (4-bit QLoRA)...")
    else:
        load_kwargs["torch_dtype"] = dtype
        print(f"Loading model ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Resume support — skip already-generated questions
    done_qs = load_done(args.output)
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                q = d.get("question", "").strip()
                if q and q not in done_qs:
                    records.append(d)

    print(f"To generate: {len(records)}  (skipped {len(done_qs)} already done)")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as f_out:
        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]

            # Build chat prompts
            prompts = []
            for rec in batch:
                msgs = [
                    {"role": "system", "content": GSM8K_SYSTEM},
                    {"role": "user",   "content": GSM8K_USER.format(question=rec["question"])},
                ]
                try:
                    p = tokenizer.apply_chat_template(
                        msgs, enable_thinking=True,
                        tokenize=False, add_generation_prompt=True,
                    )
                except TypeError:
                    p = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                prompts.append(p)

            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = enc["input_ids"].shape[1]
            for j, rec in enumerate(batch):
                gen_ids  = out[j][prompt_len:]
                raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                model_answer = extract_model_answer(raw_text)

                f_out.write(json.dumps({
                    "question":     rec["question"],
                    "model_answer": model_answer,
                    "answer":       rec["answer"],
                }, ensure_ascii=False) + "\n")
                f_out.flush()

            done = min(i + args.batch_size, len(records))
            print(f"  {done}/{len(records)}", end="\r")

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
