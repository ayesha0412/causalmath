#!/usr/bin/env python3
"""
Generate detailed COT chains from Qwen3-14B with batched GPU inference.
FIXED VERSION: Comprehensive logging, error handling, and resilience.

Forces explicit arithmetic at every step so PNS rollouts identify
causally necessary steps that still retain full working.
Outputs model_answer field so expt/run_algo_o.py can run PNS directly.

Usage:
  python dpo/gen_detailed_for_dpo_fixed.py --batch_size 2
"""
import os, sys, json, argparse, traceback, time
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
    p.add_argument("--input",      default="data/dpo/gsm8k_causal_train.jsonl")
    p.add_argument("--output",     default="data/dpo/detailed_cot_for_dpo.jsonl")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Questions per GPU batch. Start with 1, try 2 if stable.")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Max tokens per response. 4096 for detailed steps.")
    return p.parse_args()


def build_prompt(tok, question):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
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

    print(f"\n[INIT] Loading records from {args.input}...")
    records = load_records(args.input)
    print(f"[INFO] Input questions: {len(records)}")

    # Resume support
    done = set()
    done_questions = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            q = r["question"].strip()
                            done.add(q)
                            done_questions[q] = r
                        except json.JSONDecodeError:
                            pass
            print(f"[RESUME] Found {len(done)} already processed questions")
        except Exception as e:
            print(f"[WARN] Could not load previous output: {e}")

    records = [r for r in records if r["question"].strip() not in done]
    print(f"[INFO] {len(records)} remaining to process ({len(done)} already done)")

    if not records:
        print("[INFO] All done!")
        return

    print(f"\n[LOAD] Loading {args.model} (4-bit, batch_size={args.batch_size})...")
    try:
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
        print(f"[LOAD] VRAM after load: {vram:.1f} GB")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        return

    # Test tokenization on first record
    print(f"[TEST] Testing tokenization on first record...")
    try:
        test_prompt = build_prompt(tok, records[0]["question"])
        test_enc = tok(test_prompt, return_tensors="pt")
        print(f"[TEST] Tokenization OK: {test_enc['input_ids'].shape[1]} tokens")
    except Exception as e:
        print(f"[ERROR] Tokenization failed: {e}")
        traceback.print_exc()
        return

    print(f"[PROCESS] Processing {len(records)} questions in batches of {args.batch_size}\n")

    correct = total = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    start_time = time.time()

    try:
        with open(args.output, "a", encoding="utf-8") as fout:
            for batch_idx, batch_start in enumerate(range(0, len(records), args.batch_size)):
                batch = records[batch_start : batch_start + args.batch_size]
                batch_num = batch_idx + 1
                total_batches = (len(records) + args.batch_size - 1) // args.batch_size

                try:
                    # Build prompts
                    prompts = [build_prompt(tok, r["question"]) for r in batch]

                    # Tokenize with error handling
                    try:
                        enc = tok(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048,
                        ).to("cuda")
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"\n[ERROR] CUDA OOM at batch {batch_num}. Try reducing batch_size.")
                            print(f"[INFO] Collected {total} records before OOM. Saving...")
                            break
                        else:
                            raise

                    in_len = enc["input_ids"].shape[1]

                    # Generate with error handling
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                **enc,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tok.eos_token_id,
                            )
                    except Exception as gen_err:
                        print(f"\n[ERROR] Generation failed at batch {batch_num}/{total_batches}: {gen_err}")
                        traceback.print_exc()
                        print(f"[INFO] Skipping this batch, continuing...")
                        continue

                    # Process outputs
                    for j, rec in enumerate(batch):
                        try:
                            new_toks  = out[j][in_len:]
                            response  = tok.decode(new_toks, skip_special_tokens=True).strip()

                            # Extract clean chain after </think> if present
                            if "</think>" in response:
                                chain_for_pns = response.split("</think>", 1)[1].strip()
                            else:
                                chain_for_pns = response

                            # Verify structure
                            step_count = len([s for s in chain_for_pns.split("\n\n") if s.strip()])
                            if step_count < 2:
                                print(f"\n[WARN] Only {step_count} step(s) for Q{len(done)+total+1}")

                            # Extract and check correctness
                            extracted  = _extract_boxed(response)
                            gt         = str(rec["answer"]).strip()
                            is_correct = bool(extracted and _normalize(extracted) == _normalize(gt))

                            correct += int(is_correct)
                            total   += 1

                            # Write output
                            output_record = {
                                "question":      rec["question"],
                                "answer":        gt,
                                "full_response": response,
                                "model_answer":  chain_for_pns,
                                "extracted":     extracted,
                                "is_correct":    is_correct,
                                "token_count":   int((new_toks != tok.pad_token_id).sum().item()),
                            }
                            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                            fout.flush()  # Explicit flush after each record

                        except Exception as item_err:
                            print(f"\n[ERROR] Failed to process item {j} in batch {batch_num}: {item_err}")
                            traceback.print_exc()
                            continue

                    # Progress update after EVERY batch (not every 10)
                    elapsed = time.time() - start_time
                    done_so_far = len(done) + total
                    total_qs    = len(done) + len(records)
                    if total > 0:
                        acc = correct / total * 100
                        rate = total / elapsed * 60  # questions per minute
                        eta_mins = (total_qs - done_so_far) / rate if rate > 0 else 0
                        print(f"[{batch_num:3d}/{total_batches:3d}] "
                              f"Done: {done_so_far:4d}/{total_qs}  "
                              f"Acc: {acc:.1f}%  "
                              f"Rate: {rate:.1f} q/min  "
                              f"ETA: {eta_mins:.0f} min  "
                              f"Tokens: {out.shape[1]-in_len}",
                              flush=True)

                except Exception as batch_err:
                    print(f"\n[ERROR] Batch {batch_num} processing failed: {batch_err}")
                    traceback.print_exc()
                    print(f"[INFO] Continuing to next batch...")
                    continue

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Stopped by user. {total} records written.")
    except Exception as e:
        print(f"\n[FATAL] Unhandled error in main loop: {e}")
        traceback.print_exc()

    print(f"\n[DONE] Completed. Written: {total} records  |  Correct: {correct}/{total} = {correct/total*100:.1f}%")

    # Print final stats
    try:
        lines = [json.loads(l) for l in open(args.output, encoding="utf-8") if l.strip()]
        if lines:
            toks  = [r["token_count"] for r in lines]
            print(f"[STATS] Avg tokens: {sum(toks)/len(toks):.0f}  Min: {min(toks)}  Max: {max(toks)}")
            print(f"[STATS] Total records in file: {len(lines)}")
    except Exception as e:
        print(f"[WARN] Could not compute final stats: {e}")


if __name__ == "__main__":
    main()
