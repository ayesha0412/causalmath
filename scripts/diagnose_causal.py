import json

# Wrong causal examples
path = 'data/results_sft/fast/qwen3_causal_gsm8k.jsonl'
recs = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
wrong = [r for r in recs if not r['is_correct']][:2]

print("=== WRONG causal answers ===")
for r in wrong:
    q  = r['question'][:70]
    gt = r['ground_truth']
    ex = r['extracted_answer']
    resp = r['model_answer'][:300]
    print(f"Q: {q}...")
    print(f"GT: {gt}  |  Got: {ex!r}")
    print(f"Response: {resp!r}")
    print()

# Training data sample - what did causal model learn?
print("=== CAUSAL TRAINING SAMPLE ===")
path2 = 'data/sft/fast/causal_train.jsonl'
recs2 = [json.loads(l) for l in open(path2, encoding='utf-8') if l.strip()]
s = recs2[0]
q = s['messages'][0]['content'][:60]
a = s['messages'][1]['content'][:300]
print(f"Q: {q}")
print(f"A: {a!r}")

# Token length distribution of training data
toks = [len(s['messages'][1]['content'].split()) for s in recs2]
avg = sum(toks) / len(toks)
short = sum(1 for t in toks if t < 30)
print(f"\nTraining samples: {len(toks)}")
print(f"Avg words in answer: {avg:.0f}")
print(f"Very short (<30 words): {short} ({short/len(toks)*100:.1f}%)")
