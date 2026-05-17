import json

path = 'data/results_sft/fast/qwen3_causal_gsm8k.jsonl'
recs = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]

print('=== CAUSAL GSM8K - sample outputs ===')
for i, r in enumerate(recs[:3]):
    q  = r['question'][:60]
    gt = r['ground_truth']
    ex = r['extracted_answer']
    ok = r['is_correct']
    tk = r['token_count']
    st = r['step_count']
    resp = r['model_answer'][:300]
    print(f'\n[{i+1}] Q: {q}...')
    print(f'     GT: {gt}  |  extracted: {ex}  |  correct: {ok}')
    print(f'     tokens: {tk}  steps: {st}')
    print(f'     response: {resp!r}')

# Step count distribution
from collections import Counter
step_dist = Counter(r['step_count'] for r in recs)
print(f'\nStep count distribution: {dict(sorted(step_dist.items()))}')
print(f'Avg tokens: {sum(r["token_count"] for r in recs)/len(recs):.1f}')
