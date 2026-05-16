import json, sys
from causalmath.algorithm.equivalence import _extract_boxed, _normalize

# Check PNS chains - are they actually correct?
pns_path = "data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl"
recs = [json.loads(l) for l in open(pns_path, encoding='utf-8') if l.strip()]

total = correct = wrong = no_answer = 0
wrong_examples = []

for r in recs:
    gt = str(r.get('answer', '')).strip()
    chain = r.get('metrics', {}).get('final_chain', [])
    if isinstance(chain, list):
        chain_text = " ".join(chain)
    else:
        chain_text = str(chain)

    # also check model_answer field
    model_ans = r.get('model_answer', '')
    full_text = chain_text + " " + model_ans

    extracted = _extract_boxed(full_text)
    total += 1

    if not extracted:
        no_answer += 1
    elif _normalize(extracted) == _normalize(gt):
        correct += 1
    else:
        wrong += 1
        if len(wrong_examples) < 3:
            wrong_examples.append({
                'q': r['question'][:60],
                'gt': gt,
                'extracted': extracted,
                'chain': chain_text[:150]
            })

print(f"PNS file: {pns_path}")
print(f"Total:      {total}")
print(f"Correct:    {correct} ({correct/total*100:.1f}%)")
print(f"Wrong:      {wrong}  ({wrong/total*100:.1f}%)  <-- these corrupt DPO chosen")
print(f"No answer:  {no_answer}")

if wrong_examples:
    print("\n=== Wrong PNS chain examples ===")
    for e in wrong_examples:
        print(f"  Q: {e['q']}...")
        print(f"  GT: {e['gt']}  |  Got: {e['extracted']}")
        print(f"  Chain: {e['chain']}")
        print()

# Also check PS=1 subset specifically
ps1 = [r for r in recs if r.get('metrics', {}).get('PS(chain)', 0) == 1
       and r.get('metrics', {}).get('final_chain')]
print(f"\nPS=1 subset: {len(ps1)} records")
ps1_correct = 0
for r in ps1:
    gt = str(r.get('answer', '')).strip()
    chain = r.get('metrics', {}).get('final_chain', [])
    chain_text = " ".join(chain) if isinstance(chain, list) else str(chain)
    model_ans = r.get('model_answer', '')
    extracted = _extract_boxed(chain_text + " " + model_ans)
    if extracted and _normalize(extracted) == _normalize(gt):
        ps1_correct += 1

print(f"PS=1 correct: {ps1_correct}/{len(ps1)} ({ps1_correct/len(ps1)*100:.1f}%)")
print(f"PS=1 wrong:   {len(ps1)-ps1_correct} ({(len(ps1)-ps1_correct)/len(ps1)*100:.1f}%)")
