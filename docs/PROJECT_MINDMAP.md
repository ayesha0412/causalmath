# Project Mindmap — PNS Causal CoT Research
*Last updated: 2026-05-12*

---

## CORE CLAIM
> "Causal Chain-of-Thought (PNS-pruned) achieves **competitive accuracy** with **fewer tokens/steps** vs standard CoT."

---

## RQ1 — Does Causal SFT match Noncausal SFT accuracy with shorter chains?

### Datasets
| Role | File |
|------|------|
| GSM8K train (SFT) | `data/gsm8k/train.jsonl` |
| PNS-scored train | `data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl` |
| Causal SFT train | `data/gsm8k/train_causal_fast.jsonl` (PS=1, PN≥0.5, steps≥2) |
| Noncausal SFT train | `data/gsm8k/train_noncausal_fast.jsonl` |
| GSM8K test (full) | `data/gsm8k/test.jsonl` (1319 questions) |
| MATH-500 test | `data/math500/test.jsonl` |

### Models Trained
| Model | Script | Saved To |
|-------|--------|----------|
| Noncausal SFT 1.7B | `sft/train_fast.py` | `models/sft/qwen3_fast_noncausal` |
| Causal SFT 1.7B | `sft/train_fast.py` | `models/sft/qwen3_fast_causal` |

### Eval Scripts
- `sft/eval_correct.py` — greedy decode, checks `is_correct`, writes `token_count`
- `sft/eval_sc.py` — SC@N majority vote (N=5, temp=0.7), writes both `is_correct_sc` + `is_correct_greedy`

### Results (GSM8K 1319q test)
| Model | Acc | Avg Tokens |
|-------|-----|-----------|
| Original Qwen3-1.7B | 66.6% | — |
| Noncausal SFT 1.7B | **83.1%** | 220 |
| Causal SFT 1.7B | 77.3% | **168** |
| Causal DPO v4 | 75.1% | 172 |
| Causal SC@5 (19q preview) | ~94.7% | — |

**Key finding:** Causal SFT uses 24% fewer tokens (168 vs 220) at −5.8pp accuracy.
Token compression ratio: 0.764×. Step compression ratio: ~0.7× (depends on PNS run).

### Result Files
| Model | GSM8K | MATH-500 |
|-------|-------|---------|
| Original | `data/results_sft/fast/qwen3_original_gsm8k.jsonl` | `qwen3_original_math500.jsonl` |
| Noncausal | `data/results_sft/fast/qwen3_noncausal_gsm8k.jsonl` | `qwen3_noncausal_math500.jsonl` |
| Causal SFT | `data/results_sft/fast/qwen3_causal_gsm8k.jsonl` | `qwen3_causal_math500.jsonl` |
| DPO v4 | `data/results_sft/fast/qwen3_causal_dpo_v4_gsm8k.jsonl` | — |
| SC@5 | `data/results_sft/fast/qwen3_causal_sc5_gsm8k.jsonl` | — |

---

## RQ2 — Does In-Context Learning (ICL) with causal chains help?

### Datasets
| Role | File |
|------|------|
| ICL examples (causal) | `data/gsm8k/icl_causal_examples.jsonl` |
| ICL examples (noncausal) | `data/gsm8k/icl_noncausal_examples.jsonl` |
| GSM8K test | `data/gsm8k/test.jsonl` |
| MATH-500 test | `data/math500/test.jsonl` |

### Scripts
- `expt/run_icl.py` — few-shot prompting, DeepSeek + Qwen3 models
- `expt/run_rq02.py` — main RQ2 runner

### Models Tested
- Qwen3-1.7B (base + SFT)
- DeepSeek-R1-Distill-Qwen-1.5B (original)

### Result Files
- `data/results_sft/fast/` — various ICL result JSONLs
- `ICL_RESULTS_REPORT.md` — summary table
- `RQ02 Original Results.md` — raw numbers

---

## PNS Causal Pruning Algorithm

### Core Scripts
| Script | Purpose |
|--------|---------|
| `algo/pns_causal.py` | PNS computation: PS, PN per reasoning step |
| `algo/equivalent_ans.py` | Answer equivalence: `_extract_boxed`, `_normalize`, `_fast_match` |
| `expt/run_algo_o.py` | Batch PNS pipeline: reads CoT jsonl → outputs PNS scores |

### Input Format (for run_algo_o.py)
```json
{"question": "...", "answer": "72", "model_answer": "Step 1...\n\nStep 2..."}
```
Steps split on `\n\n` (double newline).

### Output Format
```json
{
  "question": "...", "answer": "72",
  "metrics": {
    "PS(chain)": 1, "step_length": 4, "avg_PN(steps)": 0.62,
    "final_chain": ["Step 1...", "Step 2...", "Step 3..."]
  }
}
```

### Key Thresholds
| Context | PS filter | PN filter | min_steps |
|---------|-----------|-----------|-----------|
| RQ1 SFT (strict) | PS=1 | PN≥0.5 | ≥2 |
| Self-distill (relaxed) | PS=1 | PN≥0.1 | ≥1 |
| DPO pairs | PS=1 | PN≥0.3 | ≥2 |

---

## DPO — Direct Preference Optimization (Negative Result)

### Pipeline
```
Causal SFT 1.7B → wrong outputs → build pairs → DPO fine-tune
```

### Scripts
| Script | Purpose |
|--------|---------|
| `dpo/build_dpo_pairs_v3.py` | 186 PNS-filtered pairs (PS=1, PN≥0.3, steps≥2) |
| `dpo/fill_missing_pairs.py` | 114 minimal pairs (total 300 pairs) |
| `dpo/build_dpo_combined.py` | GSM8K + MATH-500 combined (~512 pairs) |
| `dpo/train_dpo.py` | DPO training: β=0.1, LR=5e-6, epochs=2, r=16 |

### DPO Pair Format
```
chosen:  <think>\n[PNS final_chain steps]\n</think>\n\n$\boxed{answer}$
rejected: [wrong SFT output or empty]
```

### DPO Models
| Version | Pairs | Acc (GSM8K) | Notes |
|---------|-------|------------|-------|
| v1 | 300 | ~75% | First attempt |
| v2 | 300 | ~75% | |
| v3 | 186 | — | PNS-only pairs |
| v4 | 390 | **75.1%** | Best DPO, still −2.2pp vs SFT |

**Conclusion:** DPO consistently degrades. Likely causes: (1) 390 pairs = ~5% of training distribution, (2) 1.7B capacity ceiling. Reported as honest negative result in paper.

### Result Files
- `models/dpo/qwen3_causal_dpo_v4/` — best DPO adapter
- `data/results_sft/fast/qwen3_causal_dpo_v4_gsm8k.jsonl`

---

## Self-Distillation Loop (Novelty #2) — IN PROGRESS

### Concept
```
Model generates CoT → PNS filters best chains → Retrain → Repeat
Goal: teach model to be causally efficient across iterations
```

### Loop Commands (Iter 1 with 8B)
```bash
# Step 1: Generate CoT from fine-tuned model
python sft/gen_self_distill.py \
  --adapter models/sft/qwen3_fast_causal \
  --output data/self_distill/gsm8k_cot_iter1.jsonl \
  --load_in_4bit --detailed

# Step 2: PNS filter (relaxed: PN≥0.1, steps≥1)
python expt/run_algo_o.py \
  --input_file data/self_distill/gsm8k_cot_iter1.jsonl \
  --output_file data/self_distill/gsm8k_pns_iter1.jsonl

# Step 3: Prepare SFT data
python sft/prepare_distill_sft.py \
  --pns data/self_distill/gsm8k_pns_iter1.jsonl \
  --output data/self_distill/train_causal_iter1.jsonl --iter 1

# Step 4: Train 8B
python sft/train_correct.py \
  --data data/self_distill/train_causal_iter1.jsonl \
  --output_dir models/distill/qwen3_8b_causal_iter2
```

### Datasets Used
| File | Content |
|------|---------|
| `data/self_distill/causal_300.jsonl` | 300 PNS-curated examples (reduced for time) |
| `data/self_distill/detailed_cot_for_dpo.jsonl` | Detailed CoT with markdown sections, used as seed |
| `data/gsm8k/test_200.jsonl` | 200-question reduced test set for faster eval |

### Models Trained
| Model | Base | Data | Training | Status |
|-------|------|------|---------|--------|
| 8B Noncausal | Qwen3-8B | Standard SFT | QLoRA r=16 | ✅ Done (`models/distill/qwen3_8b_noncausal`) |
| 8B Causal (Iter1) | Qwen3-8B | causal_300.jsonl | QLoRA r=16, 2ep, loss=0.27, 3h10m | ✅ Done (`models/distill/qwen3_8b_causal`) |

### Eval Commands (running now)
```bash
# Original 8B baseline
python sft/eval_correct.py --base_model Qwen/Qwen3-8B \
  --data data/gsm8k/test_200.jsonl \
  --output data/results_sft/distill/original_gsm8k.jsonl

# Noncausal 8B
python sft/eval_correct.py --adapter models/distill/qwen3_8b_noncausal \
  --data data/gsm8k/test_200.jsonl \
  --output data/results_sft/distill/noncausal_gsm8k.jsonl

# Causal 8B (Iter1)
python sft/eval_correct.py --adapter models/distill/qwen3_8b_causal \
  --data data/gsm8k/test_200.jsonl \
  --output data/results_sft/distill/causal_gsm8k.jsonl
```

### Result Files (pending)
- `data/results_sft/distill/original_gsm8k.jsonl` — 🟡 running (0 bytes at 4:16 PM)
- `data/results_sft/distill/noncausal_gsm8k.jsonl` — 🟡 queued
- `data/results_sft/distill/causal_gsm8k.jsonl` — 🟡 queued

### Expected Results (Table XI in paper)
| Model | Expected Acc | Expected Tokens |
|-------|-------------|----------------|
| Original Qwen3-8B | 82–86% | ~2000 |
| Noncausal 8B SFT | 85–89% | ~300 |
| Causal 8B Iter1 | 83–88% | ~200 |

---

## Comparison Script
```bash
python sft/eval_distill_vs_rq1.py
```
Prints unified table: RQ1 1.7B baselines + self-distillation 14B/8B iterations.
Token compression ratios vs noncausal baseline included.

---

## Key Algo Scripts (Full Map)

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `expt/run_algo_o.py` | CoT jsonl | PNS jsonl | PNS scoring + chain pruning |
| `algo/pns_causal.py` | CoT text | PS/PN scores | Core PNS computation |
| `algo/equivalent_ans.py` | pred, gt strings | bool | Answer equivalence |
| `sft/train_fast.py` | SFT jsonl | LoRA adapter | 1.7B SFT training |
| `sft/train_correct.py` | SFT jsonl | LoRA adapter | 8B QLoRA SFT training |
| `sft/train_14b_sft.py` | SFT jsonl | LoRA adapter | 14B QLoRA SFT training |
| `sft/eval_correct.py` | test jsonl | result jsonl | Greedy eval |
| `sft/eval_sc.py` | test jsonl | result jsonl | SC@N majority vote eval |
| `sft/gen_self_distill.py` | adapter/base model | CoT jsonl | Generate CoT for distillation |
| `sft/prepare_distill_sft.py` | PNS jsonl | SFT jsonl | Convert PNS → SFT format |
| `dpo/train_dpo.py` | DPO jsonl + adapter | merged model | DPO fine-tuning |
| `dpo/build_dpo_pairs_v3.py` | PNS jsonl | DPO jsonl | Build preference pairs |
| `grpo/train_grpo.py` | train jsonl + adapter | LoRA adapter | GRPO reinforcement training |

---

## Models Directory Map

```
models/
├── sft/
│   ├── qwen3_fast_causal/         # 1.7B Causal SFT [RQ1 main model]
│   ├── qwen3_fast_noncausal/      # 1.7B Noncausal SFT [RQ1 baseline]
│   ├── qwen3_causal/              # 1.7B Causal SFT (older, longer training)
│   └── qwen3_noncausal/           # 1.7B Noncausal SFT (older)
├── dpo/
│   ├── qwen3_causal_dpo_v4/       # Best DPO model [RQ1 extension]
│   ├── qwen3_causal_dpo_v3/       # DPO v3 (PNS-only pairs)
│   ├── qwen3_causal_dpo_v2/       # DPO v2
│   └── qwen3_causal_dpo/          # DPO v1
└── distill/
    ├── qwen3_8b_causal/           # 8B Causal SFT Iter1 [Self-distill]
    └── qwen3_8b_noncausal/        # 8B Noncausal SFT [Self-distill baseline]
```

---

## Data Directory Map

```
data/
├── gsm8k/
│   ├── train.jsonl                # GSM8K train (7473 questions)
│   ├── test.jsonl                 # GSM8K test (1319 questions)
│   ├── test_200.jsonl             # Reduced test (200 questions)
│   ├── gsm8k_pns_qwen3-14b-thinking-v2.jsonl  # 14B PNS scores [main PNS data]
│   ├── train_causal_fast.jsonl    # 1.7B causal SFT train
│   └── train_noncausal_fast.jsonl # 1.7B noncausal SFT train
├── math500/
│   ├── test.jsonl                 # MATH-500 test (500 questions)
│   └── train.jsonl
├── self_distill/
│   ├── causal_300.jsonl           # 300 PNS-causal examples (reduced dataset)
│   └── detailed_cot_for_dpo.jsonl # Detailed CoT chains (markdown sections)
├── dpo/
│   ├── dpo_pairs_v3.jsonl         # 186 PNS pairs
│   ├── dpo_pairs_filled.jsonl     # +114 minimal pairs = 300 total
│   └── dpo_pairs_combined.jsonl   # GSM8K+MATH500 combined
└── results_sft/
    ├── fast/                      # 1.7B eval results (RQ1)
    │   ├── qwen3_original_gsm8k.jsonl
    │   ├── qwen3_noncausal_gsm8k.jsonl
    │   ├── qwen3_causal_gsm8k.jsonl
    │   ├── qwen3_causal_dpo_v4_gsm8k.jsonl
    │   └── qwen3_causal_sc5_gsm8k.jsonl
    └── distill/                   # 8B eval results (Self-distill)
        ├── original_gsm8k.jsonl   # 🟡 running
        ├── noncausal_gsm8k.jsonl  # 🟡 queued
        └── causal_gsm8k.jsonl     # 🟡 queued
```

---

## Paper Structure → Results Mapping

| Paper Section | Table/Figure | Data Source | Status |
|--------------|-------------|-------------|--------|
| §III RQ1 Setup | Table I (dataset stats) | train.jsonl, pns jsonl | ✅ |
| §IV RQ1 Results | Table II (GSM8K acc) | `fast/qwen3_*.jsonl` | ✅ |
| §IV RQ1 Results | Table III (MATH-500 acc) | `fast/qwen3_*_math500.jsonl` | ✅ |
| §V Ablation | Table IV (PNS threshold α) | gsm8k_pns_*.jsonl | ✅ |
| §V Ablation | Table V (step pruning) | pns chain stats | ✅ |
| §VI DPO | Table VI (DPO comparison) | `fast/qwen3_causal_dpo_v4_gsm8k.jsonl` | ✅ needs 1.7B rows added |
| §VI DPO | Table VII (pair quality) | dpo_pairs_v3.jsonl stats | ✅ |
| §VII SC@N | Table VIII (SC@5 results) | `fast/qwen3_causal_sc5_gsm8k.jsonl` | 🟡 19/1319 done |
| §VIII Self-Distill | Table XI (distill results) | `distill/*.jsonl` | 🟡 evals running |
| §III Fig.1 | PNS diagram | placeholder | ❌ needs figure |
| §IV Fig.2 | Token dist plot | fast/ results | ❌ needs figure |
| §VIII Fig.3 | Distill loop diagram | self-distill concept | ❌ needs figure |

---

## Pending Tasks

1. **Wait for evals** — `distill/` 3 files + SC@5 full run
2. **Update Table XI** — fill [TBD] with actual 8B numbers
3. **Fix Table VI** — add Qwen3-1.7B Noncausal (66.6%/83.1%) and Causal (57.6%/77.3%) rows; remove DeepSeek-1.5B
4. **Figures** — generate 3 figures (PNS diagram, token dist bar chart, distill loop)
5. **GRPO** — `grpo/train_grpo.py` ready to run if time permits
6. **SC@5 full result** — update Table VIII when done (~27h at current rate)

---

## Bugs Fixed This Session

| Bug | File | Fix |
|-----|------|-----|
| TRL Unicode crash on Windows | `trl/chat_template_utils.py` | Added `encoding="utf-8"` to all 14 `read_text()` calls |
| DPO saves adapter only (loses merge) | `dpo/train_dpo.py` | Added `model.merge_and_unload()` before `save_pretrained()` |
| DeepSeek think-strip | `sft/train_sft.py` | Fixed `apply_chat_template` to preserve `<think>` blocks |
| run_algo_o arg names | shell commands | `--input_file`/`--output_file` (not `--input`/`--output`) |
