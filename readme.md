# Keeping Only What Counts
### Causal PNS-Guided Chain-of-Thought Optimisation with Automatic DPO Preference Alignment and Iterative Self-Distillation

**Ayesha Tahir Awan · Mariam Zahid · Ahmed Khan**  
Department of Artificial Intelligence & Data Science, School of Computing  
FAST-NUCES, Islamabad, Pakistan

[![Paper](https://img.shields.io/badge/Base%20Paper-arXiv%3A2506.09853-b31b1b)](https://arxiv.org/abs/2506.09853)
[![Code](https://img.shields.io/badge/GitHub-causalmath-blue)](https://github.com/ayesha0412/causalmath)

---

## What Is This About?

Large language models produce reasoning chains that *look* thorough but are often full of steps that do not actually matter. Restate the problem, define a variable you never use, verify an obvious intermediate — none of these change the answer, yet the model generates them every time and we pay for every token.

Yu et al. (NeurIPS 2025) showed that the **Probability of Necessity and Sufficiency (PNS)** — a criterion from Pearl's causal calculus — can identify exactly which reasoning steps are load-bearing and which are redundant. Prune the redundant ones and you get shorter chains, at the same or better accuracy.

Their results were impressive. They were also produced on multi-GPU VLLM clusters that most research groups cannot access.

**This project asks: how much of that holds on a single consumer GPU?**

We replicate the full PNS pipeline on one NVIDIA RTX 5080 (16 GB), at model scales five times smaller than the original paper, and then go further — introducing two novel extensions that the original paper did not explore:

1. **Automatic DPO alignment** using PNS-pruned traces as preference signals, eliminating human annotation entirely.
2. **Iterative self-distillation** where the model filters and retrains on its own causal chains, removing the need for a teacher after the first round.

---

## Research Questions

The work is structured around three questions drawn directly from the paper:

> **RQ1 — Token Efficiency**  
> Can the PNS algorithm reduce CoT token usage on a single consumer GPU without degrading reasoning accuracy, and do the reductions replicate at a model scale five times smaller than the baseline paper?

> **RQ2 — Downstream Quality**  
> Do PNS-optimised CoT traces produce better downstream task performance when used as (a) in-context learning exemplars and (b) supervised fine-tuning data, compared to unoptimised traces?

> **RQ3 — Causal Preference Signals (Novel)**  
> Can PNS-derived causal preference signals, applied through DPO training and iterative self-distillation, improve model accuracy beyond the SFT baseline without any human annotation?

---

## Key Findings

**RQ1 → Yes.** Qwen3-14B achieves 21.4% token reduction on GSM-8k and 33.3% on MATH-500 with zero accuracy degradation — entirely on a single RTX 5080. The MATH-500 number closely tracks the paper's 36.6%, confirming the algorithm works at smaller scale. Full results in [`data/results_sft/`](data/results_sft/).

**RQ2 → Yes for GSM-8k.** Under in-context learning, Fast-Solve reaches 93.18% on GSM-8k with Qwen3-8B. Under SFT, Qwen3-1.7B trained on noncausal chains matches the baseline paper's noncausal GSM-8k accuracy at five times smaller model scale. PNS-pruned training data produces 52.1 fewer tokens per response at inference time compared to noncausal training — a measurable and consistent efficiency gain.

**RQ3 → Mixed, and instructive.** DPO on 390 automatically-constructed PNS preference pairs yields a slight accuracy drop at 1.7B scale — not because the preference signal is wrong, but because 390 pairs covering 5% of the training distribution is too sparse for a model already near its capacity ceiling. The self-distillation loop tells a cleaner story: one round of teacher-seeded PNS distillation produces a Qwen3-8B student at 90% accuracy generating 2 reasoning steps and 143 tokens on average — compared to 29 steps and 2,048 truncated tokens before fine-tuning. Iteration 2 (self-generated, no teacher) holds that performance exactly.

The DPO negative result is a genuine finding: it establishes the minimum scale and data requirements for PNS-derived preferences to transfer, pointing directly to the right next experiment.

---

## Novel Contributions

### Automatic DPO via PNS (`dpo/`)

Standard DPO needs human annotators or an expensive reward model to label which response is better. We replace that entirely with the PNS score.

For every training question where Algorithm 1 produces a valid pruned chain:
- **Chosen** response = the PNS-pruned correct chain (causally grounded, shorter)
- **Rejected** response = the SFT model's own incorrect output on the same question

This gives contrastive training signal that is free, reproducible, and causally principled. See [`dpo/build_dpo_pairs.py`](dpo/build_dpo_pairs.py) and [`dpo/train_dpo.py`](dpo/train_dpo.py).

### Iterative Self-Distillation Loop (`Self_distill/`)

The self-distillation loop tests whether PNS filtering can function as a data flywheel — where a model curates and retrains on its own outputs without any external teacher after the first iteration.

```
Qwen3-14B (teacher)
      │  generates CoT chains
      ▼
  PNS Filter (α=0.5, k=3)
      │  retains causally necessary steps
      ▼
Qwen3-8B fine-tuned (iter 1)  ──► 90% accuracy, 143 tokens, 2 steps
      │  generates its own chains
      ▼
  PNS Filter
      │
Qwen3-8B fine-tuned (iter 2)  ──► 90% accuracy, 137 tokens, 2 steps
      │  no teacher, no annotation
      ▼
      ...
```

After one teacher-seeded round, the model has internalised causal necessity well enough that its own outputs have little left to prune — PNS yield drops from 86% to 62% because the chains are already compact. See [`Self_distill/scripts/`](Self_distill/scripts/).

---

## How It Compares to the Baseline Paper

| Aspect | Yu et al. [4] | This Work |
|--------|--------------|-----------|
| Hardware | Multi-GPU VLLM | Single RTX 5080, 16 GB |
| PNS model | QwQ-32B + Qwen2.5-72B | Qwen3-14B |
| GSM-8k token reduction | 70.2% | 21.4% |
| MATH-500 token reduction | 36.6% | **33.3%** ✓ |
| Accuracy degradation | 0.0 pp | 0.0 pp |
| DPO extension | ✗ | ✓ (novel) |
| Self-distillation | ✗ | ✓ (novel) |

The MATH-500 gap closes entirely because harder problems contain more genuine redundancy. The GSM-8k gap reflects that Qwen3-14B already writes compact 53-token chains compared to QwQ-32B's 113-token chains — there is simply less to prune. This is a verbosity difference, not an algorithmic one.

---

## Repository Structure

```
causalmath/
│
├── algo/                    # PNS algorithm (pnps_cot.py, equivalent_ans.py)
├── expt/                    # RQ1 + RQ2-ICL runners and shell scripts
│
├── sft/                     # RQ2b: SFT training pipeline
│   ├── prepare_sft_correct.py
│   ├── train_sft.py
│   └── eval_correct.py
│
├── dpo/                     # RQ3a: Novel DPO pipeline
│   ├── build_dpo_pairs.py   # Constructs PNS preference pairs
│   ├── train_dpo.py         # DPO training via TRL DPOTrainer
│   └── eval_dpo.py
│
├── Self_distill/            # RQ3b: Novel self-distillation loop
│   └── scripts/
│       ├── generate_teacher.py
│       ├── apply_pns_filter.py
│       ├── train_iter.py
│       └── eval_correct.py
│
├── inference/               # Colab-ready inference notebooks
│   ├── inference_DPO.ipynb
│   ├── inference_SFT.ipynb
│   └── inference_Self_Distill.ipynb
│
├── data/
│   ├── dpo/                 # Preference pairs (JSONL)
│   ├── self_distill/        # Per-iteration training data
│   └── results_sft/         # Evaluation outputs and summaries
│
└── models/
    ├── sft/                 # LoRA adapters (causal, noncausal)
    └── dpo/                 # Merged DPO checkpoint
```

---

## Quickstart

### Setup

```bash
git clone https://github.com/ayesha0412/causalmath.git
cd causalmath
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

### RQ1 — PNS Pruning

```bash
# Generate CoT traces and apply PNS optimisation
cd expt
bash run_algo_o.sh
```

### RQ2a — In-Context Learning

```bash
bash run_icl.sh --dataset gsm8k --shots 0 1 3 5
```

### RQ2b — Supervised Fine-Tuning

```bash
# Causal condition (PNS-pruned chains)
python sft/prepare_sft_correct.py --condition causal
python sft/train_sft.py --condition causal
python sft/eval_correct.py --condition causal

# Noncausal condition (full CoT chains)
python sft/prepare_sft_correct.py --condition noncausal
python sft/train_sft.py --condition noncausal
```

### RQ3a — DPO

```bash
python dpo/build_dpo_pairs.py
python dpo/train_dpo.py
python dpo/eval_dpo.py
```

### RQ3b — Self-Distillation

```bash
# Iter 1: teacher-seeded
python Self_distill/scripts/generate_teacher.py
python Self_distill/scripts/apply_pns_filter.py --iter 1
python Self_distill/scripts/train_iter.py --iter 1

# Iter 2: self-generated, no teacher
python Self_distill/scripts/generate_teacher.py --model iter1
python Self_distill/scripts/apply_pns_filter.py --iter 2
python Self_distill/scripts/train_iter.py --iter 2
```

---

## Inference Notebooks

Three Colab-ready notebooks are provided for interactive testing of trained checkpoints against base models. Each uses a small set of examples designed to fit within Colab's T4 memory:

| Notebook | What it compares |
|----------|-----------------|
| [`inference_DPO.ipynb`](inference/inference_DPO.ipynb) | DPO-aligned model vs base Qwen3-1.7B |
| [`inference_SFT.ipynb`](inference/inference_SFT.ipynb) | Causal SFT vs Noncausal SFT vs base |
| [`inference_Self_Distill.ipynb`](inference/inference_Self_Distill.ipynb) | Iter 1 vs Iter 2 vs base Qwen3-8B |

> These are sanity-check runs to verify model behaviour. Full benchmark evaluation results are stored in `data/results_sft/`.

---

## Citation

**Base paper:**
```bibtex
@inproceedings{yu2025causal,
  title     = {Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning},
  author    = {Yu, Xingchen and Wang, Zihao and Yang, Liang and Li, Haoran and
               Liu, Anni and Xue, Xianglong and Wang, Jian and Yang, Meng},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2506.09853}
}
```

**This reproduction:**
```bibtex
@misc{awan2025pnscot,
  title       = {Keeping Only What Counts: Causal PNS-Guided CoT Optimisation
                 with Automatic DPO Preference Alignment and Iterative Self-Distillation},
  author      = {Awan, Ayesha Tahir and Zahid, Mariam and Khan, Ahmed},
  institution = {FAST-NUCES, Islamabad},
  year        = {2025}
}
```

---

© 2025 Ayesha Tahir Awan, Mariam Zahid, Ahmed Khan — FAST-NUCES, Islamabad. All rights reserved.
