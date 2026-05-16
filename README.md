# Keeping Only What Counts

### Causal PNS-Guided Chain-of-Thought Optimisation with Automatic DPO Preference Alignment and Iterative Self-Distillation

**Ayesha Tahir Awan · Ahmed Khan · Mariam Zahid**
Supervised by **Dr. Zohair Ahmed**
Department of Artificial Intelligence & Data Science, School of Computing
FAST-NUCES, Islamabad, Pakistan

[![Paper](https://img.shields.io/badge/Base%20Paper-arXiv%3A2506.09853-b31b1b)](https://arxiv.org/abs/2506.09853)
[![Code](https://img.shields.io/badge/GitHub-causalmath-blue)](https://github.com/ayesha0412/causalmath)
[![Interactive Demo](https://img.shields.io/badge/Interactive%20Demo-0xahmedk.github.io%2Fcot-brightgreen)](https://0xahmedk.github.io/cot)

Large language models produce reasoning chains that look thorough but are full of steps that do not actually matter. This project replicates and extends Yu et al.'s PNS-based chain-of-thought pruning algorithm on a single consumer GPU, then introduces two novel extensions — automatic DPO alignment using PNS-derived preference signals, and an iterative self-distillation loop that removes the need for a teacher after the first round.

> **Interactive explainer** — An interactive blog walkthrough of the Causal CoT concept with live demos is live at **[0xahmedk.github.io/cot](https://0xahmedk.github.io/cot)**.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

### Problem Statement

Modern LLMs emit verbose chain-of-thought reasoning by default. On GSM8K, Qwen3-8B generates 808 tokens per answer on average under standard CoT prompting — many of those tokens restate the problem, define unused variables, or verify obvious intermediates. Every token costs inference time and compute. The question is: which steps are genuinely load-bearing, and which can be removed without degrading the answer?

### Approach

Yu et al. (NeurIPS 2025) showed that the **Probability of Necessity and Sufficiency (PNS)** from Pearl's causal calculus can identify exactly which reasoning steps are causally necessary:

- **PS (Probability of Sufficiency)**: whether the chain as a whole reaches the correct answer
- **PN (Probability of Necessity)**: for each step, the probability that removing it would break the answer

A step with PN below a threshold is redundant and can be pruned. This project:

1. **Replicates** the full PNS pipeline on a single NVIDIA RTX 5080 (16 GB), at model scales five times smaller than the original paper
2. **Extends** the pipeline with automatic DPO alignment (PNS-pruned traces as preference signals, no human annotation)
3. **Extends** the pipeline with iterative self-distillation (model curates and retrains on its own causal chains)

The work is structured around three research questions:

- **RQ1 — Token Efficiency**: Can PNS reduce CoT token usage on a single consumer GPU without degrading accuracy?
- **RQ2 — Downstream Quality**: Do PNS-optimised traces improve in-context learning and SFT performance?
- **RQ3 — Causal Preference Signals** (novel): Can PNS-derived signals drive DPO alignment and iterative self-distillation?

### Key Results

| Metric                                      | Value                                 |
| ------------------------------------------- | ------------------------------------- |
| MATH-500 token reduction (vs baseline)      | **33.3%** (paper: 36.6%)              |
| GSM-8K token reduction                      | **21.4%**                             |
| Accuracy degradation from pruning           | **0.0 pp**                            |
| Self-distill iter1: tokens vs noncausal SFT | **143 vs 261 tokens** (−45%)          |
| Self-distill iter2 accuracy                 | **90.0%** (same as iter1, no teacher) |
| Fast-Solve ICL accuracy on GSM-8K           | **93.18%**                            |

---

## Project Structure

```
causalmath/
│
├── src/causalmath/                # installable Python package (pip install -e .)
│   ├── algorithm/                 # Core PNS algorithm
│   │   ├── pns_cot.py             # Main PNS computation: calculate_ps_pn(), parse_nodes()
│   │   ├── equivalence.py         # Answer equivalence: _extract_boxed(), _fast_match(), LLM fallback
│   │   └── __init__.py
│   ├── models/                    # LLM clients
│   │   ├── ollama_client.py       # Ollama/OpenAI-compatible wrapper: cerebras_query()
│   │   ├── base_model.py          # Legacy Azure/GPT-3.5 wrapper (unused in main pipeline)
│   │   └── __init__.py
│   ├── data/                      # Data preparation pipelines
│   │   ├── prepare_sft.py         # Build causal/noncausal SFT JSONL from PNS output
│   │   ├── prepare_distill.py     # Convert PNS output to SFT format for distillation
│   │   ├── generate_cot.py        # Generate CoT from a fine-tuned adapter (for distillation)
│   │   ├── generate_cot_distill.py# Generate CoT from HF adapter path
│   │   ├── generate_detailed.py   # Generate detailed CoT for DPO chosen side
│   │   ├── generate_rejected.py   # Generate wrong outputs from SFT model (DPO rejected side)
│   │   ├── build_dpo_pairs.py     # Build PNS preference pairs (chosen=causal, rejected=wrong)
│   │   ├── build_dpo_combined.py  # Combine GSM8K + MATH-500 DPO pairs
│   │   ├── prepare_dpo_pairs_causal.py  # Causal-only pair builder
│   │   ├── prepare_noncausal.py   # Build matched noncausal baseline for self-distillation
│   │   └── __init__.py
│   ├── training/                  # SFT and DPO training
│   │   ├── train_sft.py           # QLoRA SFT with correct label masking (LabelMaskingCollator)
│   │   ├── train_sft_14b.py       # QLoRA SFT variant for Qwen3-14B
│   │   ├── train_dpo.py           # DPO fine-tuning via TRL DPOTrainer
│   │   ├── train_dpo_alt.py       # Alternative DPO training script
│   │   └── __init__.py
│   ├── evaluation/                # Evaluation scripts
│   │   ├── eval_correct.py        # Greedy batch evaluation with answer extraction
│   │   ├── eval_self_consistency.py  # SC@N majority-vote evaluation
│   │   ├── eval_dpo.py            # DPO comparison evaluation
│   │   ├── eval_distill_vs_rq1.py # Self-distill vs RQ1 SFT comparison
│   │   ├── print_results.py       # Quick accuracy/token summary from result JSONL files
│   │   ├── print_dpo_results.py   # DPO evaluation summary printer
│   │   ├── summarize_results.py   # Unified results table across all conditions
│   │   ├── plot_pn_aime.py        # PN distribution plots for AIME
│   │   ├── plot_pn_common.py      # PN distribution plots for CommonsenseQA
│   │   ├── stats.py               # Summary statistics over PNS output files
│   │   └── __init__.py
│   └── utils/                     # Shared utilities
│       ├── prompts.py             # System prompts (math_prompt, common_prompt)
│       ├── append_index.py        # Append line indices to JSONL files
│       └── __init__.py
│
├── scripts/                       # Entry-point CLI scripts
│   ├── run_curation.py            # Batch PNS pipeline: JSONL in → PNS scores out
│   ├── run_curation_posthoc.py    # Post-hoc PNS on existing CoT files
│   ├── run_training.py            # Main SFT training entry point
│   ├── run_self_distill.py        # Full 7-step iterative self-distillation pipeline
│   ├── run_self_distill_detailed.py  # Detailed self-distillation runner (Linux/GPU)
│   ├── run_inference.py           # Standalone inference script with --compare mode
│   ├── run_get_response.py        # CoT generation via Ollama (input to PNS stage)
│   ├── run_icl.py                 # ICL experiment runner (all methods)
│   ├── run_icl_all.py             # Runs all ICL conditions sequentially
│   ├── check_causal_outputs.py    # Inspect causal chain outputs
│   ├── check_pns_correctness.py   # Verify PNS scores on sample questions
│   ├── diagnose_causal.py         # Diagnostic tool for causal chain quality
│   ├── download_datasets.py       # Download GSM8K, MATH-500, CommonsenseQA from HuggingFace
│   ├── run_sft_fast.ps1           # PowerShell: full SFT pipeline (prepare + train + eval)
│   ├── run_dpo_pipeline.ps1       # PowerShell: full DPO pipeline
│   └── run_dpo_gsm8k.ps1          # PowerShell: DPO on GSM8K
│
├── configs/                       # YAML configuration files
│   ├── curation_config.yaml       # PNS thresholds, rollouts, dataset paths
│   ├── training_config.yaml       # SFT / DPO / self-distill hyperparameters
│   └── eval_config.yaml           # Dataset paths, SC@N settings, model registry
│
├── data/
│   ├── raw/                       # Original datasets (gitignored — download via scripts/)
│   │   ├── gsm8k/                 # GSM8K test set (1,319 questions)
│   │   ├── math500/               # MATH-500 test set (500 questions)
│   │   └── commonsenseqa/         # CommonsenseQA test set (1,221 questions)
│   └── processed/                 # Curation pipeline outputs (committed)
│       ├── sft/                   # causal_train.jsonl, noncausal_train.jsonl
│       ├── dpo/                   # DPO preference pair JSONL files
│       └── self_distill/          # Per-iteration training data (iter_1/, iter_2/)
│
├── notebooks/                     # Colab-ready inference notebooks
│   ├── 01_inference_pns.ipynb     # Interactive PNS pruning demo
│   ├── 02_inference_icl.ipynb     # ICL comparison (standard CoT vs causal examples)
│   ├── 03_inference_SFT.ipynb     # Causal SFT vs noncausal SFT vs base Qwen3-1.7B
│   ├── 04_inference_DPO.ipynb     # DPO-aligned model vs SFT baseline vs base
│   ├── 05_inference_Self_Distill.ipynb  # Iter1 vs iter2 vs base Qwen3-8B
│   └── self-distill/
│       └── inference-distill.ipynb
│
├── outputs/
│   ├── checkpoints/               # LoRA adapter checkpoints (gitignored)
│   │   └── dpo/qwen3_causal_dpo_v4/   # DPO checkpoint (adapter_config, trainer_state)
│   ├── logs/                      # Timestamped training and eval logs
│   └── results/                   # Evaluation outputs (JSONL, CSV, JSON)
│       ├── icl/                   # ICL results + comparison CSV
│       ├── sft/                   # SFT/DPO/distill evaluation results
│       ├── dpo/                   # DPO-specific result files
│       └── self_distill/          # Self-distillation evaluation results
│
├── docs/                          # Documentation and paper assets
│   ├── figures/                   # Paper figures (PNG)
│   ├── PROJECT_MINDMAP.md         # Internal research notes and result tracking
│   └── SELF_DISTILLATION_RESULTS.md  # Self-distillation results summary
│
├── tests/                         # pytest unit tests
│   ├── test_algorithm.py          # Tests for equivalence helpers and CoT parsing
│   ├── test_utils.py              # Tests for prompts and shared utilities
│   └── test_api.py                # API wrapper tests
│
├── pyproject.toml                 # Package metadata + pip install -e .
├── requirements.txt               # Core pip dependencies
├── requirements-dev.txt           # Dev dependencies (pytest, ruff, mypy)
└── .env.example                   # Environment variable template
```

---

## Pipeline Architecture

The project consists of four sequential stages. Each stage depends on the outputs of the previous one.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0: DATA PREPARATION                                      │
│  downlaod_datasets.py                                           │
│  → data/gsm8k/test.jsonl  (1319 questions)                     │
│  → data/MATH-500/test.jsonl  (500 questions)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: PNS PRUNING  (RQ1)                                    │
│                                                                 │
│  expt/get_response.py         ← Qwen3-14B via Ollama           │
│        │  generates CoT JSONL                                   │
│        ▼                                                        │
│  expt/run_algo_o.py           ← algo/pnps_cot.py::calculate_ps_pn │
│        │  for each step: generate rollout → compute PN         │
│        │  prune if PN < threshold (default 0.5)               │
│        ▼                                                        │
│  data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl             │
│  data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐   ┌────────────────────────────────────────┐
│  STAGE 2a: ICL      │   │  STAGE 2b: SFT  (RQ2b)               │
│  (RQ2a)             │   │                                        │
│  expt/run_icl.py    │   │  sft/prepare_sft_fast.py              │
│  Conditions:        │   │    filters PS=1, joins final_chain     │
│  - standard_cot     │   │    → data/sft/fast/causal_train.jsonl  │
│  - fast_solve       │   │    → data/sft/fast/noncausal_train.jsonl│
│  - reduction        │   │                                        │
│  - cod              │   │  train.py --condition causal/noncausal  │
│  - ours_icl         │   │    QLoRA on Qwen3-1.7B                 │
└─────────────────────┘   │    LoRA r=16, α=32, epochs=3          │
                           └──────────────┬─────────────────────────┘
                                          │
                             ┌────────────┴────────────┐
                             ▼                         ▼
              ┌────────────────────────┐  ┌────────────────────────────┐
              │  STAGE 3a: DPO  (RQ3a)│  │  STAGE 3b: SELF-DISTILL    │
              │                        │  │  (RQ3b)                    │
              │  dpo/gen_rejected.py   │  │                            │
              │    generates wrong SFT │  │  Self_distill/scripts/     │
              │    outputs             │  │  self_distill_loop.py      │
              │                        │  │  --iter 0 (14B teacher)    │
              │  dpo/build_dpo_pairs_  │  │  --iter 1 (8B self)        │
              │  v3.py                 │  │  --iter 2 (8B self, again) │
              │    chosen = PNS chain  │  │                            │
              │    rejected = wrong    │  │  Each iteration:           │
              │    output              │  │  1. Sample new questions   │
              │                        │  │  2. Generate CoT           │
              │  dpo/train_dpo.py      │  │  3. PNS prune              │
              │    DPO β=0.1, lr=5e-5  │  │  4. Build SFT data         │
              │    from SFT checkpoint │  │  5. Build cumulative set   │
              └────────────────────────┘  │  6. Train QLoRA            │
                                          │  7. Evaluate               │
                                          └────────────────────────────┘
```

### Data Flow

Input JSONL format (for PNS stage):

```json
{
  "question": "Natalia sold clips to 48 friends in April...",
  "answer": "72",
  "model_answer": "Step 1: ...\n\nStep 2: ..."
}
```

Steps are delimited by `\n\n` (double newline). The `model_answer` field holds the full CoT text.

PNS output format:

```json
{
  "question": "...",
  "answer": "72",
  "metrics": {
    "PS(chain)": 1,
    "step_length": 2,
    "avg_PN(steps)": 0.62,
    "final_chain": ["Step 1: ...", "Step 2: ..."],
    "pn_per_step": [0.8, 1.0],
    "token_length": 45,
    "total_rollout_calls": 10
  }
}
```

SFT training format (for `train.py`):

```json
{
  "messages": [
    { "role": "user", "content": "Natalia sold clips to 48 friends..." },
    {
      "role": "assistant",
      "content": "<think>\nStep 1: ...\n\nStep 2: ...\n</think>\n\n$\\boxed{72}$"
    }
  ]
}
```

DPO preference pair format (for `dpo/train_dpo.py`):

```json
{
  "prompt": "Natalia sold clips to 48 friends...",
  "chosen": "<think>\nStep 1 (causally necessary)...\n</think>\n\n$\\boxed{72}$",
  "rejected": "<think>\nWrong reasoning...\n</think>\n\n$\\boxed{48}$"
}
```

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- CUDA 12.1+ with a GPU of at least 16 GB VRAM (tested on NVIDIA RTX 5080)
- [Ollama](https://ollama.com/) installed and running locally (for PNS rollouts)

### 1. Clone the repository

```bash
git clone https://github.com/ayesha0412/causalmath.git
cd causalmath
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# or
venv\Scripts\activate             # Windows
```

### 3. Install Python dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Full dependency list (`requirements.txt`):

```
torch>=2.1.0
transformers>=4.45.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.26.0
datasets>=2.18.0
trl>=0.8.0
sentencepiece
protobuf
scipy
numpy
pandas
tqdm
jsonlines
```

Additional packages needed (install manually):

```bash
pip install openai python-dotenv colorama
```

### 4. Pull the Qwen3-14B model via Ollama

The PNS algorithm uses Qwen3-14B running locally via Ollama for Monte-Carlo rollouts:

```bash
ollama pull qwen3:14b
```

Verify it's running:

```bash
ollama run qwen3:14b "Hello, what is 2+2?"
```

Ollama must be running on `http://localhost:11434` before any PNS script is executed.

### 5. Configure the model (optional)

Create a `.env` file in the project root to override the default Ollama model:

```bash
QWEN_MODEL=qwen3:14b   # default; change to qwen3:8b for lighter hardware
```

---

## Configuration

### PNS Algorithm Parameters (`expt/run_algo_o.py`)

| Parameter            | Flag             | Default | Description                                                                   |
| -------------------- | ---------------- | ------- | ----------------------------------------------------------------------------- |
| PN threshold         | `--threshold`    | `0.5`   | Steps with PN below this are pruned. Lower = more pruning.                    |
| Monte-Carlo rollouts | `--rollouts`     | `5`     | Rollouts per step for PN estimation. More = more accurate, slower.            |
| Parallel workers     | `--workers`      | `2`     | Questions processed in parallel. Limited by Ollama `OLLAMA_NUM_PARALLEL`.     |
| Rollout method       | `--prompt_based` | off     | If set, uses prompt-intervention (do_type=1) instead of context-continuation. |
| Batch size           | `--batch_size`   | `20`    | Questions per batch before flushing output.                                   |

**Threshold guide**:

- `PS=1, PN≥0.5`: strict — used for RQ1 SFT training, retains only clearly necessary steps
- `PS=1, PN≥0.3`: moderate — used for DPO pair building
- `PS=1, PN≥0.1`: relaxed — used for self-distillation (model chains are already short)

### SFT Training Parameters (`train.py`)

| Parameter             | Flag            | Default           | Description                                      |
| --------------------- | --------------- | ----------------- | ------------------------------------------------ |
| Training condition    | `--condition`   | required          | `causal` (PNS-filtered) or `noncausal` (raw CoT) |
| Base model            | `--model`       | `Qwen/Qwen3-1.7B` | HuggingFace model ID                             |
| Training data         | `--data`        | required          | Path to JSONL file with `messages` field         |
| Output directory      | `--output_dir`  | `models/sft/run`  | Where LoRA adapter is saved                      |
| Epochs                | `--epochs`      | `3`               | Training epochs                                  |
| Batch size            | `--batch_size`  | `4`               | Per-device batch size                            |
| Gradient accumulation | `--grad_accum`  | `4`               | Effective batch = batch_size × grad_accum        |
| Learning rate         | `--lr`          | `5e-4`            | Initial learning rate                            |
| LoRA rank             | `--lora_r`      | `16`              | LoRA rank (higher = more capacity)               |
| LoRA alpha            | `--lora_alpha`  | `32`              | LoRA scaling factor                              |
| Max sequence length   | `--max_seq_len` | `1024`            | Longer = more memory                             |
| Seed                  | `--seed`        | `42`              | Random seed                                      |

### DPO Training Parameters (`dpo/train_dpo.py`)

| Parameter        | Flag            | Default                       | Description                              |
| ---------------- | --------------- | ----------------------------- | ---------------------------------------- |
| SFT adapter      | `--sft_adapter` | required                      | Path to SFT checkpoint to start DPO from |
| DPO pairs        | `--data`        | `data/dpo/dpo_pairs.jsonl`    | Preference pair JSONL                    |
| Output directory | `--output_dir`  | `models/dpo/qwen3_causal_dpo` | Save path                                |
| Beta             | `--beta`        | `0.1`                         | KL divergence penalty strength           |
| Epochs           | `--epochs`      | `2`                           | Training epochs                          |
| Learning rate    | `--lr`          | `5e-5`                        | Lower than SFT — DPO is fine-tuning      |

### Self-Distillation Parameters (`Self_distill/scripts/self_distill_loop.py`)

| Parameter               | Flag                                                      | Default           | Description                                                    |
| ----------------------- | --------------------------------------------------------- | ----------------- | -------------------------------------------------------------- |
| Iteration index         | `--iter`                                                  | required          | `0` = seed from 14B teacher; `1+` = self-generated             |
| Questions per iteration | `--n_questions`                                           | `500`             | New GSM8K questions to sample                                  |
| Base model              | `--base_model`                                            | `Qwen/Qwen3-1.7B` | HF model to fine-tune each iteration                           |
| Learning rate           | `--lr`                                                    | auto              | Defaults to `2e-4` (iter 0), `1e-4` (iter 1), `5e-5` (iter 2+) |
| Skip flags              | `--skip_cot`, `--skip_pns`, `--skip_train`, `--skip_eval` | off               | Resume partial runs                                            |

---

## Usage

### Step 0: Download datasets

```bash
python downlaod_datasets.py
```

This downloads MATH-500 and CommonsenseQA from HuggingFace into `data/`. GSM8K is already included in `data/gsm8k/test.jsonl`.

---

### Step 1: RQ1 — Generate CoT chains and run PNS pruning

First generate CoT responses from Qwen3-14B (requires Ollama running):

```bash
python expt/get_response.py data/gsm8k/test.jsonl data/gsm8k/gsm8k_cot_qwen3-14b.jsonl --workers 2
```

Then run PNS pruning on the generated chains:

```bash
python expt/run_algo_o.py \
    --input_file  data/gsm8k/gsm8k_cot_qwen3-14b.jsonl \
    --output_file data/gsm8k/gsm8k_pns_qwen3-14b.jsonl \
    --threshold   0.5 \
    --rollouts    5 \
    --workers     2
```

For MATH-500:

```bash
python expt/run_algo_o.py \
    --input_file  data/MATH-500/math500_cot_qwen3-14b.jsonl \
    --output_file data/MATH-500/math500_pns_qwen3-14b.jsonl \
    --threshold   0.5 \
    --rollouts    5 \
    --workers     2
```

**Resume a partial run** (if interrupted):

```bash
python expt/run_algo_o.py \
    --input_file  data/gsm8k/gsm8k_cot_qwen3-14b.jsonl \
    --output_file data/gsm8k/gsm8k_pns_qwen3-14b.jsonl \
    --lines_processed 450 \
    --append
```

Expected output: one JSON line per input question with a `metrics` dict containing `PS(chain)`, `final_chain`, `pn_per_step`, and token/step counts.

---

### Step 2a: RQ2a — In-context learning experiments

```bash
# PNS-guided ICL exemplars on GSM8K
python expt/run_icl.py \
    --input   data/gsm8k/test.jsonl \
    --output  data/results_icl/gsm8k_ours_icl_qwen3-8b.jsonl \
    --method  ours_icl --dataset gsm8k --model qwen3:8b

# Standard CoT baseline
python expt/run_icl.py \
    --input   data/gsm8k/test.jsonl \
    --output  data/results_icl/gsm8k_standard_cot_qwen3-8b.jsonl \
    --method  standard_cot --dataset gsm8k --model qwen3:8b

# Run all ICL conditions at once
python expt/run_rq02_icl_all.py --dataset gsm8k --model qwen3:8b
```

Available `--method` values: `standard_cot`, `fast_solve`, `reduction`, `cod`, `ours_icl`

---

### Step 2b: RQ2b — Supervised fine-tuning

Prepare training data from PNS output:

```bash
python sft/prepare_sft_fast.py \
    --n_gsm8k_causal 1100 \
    --n_math_causal  500 \
    --out_dir        data/sft/fast
```

This produces:

- `data/sft/fast/causal_train.jsonl` — PNS-filtered chains (PS=1 only)
- `data/sft/fast/noncausal_train.jsonl` — raw Qwen3-14B chains

Train the causal condition:

```bash
python train.py \
    --condition  causal \
    --model      Qwen/Qwen3-1.7B \
    --data       data/sft/fast/causal_train.jsonl \
    --output_dir models/sft/qwen3_fast_causal \
    --epochs     3 \
    --lora_r     16
```

Train the noncausal baseline:

```bash
python train.py \
    --condition  noncausal \
    --model      Qwen/Qwen3-1.7B \
    --data       data/sft/fast/noncausal_train.jsonl \
    --output_dir models/sft/qwen3_fast_noncausal \
    --epochs     3
```

---

### Step 3a: RQ3a — DPO alignment (novel)

Generate wrong outputs from the SFT model (to use as `rejected` side):

```bash
python dpo/gen_rejected.py \
    --adapter    models/sft/qwen3_fast_causal \
    --data       data/gsm8k/test.jsonl \
    --output     dpo/data/rejected_chains_gsm8k.jsonl
```

Build DPO preference pairs:

```bash
python dpo/build_dpo_pairs_v3.py \
    --pns       data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl \
    --sft_eval  data/results_sft/fast/qwen3_causal_gsm8k.jsonl \
    --output    dpo/data/dpo_pairs_v3.jsonl \
    --min_steps 2 \
    --min_pn    0.3
```

Train DPO:

```bash
python dpo/train_dpo.py \
    --sft_adapter models/sft/qwen3_fast_causal \
    --data        dpo/data/dpo_pairs_v3.jsonl \
    --output_dir  models/dpo/qwen3_causal_dpo \
    --beta        0.1 \
    --epochs      2 \
    --lr          5e-5
```

---

### Step 3b: RQ3b — Iterative self-distillation (novel)

Iteration 0 — seed from Qwen3-14B teacher:

```bash
python Self_distill/scripts/self_distill_loop.py \
    --iter        0 \
    --n_questions 500 \
    --base_model  Qwen/Qwen3-8B
```

Iteration 1 — self-generated (no teacher):

```bash
python Self_distill/scripts/self_distill_loop.py \
    --iter        1 \
    --n_questions 500 \
    --base_model  Qwen/Qwen3-8B
```

Iteration 2:

```bash
python Self_distill/scripts/self_distill_loop.py \
    --iter        2 \
    --n_questions 500 \
    --base_model  Qwen/Qwen3-8B
```

Skip individual stages when resuming after interruption:

```bash
python Self_distill/scripts/self_distill_loop.py \
    --iter 1 --skip_cot --skip_pns  # resume from training step
```

---

### Inference notebooks

Five Colab-ready notebooks are in `inference/` for interactive comparison of trained models against their base versions:

| Notebook                          | What it tests                                  |
| --------------------------------- | ---------------------------------------------- |
| `01_inference_pns.ipynb`          | Interactive PNS pruning on custom questions    |
| `02_inference_icl.ipynb`          | ICL: causal exemplars vs standard CoT          |
| `03_inference_SFT.ipynb`          | Causal SFT vs noncausal SFT vs base Qwen3-1.7B |
| `04_inference_DPO.ipynb`          | DPO-aligned model vs SFT baseline vs base      |
| `05_inference_Self_Distill.ipynb` | Iter1 vs iter2 vs base Qwen3-8B                |

---

### Evaluation

Evaluate any trained adapter on GSM8K:

```bash
python Self_distill/scripts/eval_correct.py \
    --adapter    models/sft/qwen3_fast_causal \
    --input      data/gsm8k/test.jsonl \
    --output     data/results_sft/fast/qwen3_causal_gsm8k.jsonl \
    --dataset    gsm8k
```

Print a results summary across all conditions:

```bash
python sft/summarize_all_results.py
python sft/print_results_fast.py
```

---

## Dataset

### GSM8K

- **Source**: `openai/gsm8k` on HuggingFace
- **Size**: 7,473 training questions, 1,319 test questions
- **Format**: Grade school math word problems with step-by-step answers
- **Used for**: RQ1 PNS scoring, SFT training (both conditions), DPO pair building, self-distillation
- **Local path**: `data/gsm8k/test.jsonl`

```json
{ "question": "Natalia sold clips to 48 of her friends...", "answer": "72" }
```

### MATH-500

- **Source**: `HuggingFaceH4/MATH-500` on HuggingFace
- **Size**: 500 test questions across 7 math categories (algebra, number theory, etc.)
- **Format**: Competition math with LaTeX answers wrapped in `\boxed{}`
- **Used for**: RQ1 MATH-500 token reduction verification, SFT training
- **Local path**: `data/MATH-500/test.jsonl`

```json
{ "problem": "Evaluate $\\lceil{\\sqrt{20}}\\rceil^2$.", "answer": "25" }
```

### CommonsenseQA

- **Source**: `tau/commonsense_qa` on HuggingFace
- **Size**: 1,221 test questions
- **Format**: Multiple-choice (A–E) commonsense reasoning
- **Used for**: Extension to non-math domain (uses `common_prompt` instead of `math_prompt`)
- **Local path**: `data/commonsenseqa/test.jsonl`

### Downloading all datasets

```bash
python downlaod_datasets.py
```

### Pre-scored PNS data

The repository includes pre-computed PNS outputs from Qwen3-14B:

- `data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl` — 1,319 GSM8K test questions with PNS scores
- `data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl` — 500 MATH-500 test questions with PNS scores

These can be used directly to skip the PNS scoring step.

---

## Model Details

### Inference / PNS Rollouts: Qwen3-14B (via Ollama)

The PNS algorithm requires a capable model to generate alternative reasoning steps and evaluate them. This project uses Qwen3-14B running locally via Ollama:

- **Model**: `qwen3:14b` (default, configurable via `QWEN_MODEL` env var)
- **API**: OpenAI-compatible endpoint at `http://localhost:11434/v1`
- **Thinking mode**: Disabled (`think=False`) for rollouts — deliberation weakens the model enough to give realistic PN signal
- **Context window**: 4,096 tokens (`num_ctx=4096`)
- **Concurrency**: 2 parallel requests (matches `OLLAMA_NUM_PARALLEL=2`)
- **Client**: `qwen_local.py::cerebras_query()` with 5-retry logic and 30-minute timeout

### Fine-tuned Models: Qwen3-1.7B / Qwen3-8B with QLoRA

All fine-tuned models use Parameter-Efficient Fine-Tuning via LoRA:

| Setting            | Value                          |
| ------------------ | ------------------------------ |
| Fine-tuning method | QLoRA (4-bit NF4 quantization) |
| LoRA rank (r)      | 16                             |
| LoRA alpha (α)     | 32                             |
| Target modules     | `q_proj`, `v_proj`             |
| LoRA dropout       | 0.05                           |
| Optimizer          | AdamW with cosine LR schedule  |
| Quantization       | `bitsandbytes` 4-bit NF4       |

**SFT training objectives**:

- Causal condition: train on PNS-pruned `final_chain` (PS=1 examples only)
- Noncausal condition: train on raw Qwen3-14B chains (no PNS filtering)
- Both conditions wrap output in `<think>...</think>\n\n\boxed{answer}` format

**DPO training objective**:

- Starts from merged SFT checkpoint (not raw base)
- Uses TRL `DPOTrainer` with KL penalty (β=0.1)
- Chosen: PNS-pruned correct chain; Rejected: SFT model's wrong output on the same question
- Saves full merged weights (SFT merge + DPO LoRA merge) to avoid adapter stacking

**Answer equivalence checking** (`algo/equivalent_ans.py`):

1. Fast path: extract `\boxed{}` content with brace-depth counter, normalize LaTeX (`\dfrac` → `\frac`, strip `\circ`, etc.)
2. LLM fallback: yes/no prompt to Qwen3-14B via Ollama if fast path returns None

---

## Results

### RQ1 — PNS Token Efficiency (Qwen3-14B, single RTX 5080)

| Dataset  | Original tokens | PNS tokens | Reduction | Accuracy change |
| -------- | --------------- | ---------- | --------- | --------------- |
| GSM-8K   | ~113 (14B avg)  | ~89        | **21.4%** | 0.0 pp          |
| MATH-500 | ~300+           | ~200+      | **33.3%** | 0.0 pp          |

Compared to the baseline paper (Yu et al., multi-GPU, QwQ-32B + Qwen2.5-72B):

| Metric                   | Yu et al. | This Work   |
| ------------------------ | --------- | ----------- |
| GSM-8K token reduction   | 70.2%     | 21.4%       |
| MATH-500 token reduction | 36.6%     | **33.3%** ✓ |
| Accuracy degradation     | 0.0 pp    | 0.0 pp      |

The GSM-8K gap is a verbosity difference: Qwen3-14B already writes compact 53-token chains vs QwQ-32B's 113-token chains, leaving less to prune. MATH-500 closes the gap because harder problems contain more genuine redundancy.

### RQ2a — In-Context Learning (Qwen3-8B, 1,319 GSM-8K questions)

| Method         | Accuracy   | Avg Tokens | Avg Steps |
| -------------- | ---------- | ---------- | --------- |
| Standard CoT   | 82.94%     | 808        | 3.74      |
| Reduction      | 91.66%     | 528        | 1.09      |
| CoD            | 92.72%     | 415        | 1.00      |
| **Fast-Solve** | **93.18%** | **458**    | **1.05**  |
| Ours (ICL)     | 90.83%     | 604        | 1.95      |

| Method       | Accuracy   | Avg Tokens | Avg Steps |
| ------------ | ---------- | ---------- | --------- |
| Standard CoT | 87.20%     | 3,136      | 23.14     |
| Fast-Solve   | **89.20%** | 2,290      | 11.66     |
| CoD          | 88.20%     | 1,903      | 4.25      |
| Ours (ICL)   | 88.00%     | 2,795      | 19.22     |

_(MATH-500, 500 questions, Qwen3-8B)_

### RQ2b — Supervised Fine-Tuning (Qwen3-1.7B, 1,319 GSM-8K questions)

| Model               | Accuracy  | Avg Tokens | Notes                      |
| ------------------- | --------- | ---------- | -------------------------- |
| Original Qwen3-1.7B | 66.6%     | —          | Base model                 |
| Noncausal SFT       | **83.1%** | 220        | Raw 14B chains             |
| **Causal SFT**      | 77.3%     | **168**    | PNS-filtered (−24% tokens) |
| Causal DPO v4       | 75.1%     | 172        | DPO on top of causal SFT   |

### RQ3b — Iterative Self-Distillation (Qwen3-8B, 50-question GSM-8K sample)

| Condition              | Accuracy  | Avg Tokens | Avg Steps | Token Reduction |
| ---------------------- | --------- | ---------- | --------- | --------------- |
| Original Qwen3-8B      | 66.0%     | 2,048\*    | 29.1      | —               |
| Noncausal SFT          | 92.0%     | 261        | 5.6       | −87%            |
| **Causal SFT iter1**   | **90.0%** | **143**    | **2.0**   | **−93%**        |
| **Self-distill iter2** | **90.0%** | **137**    | **2.1**   | **−93%**        |

\*Base model hits the 2,048 token generation cap (responses were truncated).

Self-distillation PNS yield across iterations:

| Iteration           | Source                | Questions | PS=1 kept | Yield | Avg steps (before → after pruning) |
| ------------------- | --------------------- | --------- | --------- | ----- | ---------------------------------- |
| Iter1 (14B teacher) | Qwen3-14B             | 1,319     | 1,133     | 86%   | ~4–5 → ~2–3                        |
| Iter2 (8B self)     | Qwen3-8B Causal iter1 | 100       | 62        | 62%   | 1.6 → 1.1                          |

The drop from 86% to 62% yield is expected: the iter1 model already generates concise chains, leaving PNS with little to prune. This is the self-distillation convergence signal.

### DPO — Honest Negative Result

DPO consistently degrades accuracy by ~2.2pp relative to the SFT baseline across four versions (v1–v4) with 300–390 pairs. Root causes: (1) 390 pairs cover ~5% of the training distribution — too sparse for a 1.7B model already near its capacity ceiling; (2) the model cannot fully leverage the preference signal at this scale. This is reported as a genuine finding: it establishes the minimum scale and data requirements for PNS-derived preferences to transfer.

---

## Troubleshooting

### Ollama timeout or connection refused

```
RuntimeError: Ollama failed after 5 attempts
```

Ensure Ollama is running before starting any PNS or inference script:

```bash
ollama serve &
ollama run qwen3:14b "test"   # warm up the model
```

### `\n\n` not splitting CoT into steps

If `step_length` is 1 for every question, the CoT contains literal `\\n\\n` escape sequences instead of real newlines. `run_algo_o.py` handles this automatically with:

```python
model_cot = model_cot.replace("\\n\\n", "\n\n")
```

If generating your own CoT files, ensure steps are separated by actual double newlines.

### Chain reverts to original after pruning

The PNS algorithm reverts to the original chain if pruning breaks a previously correct answer (`PS` drops from 1 to 0). This is correct behavior per Algorithm 1 of the paper — the revert prevents accuracy degradation.

### DPOTrainer `log()` TypeError

```
TypeError: log() takes 2 positional arguments but 3 were given
```

This is a version mismatch between transformers and TRL. The fix is already applied in `dpo/train_dpo.py` via a monkey-patch of `trainer.log`. Ensure `trl>=0.8.0` and `transformers>=4.45.0`.

### TRL Unicode crash on Windows

```
UnicodeDecodeError in trl/chat_template_utils.py
```

All `read_text()` calls in TRL need `encoding="utf-8"`. Add this to the affected lines in your TRL installation, or switch to Linux/WSL2 where UTF-8 is the default.

### CUDA out of memory during SFT

Reduce `--max_seq_len` to 512 or 768, or lower `--batch_size` to 1 and increase `--grad_accum`. The 8B model requires 4-bit quantization (`--load_in_4bit`) on 16 GB VRAM.

### PN always 0 (steps never pruned)

This happens when the rollout model re-reads the original question and re-derives the answer independently of the partial chain, making every continuation correct regardless of the candidate step. The `evaluate_replacement_step()` function uses a strong `CRITICAL: Continue STRICTLY from the given steps` constraint to prevent this. If you see PN≈0 across all steps, the Ollama model may not be respecting the continuation instruction — try `thinking=False` (already set in `qwen_local.py`).

---

## Contributing

### Issues

Report bugs or request features by opening a GitHub issue at the repository URL. When reporting a bug, include:

- The exact command you ran
- The relevant section of the log output
- Your OS, GPU, CUDA version, and Python version

### Pull Requests

1. Fork the repository and create a branch from `main`
2. Follow the existing code style: flat functions over classes, explicit argument lists, no undocumented magic values
3. Keep changes minimal — do not refactor unrelated code in the same PR
4. For PNS algorithm changes, test on at least 5 GSM8K questions and report before/after PN values

### Code conventions observed in this project

- JSONL for all data files (one JSON object per line, UTF-8)
- `argparse` for all script CLIs — every parameter must have a flag
- `ThreadPoolExecutor(max_workers=2)` for parallelism (matches Ollama's parallel capacity)
- All output files are append-safe — scripts check `lines_processed` and resume mid-file
- LoRA adapters saved separately from merged models — DPO always merges before saving

---

## License

TODO: Add a LICENSE file. No license file was detected in this repository.

---

## Citation

**Base paper** (Yu et al., NeurIPS 2025):

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

**This reproduction and extension**:

```bibtex
@misc{awan2025pnscot,
  title       = {Keeping Only What Counts: Causal PNS-Guided CoT Optimisation
                 with Automatic DPO Preference Alignment and Iterative Self-Distillation},
  author      = {Awan, Ayesha Tahir and Khan, Ahmed and Zahid, Mariam},
  institution = {FAST-NUCES, Islamabad},
  year        = {2025}
}
```

---

© 2025 Ayesha Tahir Awan, Mariam Zahid, Ahmed Khan — FAST-NUCES, Islamabad. All rights reserved.
