# ============================================================
# DPO Pipeline - GSM8K ONLY (fast version)
#
# chosen=COT (correct+verbose)  rejected=SFT-causal wrong chains
# Only uses GSM8K data - skips MATH-500 to save time
#
# Prerequisite: run_sft_fast.ps1 must have completed (SFT model needed)
# ============================================================

$PYTHON    = "C:\Users\ayesha\anaconda3\envs\dpo_env\python.exe"
$RESULTS   = "data/results_sft/fast"
$LOGS      = "logs"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"

New-Item -ItemType Directory -Force -Path $RESULTS    | Out-Null
New-Item -ItemType Directory -Force -Path $LOGS       | Out-Null
New-Item -ItemType Directory -Force -Path "data/dpo"  | Out-Null

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DPO PIPELINE  (GSM8K only)" -ForegroundColor Cyan
Write-Host "  chosen=COT  |  rejected=wrong SFT-causal" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Cyan

# Pre-check
if (-not (Test-Path "models/sft/qwen3_fast_causal/adapter_config.json")) {
    Write-Host "ERROR: SFT model not found at models/sft/qwen3_fast_causal" -ForegroundColor Red
    exit 1
}
Write-Host "  SFT model found" -ForegroundColor Green

# ── Step 0: Filter causal_train.jsonl to GSM8K only ─────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  0/4  FILTER GSM8K CAUSAL TRAIN DATA" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON -c @"
import json, re

# GSM8K questions are numeric (e.g. 'Janet's ducks lay...')
# MATH questions contain LaTeX (dollar signs, \frac, etc.)
def is_math(q):
    return bool(re.search(r'\\\w+|\$.*\$|\^|\\frac|\\sqrt', q))

recs = [json.loads(l) for l in open('data/sft/fast/causal_train.jsonl', encoding='utf-8') if l.strip()]
gsm = [r for r in recs if not is_math(r['messages'][0]['content'])]
with open('data/dpo/gsm8k_causal_train.jsonl', 'w', encoding='utf-8') as f:
    for r in gsm:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Total causal: {len(recs)}  GSM8K only: {len(gsm)}')
"@

$gsm_count = (Get-Content data/dpo/gsm8k_causal_train.jsonl | Measure-Object -Line).Lines
Write-Host "  GSM8K causal records: $gsm_count" -ForegroundColor Green

# ── Stage 1: Generate rejected chains (GSM8K only) ──────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  1/4  GENERATE REJECTED CHAINS (GSM8K)" -ForegroundColor Yellow
Write-Host "  Wrong answers from SFT-causal, temp=0.8" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON dpo/gen_rejected.py `
    --adapter       models/sft/qwen3_fast_causal `
    --input         data/dpo/gsm8k_causal_train.jsonl `
    --output        data/dpo/rejected_chains_gsm8k.jsonl `
    --n_samples     3 `
    --temperature   0.8 `
    --max_questions 800 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_gen_rejected_$timestamp.txt"

$rejected_count = (Get-Content data/dpo/rejected_chains_gsm8k.jsonl | Measure-Object -Line).Lines
Write-Host "  Rejected chains found: $rejected_count" -ForegroundColor Green

# ── Stage 2: Build DPO pairs (GSM8K) ────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  2/4  BUILD DPO PREFERENCE PAIRS" -ForegroundColor Yellow
Write-Host "  chosen=COT  rejected=wrong SFT-causal" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON dpo/prepare_dpo_pairs.py `
    --rejected  data/dpo/rejected_chains_gsm8k.jsonl `
    --output    data/dpo/dpo_pairs_gsm8k.jsonl `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_prepare_pairs_$timestamp.txt"

$pairs_count = (Get-Content data/dpo/dpo_pairs_gsm8k.jsonl | Measure-Object -Line).Lines
Write-Host "  DPO pairs ready: $pairs_count" -ForegroundColor Green

# ── Stage 3: DPO training ────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  3/4  DPO TRAINING" -ForegroundColor Yellow
Write-Host "  Starting from SFT-causal checkpoint" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON dpo/train_dpo.py `
    --sft_adapter   models/sft/qwen3_fast_causal `
    --data          data/dpo/dpo_pairs_gsm8k.jsonl `
    --output_dir    models/dpo/qwen3_causal_dpo `
    --beta          0.1 `
    --epochs        2 `
    --batch_size    1 `
    --grad_accum    8 `
    --lr            5e-5 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_train_$timestamp.txt"

# ── Stage 4: Eval DPO on GSM8K only ─────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  4/4  EVAL DPO - GSM8K" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON sft/eval_correct.py `
    --adapter        models/dpo/qwen3_causal_dpo `
    --input          data/gsm8k/test.jsonl `
    --output         "$RESULTS/qwen3_causal_dpo_gsm8k.jsonl" `
    --dataset        gsm8k `
    --max_new_tokens 4096 `
    --batch_size     4 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_eval_gsm8k_$timestamp.txt"

# ── Print results ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  RESULTS: Original vs SFT vs DPO (GSM8K)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

& $PYTHON dpo/print_dpo_results.py --results_dir $RESULTS `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_results_$timestamp.txt"

Write-Host ""
Write-Host "Done! DPO pipeline (GSM8K) complete." -ForegroundColor Green
