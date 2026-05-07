# ============================================================
# Combined Causal SFT + DPO Pipeline
#
# Novelty: DPO with COT-chosen preference learning
#   - Causal SFT compresses reasoning (PNS chains ~58 words avg)
#   - 1.7B model drops accuracy on hard problems (skips steps)
#   - DPO fix: chosen=COT (correct + verbose), rejected=SFT wrong chain
#   - DPO teaches: prefer correct step-by-step over wrong compressed output
#   - Result: accuracy recovers while staying more efficient than original
#
# Stages:
#   1. Generate rejected chains (wrong answers from SFT-causal model)
#   2. Build DPO preference pairs (COT chosen vs wrong rejected)
#   3. DPO training on top of SFT-causal checkpoint
#   4. Evaluate: Original vs SFT-Causal vs DPO-Causal
# ============================================================

$PYTHON  = "C:\Users\ayesha\anaconda3\envs\dpo_env\python.exe"
$RESULTS = "data/results_sft/fast"
$LOGS    = "logs"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"

New-Item -ItemType Directory -Force -Path $RESULTS | Out-Null
New-Item -ItemType Directory -Force -Path $LOGS    | Out-Null
New-Item -ItemType Directory -Force -Path "data/dpo" | Out-Null

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COMBINED SFT + DPO PIPELINE" -ForegroundColor Cyan
Write-Host "  Causal SFT -> DPO (Causal preference)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Pre-check: SFT model must exist
if (-not (Test-Path "models/sft/qwen3_fast_causal/adapter_config.json")) {
    Write-Host "ERROR: SFT model not found. Run run_sft_fast.ps1 first." -ForegroundColor Red
    exit 1
}
Write-Host "  SFT model found: models/sft/qwen3_fast_causal" -ForegroundColor Green

# ── Stage 1: Generate rejected chains ───────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  1/4 GENERATE REJECTED CHAINS" -ForegroundColor Yellow
Write-Host "  (wrong answers from SFT model, temp=0.8)" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON dpo/gen_rejected.py `
    --adapter       models/sft/qwen3_fast_causal `
    --input         data/sft/fast/causal_train.jsonl `
    --output        data/dpo/rejected_chains.jsonl `
    --n_samples     3 `
    --temperature   0.8 `
    --max_questions 800 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_gen_rejected_$timestamp.txt"

$rejected_count = (Get-Content data/dpo/rejected_chains.jsonl | Measure-Object -Line).Lines
Write-Host "  Rejected chains found: $rejected_count" -ForegroundColor Green

# ── Stage 2: Build DPO pairs ─────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  2/4 BUILD DPO PREFERENCE PAIRS" -ForegroundColor Yellow
Write-Host "  chosen=COT (correct+verbose), rejected=wrong SFT-causal" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON dpo/prepare_dpo_pairs.py `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_prepare_pairs_$timestamp.txt"

$pairs_count = (Get-Content data/dpo/dpo_pairs.jsonl | Measure-Object -Line).Lines
Write-Host "  DPO pairs ready: $pairs_count" -ForegroundColor Green

# ── Stage 3: DPO training ────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  3/4 DPO TRAINING (SFT -> DPO)" -ForegroundColor Yellow
Write-Host "  Starting from SFT-causal checkpoint" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Yellow

& $PYTHON dpo/train_dpo.py `
    --sft_adapter   models/sft/qwen3_fast_causal `
    --data          data/dpo/dpo_pairs.jsonl `
    --output_dir    models/dpo/qwen3_causal_dpo `
    --beta          0.1 `
    --epochs        2 `
    --batch_size    1 `
    --grad_accum    8 `
    --lr            5e-5 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_train_$timestamp.txt"

# ── Stage 4: Evaluate DPO model ──────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  4/4 EVAL DPO MODEL" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

Write-Host "  >> GSM8K"
& $PYTHON sft/eval_correct.py `
    --adapter        models/dpo/qwen3_causal_dpo `
    --input          data/gsm8k/test.jsonl `
    --output         "$RESULTS/qwen3_causal_dpo_gsm8k.jsonl" `
    --dataset        gsm8k `
    --max_new_tokens 4096 `
    --batch_size     4 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_eval_gsm8k_$timestamp.txt"

Write-Host "  >> MATH-500"
& $PYTHON sft/eval_correct.py `
    --adapter        models/dpo/qwen3_causal_dpo `
    --input          data/MATH-500/test.jsonl `
    --output         "$RESULTS/qwen3_causal_dpo_math500.jsonl" `
    --dataset        math500 `
    --max_new_tokens 8192 `
    --batch_size     2 `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_eval_math500_$timestamp.txt"

# ── Final comparison table ───────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  FINAL RESULTS: Original vs SFT vs DPO" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

& $PYTHON dpo/print_dpo_results.py --results_dir $RESULTS `
    2>&1 | Tee-Object -FilePath "$LOGS/dpo_results_$timestamp.txt"

Write-Host ""
Write-Host "Pipeline complete!" -ForegroundColor Green
