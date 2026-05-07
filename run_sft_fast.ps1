# ============================================================
# Fast SFT Pipeline: Qwen3-1.7B  |  GSM8K + MATH-500
# Causal (PNS PS=1) vs Noncausal (COT) vs Baseline
#
# Data splits (NO leakage):
#   GSM8K train:  1100 causal + 1210 noncausal
#   GSM8K test:   data/gsm8k/test.jsonl (1319 standard benchmark)
#   MATH train:    346 causal +  397 noncausal  (from 400-example split)
#   MATH test:    data/MATH-500/test.jsonl (100 held-out)
#
# Output:
#   models/sft/qwen3_fast_causal/
#   models/sft/qwen3_fast_noncausal/
#   data/results_sft/fast/qwen3_{original,causal,noncausal}_{gsm8k,math500}.jsonl
#   logs/sft_fast_train_*.txt
#   logs/sft_fast_eval_*.txt
# ============================================================

$PYTHON  = "C:\Users\ayesha\anaconda3\envs\dpo_env\python.exe"
$RESULTS = "data/results_sft/fast"
$LOGS    = "logs"
$MODEL   = "Qwen/Qwen3-1.7B"

New-Item -ItemType Directory -Force -Path $RESULTS | Out-Null
New-Item -ItemType Directory -Force -Path $LOGS    | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"

# ── Helper ───────────────────────────────────────────────────────────────────
function Run-Step($label, $logFile, $scriptBlock) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  $label" -ForegroundColor Cyan
    Write-Host "  Log: $logFile" -ForegroundColor DarkGray
    Write-Host "========================================" -ForegroundColor Cyan
    & $scriptBlock 2>&1 | Tee-Object -FilePath $logFile
    Write-Host "  Done: $label" -ForegroundColor Green
}

# ── 0. Write data manifest ────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  0/7 DATA MANIFEST" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

$manifest = @"
# SFT Fast Pipeline - Data Manifest
# Generated: $timestamp

## Training Data
- Causal:    data/sft/fast/causal_train.jsonl
  - GSM8K: 1100 (PNS, PS=1 filter)
  - MATH:   346 (PNS, PS=1 filter, from 400-example training split)
  - Total: 1446

- Noncausal: data/sft/fast/noncausal_train.jsonl
  - GSM8K: 1210 (115 new Qwen3-14B + 1100 COT)
  - MATH:   397 (COT from 400-example training split)
  - Total: 1607

## Test Data (NO overlap with training)
- GSM8K:    data/gsm8k/test.jsonl (136 held-out, zero overlap)
- MATH-500: data/MATH-500/test.jsonl (100 held-out examples)

## Sources
- GSM8K PNS:  data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl
- GSM8K COT:  data/gsm8k/gsm8k_cot_qwen3-14b-thinking-v2.jsonl
- GSM8K NEW:  data/gsm8k/gsm8k_cot_train_detailed.jsonl (Qwen3-14B, new)
- MATH PNS:   data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl
- MATH COT:   data/MATH-500/test_math500_answered_qwen3-14B-thinking-v2.jsonl
- MATH split: data/MATH-500/train_math500_split.jsonl (400 training questions)

## Model
- Base: Qwen/Qwen3-1.7B
- LoRA r=16, alpha=32, target=q_proj+v_proj
- Epochs: 3, batch_size: 2, grad_accum: 4, lr: 2e-4
"@

$manifest | Out-File -FilePath "$LOGS/data_manifest_$timestamp.txt" -Encoding utf8
Write-Host $manifest

# ── 1. Prepare data ──────────────────────────────────────────────────────────
Run-Step "1/7 PREPARE DATA" "$LOGS/sft_fast_prepare_$timestamp.txt" {
    & $PYTHON sft/prepare_sft_fast.py
}

# Verify training files
Write-Host "`nTraining file sizes:"
(Get-Content data/sft/fast/causal_train.jsonl   | Measure-Object -Line).Lines | ForEach-Object { Write-Host "  causal_train.jsonl:    $_ records" }
(Get-Content data/sft/fast/noncausal_train.jsonl | Measure-Object -Line).Lines | ForEach-Object { Write-Host "  noncausal_train.jsonl: $_ records" }

# ── 2. Train causal ──────────────────────────────────────────────────────────
Run-Step "2/7 TRAIN CAUSAL (PNS PS=1 chains)" "$LOGS/sft_fast_train_causal_$timestamp.txt" {
    & $PYTHON sft/train_correct.py `
        --model       $MODEL `
        --data        data/sft/fast/causal_train.jsonl `
        --output_dir  models/sft/qwen3_fast_causal `
        --condition   causal `
        --epochs      3 `
        --batch_size  2 `
        --grad_accum  4 `
        --lr          2e-4 `
        --max_seq_len 2048
}

# ── 3. Train noncausal ───────────────────────────────────────────────────────
Run-Step "3/7 TRAIN NONCAUSAL (COT chains)" "$LOGS/sft_fast_train_noncausal_$timestamp.txt" {
    & $PYTHON sft/train_correct.py `
        --model       $MODEL `
        --data        data/sft/fast/noncausal_train.jsonl `
        --output_dir  models/sft/qwen3_fast_noncausal `
        --condition   noncausal `
        --epochs      3 `
        --batch_size  2 `
        --grad_accum  4 `
        --lr          2e-4 `
        --max_seq_len 2048
}

# ── 4. Eval baseline ─────────────────────────────────────────────────────────
Run-Step "4/7 EVAL BASELINE (original Qwen3-1.7B)" "$LOGS/sft_fast_eval_baseline_$timestamp.txt" {
    Write-Host "  >> GSM8K (136 held-out)"
    & $PYTHON sft/eval_correct.py `
        --base_model     $MODEL `
        --input          data/gsm8k/test.jsonl `
        --output         "$RESULTS/qwen3_original_gsm8k.jsonl" `
        --dataset        gsm8k `
        --max_new_tokens 4096 `
        --batch_size     4

    Write-Host "  >> MATH-500 (100 held-out)"
    & $PYTHON sft/eval_correct.py `
        --base_model     $MODEL `
        --input          data/MATH-500/test.jsonl `
        --output         "$RESULTS/qwen3_original_math500.jsonl" `
        --dataset        math500 `
        --max_new_tokens 8192 `
        --batch_size     2
}

# ── 5. Eval causal ───────────────────────────────────────────────────────────
Run-Step "5/7 EVAL CAUSAL ADAPTER" "$LOGS/sft_fast_eval_causal_$timestamp.txt" {
    Write-Host "  >> GSM8K (136 held-out)"
    & $PYTHON sft/eval_correct.py `
        --adapter        models/sft/qwen3_fast_causal `
        --input          data/gsm8k/test.jsonl `
        --output         "$RESULTS/qwen3_causal_gsm8k.jsonl" `
        --dataset        gsm8k `
        --max_new_tokens 4096 `
        --batch_size     4

    Write-Host "  >> MATH-500 (100 held-out)"
    & $PYTHON sft/eval_correct.py `
        --adapter        models/sft/qwen3_fast_causal `
        --input          data/MATH-500/test.jsonl `
        --output         "$RESULTS/qwen3_causal_math500.jsonl" `
        --dataset        math500 `
        --max_new_tokens 8192 `
        --batch_size     2
}

# ── 6. Eval noncausal ────────────────────────────────────────────────────────
Run-Step "6/7 EVAL NONCAUSAL ADAPTER" "$LOGS/sft_fast_eval_noncausal_$timestamp.txt" {
    Write-Host "  >> GSM8K (136 held-out)"
    & $PYTHON sft/eval_correct.py `
        --adapter        models/sft/qwen3_fast_noncausal `
        --input          data/gsm8k/test.jsonl `
        --output         "$RESULTS/qwen3_noncausal_gsm8k.jsonl" `
        --dataset        gsm8k `
        --max_new_tokens 4096 `
        --batch_size     4

    Write-Host "  >> MATH-500 (100 held-out)"
    & $PYTHON sft/eval_correct.py `
        --adapter        models/sft/qwen3_fast_noncausal `
        --input          data/MATH-500/test.jsonl `
        --output         "$RESULTS/qwen3_noncausal_math500.jsonl" `
        --dataset        math500 `
        --max_new_tokens 8192 `
        --batch_size     2
}

# ── 7. Print results table ───────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  7/7 RESULTS TABLE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

& $PYTHON sft/print_results_fast.py --results_dir $RESULTS 2>&1 | Tee-Object -FilePath "$LOGS/sft_fast_results_$timestamp.txt"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL FILES FOR REPRODUCIBILITY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Training data:"
Write-Host "  data/sft/fast/causal_train.jsonl"
Write-Host "  data/sft/fast/noncausal_train.jsonl"
Write-Host "Test data:"
Write-Host "  data/gsm8k/test.jsonl"
Write-Host "  data/MATH-500/test.jsonl"
Write-Host "Results (JSONL, one record per question):"
Write-Host "  $RESULTS/qwen3_original_gsm8k.jsonl"
Write-Host "  $RESULTS/qwen3_causal_gsm8k.jsonl"
Write-Host "  $RESULTS/qwen3_noncausal_gsm8k.jsonl"
Write-Host "  $RESULTS/qwen3_original_math500.jsonl"
Write-Host "  $RESULTS/qwen3_causal_math500.jsonl"
Write-Host "  $RESULTS/qwen3_noncausal_math500.jsonl"
Write-Host "Logs:"
Write-Host "  $LOGS/data_manifest_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_prepare_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_train_causal_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_train_noncausal_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_eval_baseline_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_eval_causal_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_eval_noncausal_$timestamp.txt"
Write-Host "  $LOGS/sft_fast_results_$timestamp.txt"
Write-Host ""
Write-Host "Pipeline complete!" -ForegroundColor Green
