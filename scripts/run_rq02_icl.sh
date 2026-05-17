#!/usr/bin/env bash
# RQ02 ICL — run all 5 methods × 2 models × 2 datasets on RTX 5080
# Models: qwen3-14b (primary), qwen3-8b (secondary), llama3.1-8b (paper baseline)
# Datasets: math500, gsm8k
#
# Before running:
#   ollama pull qwen3:8b
#   ollama pull llama3.1:8b
#   mkdir -p data/results_icl
#
# Usage:
#   bash expt/run_rq02_icl.sh 2>&1 | tee expt/rq02_icl_log.txt

set -e
PYTHON="/c/Users/ayesha/anaconda3/envs/pns_venv/python.exe"
RUN="$PYTHON expt/run_icl.py"
OUTDIR="data/results_icl"
mkdir -p "$OUTDIR"

# ── Datasets & their PNS files ────────────────────────────────────────────────
declare -A PNS_FILES=(
    ["math500"]="data/gsm8k/gsm8k_pns_qwen3-14b-thinking-v2.jsonl"
    ["gsm8k"]="data/MATH-500/math500_pns_qwen3-v2-pb-fixed.jsonl"
)
declare -A INPUTS=(
    ["math500"]="data/MATH-500/test.jsonl"
    ["gsm8k"]="data/gsm8k/test.jsonl"
)

METHODS=("standard_cot" "fast_solve" "reduction" "cod" "ours_icl")
MODELS=("qwen3-14b" "qwen3-8b" "llama3.1-8b")
DATASETS=("math500" "gsm8k")

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    INPUT="${INPUTS[$DATASET]}"
    PNS="${PNS_FILES[$DATASET]}"
    for METHOD in "${METHODS[@]}"; do
      OUTPUT="$OUTDIR/${DATASET}_${METHOD}_${MODEL}.jsonl"
      echo ""
      echo "════════════════════════════════════════════"
      echo "  Model=$MODEL  Dataset=$DATASET  Method=$METHOD"
      echo "  Output: $OUTPUT"
      echo "════════════════════════════════════════════"

      if [ "$METHOD" = "ours_icl" ]; then
        $RUN \
          --input "$INPUT" \
          --output "$OUTPUT" \
          --method "$METHOD" \
          --dataset "$DATASET" \
          --model "$MODEL" \
          --pns_file "$PNS" \
          --n_examples 3 \
          --workers 1
      else
        $RUN \
          --input "$INPUT" \
          --output "$OUTPUT" \
          --method "$METHOD" \
          --dataset "$DATASET" \
          --model "$MODEL" \
          --workers 1
      fi
    done

    echo ""
    echo "── Results for Model=$MODEL Dataset=$DATASET ──"
    $PYTHON expt/eval_icl.py \
      --results_dir "$OUTDIR" \
      --model "$MODEL" \
      --dataset "$DATASET"
  done
done

echo ""
echo "All ICL experiments complete."
