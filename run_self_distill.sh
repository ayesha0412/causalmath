#!/usr/bin/env bash
# ============================================================
#  Self-Distillation Loop  —  3 iterations on GSM8K
#
#  PRE-REQUISITES (two terminals needed):
#    Terminal A — Ollama (qwen3:14b for CoT + PNS rollouts):
#      ollama serve --port 11436
#      ollama run qwen3:14b   (first time only, to pull model)
#
#    Terminal B — this script:
#      conda activate dpo_env
#      cd <project root>
#      bash run_self_distill.sh
#
#  WHAT EACH ITERATION DOES:
#    iter 0  seed  — Ollama generates 500 CoT chains, PNS prunes them,
#                    trains Qwen3-1.7B on self-curated data
#    iter 1  self  — Fine-tuned model (iter 0) generates 500 new chains,
#                    PNS prunes, retrains on iter 0 + iter 1 data combined
#    iter 2  self  — Fine-tuned model (iter 1) generates 500 more chains,
#                    PNS prunes, retrains on all 3 iters combined
#
#  SKIP FLAGS (resume partial runs):
#    --skip_cot    CoT JSONL already exists
#    --skip_pns    PNS JSONL already exists
#    --skip_train  Skip training (eval only)
#    --skip_eval   Skip evaluation
#
#  USAGE:
#    bash run_self_distill.sh              # full 3-iter loop (iter 0-2)
#    bash run_self_distill.sh 1 2          # only iter 1 and 2
#    bash run_self_distill.sh 0 0          # only iter 0
# ============================================================

set -e

PYTHON="C:/Users/ayesha/anaconda3/envs/dpo_env/python.exe"
BASE_MODEL="Qwen/Qwen3-1.7B"
N_QUESTIONS=500     # questions to sample per iteration (7473 available)
SEED=42

START_ITER=${1:-0}
END_ITER=${2:-2}

echo "============================================================"
echo "  SELF-DISTILLATION LOOP"
echo "  Base model      : $BASE_MODEL"
echo "  Questions/iter  : $N_QUESTIONS"
echo "  Iterations      : $START_ITER → $END_ITER"
echo "  State file      : data/self_distill/state.json"
echo "============================================================"
echo ""

for ITER in $(seq "$START_ITER" "$END_ITER"); do
    echo ""
    echo "############################################################"
    echo "  ITERATION $ITER / $END_ITER   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"

    $PYTHON sft/self_distill_loop.py \
        --iter        "$ITER"        \
        --n_questions "$N_QUESTIONS" \
        --base_model  "$BASE_MODEL"  \
        --seed        "$SEED"

    echo "  ✓ Iteration $ITER done"
done

echo ""
echo "============================================================"
echo "  ALL ITERATIONS COMPLETE"
echo ""
echo "  Adapters : models/self_distill/iter_{0,1,2}/"
echo "  Data     : data/self_distill/"
echo "  Results  : data/results_sft/self_distill/"
echo "============================================================"
echo ""

# Print accuracy table across iterations
echo "  Accuracy summary:"
for ITER in $(seq "$START_ITER" "$END_ITER"); do
    RESULT="data/results_sft/self_distill/iter${ITER}_gsm8k.jsonl"
    if [[ -f "$RESULT" ]]; then
        $PYTHON - <<PYEOF
import json
results = [json.loads(l) for l in open("$RESULT", encoding="utf-8") if l.strip()]
correct = sum(1 for r in results if r.get("is_correct", False))
total   = len(results)
acc     = 100 * correct / total if total else 0
print(f"  iter $ITER : {correct}/{total} = {acc:.1f}%  GSM8K")
PYEOF
    fi
done
echo ""
