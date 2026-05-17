# Model Checkpoints

Model weights are hosted on Google Drive due to file size constraints (GitHub LFS not configured).

## Available Checkpoints

| Checkpoint | Description | Size | Drive Link |
|------------|-------------|------|------------|
| `qwen3_fast_causal/` | Qwen3-1.7B LoRA adapter — Causal SFT (PNS-pruned chains) | ~50 MB | See below |
| `qwen3_fast_noncausal/` | Qwen3-1.7B LoRA adapter — Noncausal SFT (full CoT chains) | ~50 MB | See below |
| `qwen3_causal_dpo_v4/` | Qwen3-1.7B fully merged DPO model (SFT + DPO merged) | ~3.3 GB | See below |
| `self_distill/iter1/` | Qwen3-8B LoRA adapter — Causal iter1 (teacher-seeded) | ~200 MB | See below |
| `self_distill/iter2/` | Qwen3-8B LoRA adapter — Self-distill iter2 (no teacher) | ~200 MB | See below |

## Loading Instructions

### SFT LoRA Adapter (Causal / Noncausal)
```python
from src.model import load_sft_adapter
model, tokenizer = load_sft_adapter(
    base_model_name="Qwen/Qwen3-1.7B",
    adapter_path="checkpoints/qwen3_fast_causal"
)
```

### Merged DPO Model
```python
from src.model import load_merged_dpo_model
model, tokenizer = load_merged_dpo_model(
    model_path="checkpoints/qwen3_causal_dpo_v4",
    quantise=True
)
```

### Self-Distillation (Qwen3-8B, 4-bit)
```python
from src.model import load_sft_adapter
model, tokenizer = load_sft_adapter(
    base_model_name="Qwen/Qwen3-8B",
    adapter_path="checkpoints/self_distill/iter2",
    quantise=True   # 4-bit NF4 for 8B model
)
```

## Google Drive

Model weights are available in the shared Google Drive folder linked in the project submission.
Download and place the contents into this `checkpoints/` directory before running inference.

See `inference.py` or the notebooks in `Inference Notebooks/` for full download automation via `gdown`.
