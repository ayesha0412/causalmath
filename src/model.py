"""
src/model.py
Model loading utilities for PNS-guided CoT experiments.
Supports: base model, LoRA SFT adapter, merged DPO model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def get_bnb_config(load_in_4bit: bool = True) -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config for large models (8B+)."""
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(model_name: str, quantise: bool = False, device_map: str = "auto"):
    """
    Load a base causal LM.

    Args:
        model_name: HuggingFace model ID or local path.
        quantise:   If True, apply 4-bit NF4 quantisation (for 8B+ models).
        device_map: Device placement strategy.

    Returns:
        (model, tokenizer)
    """
    bnb_cfg = get_bnb_config() if quantise else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16 if not quantise else None,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_sft_adapter(base_model_name: str, adapter_path: str, quantise: bool = False):
    """
    Load a LoRA SFT adapter on top of a base model.

    Args:
        base_model_name: HuggingFace model ID for the base.
        adapter_path:    Local path to the LoRA adapter directory.
        quantise:        Whether to load base in 4-bit.

    Returns:
        (peft_model, tokenizer)
    """
    base_model, tokenizer = load_base_model(base_model_name, quantise=quantise)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer


def load_merged_dpo_model(model_path: str, quantise: bool = True):
    """
    Load a fully merged DPO model (SFT + DPO merged into base weights).
    No PEFT required — weights are already merged.

    Args:
        model_path: Local directory containing model.safetensors + config.json.
        quantise:   Whether to apply 4-bit quantisation.

    Returns:
        (model, tokenizer)
    """
    return load_base_model(model_path, quantise=quantise)


def model_info(model) -> dict:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
    }
