"""
src/dataset.py
Dataset loading and preprocessing utilities.
Handles GSM-8k, MATH-500, and SFT/DPO data formats.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset


# ── Benchmark Loaders ────────────────────────────────────────

def load_gsm8k(split: str = "test"):
    """Load GSM-8k from HuggingFace. Returns list of {question, answer} dicts."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]


def load_math500(split: str = "test"):
    """Load MATH-500 from HuggingFace. Returns list of {problem, solution} dicts."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    return [{"question": row["problem"], "answer": row["solution"]} for row in ds]


# ── JSONL Helpers ────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: str):
    """Save a list of dicts as a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Answer Extraction ─────────────────────────────────────────

def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} LaTeX notation (MATH-500)."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def extract_hash_answer(text: str) -> str | None:
    """Extract answer after #### delimiter (GSM-8k)."""
    match = re.search(r"####\s*([\d,.\-]+)", text)
    return match.group(1).replace(",", "").strip() if match else None


def extract_answer(text: str) -> str | None:
    """Try \\boxed{} first, then #### fallback."""
    return extract_boxed_answer(text) or extract_hash_answer(text)


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> block from Qwen3 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── SFT Data Formatting ───────────────────────────────────────

def format_sft_sample(question: str, cot: str, answer: str,
                       enable_thinking: bool = False) -> dict:
    """
    Format a single training sample for SFT.
    enable_thinking=True  → Noncausal condition (model uses <think> block)
    enable_thinking=False → Causal condition (compact non-thinking output)
    """
    system = "You are a helpful math assistant. Solve step by step."
    if enable_thinking:
        assistant_content = f"<think>\n{cot}\n</think>\n\nThe answer is {answer}."
    else:
        assistant_content = f"{cot}\n\nThe answer is {answer}."
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def format_dpo_sample(question: str, chosen: str, rejected: str) -> dict:
    """Format a preference pair for DPO training."""
    system = "You are a helpful math assistant. Solve step by step."
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }
