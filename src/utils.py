"""
src/utils.py
Shared utility functions for evaluation, logging, and metrics.
"""

import json
import csv
import time
from pathlib import Path
from typing import Optional


# ── Answer Comparison ─────────────────────────────────────────

def normalise_number(s: str) -> Optional[float]:
    """Try to parse a string as a float for numeric comparison."""
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def answers_match(pred: str, gold: str) -> bool:
    """
    Check if predicted answer matches gold answer.
    Tries numeric equality first, falls back to string equality.
    """
    if pred is None:
        return False
    pred_n, gold_n = normalise_number(pred), normalise_number(gold)
    if pred_n is not None and gold_n is not None:
        return abs(pred_n - gold_n) < 1e-6
    return str(pred).strip().lower() == str(gold).strip().lower()


# ── Token Counting ────────────────────────────────────────────

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a string using a HuggingFace tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_steps(cot: str) -> int:
    """Estimate reasoning step count by counting sentence-ending punctuation."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", cot.strip())
    return max(1, len([s for s in sentences if len(s.strip()) > 5]))


# ── Results Logging ───────────────────────────────────────────

def save_results_json(results: dict, path: str):
    """Save a results dict as pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {path}")


def append_to_csv(row: dict, path: str):
    """Append a row dict to a CSV file, creating headers if new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Evaluation Summary ────────────────────────────────────────

def compute_metrics(predictions: list, golds: list) -> dict:
    """
    Compute accuracy and token stats from parallel prediction/gold lists.

    Each element should be a dict with keys: 'pred_answer', 'gold_answer',
    'tokens' (optional), 'steps' (optional).
    """
    assert len(predictions) == len(golds), "Lengths must match"
    correct = sum(
        1 for p, g in zip(predictions, golds)
        if answers_match(p.get("pred_answer"), g)
    )
    total = len(golds)
    tokens = [p.get("tokens", 0) for p in predictions if p.get("tokens")]
    steps = [p.get("steps", 0) for p in predictions if p.get("steps")]
    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "avg_steps": round(sum(steps) / len(steps), 2) if steps else None,
    }


# ── Timing ───────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer for experiment logging."""
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> str:
        secs = int(time.time() - self._start)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def reset(self):
        self._start = time.time()
