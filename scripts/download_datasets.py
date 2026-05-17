"""
download_datasets.py
====================
Run from repo root:
    python download_datasets.py

Downloads and saves test.jsonl for:
    - MATH-500
    - CommonsenseQA
    - AIME
"""

import json
import os
import time
from datasets import load_dataset

# Set reasonable timeouts for HuggingFace Hub downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # 2 minutes per file
os.environ["HF_DATASETS_OFFLINE"] = "0"


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def load_dataset_with_retry(repo_id, split=None, config=None, max_retries=3):
    """Load dataset with retry logic for network issues."""
    for attempt in range(max_retries):
        try:
            if config and split:
                return load_dataset(repo_id, config, split=split, trust_remote_code=True)
            elif split:
                return load_dataset(repo_id, split=split, trust_remote_code=True)
            else:
                return load_dataset(repo_id, trust_remote_code=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"    Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}")
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


# ── MATH-500 ──────────────────────────────────────────────────────────────────
def download_math500():
    print("\n[1/3] Downloading MATH-500...")
    try:
        ds = load_dataset_with_retry("HuggingFaceH4/MATH-500", split="test")
        records = []
        for item in ds:
            records.append({
                "question": item["problem"],
                "answer":   item["answer"],
            })
        save_jsonl(records, "data/MATH-500/test.jsonl")
    except Exception as e:
        print(f"  HuggingFaceH4/MATH-500 failed ({type(e).__name__}: {e}), trying fallback...")
        try:
            ds = load_dataset_with_retry("lighteval/MATH", config="all", split="test")
            records = []
            for item in list(ds)[:500]:
                records.append({
                    "question": item["problem"],
                    "answer":   item["solution"],
                })
            save_jsonl(records, "data/MATH-500/test.jsonl")
        except Exception as e2:
            print(f"  Fallback also failed: {type(e2).__name__}: {e2}")


# ── CommonsenseQA ─────────────────────────────────────────────────────────────
def download_csqa():
    print("\n[2/3] Downloading CommonsenseQA...")
    try:
        ds = load_dataset_with_retry("tau/commonsense_qa", split="validation")
        records = []
        for item in ds:
            labels = item["choices"]["label"]
            texts  = item["choices"]["text"]
            choices_str = "  ".join(
                f"{l}. {t}" for l, t in zip(labels, texts)
            )
            question = (
                item["question"]
                + "\nChoices: "
                + choices_str
            )
            records.append({
                "question": question,
                "answer":   item["answerKey"],
            })
        save_jsonl(records, "data/commonsenseqa/test.jsonl")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {e}")


# ── AIME ──────────────────────────────────────────────────────────────────────
def download_aime():
    print("\n[3/3] Downloading AIME...")
    try:
        ds = load_dataset_with_retry("Maxwell-Jia/AIME_1983_2024", split="train")
        records = []
        for item in ds:
            records.append({
                "question": item["Problem"],
                "answer":   str(item["Answer"]),
            })
        save_jsonl(records, "data/AIME/test.jsonl")
    except Exception as e:
        print(f"  Maxwell-Jia failed ({type(e).__name__}: {e}), trying fallback...")
        try:
            ds = load_dataset_with_retry("di-zhang-fdu/AIME_1983_2024", split="train")
            records = []
            for item in ds:
                records.append({
                    "question": item["problem"],
                    "answer":   str(item["answer"]),
                })
            save_jsonl(records, "data/AIME/test.jsonl")
        except Exception as e2:
            print(f"  Fallback also failed: {type(e2).__name__}: {e2}")


# ── Verify ────────────────────────────────────────────────────────────────────
def verify():
    print("\n── Verification ─────────────────────────────")
    paths = [
        "data/gsm8k/test.jsonl",
        "data/MATH-500/test.jsonl",
        "data/commonsenseqa/test.jsonl",
        "data/AIME/test.jsonl",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                lines = [l for l in f if l.strip()]
            # show first record keys
            sample = json.loads(lines[0]) if lines else {}
            print(f"  ✅ {p:<40} {len(lines):>5} records  keys={list(sample.keys())}")
        else:
            print(f"  ❌ {p}  NOT FOUND")


if __name__ == "__main__":
    download_math500()
    download_csqa()
    download_aime()
    verify()