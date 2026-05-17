#!/usr/bin/env python3
import os, time, threading, re
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11436/v1")
DEFAULT_MODEL   = os.getenv("QWEN_MODEL", "qwen3:14b")

_stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0}
_lock  = threading.Lock()

print(f"[INIT] Using Ollama OpenAI-compatible API")
print(f"[INIT] Model: {DEFAULT_MODEL}")
print(f"[INIT] Endpoint: {OLLAMA_BASE_URL}")

# Verify Ollama is reachable
try:
    import requests as _req
    _base = OLLAMA_BASE_URL.replace("/v1", "")
    _req.get(f"{_base}/", timeout=5)
    print(f"[INIT] Ollama is running ✓")
except Exception:
    print(f"[WARN] Could not verify Ollama at {OLLAMA_BASE_URL}")


def cerebras_query(
    messages:    List[Dict[str, str]],
    model:       str   = None,
    temperature: float = 0.6,
    top_p:       float = 0.95,
    top_k:       int   = 20,
    max_tokens:  int   = 4096,
    max_retries: int   = 5,
    thinking:    bool  = True,
) -> str:
    import openai
    model = model or DEFAULT_MODEL
    with _lock:
        _stats["total_calls"] += 1

    for attempt in range(1, max_retries + 1):
        try:
            client = openai.OpenAI(
                api_key="ollama",
                base_url=OLLAMA_BASE_URL,
                timeout=1800.0,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "num_ctx": 4096,
                    "top_k": top_k,
                    "think": thinking,
                },
            )
            text = resp.choices[0].message.content.strip()

            # Strip internal <think>...</think> deliberation block
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            elif "<think>" in text:
                text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

            with _lock:
                _stats["successful_calls"] += 1
            return text

        except Exception as e:
            print(f"\n  ✗  Attempt {attempt}/{max_retries}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(3)

    with _lock:
        _stats["failed_calls"] += 1
    raise RuntimeError(f"Ollama failed after {max_retries} attempts")


def get_api_stats() -> Dict:
    with _lock:
        return dict(_stats)
