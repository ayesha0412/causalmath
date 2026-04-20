#!/usr/bin/env python3
import os, time, threading
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL   = os.getenv("QWEN_MODEL", "qwen2.5:14b")

_stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0}
_lock  = threading.Lock()

def cerebras_query(
    messages:    List[Dict[str, str]],
    model:       str   = None,
    temperature: float = 0.7,
    max_tokens:  int   = 4096,
    max_retries: int   = 5,
) -> str:
    import openai
    model = model or DEFAULT_MODEL
    with _lock:
        _stats["total_calls"] += 1

    for attempt in range(1, max_retries + 1):
        try:
            client = openai.OpenAI(
                api_key="ollama",           # Ollama doesn't need a real key
                base_url=OLLAMA_BASE_URL,
                timeout=300.0,             # local models can be slow
            )
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            with _lock:
                _stats["successful_calls"] += 1
            return resp.choices[0].message.content.strip()

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