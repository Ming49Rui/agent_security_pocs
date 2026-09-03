#!/usr/bin/env python3
"""msc_llm.py — shared LLM client for MSC-RiskBench v3.0 (attack side).

Mirrors the proven OpenAI-compatible JSON client used by benchmark/tpvs_scanner.py
so capability profiling and chain-coherence judging use *LLM reasoning* rather than
hardcoded string/regex rules (per unified spec §11.3 and project convention).

Configuration (env vars, first match wins):
  API key  : MSC_LLM_API_KEY | DASHSCOPE_API_KEY | DEEPSEEK_API_KEY | OPENAI_API_KEY
  base URL : MSC_LLM_BASE_URL | OPENAI_BASE_URL
             (default DashScope OpenAI-compatible endpoint)
  model    : MSC_LLM_MODEL | QWEN_MODEL | OPENAI_MODEL  (default qwen-plus)

A tiny on-disk JSON cache keyed by a content hash avoids re-paying for identical
prompts across the 5 threats × 6 scenarios (skills overlap heavily).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from socket import timeout as SocketTimeout

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# Project root = parent of scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Does not print values.

    Supports lowercase keys used in this project's .env (api_key, base_url,
    model) as well as the usual uppercase OPENAI_* names. Quotes are stripped.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


_DOTENV = _load_dotenv(_PROJECT_ROOT / ".env")


def _first_env(*names: str) -> str | None:
    """Resolve a config value. The project's .env is the source of truth, so it
    takes precedence over ambient OS env vars (which may hold leftover keys for
    a different provider). Falls back to OS env if .env lacks the value."""
    lowered = {k.lower(): v for k, v in _DOTENV.items()}
    for n in names:
        if n in _DOTENV and _DOTENV[n]:
            return _DOTENV[n]
        if n.lower() in lowered and lowered[n.lower()]:
            return lowered[n.lower()]
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


class LLMClient:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, cache_path: Path | None = None,
                 temperature: float = 0.0, max_retries: int = 3,
                 request_timeout: float = 60.0):
        self.model = model or _first_env("MSC_LLM_MODEL", "QWEN_MODEL", "OPENAI_MODEL", "model") or DEFAULT_MODEL
        self.base_url = (base_url or _first_env("MSC_LLM_BASE_URL", "OPENAI_BASE_URL", "base_url")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or _first_env(
            "MSC_LLM_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "api_key")
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key. Set api_key in .env or one of "
                "MSC_LLM_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY.")
        self.temperature = temperature
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.cache_path = cache_path
        self._cache: dict[str, Any] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        self._dirty = 0

    # -- cache -------------------------------------------------------------
    def _key(self, messages: list[dict[str, str]]) -> str:
        blob = json.dumps([self.model, messages], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def flush(self) -> None:
        if self.cache_path and self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
            self._dirty = 0

    # -- request -----------------------------------------------------------
    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        key = self._key(messages)
        if key in self._cache:
            return self._cache[key]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=self.request_timeout) as resp:
                    body = resp.read().decode("utf-8")
                content = json.loads(body)["choices"][0]["message"]["content"]
                result = json.loads(content)
                self._cache[key] = result
                self._dirty += 1
                if self._dirty >= 10:
                    self.flush()
                return result
            except (HTTPError, URLError, SocketTimeout, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"LLM request failed after {self.max_retries} tries: {last_err}")
