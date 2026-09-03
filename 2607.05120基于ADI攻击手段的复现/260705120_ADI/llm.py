# -*- coding: utf-8 -*-
"""
llm.py — LLM 客户端 (OpenAI 兼容接口, 零第三方依赖)
====================================================
只依赖 requests。用于:
  * Layer 1: 单轮问答(测概率性分隔符注入 ASR)
  * Layer 2: 智能环节(逐个工具调用)
"""
import json
import time
from typing import List, Dict, Optional

import requests

import config


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.API_KEY}",
    }


# 代理策略: 默认直连(绕过系统代理)。注册表残留 127.0.0.1:7897 但 ProxyEnable=0,
# 直连验证可用(HTTP 200 ~1s)。若以后挂梯子且想走系统代理, 设 ADI_USE_PROXY=1。
import os as _os

_USE_SYSTEM_PROXY = _os.environ.get("ADI_USE_PROXY", "0") == "1"
_SESSION = requests.Session()
_SESSION.trust_env = _USE_SYSTEM_PROXY


def chat(messages: List[Dict[str, str]], temperature: float = 0.0,
         max_tokens: Optional[int] = None, retries: int = 8,
         timeout: int = 180) -> str:
    """调用 chat/completions, 返回助手文本。429/网络错误按服务端要求退避重试。"""
    body: Dict = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = _SESSION.post(
                f"{config.BASE_URL}/chat/completions",
                json=body, headers=_headers(), timeout=timeout,
            )
            if r.status_code == 429:
                wait = 60.0
                try:  # 服务端指定了重试时间(秒)
                    wait = float(r.json().get("retry_after", 60))
                except Exception:
                    pass
                raise RateLimited(wait)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except RateLimited as ex:  # 限流: 按 retry_after 等待
            last_err = ex
            print(f"    [429] 限流, 等待 {ex.wait:.0f}s (第{i+1}/{retries}次)")
            time.sleep(ex.wait)
        except Exception as ex:  # 网络抖动等
            last_err = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM 调用多次失败: {last_err}")


class RateLimited(Exception):
    def __init__(self, wait: float = 60.0):
        super().__init__(f"rate limited, retry after {wait}s")
        self.wait = wait


def ask_once(prompt: str, system: str = "You are a helpful assistant.",
             temperature: float = 0.0) -> str:
    return chat([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ], temperature=temperature)