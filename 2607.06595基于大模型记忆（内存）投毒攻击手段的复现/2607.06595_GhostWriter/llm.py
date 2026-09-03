# -*- coding: utf-8 -*-
"""
llm.py — LLM 客户端 (OpenAI 兼容接口, 零第三方依赖)
====================================================
只依赖 requests。统一封装:
  * chat()      普通文本回复
  * ask_json()  要求模型按 JSON 返回并解析(用于打分/判定)
  * tag()       为记忆/查询生成关键词标签(论文 §3.1 的 tag 检索机制)
注意: 服务端可能返回 reasoning_content 字段, 只取 message.content。
"""
import json
import time
from typing import List, Dict, Optional

import requests

import config


def _headers() -> Dict[str, str]:
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}"}


# 直连策略: 绕过系统代理(与已有项目 260705120_ADI 同款处理)。
import os as _os
_USE_SYSTEM_PROXY = _os.environ.get("GW_USE_PROXY", "0") == "1"
_SESSION = requests.Session()
_SESSION.trust_env = _USE_SYSTEM_PROXY


class RateLimited(Exception):
    """服务端限流, 携带建议等待秒数。"""

    def __init__(self, wait: float = 60.0):
        super().__init__(f"rate limited, retry after {wait}s")
        self.wait = wait


def chat(messages: List[Dict[str, str]], temperature: float = 0.0,
         max_tokens: Optional[int] = None, retries: int = 8,
         timeout: int = 180) -> str:
    """调用 chat/completions, 返回助手 text 内容。429 按服务端要求退避重试。"""
    body: Dict = {"model": config.MODEL, "messages": messages,
                  "temperature": temperature}
    if max_tokens:
        body["max_tokens"] = max_tokens

    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = _SESSION.post(f"{config.BASE_URL}/chat/completions",
                              json=body, headers=_headers(), timeout=timeout)
            if r.status_code == 429:
                wait = 60.0
                try:
                    wait = float(r.json().get("retry_after", 60))
                except Exception:
                    pass
                raise RateLimited(wait)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"].strip()
        except RateLimited as ex:
            last_err = ex
            print(f"    [429] 限流, 等待 {ex.wait:.0f}s (第{i+1}/{retries}次)")
            time.sleep(ex.wait)
        except Exception as ex:
            last_err = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM 调用多次失败: {last_err}")


def ask_once(prompt: str, system: str = "You are a helpful assistant.",
             temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
    """单轮提问的便捷封装。"""
    return chat([{"role": "system", "content": system},
                 {"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens)


def ask_json(prompt: str, system: str, temperature: float = 0.0) -> dict:
    """请求模型只输出 JSON 对象, 解析失败时自动重试(剔除多余文字)。"""
    sys_p = system + "\n必须只输出一个 JSON 对象, 不要输出任何其他文字或代码块标记。"
    for _ in range(3):
        raw = ask_once(prompt, sys_p, temperature=temperature)
        # 剥掉 ```json ... ``` 包裹
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            if "{" in raw and "}" in raw:
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"模型未返回合法 JSON: {raw[:200]}")


def tag(text: str, num: int = None) -> List[str]:
    """生成用于记忆检索的关键词标签(小写)。

    带磁盘缓存(cache/tag_cache.json): 防御实验会为每个配置重建记忆快照,
    同一文本的标签只请求一次, 避免重复烧 token。
    """
    num = num or config.TAG_NUM
    key = f"{num}:{text[:500]}"
    from cache_tags import get, put
    cached = get(key)
    if cached is not None:
        return cached
    prompt = (
        f"Extract {num} short lowercase keywords/phrases (max 2 words each) that "
        f"best represent the following content for later retrieval. "
        f"Return ONLY a JSON array of strings.\n\nContent:\n{text[:3000]}"
    )
    result: List[str] = []
    try:
        data = ask_json(prompt, "You are a tagging assistant.")
        if isinstance(data, list):
            result = [str(x).strip().lower() for x in data[:num]]
        elif isinstance(data, dict) and "tags" in data:
            result = [str(x).strip().lower() for x in data["tags"][:num]]
    except Exception:
        result = []
    put(key, result)
    return result