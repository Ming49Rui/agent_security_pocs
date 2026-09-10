# -*- coding: utf-8 -*-
"""OpenAI 兼容 LLM 客户端: 绕过系统代理, 带重试与并发工具。"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import API_BASE, API_KEY


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, retries=6, retry_delay=2.0, timeout=180):
        self.retries = retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def chat(self, model, messages, temperature=0.0, max_tokens=1024):
        s = requests.Session()
        s.trust_env = False  # 绕过系统代理 (SCHANNEL 抖动)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last = None
        for i in range(self.retries):
            try:
                r = s.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json=payload, timeout=self.timeout,
                )
                if r.status_code == 200:
                    j = r.json()
                    content = j["choices"][0]["message"].get("content") or ""
                    content = content.strip()
                    if content:
                        return content
                    # 空内容: 多为模型静默过滤/截断, 重试一次再放弃
                    last = LLMError("empty content (silent filter?)")
                    continue
                last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            except Exception as e:  # 网络抖动/限流统一重试
                last = e
            time.sleep(self.retry_delay * (i + 1))
        raise LLMError(f"chat failed after {self.retries} retries: {last}")

    def map_parallel(self, fn, items, workers=10):
        """对 items 并发执行 fn(item), 保持输入顺序返回结果列表。"""
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results


# ---------------- 解析工具 ----------------

BRACKET_RE = re.compile(r"\[\[(.*?)\]\]", re.S)


def parse_brackets(text):
    """按 [[...]] 切块, 去序号前缀 (如 "1. xxx" / "1."), 返回去重后的内容列表。"""
    items = []
    seen = set()
    for m in BRACKET_RE.findall(text or ""):
        s = re.sub(r"^\s*\d+\s*[.)、:]\s*", "", m.strip())
        s = re.sub(r"\s+", " ", s).strip()
        if s and s not in seen:
            seen.add(s)
            items.append(s)
    return items


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)