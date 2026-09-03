# -*- coding: utf-8 -*-
"""OpenAI 兼容 LLM 客户端（带重试），支持 function calling。"""
import json
import time

import requests

from config import API_BASE, API_KEY, MODEL, TEMPERATURE


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, model=MODEL, temperature=TEMPERATURE, retries=8, retry_delay=2.0):
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.retry_delay = retry_delay

    def _post(self, payload):
        # trust_env=False: 绕过可能存在的系统代理/SCHANNEL 抖动，直接 TCP+TLS
        s = requests.Session()
        s.trust_env = False
        resp = s.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json()
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:400]}")

    def chat(self, messages, tools=None, tool_choice="auto", return_full=False):
        """单次对话调用。返回 assistant message dict。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        last = None
        for i in range(self.retries):
            try:
                resp = self._post(payload)
                return resp if return_full else resp["choices"][0]["message"]
            except Exception as e:  # 网络抖动/限流统一重试
                last = e
                time.sleep(self.retry_delay)
        raise LLMError(f"chat failed after {self.retries} retries: {last}")

    @staticmethod
    def parse_tool_calls(msg):
        """把 assistant message 的 tool_calls 解析成 [(call_id, name, args_dict), ...]"""
        out = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append((tc.get("id"), fn.get("name"), args))
        return out