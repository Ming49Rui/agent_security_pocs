# -*- coding: utf-8 -*-
"""AgentBreaker 轻量复现 —— 极简 LLM 客户端（Anthropic Messages 格式）.

复用 ZCode 已配置好的模型服务（~/.zcode/v2/config.json 中的 provider）,
不需要额外安装 SDK, 也不在代码/日志中输出密钥.
"""
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request

ZCODE_CONFIG = os.path.expanduser(r"~/.zcode/v2/config.json")
DEFAULT_PROVIDER = "首序（deep）"
_CTX = ssl.create_default_context()


class LLMError(RuntimeError):
    pass


def load_provider(name=DEFAULT_PROVIDER):
    """从 ZCode 配置读取指定 provider 的 baseURL 和 apiKey."""
    with open(ZCODE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    for v in cfg.get("provider", {}).values():
        if isinstance(v, dict) and v.get("name") == name:
            opts = v.get("options", {})
            return opts.get("baseURL"), opts.get("apiKey")
    raise LLMError(f"未在 ZCode 配置中找到 provider: {name}")


def chat(base_url, api_key, model, system="", msgs=None, temperature=0.0,
         max_tokens=1024, timeout=240, tries=6, sleep=2.0):
    """调用 {base}/messages (Anthropic 格式), 返回文本.

    重试策略: 网络错误/5xx 指数退避; 429(限流) 等待更久再试.
    """
    body = {"model": model, "max_tokens": max_tokens,
            "temperature": temperature, "messages": msgs or []}
    if system:
        body["system"] = system
    last_err = None
    payload = json.dumps(body).encode("utf-8")
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/messages", data=payload, method="POST",
                headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parts = [b.get("text", "") for b in data.get("content", [])
                     if b.get("type") == "text"]
            txt = "\n".join(p for p in parts if p).strip()
            if txt:
                return txt
            last_err = "empty response"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 429:
                time.sleep(sleep * 8 * (attempt + 1))  # 限流: 等久一点
                continue
            if e.code < 500:
                raise LLMError(f"LLM 调用失败 ({model}): {last_err} {e.reason}")
            time.sleep(sleep * (attempt + 1))
            continue
        except Exception as e:  # noqa: BLE001 —— 网络/超时/解析错误统一重试
            last_err = f"{type(e).__name__}: {e}"
        if attempt < tries - 1:
            time.sleep(sleep * (attempt + 1))
    raise LLMError(f"LLM 调用失败 ({model}): {last_err}")


def extract_json(text):
    """从模型输出中提取第一个完整 JSON 数组/对象（容忍 ```json 围栏与前后缀文本）."""
    text = re.sub(r"```(?:json)?", "", text or "")
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError("模型输出中未找到合法 JSON: " + (text[:200] or "<空>"))