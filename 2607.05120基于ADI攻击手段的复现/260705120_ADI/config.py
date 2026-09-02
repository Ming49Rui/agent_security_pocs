# -*- coding: utf-8 -*-
"""
config.py — API 配置
====================
从环境变量读取, 也支持 .env 文件(与代码同目录), 优先环境变量。
密钥不写死在代码里, .env 已被 .gitignore 默认排除(如使用 git)。
"""
import os


def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://model.shouxu.tech/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "deepseek-v4-flash-0731")


def check() -> None:
    if not API_KEY:
        raise RuntimeError(
            "缺少 OPENAI_API_KEY。请在 .env 中配置, 或设置环境变量: \n"
            "  OPENAI_BASE_URL=https://model.shouxu.tech/v1\n"
            "  OPENAI_API_KEY=sk-xxx\n"
            "  AGENT_MODEL=deepseek-v4-flash-0731"
        )