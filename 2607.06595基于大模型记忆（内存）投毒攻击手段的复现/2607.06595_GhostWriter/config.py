# -*- coding: utf-8 -*-
"""
config.py — GhostWriter 复现项目配置
====================================
读取顺序: 环境变量 > .env 文件(与代码同目录)。
密钥不写死在代码里, .env 已加入 .gitignore。
"""
import os


def _load_dotenv(path: str) -> None:
    """把 .env 中的 KEY=VALUE 行读入环境变量(不覆盖已有环境变量)。"""
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

# ---- LLM ----
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://model.shouxu.tech/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "deepseek-v4-flash-0731")

# ---- 检索 ----
RETRIEVAL_TOP_K = 3          # 论文 Protocol 3: 检索窗口 k=3
TAG_NUM = 4                  # 每条记忆生成的关键词标签个数

# ---- 攻击 ----
SIM_SCORE_THRESHOLD = 0.55   # 论文 Protocol 2: 载荷相似度阈值 τ(黑盒优化用)


def check() -> None:
    """启动前置检查, 缺 API key 时报错。"""
    if not API_KEY:
        raise RuntimeError(
            "缺少 OPENAI_API_KEY。请在 .env 中配置, 或设置环境变量:\n"
            "  OPENAI_BASE_URL=https://model.shouxu.tech/v1\n"
            "  OPENAI_API_KEY=sk-xxx\n"
            "  AGENT_MODEL=deepseek-v4-flash-0731"
        )