# -*- coding: utf-8 -*-
"""
cache_mem.py — 事件→记忆抽取结果的磁盘缓存
===========================================
工作周事件是固定不变的, 而每个防御配置都要重建记忆快照, 会对同一批
事件重复做"事实抽取/摘要生成"。这些抽取结果与防御无关(防御只影响
"是否入库"), 因此按 (agent 类型, 事件 id) 缓存, 大幅减少 LLM 调用。
结构: {"fact|m1|body前100字符": [候选记忆列表, ...], ...}
"""
import json
import os

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "mem_cache.json")

_cache = None  # 懒加载


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except FileNotFoundError:
            _cache = {}
    return _cache


def _save() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)


def get(key: str):
    return _load().get(key)


def put(key: str, value) -> None:
    _load()[key] = value
    _save()