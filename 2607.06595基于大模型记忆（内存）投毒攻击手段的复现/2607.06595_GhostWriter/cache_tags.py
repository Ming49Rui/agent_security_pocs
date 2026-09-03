# -*- coding: utf-8 -*-
"""
cache_tags.py — 标签磁盘缓存(小巧, 零依赖)
===========================================
llm.tag() 的缓存后端: key = "个数:文本前500字符" → tags 列表。
存成 JSON 文件(cache/tag_cache.json), 进程间与多次运行间复用。
"""
import json
import os

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "tag_cache.json")

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
    c = _load()
    c[key] = value
    _save()