# -*- coding: utf-8 -*-
"""
retriever.py — 记忆检索 (论文 §3.1 的 tag 匹配机制)
====================================================
论文描述 Agent 的检索: "Retrieval typically proceeds by matching the tags
in the user's query to those in the memories." (第 3.1 节)

因此本项目不引入向量库, 用两种信号混合打分:
  1. 标签重叠分  —— 记忆标签 ∩ 查询标签的 Jaccard 系数
  2. 词重叠分    —— 记忆文本 ∩ 查询文本的词级重叠率(兜底, 减少对 LLM 的依赖)
最终分数 = 0.7 * 标签重叠 + 0.3 * 词重叠, 取 top-k 返回。
若以后想替换成真正的 embedding(all-MiniLM-L6-v2), 只需替换 score() 的实现。
"""
import re
from typing import List, Dict

import config
import llm

_STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with",
         "at", "by", "from", "is", "are", "was", "were", "be", "been", "it",
         "this", "that", "your", "my", "i", "you", "we", "our", "please",
         "the", "as", "do", "does", "did", "will", "would", "can", "could",
         "should", "have", "has", "had", "s", "t", "about", "not", "no",
         "yes", "ok", "via", "re", "per", "am", "pm"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP}


def _tag_overlap(mem_tags: List[str], q_tags: List[str]) -> float:
    """标签集合的 Jaccard 系数, 空集合时返回 0。"""
    a, b = set(t.strip().lower() for t in mem_tags), set(q_tags)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _word_overlap(mem_text: str, query: str) -> float:
    """词级重叠率: 查询词中有多少比例出现在记忆文本中(0~1)。"""
    mw, qw = _tokens(mem_text), _tokens(query)
    if not qw:
        return 0.0
    return len(qw & mw) / len(qw)


class Retriever:
    """对一组记忆执行 top-k 检索。分数字段写入记忆对象便于调试。"""

    def __init__(self, k: int = None):
        self.k = k or config.RETRIEVAL_TOP_K

    def score(self, memory: dict, query_tags: List[str], query: str) -> dict:
        """返回 {score, tag_overlap, word_overlap}。"""
        t = _tag_overlap(memory.get("tags", []), query_tags)
        w = _word_overlap(memory.get("text", ""), query)
        return {"score": 0.7 * t + 0.3 * w,
                "tag_overlap": t, "word_overlap": w}

    def retrieve(self, memories: List[dict], query: str,
                 query_tags: List[str]) -> List[dict]:
        """按分数降序取 top-k, 返回带分数副本(不修改原记忆)。"""
        scored = []
        for m in memories:
            sc = self.score(m, query_tags, query)
            m_c = dict(m)
            m_c["_score"] = sc["score"]
            m_c["_tag_overlap"] = sc["tag_overlap"]
            m_c["_word_overlap"] = sc["word_overlap"]
            scored.append(m_c)
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[: self.k]


def query_tags(query: str) -> List[str]:
    """查询的标签可预先算好复用(攻击实验里同一触发词会跑多次)。"""
    return llm.tag(query)