# -*- coding: utf-8 -*-
"""
agents/fact_agent.py — 事实抽取型记忆 Agent (仿 Mem0 / A-Mem)
==============================================================
对应论文: Agent 收到事件后 "extract information and save only facts"
(论文 6.1 节对 Mem0 等 fact-based 记忆的描述)。

记忆表示: 每条记忆 = 一条原子事实(短文本)。
  * 事件 → LLM 抽取 1~3 条事实
  * 该事实若能由更权威/更新的记忆覆盖, 归并(简化实现: 直接追加)
抽取结果按 (agent类型, 事件id) 磁盘缓存 —— 防御实验每个配置都要重建
工作周快照, 事件内容固定不变, 无需重复调用 LLM 抽取。
"""
from typing import List, Dict

import llm
import cache_mem
from agents.base import BaseAgent


class FactAgent(BaseAgent):
    name = "fact"

    def _build_memories(self, event: Dict[str, str]) -> List[Dict]:
        # 缓存命中直接返回: 同一事件在构建多个快照时只会抽取一次
        ckey = f"fact|{event.get('id', '')}|{event.get('body', '')[:60]}"
        cached = cache_mem.get(ckey)
        if cached is not None:
            return list(cached)

        """把事件正文交给 LLM 抽取独立事实, 每条事实生成一条记忆。"""
        system = (
            "You extract atomic facts from emails/calendar events. "
            "Return ONLY a JSON array of strings, each a single concise fact. "
            "Ignore marketing fluff; keep verifiable facts like names, times, "
            "addresses, deadlines, policies."
        )
        prompt = (
            f"Event: [{event['kind']}] from {event.get('sender', '?')}\n"
            f"Subject: {event.get('subject', '')}\n"
            f"Body: {event.get('body', '')}\n\n"
            "Extract up to 3 atomic facts (short, factual, lowercase-ish)."
        )
        facts = llm.ask_json(prompt, system) or []
        if not isinstance(facts, list):
            facts = [facts]
        memories = [{"text": str(f)[:500], "meta": "fact",
                     "source": event.get("sender", "unknown"),
                     "event": event.get("id", "")} for f in facts[:3]]
        cache_mem.put(ckey, memories)
        return memories