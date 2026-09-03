"""把下游后注入记录转换成 followup_eval 数据集格式。

公开发布包自带原始记录（datasets/downstream/llm_behaviour.jsonl），但没有
followup_eval 可直接消费的 json。本构建器把它们映射成
sleeper_eval.followup_eval 所需的 "multiturn_queries" 记录格式。

每个样本的记忆池从约 56 条裁减到 MAX_BENIGN_MEMORIES（20）条，以控制上下文
长度与复现成本；注入的目标记忆始终追加在最后。

输出：followup/eval_in/merged_eval_in_local_behaviour.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "datasets" / "downstream" / "llm_behaviour.jsonl"
OUT = REPO_ROOT / "followup" / "eval_in" / "merged_eval_in_local_behaviour.json"
MAX_BENIGN_MEMORIES = 20


def main() -> None:
    records = []
    with open(SRC, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            goal_text = str(raw["goal"])
            memories = list(raw.get("memories") or [])[:MAX_BENIGN_MEMORIES]
            meta = raw.get("multi_turn_meta") or {}
            split = meta.get("split") or "goal_adjacent"
            queries = [q.strip() for q in raw.get("multi_turn_queries") or [] if q.strip()]
            records.append(
                {
                    "goal": {"goal_text": goal_text, "goal_id": f"local_goal_{i}"},
                    "document": {
                        "doc_id": f"local_doc_{i}",
                        "domain": raw.get("goal_domain", ""),
                    },
                    "query": {},
                    "multi_turn_meta": {
                        "query_style": meta.get("query_style"),
                        "split": split,
                    },
                    "preexisting_memories": {"memories": memories},
                    "multi_turn_queries": queries,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter

    print(f"wrote {len(records)} records -> {OUT}")
    print("split distribution:", Counter(r["multi_turn_meta"]["split"] for r in records))
    print("turn counts:", Counter(len(r["multi_turn_queries"]) for r in records))


if __name__ == "__main__":
    main()