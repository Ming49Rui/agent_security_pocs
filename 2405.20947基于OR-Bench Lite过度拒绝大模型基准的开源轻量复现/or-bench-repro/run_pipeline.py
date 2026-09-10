# -*- coding: utf-8 -*-
"""执行 OR-Bench 数据生成管线 (缓存: 已生成的 JSON 会跳过)。"""
import os
import time

from config import SAFE_CACHE_DIR
from llm import dump_json
from pipeline import generate_seeds, rewrite_seeds, moderate_batch, split_safe_toxic


def main():
    t0 = time.time()
    os.makedirs(SAFE_CACHE_DIR, exist_ok=True)

    seed_path = os.path.join(SAFE_CACHE_DIR, "seeds.json")
    seeds = None
    if os.path.exists(seed_path):
        from llm import load_json
        seeds = load_json(seed_path)
        print(f"[cache] 复用种子 {len(seeds)} 条")
    if not seeds:
        seeds = generate_seeds()

    rw_path = os.path.join(SAFE_CACHE_DIR, "rewritten.json")
    cands = None
    if os.path.exists(rw_path):
        from llm import load_json
        cands = load_json(rw_path)
        print(f"[cache] 复用改写候选 {len(cands)} 条")
    if not cands:
        cands = rewrite_seeds(seeds)

    mod_path = os.path.join(SAFE_CACHE_DIR, "moderation_raw.json")
    if os.path.exists(mod_path):
        from llm import load_json
        raw = load_json(mod_path)
        print(f"[cache] 复用评审结果 {len(raw)} 条")
        # raw 里是压缩后的; 直接按压缩结构重建 safe/toxic 列表
        safe = [r for r in raw if sum(1 for _, lb in r["votes"] if lb == "none") >= 2]
        toxic = [r for r in raw if sum(1 for _, lb in r["votes"] if lb == "none") < 2]
        print(f"[cache] safe={len(safe)} toxic={len(toxic)}")
        dump_json(os.path.join(SAFE_CACHE_DIR, "safe_prompts.json"),
                  [{"category": r["category"], "prompt": r["prompt"]} for r in safe])
        dump_json(os.path.join(SAFE_CACHE_DIR, "toxic_prompts.json"),
                  [{"category": r["category"], "prompt": r["prompt"]} for r in toxic])
        print(f"[pipeline] 总计耗时 {time.time() - t0:.0f}s")
        return

    print(f"[pipeline] 开始 moderation (候选 {len(cands)} 条, 每条约 3 模型 + 1 答题)")
    results = moderate_batch(cands)
    safe_c, toxic_c = split_safe_toxic(results)
    dump_json(os.path.join(SAFE_CACHE_DIR, "safe_prompts.json"),
              [{"category": r["item"]["category"], "prompt": r["item"]["prompt"]} for r in safe_c])
    dump_json(os.path.join(SAFE_CACHE_DIR, "toxic_prompts.json"),
              [{"category": r["item"]["category"], "prompt": r["item"]["prompt"]} for r in toxic_c])
    print(f"[pipeline] safe={len(safe_c)} toxic={len(toxic_c)} 类别分布: "
          f"{ {c: sum(1 for r in safe_c if r['item']['category'] == c) for c in set(r['item']['category'] for r in safe_c)} }")
    print(f"[pipeline] 总计耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()