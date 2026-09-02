# -*- coding: utf-8 -*-
"""
full_test.py — 完整测试规程（一键跑完 + 自动汇总）
===================================================
对应 README / 代码说明中的推荐流程:
  * Layer 1: 连跑 N 遍 (默认5), 每遍 14 次模型调用, 输出平均 ASR / 效用;
  * Layer 2: 每世界重复 runs 次 (默认5), 输出攻击成功率;

用法:
    python full_test.py                 # 默认: Layer1 x5 + Layer2 runs=5
    python full_test.py --layer1-only   # 只跑 Layer 1 (快, 换模型试跑用)
    python full_test.py --layer1-runs 3 --layer2-runs 5
    python full_test.py --json result.json
"""
import argparse
import json
import os
import time
from typing import Dict, List

import config
import layer1
import layer2


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_layer1_batch(n: int) -> Dict:
    """Layer 1 连跑 n 遍, 返回每类别的平均 ASR / 效用(均值±范围)。"""
    per_category: Dict[str, List[float]] = {}   # category -> [asr_prop]
    utils: List[float] = []
    totals: List[float] = []
    print(f"\n===== Layer 1: 连跑 {n} 遍 =====")
    for i in range(1, n + 1):
        r = layer1.main(verbose=False, categories=None)
        # r["agg"] = {category: {"n":2, "util":k, "asr":k, "attacks":1}}
        for cat, a in r["agg"].items():
            per_category.setdefault(cat, []).append(
                a["asr"] / a["attacks"] if a["attacks"] else 0.0)
        util_sum = sum(a["util"] for a in r["agg"].values())
        asr_sum = sum(a["asr"] for a in r["agg"].values())
        n_attack = sum(a["attacks"] for a in r["agg"].values())
        utils.append(util_sum / len(r["agg"]))
        totals.append(asr_sum / n_attack)
        print(f"  第{i}遍: 效用={util_sum}/{len(r['agg'])}  ASR={asr_sum}/{n_attack}")
        time.sleep(1)  # 留一点间隔, 降低限流概率

    print("\n----- Layer 1 汇总 (n=%d 遍) -----" % n)
    print(f"{'类别':<16}{'平均ASR':<10}{'范围'}")
    for cat in sorted(per_category):
        v = per_category[cat]
        lo, hi = min(v), max(v)
        print(f"{cat:<16}{_mean(v):<10.1%} [{lo:.0%}~{hi:.0%}]")
    print(f"{'合计(平均)':<16}{_mean(totals):<10.1%} [{min(totals):.0%}~{max(totals):.0%}]   (n={n})")
    print(f"{'效用(平均)':<16}{_mean(utils):<10.1%}   (每遍满分=100%)")
    return {
        "n_runs": n,
        "mean_total_asr": _mean(totals),
        "asr_range": [min(totals), max(totals)],
        "mean_utility": _mean(utils),
        "per_category_asr": {c: _mean(v) for c, v in per_category.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="完整测试: Layer1 xN + Layer2 runsM")
    ap.add_argument("--layer1-only", action="store_true")
    ap.add_argument("--layer2-only", action="store_true")
    ap.add_argument("--layer1-runs", type=int, default=5)
    ap.add_argument("--layer2-runs", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    config.check()
    out: Dict = {"model": config.MODEL, "base_url": config.BASE_URL}
    t0 = time.time()

    if not args.layer2_only:
        out["layer1"] = run_layer1_batch(args.layer1_runs)

    if not args.layer1_only:
        t1 = time.time()
        print(f"\n===== Layer 2: 每世界重复 {args.layer2_runs} 次 (耗时可能较长) =====")
        r = layer2.main(verbose=False, runs=args.layer2_runs)
        out["layer2"] = {
            "runs": args.layer2_runs,
            "attack": {w: rw["attack_success_rate"] for w, rw in
                       zip(("attack", "benign"), r)},
            "example_commands": {rw["world"]: rw["executed_commands"] for rw in r},
        }
        print(f"  Layer 2 总耗时 {time.time()-t1:.0f}s")

    print(f"\n===== 总耗时 {time.time()-t0:.0f}s ({config.MODEL}) =====")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()