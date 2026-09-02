# -*- coding: utf-8 -*-
"""
run.py — ADI 复现主入口
=======================
用法:
    python run.py --layer1            # 只跑 Layer 1 (LLM 层 ASR 基准)
    python run.py --layer2            # 只跑 Layer 2 (Agent 来源注入攻击)
    python run.py                     # 两层都跑
    python run.py --json out.json     # 额外输出结构化结果
"""
import argparse
import json

import config


def main() -> None:
    ap = argparse.ArgumentParser(description="ADI (arXiv:2607.05120) reproduction")
    ap.add_argument("--layer1", action="store_true")
    ap.add_argument("--layer2", action="store_true")
    ap.add_argument("--runs", type=int, default=3, help="Layer 2 每个世界重复次数(默认3)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config.check()
    run1 = args.layer1 or not (args.layer1 or args.layer2)
    run2 = args.layer2 or not (args.layer1 or args.layer2)

    out: dict = {}
    if run1:
        import layer1
        out["layer1"] = layer1.main(verbose=not args.quiet)
    if run2:
        import layer2
        out["layer2"] = layer2.main(verbose=not args.quiet, runs=args.runs)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json}")


if __name__ == "__main__":
    main()