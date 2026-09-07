# -*- coding: utf-8 -*-
"""
证据分析工具：把 capture 到的请求体解析成“上下文源清单”。

用法：
  python analyze.py --evidence <dir> [--canaries "LABEL:canary" ...]

对每个捕获的请求，输出：
  - instructions 字段（内建系统提示）摘要
  - input 中每条消息：role、命中的 canary、内容片段
每个 canary 会单独列出“在哪个 role 中被观测到”。
"""
import argparse
import glob
import json
import os
import re
import sys


def iter_text_parts(item):
    content = item.get("content")
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("input_text") or p.get("output_text")
                if t:
                    yield str(t)


def flatten(item):
    return "\n".join(iter_text_parts(item))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--canaries", nargs="*", default=[],
                    help="LABEL:canary 列表，例如 CNFRM-AGENTS：xxx")
    args = ap.parse_args()

    canaries = []
    for c in args.canaries:
        if ":" in c:
            label, val = c.split(":", 1)
            canaries.append((label, val))
        else:
            canaries.append((c, c))

    files = sorted(glob.glob(os.path.join(args.evidence, "req_*.jsonl")))
    if not files:
        print("no evidence files")
        sys.exit(1)

    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        req = d.get("request") or {}
        print("=" * 72)
        print(f"seq={d['seq']} mock={d.get('mock')} ts={d.get('ts')}")
        instructions = req.get("instructions") or ""
        if instructions:
            print(f"[instructions] 内建系统提示 {len(instructions)} 字")
            head = re.sub(r"\s+", " ", instructions)[:140]
            print(f"  摘要: {head}...")
        for i, item in enumerate(req.get("input", [])):
            role = item.get("role", item.get("type", "?"))
            text = flatten(item)
            hits = [label for label, can in canaries if can in text]
            hit_s = f"  <<HIT: {', '.join(hits)}>>" if hits else ""
            snippet = re.sub(r"\s+", " ", text)[:130]
            print(f"  [{i}] role={role}{hit_s}")
            if text.startswith("<"):
                # 上下文中带标记的块，逐行简化
                for line in text.splitlines()[:6]:
                    print(f"      | {line[:110]}")
            else:
                print(f"      | {snippet}")
        print()

    # 汇总 canary 观测矩阵
    if canaries:
        print("=" * 72)
        print("CANARY 观测矩阵（label -> 出现于哪些 seq 的哪些 role）")
        rows = {}
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            req = d.get("request") or {}
            found = {label: [] for label, _ in canaries}
            for item in req.get("input", []):
                role = item.get("role", item.get("type", "?"))
                text = flatten(item)
                for label, can in canaries:
                    if can in text:
                        found[label].append(f"{d['seq']}:{role}")
            for label, _ in canaries:
                rows.setdefault(label, []).extend(found[label])
        for label, _ in canaries:
            locs = sorted(set(rows.get(label, [])))
            print(f"  {label:24s} -> {locs if locs else '（未观测到）'}")


if __name__ == "__main__":
    main()