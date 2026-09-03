"""从 inspect 的 .eval 日志（zstd 压缩包）中提取后注入 AUR 结果。

用法：
    .venv\\Scripts\\python.exe repro_local\\analyze_followup.py <log_dir> [--by-split]

读取 <log_dir> 下 .eval 日志中所有 samples/*.json 条目，取出每个样本的
behavior_influence_scorer 判定，并按邻近度分组（multi_turn_split =
goal_adjacent | wildchat_seed）汇总 INFLUENCED 比例。
"""

from __future__ import annotations

import argparse
import glob
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import zipfile_zstd  # noqa: F401  （为 zipfile 注册 zstd 解码器）


def iter_samples(log_dir: str | Path):
    for log in sorted(glob.glob(str(Path(log_dir) / "**" / "*.eval"), recursive=True)):
        try:
            z = zipfile.ZipFile(log)
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 {log}: {exc}")
            continue
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                try:
                    yield json.loads(z.read(name).decode("utf-8")), log
                except Exception:  # noqa: BLE001
                    continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--by-split", action="store_true")
    args = ap.parse_args()

    by_split: dict[str, list[int | None]] = defaultdict(list)
    total = []
    for s, log in iter_samples(args.log_dir):
        md = s.get("metadata") or {}
        val = None
        for v in (s.get("scores") or {}).values():
            val = v.get("answer")
        label = 1 if val == "INFLUENCED" else (0 if val == "NOT_INFLUENCED" else None)
        split = md.get("multi_turn_split") or md.get("category") or "unknown"
        by_split[split].append(label)
        total.append(label)

    def rate(vals):
        scored = [v for v in vals if v is not None]
        if not scored:
            return 0.0, 0, 0
        return sum(scored) / len(scored), sum(scored), len(scored)

    if args.by_split:
        for split in sorted(by_split):
            rate_v, n_pos, n = rate(by_split[split])
            n_none = sum(1 for v in by_split[split] if v is None)
            print(f"{split:15s} AUR={rate_v*100:6.1f}%  ({n_pos}/{n})  未评分={n_none}")
    rate_v, n_pos, n = rate(total)
    print(f"{'总计':15s} AUR={rate_v*100:6.1f}%  ({n_pos}/{n})")


if __name__ == "__main__":
    main()