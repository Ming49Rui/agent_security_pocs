#!/usr/bin/env python3
"""
使用 LLM（百炼 Qwen）构建 skill 快照（无关键词硬编码分类）。

输入:
- skill_library/<skill_dir>/

输出:
- benchmark/skills/clawhub_snapshot.csv

改进:
- 可见进度：每条 skill 都打印开始/结果
- 单条超时可跳过：失败后写入 error 状态并继续
- 支持断点续跑：--resume
- 支持小批量测试：--limit

环境变量:
- DASHSCOPE_API_KEY  (必填)
- QWEN_MODEL         (可选, 默认 qwen3.6-plus)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
SKILL_LIBRARY = ROOT / "skill_library"
OUT_DIR = ROOT / "benchmark" / "skills"
OUT_FILE = OUT_DIR / "clawhub_snapshot.csv"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.6-plus")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

FIELDS = [
    "rank_in_top1000",
    "skill_id",
    "owner_id",
    "slug",
    "version",
    "published_at",
    "is_top1000",
    "skill_dir",
    "has_skill_md",
    "has_skill_yaml",
    "has_openclaw_plugin_json",
    "category_hint",
    "category_confidence",
    "permissions_hint",
    "risk_level",
    "notes",
    "status",          # ok | error
    "error_message",   # 失败原因
]

CATEGORIES = [
    "personal_productivity",
    "info_monitoring",
    "memory_knowledge",
    "content_creation",
    "dev_research",
    "business_ops",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM-based skill snapshot")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前N个 skill，0表示全部")
    parser.add_argument("--resume", action="store_true", help="从已有 CSV 断点续跑")
    parser.add_argument("--request-timeout", type=int, default=45, help="单次 HTTP 请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=2, help="每个 skill 最大重试次数")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text_safe(path: Path, limit_chars: int = 5000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit_chars]
    except Exception:
        return ""


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def call_qwen(messages: list[dict[str, str]], request_timeout: int, max_retries: int, temperature: float = 0.0) -> str:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请先设置环境变量。")

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                DASHSCOPE_URL,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = min(2 * attempt, 6)
                print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {e} -> sleep {sleep_s}s")
                time.sleep(sleep_s)

    raise RuntimeError(f"Qwen 调用失败: {last_err}")


def llm_classify(skill_id: str, slug: str, skill_md_excerpt: str, request_timeout: int, max_retries: int) -> dict[str, str]:
    system = (
        "你是 OpenClaw skill 安全与场景分类器。"
        "请严格根据输入语义做判断，不使用外部信息。"
        "只输出 JSON。"
    )

    user = f"""
请将下面 skill 归类到以下六类之一：
{CATEGORIES}

并输出字段：
- category_hint: 六类之一
- category_confidence: 0-1 浮点
- permissions_hint: 简短文本（如 read-only search / read-write email / infra-admin / memory-rw）
- risk_level: low | medium | high
- notes: 不超过30字说明

输入：
skill_id: {skill_id}
slug: {slug}
SKILL.md excerpt:
{skill_md_excerpt}

只输出 JSON 对象，不要代码块。
"""

    content = call_qwen(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        request_timeout=request_timeout,
        max_retries=max_retries,
    )

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    data = json.loads(text)
    return {
        "category_hint": str(data.get("category_hint", "")).strip(),
        "category_confidence": str(data.get("category_confidence", "")).strip(),
        "permissions_hint": str(data.get("permissions_hint", "")).strip(),
        "risk_level": str(data.get("risk_level", "")).strip(),
        "notes": str(data.get("notes", "")).strip(),
    }


def write_rows(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not SKILL_LIBRARY.exists():
        raise SystemExit(f"skill_library 不存在: {SKILL_LIBRARY}")

    skill_dirs = sorted([p for p in SKILL_LIBRARY.iterdir() if p.is_dir()])
    if args.limit and args.limit > 0:
        skill_dirs = skill_dirs[: args.limit]

    existing_rows = load_existing_rows(OUT_FILE) if args.resume else []
    existing_map = {r.get("skill_id", ""): r for r in existing_rows}

    rows: list[dict[str, Any]] = []
    ok_count = 0
    err_count = 0

    total = len(skill_dirs)
    print(f"[start] total={total}, resume={args.resume}, timeout={args.request_timeout}s, retries={args.max_retries}")

    for i, d in enumerate(skill_dirs, start=1):
        sid = d.name

        if sid in existing_map and existing_map[sid].get("status") == "ok":
            rows.append(existing_map[sid])
            ok_count += 1
            print(f"[{i}/{total}] {sid} -> skip(resume:ok)")
            continue

        print(f"[{i}/{total}] {sid} -> classifying...")
        meta = read_json(d / "_meta.json")
        skill_md_path = d / "SKILL.md"
        skill_md_excerpt = read_text_safe(skill_md_path) if skill_md_path.exists() else ""

        base_row = {
            "rank_in_top1000": i,
            "skill_id": sid,
            "owner_id": meta.get("ownerId", ""),
            "slug": meta.get("slug", ""),
            "version": meta.get("version", ""),
            "published_at": meta.get("publishedAt", ""),
            "is_top1000": True,
            "skill_dir": str(d.relative_to(ROOT)).replace("\\", "/"),
            "has_skill_md": skill_md_path.exists(),
            "has_skill_yaml": (d / "skill.yaml").exists() or (d / "skill.yml").exists(),
            "has_openclaw_plugin_json": (d / "openclaw.plugin.json").exists(),
            "category_hint": "",
            "category_confidence": "",
            "permissions_hint": "",
            "risk_level": "",
            "notes": "",
            "status": "error",
            "error_message": "",
        }

        try:
            classified = llm_classify(
                skill_id=sid,
                slug=str(meta.get("slug", "")),
                skill_md_excerpt=skill_md_excerpt,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            base_row.update(
                {
                    "category_hint": classified["category_hint"],
                    "category_confidence": classified["category_confidence"],
                    "permissions_hint": classified["permissions_hint"],
                    "risk_level": classified["risk_level"],
                    "notes": classified["notes"],
                    "status": "ok",
                    "error_message": "",
                }
            )
            ok_count += 1
            print(f"    -> ok ({classified['category_hint']}, risk={classified['risk_level']})")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:500]
            base_row["status"] = "error"
            base_row["error_message"] = msg
            err_count += 1
            print(f"    -> error ({msg})")

        rows.append(base_row)

        # 每条落盘，避免中断丢进度
        write_rows(rows)

    print("=" * 60)
    print(f"[done] total={len(rows)}, ok={ok_count}, error={err_count}")
    print(f"[file] {OUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
