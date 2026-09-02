"""本地后注入（RR/AUR）运行器：加载 .env + Windows 补丁 + 样本数限制。

用于对 OpenAI 兼容端点运行 followup_eval 任务（在"新的会话"中检验
潜伏记忆对助手行为的影响）。

用法：
    .venv\\Scripts\\python.exe repro_local\\run_followup.py \
        --dataset-file datasets/downstream/llm_behaviour.jsonl \
        --limit 20 --log-dir .artifacts/local/followup_behaviour20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

import repro_local._win_patch  # noqa: E402,F401  （Windows fsspec 路径补丁）

from inspect_ai import eval_set  # noqa: E402

from sleeper_eval.followup_eval.task import followup_eval  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/deepseek-v4-flash-0731")
    ap.add_argument("--prompt-model", default="deepseek-v4-flash-0731")
    ap.add_argument("--provider", default="generic")
    ap.add_argument("--grader-model", default="openai/deepseek-v4-flash-0731")
    ap.add_argument("--dataset-file", default="datasets/downstream/llm_behaviour.jsonl")
    ap.add_argument("--log-dir", default=".artifacts/local/followup_behaviour")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tasks", type=int, default=1, help="并行任务数（保持 1 以避免端点限流）")
    ap.add_argument("--max-connections", type=int, default=1, help="模型连接数")
    ap.add_argument("--retry-on-error", type=int, default=5, help="模型错误（如限流）时的重试次数")
    ap.add_argument("--category", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str((log_dir.parent / ".inspect_home").resolve())
    os.environ["INSPECT_LOG_DIR"] = str(log_dir)
    os.environ["INSPECT_TRACE_FILE"] = str(log_dir / "inspect-trace.log")

    print(f"数据集：{args.dataset_file}")
    print(f"模型：{args.model}（prompt={args.prompt_model}，provider={args.provider}）")
    print(f"判分器：{args.grader_model}")
    print(f"样本数限制：{args.limit if args.limit is not None else '全部'}")
    print(f"日志目录：{log_dir}")

    try:
        from sleeper_eval.followup_eval.dataset import load_dataset as ld
        ds = ld(args.dataset_file)
        print(f"样本总数：{len(ds)}")
    except Exception as exc:  # noqa: BLE001
        print(f"（样本数不可用：{exc}）")

    success, _ = eval_set(
        [followup_eval(
            dataset_file=args.dataset_file,
            category=args.category,
            provider=args.provider,
            prompt_model=args.prompt_model,
        )],
        model=[args.model],
        model_roles={"grader": args.grader_model},
        log_dir=str(log_dir),
        limit=args.limit,
        max_tasks=args.max_tasks,
        max_connections=args.max_connections,
        retry_on_error=args.retry_on_error,
        fail_on_error=False,
        max_retries=3,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())