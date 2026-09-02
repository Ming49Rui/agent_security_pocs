#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_ROOT = (
    REPO_ROOT / "logs" / "claude_docrep_actor_critic_markers_aware_validation"
)
DEFAULT_PRICING_FILE = EXPERIMENT_ROOT / "pricing.yaml"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_SEED = 20260329


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("full", "truncated", "both"),
        default="both",
        help="Which Claude prompt variant to run.",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("full", "format_smoke"),
        default="full",
        help="Which prepared dataset from the manifest to use.",
    )
    parser.add_argument(
        "--attack",
        default="universal_v1",
        help="Attack id to pass to the task.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Subject model slug.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Selection seed used by create_samples.py.",
    )
    parser.add_argument(
        "--prepare-samples",
        action="store_true",
        help="Regenerate the smoke dataset and validation manifest before running.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to selection_seed_*.json. Default derives from --seed.",
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        default=DEFAULT_PRICING_FILE,
        help="Inspect pricing file for cost logging.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=DEFAULT_LOGS_ROOT,
        help="Root directory for timestamped run outputs.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("batch", "serial"),
        default="batch",
        help="Run one normal inspect eval per condition, or one inspect eval per sample.",
    )
    parser.add_argument("--max-connections", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--attempt-timeout", type=int, default=180)
    parser.add_argument(
        "--full-sleep-seconds",
        type=int,
        default=60,
        help="Sleep between serial samples for the full prompt.",
    )
    parser.add_argument(
        "--truncated-sleep-seconds",
        type=int,
        default=30,
        help="Sleep between serial samples for the truncated prompt.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Limit to the first N manifest sample ids for debugging.",
    )
    parser.add_argument(
        "--run-timestamp",
        default="",
        help="Override the timestamp suffix used in log dir names.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip tool-only scoring after generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing model calls.",
    )
    return parser


def default_manifest_path(seed: int) -> Path:
    return EXPERIMENT_ROOT / "data" / f"selection_seed_{seed}.json"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_subprocess(command: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(shlex.join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def prepare_samples(seed: int, dry_run: bool) -> None:
    run_subprocess(
        [
            "uv",
            "run",
            "python",
            "experiments/claude_prompt_validation/scripts/create_samples.py",
            "--seed",
            str(seed),
        ],
        dry_run=dry_run,
    )


def conditions_to_run(condition: str) -> list[str]:
    if condition == "both":
        return ["full", "truncated"]
    return [condition]


def prompt_file_for(condition: str) -> str:
    return "claude.md" if condition == "full" else "claude_truncated.md"


def log_prefix_for(condition: str, dataset_kind: str) -> str:
    prefix = f"claude_sonnet_4.6_{condition}"
    if dataset_kind == "format_smoke":
        prefix += "_format_smoke"
    return prefix


def timestamp_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def log_dir_for(logs_root: Path, prefix: str, timestamp: str) -> Path:
    return logs_root / f"{prefix}-{timestamp}"


def env_for_log_dir(log_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(REPO_ROOT / ".inspect_home")
    env["INSPECT_TRACE_FILE"] = str(log_dir / "inspect-trace.log")
    env["INSPECT_LOG_DIR"] = str(log_dir)
    env["UV_CACHE_DIR"] = ".uv-cache"
    return env


def score_log_dir(log_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(
            shlex.join(
                [
                    "uv",
                    "run",
                    "inspect",
                    "score",
                    str(log_dir / "*.eval"),
                    "--scorer",
                    "sleeper_eval/scorer.py@tool_call_scorer",
                    "--overwrite",
                ]
            )
        )
        return
    eval_files = sorted(log_dir.glob("*.eval"))
    if not eval_files:
        raise FileNotFoundError(f"No .eval files found in {log_dir}")
    env = env_for_log_dir(log_dir)
    for eval_file in eval_files:
        run_subprocess(
            [
                "uv",
                "run",
                "inspect",
                "score",
                str(eval_file),
                "--scorer",
                "sleeper_eval/scorer.py@tool_call_scorer",
                "--overwrite",
            ],
            env=env,
            dry_run=dry_run,
        )


def sample_ids_for(manifest: dict, dataset_kind: str) -> list[str]:
    key = "full_sample_ids" if dataset_kind == "full" else "format_smoke_sample_ids"
    sample_ids = list(manifest[key])
    return sample_ids


def dataset_file_for(manifest: dict, dataset_kind: str) -> str:
    return str(manifest["generated_datasets"][dataset_kind])


def run_condition(
    *,
    condition: str,
    dataset_kind: str,
    attack: str,
    model: str,
    manifest: dict,
    pricing_file: Path,
    logs_root: Path,
    max_connections: int,
    max_samples: int,
    max_retries: int,
    timeout: int,
    attempt_timeout: int,
    execution_mode: str,
    sleep_seconds: int,
    sample_limit: int,
    run_timestamp: str,
    no_score: bool,
    dry_run: bool,
) -> Path:
    dataset_file = dataset_file_for(manifest, dataset_kind)
    sample_ids = sample_ids_for(manifest, dataset_kind)
    if sample_limit > 0:
        sample_ids = sample_ids[:sample_limit]
    if not sample_ids:
        raise ValueError(f"No sample ids found for dataset kind {dataset_kind!r}")

    prefix = log_prefix_for(condition, dataset_kind)
    log_dir = log_dir_for(logs_root, prefix, run_timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = env_for_log_dir(log_dir)
    prompt_file = prompt_file_for(condition)

    print()
    print(f"Condition: {condition}")
    print(f"Dataset kind: {dataset_kind}")
    print(f"Dataset file: {dataset_file}")
    print(f"Prompt file: {prompt_file}")
    print(f"Attack: {attack}")
    print(f"Model: {model}")
    print(f"Samples: {len(sample_ids)}")
    print(f"Execution mode: {execution_mode}")
    print(f"Log dir: {log_dir}")

    base_command = [
        "uv",
        "run",
        "inspect",
        "eval",
        "sleeper_eval/task.py",
        "-T",
        f"dataset_file={dataset_file}",
        "-T",
        "defense=",
        "-T",
        f"attack={attack}",
        "-T",
        "memory_backend=tool",
        "-T",
        "provider=claude",
        "-T",
        f"prompt_model={model}",
        "-T",
        f"system_prompt_file={prompt_file}",
        "--model",
        model,
        "--model-cost-config",
        str(pricing_file),
        "--max-connections",
        str(max_connections),
        "--max-samples",
        str(max_samples),
        "--max-retries",
        str(max_retries),
        "--timeout",
        str(timeout),
        "--attempt-timeout",
        str(attempt_timeout),
        "--no-score",
        "--log-dir",
        str(log_dir),
    ]

    if execution_mode == "batch":
        command = list(base_command)
        if sample_limit > 0:
            command.extend(["--limit", str(sample_limit)])
        run_subprocess(command, env=env, dry_run=dry_run)
    else:
        for index, sample_id in enumerate(sample_ids, start=1):
            print(f"[{condition}] sample {index}/{len(sample_ids)}: {sample_id}")
            command = list(base_command)
            command.extend(["--sample-id", sample_id])
            run_subprocess(command, env=env, dry_run=dry_run)
            if index < len(sample_ids) and sleep_seconds > 0:
                if dry_run:
                    print(f"sleep {sleep_seconds}")
                else:
                    time.sleep(sleep_seconds)

    if not no_score:
        score_log_dir(log_dir, dry_run=dry_run)

    return log_dir


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = (args.manifest or default_manifest_path(args.seed)).resolve()

    if args.prepare_samples:
        prepare_samples(args.seed, args.dry_run)

    manifest = load_manifest(manifest_path)
    run_timestamp = args.run_timestamp or timestamp_now()
    pricing_file = args.pricing_file.resolve()
    logs_root = args.logs_root.resolve()

    for condition in conditions_to_run(args.condition):
        sleep_seconds = (
            args.full_sleep_seconds if condition == "full" else args.truncated_sleep_seconds
        )
        run_condition(
            condition=condition,
            dataset_kind=args.dataset_kind,
            attack=args.attack,
            model=args.model,
            manifest=manifest,
            pricing_file=pricing_file,
            logs_root=logs_root,
            max_connections=args.max_connections,
            max_samples=args.max_samples,
            max_retries=args.max_retries,
            timeout=args.timeout,
            attempt_timeout=args.attempt_timeout,
            execution_mode=args.execution_mode,
            sleep_seconds=sleep_seconds,
            sample_limit=args.sample_limit,
            run_timestamp=run_timestamp,
            no_score=args.no_score,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
