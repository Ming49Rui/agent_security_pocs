#!/usr/bin/env python3
"""Launch a full manager-replay suite for one subject-family transcript source."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys

from sleeper_eval.eval_campaign.replay_suite import (
    ReplaySuiteTarget,
    build_replay_command,
    discover_replay_suite_targets,
    replay_output_is_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "scripts" / "configs" / "final_runs" / "mem0" / "optimization"
RUN_EVAL_CAMPAIGN_SCRIPT = REPO_ROOT / "scripts" / "run_eval_campaign.py"


@dataclass(frozen=True)
class ReplaySuiteResult:
    target: ReplaySuiteTarget
    returncode: int
    log_path: Path
    skipped: bool = False
    timed_out: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="sources",
        action="append",
        required=True,
        help="Source .eval log, logs/ dir, or campaign run dir. May be repeated.",
    )
    parser.add_argument(
        "--subject-family",
        required=True,
        help=(
            "Subject-family key. For legacy flat configs this is the name prefix "
            "(e.g. gpt); for paper ablation configs it is the family folder name "
            "under the config dir (gpt, claude, gemini, generic)."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default="manager_replay_smoke_70",
        help="Replay config prefix before the subject family.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing the replay campaign configs.",
    )
    parser.add_argument(
        "--manager",
        action="append",
        default=[],
        help="Optional manager key filter, e.g. gpt54nano or grok_fast_openrouter. May be repeated.",
    )
    parser.add_argument(
        "--max-parallel-managers",
        type=int,
        default=3,
        help="Maximum number of replay configs to run at once.",
    )
    parser.add_argument(
        "--manager-timeout-minutes",
        type=float,
        default=None,
        help=(
            "Optional wall-clock timeout per manager replay. If a manager exceeds this "
            "runtime, terminate it, keep any written checkpoints, and continue the suite."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip replay targets whose output dir already has a manifest, analysis summary, and .eval log.",
    )
    parser.add_argument(
        "--launcher-log-dir",
        default="",
        help="Optional directory for suite launcher stdout/stderr logs.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Optional fresh root directory for replay outputs. Each manager replay will be "
            "written under this root as <subject-family>_<manager>_replay."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the replay suite plan and exit.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser


def _default_launcher_log_dir(
    *,
    targets: list[ReplaySuiteTarget],
    run_prefix: str,
    subject_family: str,
) -> Path:
    if targets:
        parent = targets[0].output_dir.parent
    else:
        parent = REPO_ROOT / "logs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return parent / "_launcher_logs" / f"{run_prefix}_{subject_family}_{timestamp}"


def _render_plan(
    *,
    targets: list[ReplaySuiteTarget],
    launcher_log_dir: Path,
    skip_completed: bool,
    max_parallel_managers: int,
    manager_timeout_minutes: float | None,
) -> str:
    lines = [
        "",
        "== Mem0 Replay Suite ==",
        f"targets={len(targets)}",
        f"max_parallel_managers={max_parallel_managers}",
        f"manager_timeout_minutes={manager_timeout_minutes}",
        f"skip_completed={skip_completed}",
        f"launcher_log_dir={launcher_log_dir}",
    ]
    for target in targets:
        status = "complete" if replay_output_is_complete(target.output_dir) else "pending"
        lines.append(
            f"- manager={target.manager_key} config={target.config_path} output_dir={target.output_dir} status={status}"
        )
    return "\n".join(lines)


def _target_output_dir(
    *,
    target: ReplaySuiteTarget,
    subject_family: str,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return target.output_dir
    return (output_root / f"{subject_family}_{target.manager_key}_replay").resolve()


async def _run_target(
    *,
    semaphore: asyncio.Semaphore,
    target: ReplaySuiteTarget,
    sources: list[Path],
    launcher_log_dir: Path,
    skip_completed: bool,
    subject_family: str,
    output_root: Path | None,
    timeout_seconds: float | None,
) -> ReplaySuiteResult:
    target_output_dir = _target_output_dir(
        target=target,
        subject_family=subject_family,
        output_root=output_root,
    )
    if skip_completed and replay_output_is_complete(target_output_dir):
        return ReplaySuiteResult(
            target=ReplaySuiteTarget(
                manager_key=target.manager_key,
                config_path=target.config_path,
                output_dir=target_output_dir,
            ),
            returncode=0,
            log_path=launcher_log_dir / f"{target.manager_key}.log",
            skipped=True,
        )

    command = build_replay_command(
        python_executable=sys.executable,
        script_path=RUN_EVAL_CAMPAIGN_SCRIPT,
        config_path=target.config_path,
        sources=sources,
        output_dir=target_output_dir,
    )
    log_path = launcher_log_dir / f"{target.manager_key}.log"
    async with semaphore:
        with log_path.open("wb") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(REPO_ROOT),
                env=os.environ.copy(),
                stdout=handle,
                stderr=asyncio.subprocess.STDOUT,
            )
            timed_out = False
            try:
                if timeout_seconds is None:
                    returncode = await process.wait()
                else:
                    returncode = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                timed_out = True
                handle.write(
                    (
                        f"\n[replay-suite] manager timeout after {timeout_seconds:.1f}s; "
                        "terminating child process.\n"
                    ).encode("utf-8")
                )
                handle.flush()
                process.terminate()
                try:
                    returncode = await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    handle.write(
                        b"[replay-suite] child did not exit after terminate(); killing.\n"
                    )
                    handle.flush()
                    process.kill()
                    returncode = await process.wait()
                if returncode == 0:
                    returncode = 124
    return ReplaySuiteResult(
        target=ReplaySuiteTarget(
            manager_key=target.manager_key,
            config_path=target.config_path,
            output_dir=target_output_dir,
        ),
        returncode=returncode,
        log_path=log_path,
        timed_out=timed_out,
    )


async def _run_suite(
    *,
    targets: list[ReplaySuiteTarget],
    sources: list[Path],
    launcher_log_dir: Path,
    max_parallel_managers: int,
    skip_completed: bool,
    subject_family: str,
    output_root: Path | None,
    timeout_seconds: float | None,
) -> list[ReplaySuiteResult]:
    launcher_log_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_parallel_managers)
    tasks = [
        asyncio.create_task(
            _run_target(
                semaphore=semaphore,
                target=target,
                sources=sources,
                launcher_log_dir=launcher_log_dir,
                skip_completed=skip_completed,
                subject_family=subject_family,
                output_root=output_root,
                timeout_seconds=timeout_seconds,
            )
        )
        for target in targets
    ]
    results: list[ReplaySuiteResult] = []
    for completed in asyncio.as_completed(tasks):
        result = await completed
        status = (
            "skipped"
            if result.skipped
            else ("timed_out" if result.timed_out else ("ok" if result.returncode == 0 else "failed"))
        )
        print(
            f"[replay-suite] manager={result.target.manager_key} status={status} log={result.log_path}",
            flush=True,
        )
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir = Path(args.config_dir).resolve()
    sources = [Path(source).resolve() for source in args.sources]
    targets = discover_replay_suite_targets(
        config_dir=config_dir,
        run_prefix=args.run_prefix,
        subject_family=args.subject_family,
        manager_filters=args.manager,
    )
    if not targets:
        raise SystemExit(
            f"No replay configs found in {config_dir} for run_prefix={args.run_prefix} "
            f"subject_family={args.subject_family} managers={args.manager or '(all)'}."
        )
    launcher_log_dir = (
        Path(args.launcher_log_dir).resolve()
        if args.launcher_log_dir
        else _default_launcher_log_dir(
            targets=targets,
            run_prefix=args.run_prefix,
            subject_family=args.subject_family,
        )
    )
    output_root = Path(args.output_root).resolve() if args.output_root else None
    rendered_targets = [
        ReplaySuiteTarget(
            manager_key=target.manager_key,
            config_path=target.config_path,
            output_dir=_target_output_dir(
                target=target,
                subject_family=args.subject_family,
                output_root=output_root,
            ),
        )
        for target in targets
    ]
    print(
        _render_plan(
            targets=rendered_targets,
            launcher_log_dir=launcher_log_dir,
            skip_completed=args.skip_completed,
            max_parallel_managers=args.max_parallel_managers,
            manager_timeout_minutes=args.manager_timeout_minutes,
        )
    )
    if args.dry_run:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Refusing to start replay suite without --yes in non-interactive mode."
            )
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return 1

    results = asyncio.run(
        _run_suite(
            targets=targets,
            sources=sources,
            launcher_log_dir=launcher_log_dir,
            max_parallel_managers=max(1, args.max_parallel_managers),
            skip_completed=args.skip_completed,
            subject_family=args.subject_family,
            output_root=output_root,
            timeout_seconds=(
                None
                if args.manager_timeout_minutes is None
                else max(0.0, args.manager_timeout_minutes * 60.0)
            ),
        )
    )
    failures = [result for result in results if not result.skipped and result.returncode != 0]
    print("")
    print("== Replay Suite Summary ==")
    print(f"launcher_log_dir={launcher_log_dir}")
    print(f"targets={len(results)}")
    print(f"failures={len(failures)}")
    for result in sorted(results, key=lambda item: item.target.manager_key):
        status = (
            "skipped"
            if result.skipped
            else ("timed_out" if result.timed_out else ("ok" if result.returncode == 0 else "failed"))
        )
        print(
            f"manager={result.target.manager_key} status={status} "
            f"output_dir={result.target.output_dir} log={result.log_path}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
