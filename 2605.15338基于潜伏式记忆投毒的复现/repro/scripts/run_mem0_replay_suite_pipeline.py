#!/usr/bin/env python3
"""One-command transcript -> replay-suite -> optional semantic scoring pipeline for mem0 experiments."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys

import yaml

from sleeper_eval.eval_campaign.config import CampaignConfig, load_campaign_config
from sleeper_eval.eval_campaign.preflight import preflight_campaign
from sleeper_eval.eval_campaign.replay_suite import (
    ReplaySuiteTarget,
    discover_replay_suite_targets,
    replay_output_is_complete,
)
from sleeper_eval.eval_campaign.run import run_campaign


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "replay_mem0_manager.py"


@dataclass(frozen=True)
class SuitePipelineResult:
    target: ReplaySuiteTarget
    returncode: int
    log_path: Path
    skipped: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject_config",
        help="Transcript-only mem0 campaign config, typically a subject_transcript.yaml file.",
    )
    parser.add_argument(
        "--suite-config-dir",
        default="",
        help=(
            "Optional replay suite root. Defaults to the parent of the subject-family folder, "
            "e.g. subject config .../all/subject_transcript.yaml implies suite root .../"
        ),
    )
    parser.add_argument(
        "--subject-family",
        default="",
        help="Optional subject-family override. Defaults to the subject config parent folder name.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Pipeline root output dir. Transcript logs are written under transcript/ and each "
            "manager replay is written under replays/<manager>/."
        ),
    )
    parser.add_argument(
        "--manager",
        action="append",
        default=[],
        help="Optional manager key filter, e.g. gemini31flashlite. May be repeated.",
    )
    parser.add_argument(
        "--max-parallel-managers",
        type=int,
        default=3,
        help="Maximum number of replay configs to run at once.",
    )
    parser.add_argument(
        "--score-semantic",
        action="store_true",
        help="Apply semantic scoring inline during each replay using the grader model.",
    )
    parser.add_argument(
        "--grader-model",
        default="",
        help="Optional override for semantic scoring. Defaults to each replay config grader_model.",
    )
    parser.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Skip subject transcript generation and reuse transcript/logs already present.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip manager replay outputs that already have manifest, analysis, and .eval logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved transcript/replay-suite plan and exit.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args(argv)


def _validate_subject_config(config: CampaignConfig) -> None:
    if config.memory_backend != "mem0":
        raise ValueError("Pipeline requires memory_backend=mem0.")
    if config.mem0_runtime != "transcript_only":
        raise ValueError("Pipeline requires mem0_runtime=transcript_only.")


def infer_suite_layout(
    *,
    subject_config_path: Path,
    suite_config_dir_override: str,
    subject_family_override: str,
) -> tuple[Path, str]:
    subject_family = subject_family_override.strip() or subject_config_path.parent.name
    if suite_config_dir_override.strip():
        suite_config_dir = Path(suite_config_dir_override).resolve()
    else:
        suite_config_dir = subject_config_path.parent.parent.resolve()
    return suite_config_dir, subject_family


def _pipeline_root(config: CampaignConfig, output_dir_override: str) -> Path:
    if output_dir_override.strip():
        return Path(output_dir_override).resolve()
    return Path(config.output_dir).resolve()


def _transcript_config(config: CampaignConfig, transcript_dir: Path) -> CampaignConfig:
    return config.model_copy(update={"output_dir": str(transcript_dir)})


def _write_pipeline_manifest(
    *,
    root_dir: Path,
    subject_config: Path,
    suite_config_dir: Path,
    subject_family: str,
    transcript_dir: Path,
    replay_root: Path,
    score_semantic: bool,
    grader_model: str | None,
    managers: list[str],
) -> None:
    payload = {
        "pipeline": "mem0_replay_suite",
        "subject_config": str(subject_config.resolve()),
        "suite_config_dir": str(suite_config_dir.resolve()),
        "subject_family": subject_family,
        "transcript_dir": str(transcript_dir),
        "replay_root": str(replay_root),
        "score_semantic": score_semantic,
        "grader_model": grader_model,
        "managers": managers,
    }
    (root_dir / "pipeline_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _run_transcript_phase(
    *,
    config: CampaignConfig,
    config_path: Path,
    transcript_dir: Path,
) -> None:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_config = _transcript_config(config, transcript_dir)
    preflight = preflight_campaign(transcript_config, config_path=config_path)
    success, _logs, _analysis = run_campaign(
        transcript_config,
        config_path=config_path,
        preflight=preflight,
    )
    if not success:
        raise RuntimeError(f"Transcript phase did not complete successfully for {config_path}.")


def build_suite_replay_command(
    *,
    python_executable: str,
    replay_script: Path,
    config_path: Path,
    transcript_log_dir: Path,
    output_dir: Path,
    score_semantic: bool,
    grader_model: str | None,
) -> list[str]:
    command = [
        python_executable,
        str(replay_script),
        str(config_path.resolve()),
        "--from",
        str(transcript_log_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if score_semantic:
        command.append("--score-semantic")
    if grader_model:
        command.extend(["--grader-model", grader_model])
    command.append("--yes")
    return command


def _default_launcher_log_dir(root_dir: Path, subject_family: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root_dir / "_launcher_logs" / f"{subject_family}_{timestamp}"


def _render_plan(
    *,
    subject_config_path: Path,
    transcript_preflight: object,
    transcript_dir: Path,
    transcript_log_dir: Path,
    replay_root: Path,
    suite_config_dir: Path,
    subject_family: str,
    targets: list[ReplaySuiteTarget],
    score_semantic: bool,
    grader_model: str | None,
    max_parallel_managers: int,
    skip_completed: bool,
) -> str:
    lines = [
        "== Mem0 Replay Suite Pipeline ==",
        f"subject_config={subject_config_path}",
        f"suite_config_dir={suite_config_dir}",
        f"subject_family={subject_family}",
        f"transcript_output_dir={transcript_dir}",
        f"expected_transcript_log_dir={transcript_log_dir}",
        f"replay_root={replay_root}",
        f"targets={len(targets)}",
        f"max_parallel_managers={max_parallel_managers}",
        f"score_semantic={score_semantic}",
        f"grader_model={grader_model or 'per-config default'}",
        f"skip_completed={skip_completed}",
        "",
        "== Transcript Plan ==",
        f"campaign={transcript_preflight.resolved_config['name']}",
        f"expanded_tasks={len(transcript_preflight.expanded_tasks)}",
    ]
    for target in targets:
        target_output = replay_root / target.manager_key
        status = "complete" if replay_output_is_complete(target_output) else "pending"
        lines.append(
            f"- manager={target.manager_key} config={target.config_path} output_dir={target_output} status={status}"
        )
    return "\n".join(lines)


async def _run_one_target(
    *,
    semaphore: asyncio.Semaphore,
    target: ReplaySuiteTarget,
    transcript_log_dir: Path,
    replay_root: Path,
    launcher_log_dir: Path,
    score_semantic: bool,
    grader_model: str | None,
    skip_completed: bool,
) -> SuitePipelineResult:
    output_dir = (replay_root / target.manager_key).resolve()
    log_path = launcher_log_dir / f"{target.manager_key}.log"
    if skip_completed and replay_output_is_complete(output_dir):
        return SuitePipelineResult(target=target, returncode=0, log_path=log_path, skipped=True)

    command = build_suite_replay_command(
        python_executable=sys.executable,
        replay_script=REPLAY_SCRIPT,
        config_path=target.config_path,
        transcript_log_dir=transcript_log_dir,
        output_dir=output_dir,
        score_semantic=score_semantic,
        grader_model=grader_model,
    )

    async with semaphore:
        with log_path.open("wb") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(REPO_ROOT),
                env=os.environ.copy(),
                stdout=handle,
                stderr=asyncio.subprocess.STDOUT,
            )
            returncode = await process.wait()
    return SuitePipelineResult(target=target, returncode=returncode, log_path=log_path)


async def _run_suite(
    *,
    targets: list[ReplaySuiteTarget],
    transcript_log_dir: Path,
    replay_root: Path,
    launcher_log_dir: Path,
    max_parallel_managers: int,
    score_semantic: bool,
    grader_model: str | None,
    skip_completed: bool,
) -> list[SuitePipelineResult]:
    launcher_log_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_parallel_managers)
    tasks = [
        asyncio.create_task(
            _run_one_target(
                semaphore=semaphore,
                target=target,
                transcript_log_dir=transcript_log_dir,
                replay_root=replay_root,
                launcher_log_dir=launcher_log_dir,
                score_semantic=score_semantic,
                grader_model=grader_model,
                skip_completed=skip_completed,
            )
        )
        for target in targets
    ]
    results: list[SuitePipelineResult] = []
    for completed in asyncio.as_completed(tasks):
        result = await completed
        status = "skipped" if result.skipped else ("ok" if result.returncode == 0 else "failed")
        print(
            f"[replay-suite-pipeline] manager={result.target.manager_key} status={status} log={result.log_path}",
            flush=True,
        )
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subject_config_path = Path(args.subject_config).resolve()
    subject_config = load_campaign_config(subject_config_path)
    _validate_subject_config(subject_config)
    suite_config_dir, subject_family = infer_suite_layout(
        subject_config_path=subject_config_path,
        suite_config_dir_override=args.suite_config_dir,
        subject_family_override=args.subject_family,
    )

    root_dir = _pipeline_root(subject_config, args.output_dir)
    transcript_dir = root_dir / "transcript"
    transcript_log_dir = transcript_dir / "logs"
    replay_root = root_dir / "replays"
    launcher_log_dir = _default_launcher_log_dir(root_dir, subject_family)
    root_dir.mkdir(parents=True, exist_ok=True)
    transcript_preflight = preflight_campaign(
        _transcript_config(subject_config, transcript_dir),
        config_path=subject_config_path,
    )
    targets = discover_replay_suite_targets(
        config_dir=suite_config_dir,
        run_prefix="unused_for_family_layout",
        subject_family=subject_family,
        manager_filters=args.manager,
    )
    if not targets:
        raise RuntimeError(
            f"No replay targets found in {suite_config_dir} for subject family {subject_family!r}."
        )

    grader_model = args.grader_model.strip() or None
    print(
        _render_plan(
            subject_config_path=subject_config_path,
            transcript_preflight=transcript_preflight,
            transcript_dir=transcript_dir,
            transcript_log_dir=transcript_log_dir,
            replay_root=replay_root,
            suite_config_dir=suite_config_dir,
            subject_family=subject_family,
            targets=targets,
            score_semantic=args.score_semantic,
            grader_model=grader_model,
            max_parallel_managers=args.max_parallel_managers,
            skip_completed=args.skip_completed,
        )
    )
    if args.dry_run:
        return 0
    if not args.yes:
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if not args.skip_transcript:
        _run_transcript_phase(
            config=subject_config,
            config_path=subject_config_path,
            transcript_dir=transcript_dir,
        )
    elif not transcript_log_dir.exists():
        raise RuntimeError(
            f"--skip-transcript was set but transcript logs do not exist at {transcript_log_dir}"
        )

    results = asyncio.run(
        _run_suite(
            targets=targets,
            transcript_log_dir=transcript_log_dir,
            replay_root=replay_root,
            launcher_log_dir=launcher_log_dir,
            max_parallel_managers=args.max_parallel_managers,
            score_semantic=args.score_semantic,
            grader_model=grader_model,
            skip_completed=args.skip_completed,
        )
    )
    failures = [result for result in results if not result.skipped and result.returncode != 0]

    _write_pipeline_manifest(
        root_dir=root_dir,
        subject_config=subject_config_path,
        suite_config_dir=suite_config_dir,
        subject_family=subject_family,
        transcript_dir=transcript_dir,
        replay_root=replay_root,
        score_semantic=args.score_semantic,
        grader_model=grader_model,
        managers=[target.manager_key for target in targets],
    )

    print("")
    print("== Mem0 Replay Suite Pipeline ==")
    print(f"root_dir={root_dir}")
    print(f"transcript_log_dir={transcript_log_dir}")
    print(f"replay_root={replay_root}")
    print(f"launcher_log_dir={launcher_log_dir}")
    print(f"targets={len(targets)}")
    print(f"failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
