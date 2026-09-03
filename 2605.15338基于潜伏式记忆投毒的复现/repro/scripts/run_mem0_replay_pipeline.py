#!/usr/bin/env python3
"""One-command transcript -> replay -> optional semantic scoring pipeline for mem0 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from sleeper_eval.eval_campaign.config import CampaignConfig, load_campaign_config
from sleeper_eval.eval_campaign.mem0_qdrant import mem0_qdrant_runtime
from sleeper_eval.eval_campaign.mem0_replay import (
    replay_mem0_manager_campaign,
    render_replay_plan,
)
from sleeper_eval.eval_campaign.preflight import preflight_campaign
from sleeper_eval.eval_campaign.run import run_campaign


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Transcript-only mem0 campaign config.")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Pipeline root output dir. Defaults to the config output_dir. Transcript logs are "
            "written under transcript/ and replayed outputs under replay/."
        ),
    )
    parser.add_argument(
        "--score-semantic",
        action="store_true",
        help="Apply semantic scoring inline during replay using the grader model.",
    )
    parser.add_argument(
        "--grader-model",
        default="",
        help="Optional override for semantic scoring. Defaults to config grader_model.",
    )
    parser.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Skip subject transcript generation and reuse transcript/logs already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved transcript/replay plan and exit.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args(argv)


def _validate_pipeline_config(config: CampaignConfig) -> None:
    if config.memory_backend != "mem0":
        raise ValueError("Pipeline requires memory_backend=mem0.")
    if config.mem0_runtime != "transcript_only":
        raise ValueError("Pipeline requires mem0_runtime=transcript_only.")


def _pipeline_root(config: CampaignConfig, output_dir_override: str) -> Path:
    if output_dir_override.strip():
        return Path(output_dir_override).resolve()
    return Path(config.output_dir).resolve()


def _transcript_config(config: CampaignConfig, transcript_dir: Path) -> CampaignConfig:
    return config.model_copy(update={"output_dir": str(transcript_dir)})


def _write_pipeline_manifest(
    *,
    root_dir: Path,
    config_file: Path,
    transcript_dir: Path,
    replay_dir: Path,
    score_semantic: bool,
    grader_model: str | None,
) -> None:
    payload = {
        "pipeline": "mem0_replay",
        "config_file": str(config_file.resolve()),
        "transcript_dir": str(transcript_dir),
        "replay_dir": str(replay_dir),
        "score_semantic": score_semantic,
        "grader_model": grader_model,
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_file).resolve()
    config = load_campaign_config(config_path)
    _validate_pipeline_config(config)

    root_dir = _pipeline_root(config, args.output_dir)
    transcript_dir = root_dir / "transcript"
    replay_dir = root_dir / "replay"
    transcript_log_dir = transcript_dir / "logs"
    root_dir.mkdir(parents=True, exist_ok=True)
    grader_model = args.grader_model.strip() or config.grader_model

    transcript_config = _transcript_config(config, transcript_dir)
    transcript_preflight = preflight_campaign(transcript_config, config_path=config_path)

    print("== Mem0 Replay Pipeline ==")
    print(f"config={config_path}")
    print(f"root_dir={root_dir}")
    print(f"transcript_dir={transcript_dir}")
    print(f"replay_dir={replay_dir}")
    print(f"score_semantic={args.score_semantic}")
    print(f"grader_model={grader_model if args.score_semantic else 'none'}")
    print("")
    print("== Transcript Plan ==")
    print(f"campaign={transcript_preflight.resolved_config['name']}")
    print(f"expanded_tasks={len(transcript_preflight.expanded_tasks)}")
    print(f"output_dir={transcript_dir}")
    if transcript_log_dir.exists():
        print(
            render_replay_plan(
                sources=[transcript_log_dir],
                config=config,
                output_dir=replay_dir,
                grader_model=grader_model if args.score_semantic else None,
                apply_semantic_scoring=args.score_semantic,
            )
        )
    else:
        print("")
        print("== Mem0 Replay Plan ==")
        print("source_logs=1")
        print(f"output_dir={replay_dir}")
        print(f"expected_source_log_dir={transcript_log_dir}")
        print(f"source_mem0_runtime={config.mem0_runtime}")
        print("replay_mem0_runtime=prompt_only")
        print(f"mem0_provider={config.mem0_provider}")
        print(f"mem0_model={config.mem0_model}")
        print(f"mem0_include_document_content={config.mem0_include_document_content}")
        print(f"mem0_thinking={config.mem0_thinking}")
        print(f"mem0_reasoning_effort={config.mem0_reasoning_effort}")
        print(f"replay_max_concurrency={config.replay_max_concurrency}")
        print(f"apply_semantic_scoring={args.score_semantic}")
        print(f"grader_model={grader_model if args.score_semantic else 'none'}")

    if args.dry_run:
        return 0

    if not args.yes:
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if not args.skip_transcript:
        _run_transcript_phase(
            config=config,
            config_path=config_path,
            transcript_dir=transcript_dir,
        )
    elif not transcript_log_dir.exists():
        raise RuntimeError(
            f"--skip-transcript was set but transcript logs do not exist at {transcript_log_dir}"
        )

    with mem0_qdrant_runtime(config, output_dir=replay_dir):
        summary = replay_mem0_manager_campaign(
            config=config,
            config_path=config_path,
            source_logs=[transcript_log_dir],
            output_dir=replay_dir,
            grader_model=grader_model if args.score_semantic else None,
            apply_semantic_scoring=args.score_semantic,
        )

    _write_pipeline_manifest(
        root_dir=root_dir,
        config_file=config_path,
        transcript_dir=transcript_dir,
        replay_dir=replay_dir,
        score_semantic=args.score_semantic,
        grader_model=grader_model if args.score_semantic else None,
    )

    print("")
    print("== Mem0 Replay Pipeline ==")
    print(f"root_dir={root_dir}")
    print(f"transcript_log_dir={transcript_log_dir}")
    print(f"replay_log_dir={replay_dir / 'logs'}")
    for label, path in summary.analysis_outputs.items():
        print(f"{label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
