#!/usr/bin/env python3
"""One-command transcript -> Gemini replay batch -> wait/apply pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

import batch_replay_mem0_gemini as batch_script
from sleeper_eval.eval_campaign.config import CampaignConfig, load_campaign_config
from sleeper_eval.eval_campaign.preflight import preflight_campaign
from sleeper_eval.eval_campaign.run import run_campaign


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Transcript-only mem0 campaign config.")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Pipeline root output dir. Defaults to the config output_dir. The transcript run is "
            "written under transcript/ and replayed outputs under batch_replay/."
        ),
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Use inline Gemini Batch requests instead of file-backed requests.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval for Gemini batch wait.",
    )
    parser.add_argument(
        "--display-name",
        default="mem0-gemini-batch-replay",
        help="Gemini batch display name.",
    )
    parser.add_argument(
        "--split-source-log",
        action="store_true",
        help="Submit one Gemini batch per source .eval log instead of one combined batch.",
    )
    parser.add_argument(
        "--max-active-batches",
        type=int,
        default=None,
        help=(
            "When used with split replay, keep at most this many Gemini child batches active "
            "at once before polling/applying and submitting more."
        ),
    )
    parser.add_argument(
        "--resubmit-batch",
        action="store_true",
        help="Create a new Gemini batch even if a manifest already exists.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop early if the Gemini batch reaches FAILED/CANCELLED/EXPIRED.",
    )
    parser.add_argument(
        "--score-semantic",
        action="store_true",
        help=(
            "After replay apply, run inspect score with the semantic mem0 scorer over all "
            "replayed logs, then rebuild analysis."
        ),
    )
    parser.add_argument(
        "--semantic-scorer",
        default="sleeper_eval/scorer.py@mem0_goal_match_scorer",
        help="Scorer spec to use when --score-semantic is enabled.",
    )
    return parser.parse_args(argv)


def _validate_pipeline_config(config: CampaignConfig) -> None:
    if config.memory_backend != "mem0":
        raise ValueError("Pipeline requires memory_backend=mem0.")
    if config.mem0_runtime != "transcript_only":
        raise ValueError("Pipeline requires mem0_runtime=transcript_only.")
    if config.mem0_provider != "gemini":
        raise ValueError("Pipeline currently supports only mem0_provider=gemini.")


def _render_subject_batch_summary(config: CampaignConfig) -> str:
    enabled: list[str] = []
    disabled: list[str] = []
    for model in config.models:
        batch = model.resolved_batch
        if batch:
            enabled.append(f"{model.label}={batch}")
        else:
            disabled.append(model.label)
    parts: list[str] = []
    if enabled:
        parts.append("enabled: " + ", ".join(enabled))
    if disabled:
        parts.append("disabled: " + ", ".join(disabled))
    return "; ".join(parts) if parts else "disabled"


def _pipeline_root(config: CampaignConfig, output_dir_override: str) -> Path:
    if output_dir_override.strip():
        return Path(output_dir_override).resolve()
    return Path(config.output_dir).resolve()


def _transcript_config(config: CampaignConfig, transcript_dir: Path) -> CampaignConfig:
    return config.model_copy(update={"output_dir": str(transcript_dir)})


def write_pipeline_manifest(
    *,
    root_dir: Path,
    config_file: Path,
    transcript_dir: Path,
    batch_dir: Path,
    batch_manifest_path: Path,
) -> None:
    payload = {
        "pipeline": "mem0_gemini_batch",
        "config_file": str(config_file.resolve()),
        "transcript_dir": str(transcript_dir),
        "batch_dir": str(batch_dir),
        "batch_manifest": str(batch_manifest_path),
    }
    (root_dir / "pipeline_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def run_transcript_phase(
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
        raise RuntimeError(
            f"Transcript phase did not complete successfully for {config_path}."
        )


def ensure_batch_submitted(
    *,
    config_path: Path,
    transcript_dir: Path,
    batch_dir: Path,
    inline: bool,
    display_name: str,
    split_source_log: bool,
    max_active_batches: int | None,
    poll_seconds: int,
    resubmit_batch: bool,
    stop_on_failure: bool,
) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "gemini_batch_manifest.json"
    if manifest_path.exists() and not resubmit_batch:
        print(f"Reusing existing Gemini batch manifest: {manifest_path}")
        return manifest_path

    submit_args = argparse.Namespace(
        config_file=str(config_path),
        source=[str(transcript_dir)],
        output_dir=str(batch_dir),
        manifest=str(manifest_path),
        api_key_env="GOOGLE_API_KEY",
        inline=inline,
        file=not inline,
        display_name=display_name,
        split_source_log=split_source_log or max_active_batches is not None,
        max_active_batches=max_active_batches,
        poll_seconds=poll_seconds,
        apply_on_success=max_active_batches is not None,
        stop_on_failure=stop_on_failure,
    )
    batch_script.submit_batch(submit_args)
    return manifest_path


def wait_and_apply_batch(
    *,
    manifest_path: Path,
    poll_seconds: int,
    stop_on_failure: bool,
) -> None:
    wait_args = argparse.Namespace(
        manifests=[str(manifest_path)],
        poll_seconds=poll_seconds,
        apply_on_success=True,
        stop_on_failure=stop_on_failure,
    )
    batch_script.wait_for_batches(wait_args)


def score_and_analyze_replay(
    *,
    batch_dir: Path,
    semantic_scorer: str,
) -> None:
    log_dir = batch_dir / "logs"
    analysis_dir = batch_dir / "analysis"
    log_paths = sorted(log_dir.glob("*.eval"))
    if not log_paths:
        raise RuntimeError(f"No replayed logs found for semantic scoring in {log_dir}")

    inspect_home = Path(".inspect_home").resolve()
    inspect_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(inspect_home)
    env["INSPECT_LOG_DIR"] = str(log_dir.resolve())
    env["INSPECT_TRACE_FILE"] = str((batch_dir / "inspect-score.log").resolve())

    for log_path in log_paths:
        subprocess.run(
            [
                "uv",
                "run",
                "inspect",
                "score",
                str(log_path),
                "--scorer",
                semantic_scorer,
                "--action",
                "append",
                "--overwrite",
            ],
            check=True,
            env=env,
        )

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/analyze.py",
            str(log_dir),
            "--output-dir",
            str(analysis_dir),
        ],
        check=True,
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_file).resolve()
    config = load_campaign_config(config_path)
    _validate_pipeline_config(config)

    root_dir = _pipeline_root(config, args.output_dir)
    transcript_dir = root_dir / "transcript"
    batch_dir = root_dir / "batch_replay"
    manifest_path = batch_dir / "gemini_batch_manifest.json"
    root_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Mem0 Gemini Batch Pipeline ==")
    print(f"config={config_path}")
    print(f"root_dir={root_dir}")
    print(f"transcript_dir={transcript_dir}")
    print(f"batch_dir={batch_dir}")
    print(f"subject_batch={_render_subject_batch_summary(config)}")

    run_transcript_phase(
        config=config,
        config_path=config_path,
        transcript_dir=transcript_dir,
    )
    actual_manifest = ensure_batch_submitted(
        config_path=config_path,
        transcript_dir=transcript_dir,
        batch_dir=batch_dir,
        inline=args.inline,
        display_name=args.display_name,
        split_source_log=args.split_source_log,
        max_active_batches=args.max_active_batches,
        poll_seconds=args.poll_seconds,
        resubmit_batch=args.resubmit_batch,
        stop_on_failure=args.stop_on_failure,
    )
    write_pipeline_manifest(
        root_dir=root_dir,
        config_file=config_path,
        transcript_dir=transcript_dir,
        batch_dir=batch_dir,
        batch_manifest_path=actual_manifest,
    )
    if args.max_active_batches is None:
        wait_and_apply_batch(
            manifest_path=actual_manifest,
            poll_seconds=args.poll_seconds,
            stop_on_failure=args.stop_on_failure,
        )
    if args.score_semantic:
        score_and_analyze_replay(
            batch_dir=batch_dir,
            semantic_scorer=args.semantic_scorer,
        )
    print("Pipeline completed through Gemini replay apply.")
    if args.score_semantic:
        print("Semantic scoring and final analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
