#!/usr/bin/env python3
"""Replay mem0 manager extraction on existing eval logs without rerunning subject models."""

from __future__ import annotations

import argparse
from pathlib import Path

from sleeper_eval.eval_campaign.config import load_campaign_config
from sleeper_eval.eval_campaign.mem0_qdrant import mem0_qdrant_runtime
from sleeper_eval.eval_campaign.mem0_replay import (
    replay_mem0_manager_campaign,
    render_replay_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Path to a campaign config providing target mem0 settings.")
    parser.add_argument(
        "--from",
        dest="sources",
        action="append",
        required=True,
        help="Source .eval log, logs/ dir, or campaign run dir. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional override for output_dir in the config.",
    )
    parser.add_argument(
        "--score-semantic",
        action="store_true",
        help="Apply inline semantic scoring during replay using the grader model.",
    )
    parser.add_argument(
        "--grader-model",
        default="",
        help="Optional override for the semantic scorer judge model. Defaults to config grader_model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the replay plan and exit.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config_file).resolve()
    config = load_campaign_config(config_path)
    if args.output_dir:
        config = config.model_copy(update={"output_dir": args.output_dir})
    grader_model = args.grader_model.strip() or config.grader_model
    output_dir = Path(config.output_dir).resolve()
    print(
        render_replay_plan(
            sources=args.sources,
            config=config,
            output_dir=output_dir,
            grader_model=grader_model if args.score_semantic else None,
            apply_semantic_scoring=args.score_semantic,
        )
    )
    if args.dry_run:
        return 0
    if not args.yes:
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return 1
    with mem0_qdrant_runtime(config, output_dir=output_dir):
        summary = replay_mem0_manager_campaign(
            config=config,
            config_path=config_path,
            source_logs=args.sources,
            output_dir=output_dir,
            grader_model=grader_model if args.score_semantic else None,
            apply_semantic_scoring=args.score_semantic,
        )
    print("")
    print("== Mem0 Replay ==")
    print(f"output_dir={output_dir}")
    print(f"source_log_count={summary.source_log_count}")
    print(f"replayed_log_count={summary.replayed_log_count}")
    for label, path in summary.analysis_outputs.items():
        print(f"{label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
