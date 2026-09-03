#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.analysis import samples_df

from sleeper_eval.followup_eval.dataset import load_dataset
from sleeper_eval.followup_eval.task import followup_eval

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT / "logs" / "followup-smoke"
INSPECT_HOME = ROOT / ".inspect_home"
DEFAULT_DATASET = "followup/smoke/behavioral_influence_smoke.json"
DEFAULT_MODEL = "openai/gpt-5.4"
DEFAULT_GRADER_MODEL = "openai/gpt-5.4-mini"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model slug under evaluation.")
    parser.add_argument(
        "--grader-model",
        default=DEFAULT_GRADER_MODEL,
        help="Model slug to use for the grader role.",
    )
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET, help="Follow-up dataset path.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Inspect log directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan and exit.")
    return parser


def configure_env(log_dir: Path) -> None:
    INSPECT_HOME.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(INSPECT_HOME.resolve())
    os.environ["INSPECT_TRACE_FILE"] = str((log_dir / "inspect-trace.log").resolve())
    os.environ["INSPECT_LOG_DIR"] = str(log_dir.resolve())


def main() -> int:
    args = build_parser().parse_args()
    log_dir = Path(args.log_dir)
    dataset = load_dataset(args.dataset_file)

    print(f"Log dir: {log_dir}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Samples: {len(dataset)}")
    print(f"Model: {args.model}")
    print(f"Grader model: {args.grader_model}")

    if args.dry_run:
        return 0

    configure_env(log_dir)
    success, _ = eval_set(
        [followup_eval(dataset_file=args.dataset_file, prompt_model=args.model)],
        model=[args.model],
        model_roles={"grader": args.grader_model},
        log_dir=str(log_dir),
    )

    frame = samples_df(str(log_dir))
    columns = [
        column
        for column in [
            "id",
            "metadata_category",
            "metadata_injected_memory",
            "model",
            "score_behavior_influence_scorer",
        ]
        if column in frame.columns
    ]
    if columns:
        print()
        print(frame[columns].sort_values("id"))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
