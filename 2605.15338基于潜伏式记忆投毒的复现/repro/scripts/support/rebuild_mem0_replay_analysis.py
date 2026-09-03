#!/usr/bin/env python3
"""Rebuild deduped analysis outputs for mem0 replay log directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze import write_analysis_outputs
from sleeper_eval.eval_campaign.mem0_replay import load_replay_analysis_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Replay run dir(s) or replay log dir(s).",
    )
    return parser


def resolve_dirs(path_text: str) -> tuple[Path, Path]:
    path = Path(path_text).resolve()
    if path.is_dir() and path.name == "logs":
        return path, path.parent / "analysis"
    logs_dir = path / "logs"
    if logs_dir.is_dir():
        return logs_dir, path / "analysis"
    raise ValueError(f"Could not resolve replay logs from: {path_text}")


def main() -> int:
    args = build_parser().parse_args()
    for raw_path in args.paths:
        log_dir, analysis_dir = resolve_dirs(raw_path)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        outputs = write_analysis_outputs(load_replay_analysis_frame(log_dir), analysis_dir)
        print(f"Rebuilt analysis for {log_dir}")
        for name, output_path in outputs.items():
            print(f"  {name}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
