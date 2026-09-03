#!/usr/bin/env python3
"""Merge one or more campaign summary.csv files into consolidated analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze import (
    build_slice_tables,
    render_attack_result_tables_markdown,
    summary_markdown,
)


RATE_COLUMNS = ["memory_write_rate", "semantic_match_rate"]
COUNT_COLUMN = "sample_count"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_csv",
        nargs="+",
        help="One or more analysis/summary.csv files to consolidate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write consolidated summary outputs into.",
    )
    parser.add_argument(
        "--collapse-non-english",
        action="store_true",
        help="Aggregate non-English with/without-memories into one weighted table.",
    )
    return parser


def load_summary_csvs(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("No summary CSV files provided.")
    return pd.concat(frames, ignore_index=True)


def consolidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if COUNT_COLUMN not in summary.columns:
        raise ValueError(f"Missing required column: {COUNT_COLUMN}")

    summary = summary.copy()
    summary[COUNT_COLUMN] = pd.to_numeric(summary[COUNT_COLUMN], errors="raise")
    for column in RATE_COLUMNS:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    group_columns = [
        column
        for column in summary.columns
        if column not in {COUNT_COLUMN, *RATE_COLUMNS, "metadata_campaign_name"}
    ]
    if not group_columns:
        raise ValueError("No grouping columns found in summary CSV.")

    for column in RATE_COLUMNS:
        if column in summary.columns:
            summary[f"weighted_{column}"] = summary[COUNT_COLUMN] * summary[column]

    agg_spec: dict[str, str] = {COUNT_COLUMN: "sum"}
    for column in RATE_COLUMNS:
        weighted = f"weighted_{column}"
        if weighted in summary.columns:
            agg_spec[weighted] = "sum"

    consolidated = (
        summary.groupby(group_columns, dropna=False)
        .agg(agg_spec)
        .reset_index()
        .sort_values(["evaluation_mode", "dataset_label", "model_label", "defense_label"])
    )

    for column in RATE_COLUMNS:
        weighted = f"weighted_{column}"
        if weighted in consolidated.columns:
            consolidated[column] = consolidated[weighted] / consolidated[COUNT_COLUMN]
            consolidated = consolidated.drop(columns=[weighted])

    return consolidated


def write_outputs(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    collapse_non_english: bool = False,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    slices_dir = output_path / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "summary.json"
    summary_csv = output_path / "summary.csv"
    summary_md = output_path / "summary.md"

    summary_json.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(summary_markdown(summary) + "\n", encoding="utf-8")

    written = {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
    }

    attack_result_tables = render_attack_result_tables_markdown(
        summary,
        collapse_non_english=collapse_non_english,
    )
    if attack_result_tables is not None:
        attack_result_tables_md = output_path / "attack_result_tables.md"
        attack_result_tables_md.write_text(attack_result_tables, encoding="utf-8")
        written["attack_result_tables_md"] = attack_result_tables_md

    for name, table in build_slice_tables(summary).items():
        path = slices_dir / f"{name}.csv"
        table.to_csv(path)
        written[name] = path

    return written


def main() -> None:
    args = build_parser().parse_args()
    summary = load_summary_csvs(args.summary_csv)
    consolidated = consolidate_summary(summary)
    written = write_outputs(
        consolidated,
        args.output_dir,
        collapse_non_english=args.collapse_non_english,
    )
    for key, path in written.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
