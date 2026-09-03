#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleeper_eval.dataset import infer_dataset_info, load_dataset, resolve_dataset_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_file",
        help="Dataset path (short path under datasets/, repo-relative path, or absolute path).",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print a per-sample table after the aggregate summaries.",
    )
    return parser


def print_count_table(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    table = (
        df.groupby(column, dropna=False)
        .size()
        .rename("count")
        .sort_values(ascending=False)
        .reset_index()
    )
    print(f"{column}:")
    print(table.to_string(index=False))
    print()


def print_overview(dataset_file: str, df: pd.DataFrame) -> None:
    resolved = resolve_dataset_path(dataset_file)
    info = infer_dataset_info(resolved)

    print("Overview")
    print("--------")
    print(f"dataset_file: {dataset_file}")
    print(f"resolved_path: {resolved}")
    print(f"variant: {info.variant}")
    print(f"split: {info.split}")
    print(f"dataset_memory_condition: {info.memory_condition}")
    print(f"samples: {len(df)}")

    if "doc_text_source" in df.columns:
        normalized = int((df["doc_text_source"] != "document.text").sum())
        print(f"normalized_document_text_samples: {normalized}")

    if "has_memories" in df.columns:
        print(f"samples_with_memories: {int(df['has_memories'].sum())}")

    if "has_contradiction" in df.columns:
        print(f"samples_with_contradiction: {int(df['has_contradiction'].sum())}")

    print()


def print_cross_tabs(df: pd.DataFrame) -> None:
    if {"doc_domain", "memory_condition"}.issubset(df.columns):
        print("doc_domain x memory_condition:")
        pivot = pd.crosstab(df["doc_domain"], df["memory_condition"])
        print(pivot.to_string())
        print()

    if {"doc_domain", "doc_text_source"}.issubset(df.columns):
        print("doc_domain x doc_text_source:")
        pivot = pd.crosstab(df["doc_domain"], df["doc_text_source"])
        print(pivot.to_string())
        print()

    if {"doc_format", "doc_text_source"}.issubset(df.columns):
        print("doc_format x doc_text_source:")
        pivot = pd.crosstab(df["doc_format"], df["doc_text_source"])
        print(pivot.to_string())
        print()


def print_detail(df: pd.DataFrame) -> None:
    columns = [
        column
        for column in [
            "id",
            "doc_domain",
            "doc_format",
            "doc_language_code",
            "memory_condition",
            "has_memories",
            "has_contradiction",
            "doc_text_source",
            "category_id",
            "subcategory_id",
            "domain_seed",
        ]
        if column in df.columns
    ]
    if not columns:
        return

    print("Per-sample detail")
    print("-----------------")
    print(df[columns].sort_values(["doc_domain", "id"]).to_string(index=False))


def main() -> None:
    args = build_parser().parse_args()

    dataset = load_dataset(args.dataset_file)
    rows = [
        {
            "id": sample.id,
            **sample.metadata,
        }
        for sample in dataset
    ]
    df = pd.DataFrame(rows)

    print_overview(args.dataset_file, df)
    for column in [
        "doc_domain",
        "doc_format",
        "doc_language_code",
        "memory_condition",
        "doc_text_source",
        "category_id",
        "subcategory_id",
        "has_memories",
        "has_contradiction",
    ]:
        print_count_table(df, column)

    print_cross_tabs(df)

    if args.detail:
        print_detail(df)


if __name__ == "__main__":
    main()
