"""Analyze sleeper eval logs with Inspect dataframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df, samples_df

try:
    from scripts.render_attack_result_tables import render_tables_from_rows
except ImportError:
    from render_attack_result_tables import render_tables_from_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        nargs="?",
        default="logs",
        help="Inspect log directory or individual .eval file",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print a per-sample detail table at the end.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for summary outputs (JSON/CSV/Markdown and slices).",
    )
    return parser


def format_rate(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def format_table(table: pd.DataFrame, rate_columns: list[str]) -> pd.DataFrame:
    formatted = table.copy()
    for column in rate_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_rate)
    return formatted


def metadata_to_dict(metadata: object) -> dict:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            loaded = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def add_numeric_scores(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    if "error" in numeric.columns:
        numeric["sample_has_error"] = (
            numeric["error"].fillna("").astype(str).str.strip().str.len() > 0
        )
    else:
        numeric["sample_has_error"] = False

    if "score_goal_match_scorer" in numeric.columns:
        numeric["score_goal_match_numeric"] = numeric["score_goal_match_scorer"].map(
            {"C": 1.0, "I": 0.0}
        )
    if "score_benign_save_goal_match_scorer" in numeric.columns:
        numeric["score_benign_save_goal_match_numeric"] = numeric[
            "score_benign_save_goal_match_scorer"
        ].map({"C": 1.0, "I": 0.0})
    if "score_mem0_goal_match_scorer" in numeric.columns:
        numeric["score_mem0_goal_match_numeric"] = numeric["score_mem0_goal_match_scorer"].map(
            {"C": 1.0, "I": 0.0}
        )

    if "score_tool_call_scorer" in numeric.columns:
        numeric["score_tool_call_numeric"] = pd.to_numeric(
            numeric["score_tool_call_scorer"].replace({"C": 1, "I": 0}),
            errors="coerce",
        )
    if "score_mem0_write_scorer" in numeric.columns:
        numeric["score_mem0_write_numeric"] = pd.to_numeric(
            numeric["score_mem0_write_scorer"].replace({"C": 1, "I": 0}),
            errors="coerce",
        )

    numeric["score_memory_write_numeric"] = numeric.get("score_tool_call_numeric")
    if "score_mem0_write_numeric" in numeric.columns:
        numeric["score_memory_write_numeric"] = numeric["score_memory_write_numeric"].fillna(
            numeric["score_mem0_write_numeric"]
        )

    numeric["score_semantic_match_numeric"] = numeric.get("score_goal_match_numeric")
    if "score_mem0_goal_match_numeric" in numeric.columns:
        numeric["score_semantic_match_numeric"] = numeric["score_semantic_match_numeric"].fillna(
            numeric["score_mem0_goal_match_numeric"]
        )
    if "score_benign_save_goal_match_numeric" in numeric.columns:
        numeric["score_semantic_match_numeric"] = numeric["score_semantic_match_numeric"].fillna(
            numeric["score_benign_save_goal_match_numeric"]
        )

    numeric["score_asr_numeric"] = numeric.get("score_goal_match_numeric")
    if "score_mem0_goal_match_numeric" in numeric.columns:
        numeric["score_asr_numeric"] = numeric["score_asr_numeric"].fillna(
            numeric["score_mem0_goal_match_numeric"]
        )

    return numeric


def enrich_with_eval_metadata(log_path: str, df: pd.DataFrame) -> pd.DataFrame:
    eval_frame = evals_df(log_path).copy()

    defaults = {
        "task_arg_defense": "",
        "task_arg_attack": "",
        "task_arg_dataset_file": "",
        "task_arg_memory_backend": "",
        "task_arg_mem0_runtime": "",
        "task_arg_evaluation_mode": "",
        "metadata": None,
    }
    for column, default in defaults.items():
        if column not in eval_frame.columns:
            eval_frame[column] = default

    eval_frame["task_arg_defense"] = eval_frame["task_arg_defense"].fillna("").replace(
        {"": "(none)"}
    )
    eval_frame["task_arg_attack"] = eval_frame["task_arg_attack"].fillna("").replace(
        {"": "(unspecified)"}
    )
    eval_frame["task_arg_memory_backend"] = eval_frame["task_arg_memory_backend"].fillna("").replace(
        {"": "(default)"}
    )
    eval_frame["task_arg_mem0_runtime"] = eval_frame["task_arg_mem0_runtime"].fillna("").replace(
        {"": "(n/a)"}
    )

    parsed_metadata = eval_frame["metadata"].map(metadata_to_dict)
    eval_frame["defense_system_prompt_hardening"] = parsed_metadata.map(
        lambda metadata: bool(metadata.get("system_prompt_hardening", False))
    )
    eval_frame["defense_untrusted_content_markers"] = parsed_metadata.map(
        lambda metadata: bool(metadata.get("untrusted_content_markers", False))
    )
    eval_frame["metadata_memory_backend"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("memory_backend", ""))
    ).replace({"": "(unknown)"})
    eval_frame["metadata_mem0_runtime"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("mem0_runtime", ""))
    ).replace({"": "(n/a)"})
    eval_frame["metadata_model_api"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("model_api", ""))
    ).replace({"": "(unknown)"})
    eval_frame["metadata_model_route_type"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("model_route_type", ""))
    ).replace({"": "(unknown)"})
    eval_frame["metadata_model_vendor"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("model_vendor", ""))
    ).replace({"": "(unknown)"})
    eval_frame["metadata_evaluation_mode"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("evaluation_mode", ""))
    ).replace({"": "attack"})
    eval_frame["metadata_campaign_name"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("campaign_name", ""))
    )
    eval_frame["metadata_campaign_dataset_label"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("campaign_dataset_label", ""))
    )
    eval_frame["metadata_campaign_model_label"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("campaign_model_label", ""))
    )
    eval_frame["metadata_campaign_defense_label"] = parsed_metadata.map(
        lambda metadata: str(metadata.get("campaign_defense_label", ""))
    )

    eval_frame["task_arg_evaluation_mode"] = eval_frame["task_arg_evaluation_mode"].fillna("")
    eval_frame["task_arg_evaluation_mode"] = eval_frame["task_arg_evaluation_mode"].replace({"": None})
    eval_frame["task_arg_evaluation_mode"] = eval_frame["task_arg_evaluation_mode"].fillna(
        eval_frame["metadata_evaluation_mode"]
    )

    columns = [
        column
        for column in [
            "eval_id",
            "model",
            "task",
            "task_arg_dataset_file",
            "task_arg_defense",
            "task_arg_attack",
            "task_arg_memory_backend",
            "task_arg_mem0_runtime",
            "task_arg_evaluation_mode",
            "defense_system_prompt_hardening",
            "defense_untrusted_content_markers",
            "metadata_memory_backend",
            "metadata_mem0_runtime",
            "metadata_model_api",
            "metadata_model_route_type",
            "metadata_model_vendor",
            "metadata_campaign_name",
            "metadata_campaign_dataset_label",
            "metadata_campaign_model_label",
            "metadata_campaign_defense_label",
        ]
        if column in eval_frame.columns
    ]
    return df.merge(eval_frame[columns], on="eval_id", how="left")


def load_analysis_frame(log_path: str) -> pd.DataFrame:
    df = samples_df(log_path)
    if df.empty:
        raise ValueError(f"No sample rows found for: {log_path}")
    df = add_numeric_scores(df)
    return enrich_with_eval_metadata(log_path, df)


def with_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "task_arg_evaluation_mode" in display.columns:
        display["evaluation_mode"] = display["task_arg_evaluation_mode"].fillna("attack")
    else:
        display["evaluation_mode"] = "attack"
    if "metadata_campaign_defense_label" in display.columns:
        display["defense_label"] = display["metadata_campaign_defense_label"].fillna("")
    else:
        display["defense_label"] = ""
    if "task_arg_defense" in display.columns:
        display["defense_label"] = display["defense_label"].where(
            display["defense_label"].astype(str).str.len() > 0,
            display["task_arg_defense"],
        )
    if "metadata_campaign_model_label" in display.columns:
        display["model_label"] = display["metadata_campaign_model_label"].fillna("")
    else:
        display["model_label"] = ""
    if "model" in display.columns:
        display["model_label"] = display["model_label"].where(
            display["model_label"].astype(str).str.len() > 0,
            display["model"],
        )
    if "metadata_campaign_dataset_label" in display.columns:
        display["dataset_label"] = display["metadata_campaign_dataset_label"].fillna("")
    else:
        display["dataset_label"] = ""
    if "task_arg_dataset_file" in display.columns:
        display["dataset_label"] = display["dataset_label"].where(
            display["dataset_label"].astype(str).str.len() > 0,
            display["task_arg_dataset_file"],
        )
    return display


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    display = with_display_columns(df)
    if "sample_has_error" in display.columns:
        display = display.loc[~display["sample_has_error"]].copy()
    group_columns = [
        column
        for column in [
            "evaluation_mode",
            "model_label",
            "model",
            "defense_label",
            "task_arg_defense",
            "task_arg_attack",
            "dataset_label",
            "task_arg_dataset_file",
            "task_arg_memory_backend",
            "task_arg_mem0_runtime",
            "metadata_campaign_name",
        ]
        if column in display.columns
    ]
    summary = (
        display.groupby(group_columns, dropna=False)
        .agg(
            sample_count=("id", "count"),
            memory_write_rate=("score_memory_write_numeric", "mean"),
            semantic_match_rate=("score_semantic_match_numeric", "mean"),
        )
        .reset_index()
        .sort_values(["evaluation_mode", "dataset_label", "model_label", "defense_label"])
    )
    return summary


def summary_markdown(summary: pd.DataFrame) -> str:
    formatted = summary.copy()
    for column in ["memory_write_rate", "semantic_match_rate"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_rate)
    columns = list(formatted.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def build_slice_tables(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slices: dict[str, pd.DataFrame] = {}
    for metric in ["memory_write_rate", "semantic_match_rate"]:
        if metric not in summary.columns:
            continue
        if {"model_label", "defense_label"}.issubset(summary.columns):
            slices[f"model_x_defense_{metric}"] = summary.pivot_table(
                values=metric,
                index="model_label",
                columns="defense_label",
                aggfunc="mean",
            )
        if {"dataset_label", "defense_label"}.issubset(summary.columns):
            slices[f"dataset_x_defense_{metric}"] = summary.pivot_table(
                values=metric,
                index="dataset_label",
                columns="defense_label",
                aggfunc="mean",
            )
    return slices


def render_attack_result_tables_markdown(
    summary: pd.DataFrame,
    *,
    collapse_non_english: bool = False,
) -> str | None:
    required = {
        "dataset_label",
        "defense_label",
        "task_arg_attack",
        "sample_count",
        "memory_write_rate",
        "semantic_match_rate",
    }
    if not required.issubset(summary.columns):
        return None

    cleaned = summary.copy()
    # Keep rows when at least one metric is present. This matters for paper-facing
    # tables where semantic ASR is the authoritative value even if write-rate
    # scoring is missing from some logs.
    cleaned = cleaned.dropna(subset=["sample_count"])
    cleaned = cleaned[
        ~(
            cleaned["memory_write_rate"].isna()
            & cleaned["semantic_match_rate"].isna()
        )
    ]
    if cleaned.empty:
        return None
    rows = cleaned.to_dict(orient="records")
    try:
        return render_tables_from_rows(
            rows,
            include_placeholders=False,
            collapse_non_english=collapse_non_english,
        )
    except ValueError:
        return None


def write_analysis_outputs(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    slices_dir = output_path / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_table(df)
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
    attack_result_tables_md = output_path / "attack_result_tables.md"
    attack_result_tables = render_attack_result_tables_markdown(summary)
    if attack_result_tables is not None:
        attack_result_tables_md.write_text(attack_result_tables, encoding="utf-8")
        written["attack_result_tables_md"] = attack_result_tables_md
    for name, table in build_slice_tables(summary).items():
        path = slices_dir / f"{name}.csv"
        table.to_csv(path)
        written[name] = path
    return written


def print_overview(df: pd.DataFrame) -> None:
    error_count = int(df["sample_has_error"].sum()) if "sample_has_error" in df.columns else 0
    print("Overview")
    print("--------")
    print(f"Samples: {len(df)}")
    print(f"Errored samples excluded from summaries: {error_count}")
    print(f"Eval IDs: {df['eval_id'].nunique() if 'eval_id' in df.columns else 'n/a'}")
    if "model" in df.columns:
        print(f"Models: {df['model'].nunique()} -> {sorted(df['model'].dropna().unique())}")
    if "task_arg_dataset_file" in df.columns:
        datasets = sorted(df["task_arg_dataset_file"].dropna().unique())
        print(f"Datasets: {len(datasets)} -> {datasets}")
    if "task_arg_defense" in df.columns:
        defenses = sorted(df["task_arg_defense"].dropna().unique())
        print(f"Defenses: {len(defenses)} -> {defenses}")
    if "task_arg_attack" in df.columns:
        attacks = sorted(df["task_arg_attack"].dropna().unique())
        print(f"Attacks: {len(attacks)} -> {attacks}")
    if "task_arg_memory_backend" in df.columns:
        backends = sorted(df["task_arg_memory_backend"].dropna().unique())
        print(f"Memory backends: {len(backends)} -> {backends}")
    if "task_arg_mem0_runtime" in df.columns:
        runtimes = sorted(df["task_arg_mem0_runtime"].dropna().unique())
        print(f"Memory runtimes: {len(runtimes)} -> {runtimes}")
    if "task_arg_evaluation_mode" in df.columns:
        modes = sorted(df["task_arg_evaluation_mode"].dropna().unique())
        print(f"Evaluation modes: {len(modes)} -> {modes}")
    print()


def print_group_summary(
    df: pd.DataFrame,
    metric_column: str,
    label: str,
    group_column: str,
) -> None:
    if metric_column not in df.columns or group_column not in df.columns:
        return

    table = (
        df.groupby(group_column, dropna=False)[metric_column]
        .agg(rate="mean", n="count")
        .sort_values(["rate", "n"], ascending=[False, False])
        .reset_index()
    )
    print(f"{label} by {group_column}:")
    print(format_table(table, ["rate"]).to_string(index=False))
    print()


def print_pivot_summary(
    df: pd.DataFrame,
    metric_column: str,
    label: str,
    index_column: str,
    column_column: str,
) -> None:
    if metric_column not in df.columns:
        return
    if index_column not in df.columns or column_column not in df.columns:
        return

    pivot = df.pivot_table(
        values=metric_column,
        index=index_column,
        columns=column_column,
        aggfunc="mean",
    )
    if pivot.empty:
        return

    print(f"{label} by {index_column} x {column_column}:")
    formatted_pivot = pivot.apply(lambda column: column.map(format_rate))
    print(formatted_pivot.fillna("-"))
    print()


def print_metric_sections(df: pd.DataFrame, metric_column: str, label: str) -> None:
    if metric_column not in df.columns:
        return
    if "sample_has_error" in df.columns:
        df = df.loc[~df["sample_has_error"]].copy()

    print(label)
    print("-" * len(label))
    for group_column in [
        "model",
        "model_label",
        "task_arg_defense",
        "defense_label",
        "task_arg_attack",
        "task_arg_evaluation_mode",
        "task_arg_memory_backend",
        "task_arg_mem0_runtime",
        "metadata_model_api",
        "metadata_model_route_type",
        "metadata_model_vendor",
        "task_arg_dataset_file",
        "metadata_memory_condition",
        "metadata_doc_domain",
        "metadata_category_id",
        "metadata_subcategory_id",
        "metadata_doc_text_source",
    ]:
        print_group_summary(df, metric_column, label, group_column)

    print_pivot_summary(df, metric_column, label, "model_label", "defense_label")
    print_pivot_summary(df, metric_column, label, "model", "task_arg_defense")
    print_pivot_summary(df, metric_column, label, "model", "task_arg_dataset_file")


def print_detail_table(df: pd.DataFrame) -> None:
    columns = [
        column
        for column in [
            "eval_id",
            "id",
            "model_label",
            "model",
            "defense_label",
            "task_arg_dataset_file",
            "task_arg_defense",
            "task_arg_attack",
            "task_arg_evaluation_mode",
            "score_tool_call_scorer",
            "score_goal_match_scorer",
            "score_benign_save_goal_match_scorer",
            "score_mem0_write_scorer",
            "score_mem0_goal_match_scorer",
        ]
        if column in df.columns
    ]
    if not columns:
        return

    detail = df[columns].sort_values(
        [column for column in ["model_label", "defense_label", "task_arg_dataset_file", "id"] if column in columns]
    )
    print("Per-sample detail")
    print("-----------------")
    print(detail.to_string(index=False))
    print()


def main() -> None:
    args = build_parser().parse_args()

    df = with_display_columns(load_analysis_frame(args.log_path))
    print_overview(df)

    attack_df = df[df["evaluation_mode"] == "attack"]
    benign_df = df[df["evaluation_mode"] == "benign_save"]

    if not attack_df.empty:
        print_metric_sections(attack_df, "score_asr_numeric", "ASR")
        print_metric_sections(attack_df, "score_memory_write_numeric", "Memory write rate")
    if not benign_df.empty:
        print_metric_sections(benign_df, "score_memory_write_numeric", "Benign save write success")
        print_metric_sections(benign_df, "score_semantic_match_numeric", "Benign save semantic match")
    if args.output_dir:
        written = write_analysis_outputs(df, args.output_dir)
        print("Analysis outputs")
        print("----------------")
        for label, path in written.items():
            print(f"{label}: {path}")
        print()
    if args.detail:
        print_detail_table(df)


if __name__ == "__main__":
    main()
