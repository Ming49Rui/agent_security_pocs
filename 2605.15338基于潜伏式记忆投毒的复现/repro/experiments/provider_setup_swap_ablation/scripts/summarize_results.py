#!/usr/bin/env python3
"""Summarize provider-setup-swap campaign results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sleeper_eval.eval_campaign.run import load_campaign_analysis_frame
from sleeper_eval.provider_config import infer_provider

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    EXPERIMENT_ROOT
    / "data"
    / "merged_eval_in_provider_setup_swap_true_opt_100_seed_20260507.manifest.json"
)

SUBJECT_ORDER = {
    "gpt-5.5": 0,
    "claude-sonnet-4.6": 1,
    "gemini-3.1-pro-preview": 2,
}
SETUP_ORDER = {"gpt": 0, "claude": 1, "gemini": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Campaign run directory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to <run_dir>/analysis/provider_setup_swap.",
    )
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def display_subject_label(model_slug: str) -> str:
    slug = model_slug.strip()
    if slug == "openai/gpt-5.5":
        return "gpt-5.5"
    if slug == "anthropic/claude-sonnet-4-6":
        return "claude-sonnet-4.6"
    if slug == "openrouter/google/gemini-3.1-pro-preview":
        return "gemini-3.1-pro-preview"
    return slug.split("/")[-1]


def model_lookup_by_label(config_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for model in config_payload.get("models") or []:
        label = str(model["label"])
        subject_model = str(model["model"])
        prompt_model = str(model.get("prompt_model") or subject_model)
        provider = str(model["provider"])
        lookup[label] = {
            "subject_model": subject_model,
            "prompt_model": prompt_model,
            "provider": provider,
            "subject_label": display_subject_label(subject_model),
            "subject_family": infer_provider(subject_model),
            "setup_family": provider,
        }
    return lookup


def family_lookup_from_manifest(manifest: dict[str, Any]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for sample in manifest["samples"]:
        key = (str(sample["doc_id"]), str(sample["goal_id"]))
        lookup[key] = str(sample["family"])
    return lookup


def ensure_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in analysis frame: {missing}")


def summarize_subset(
    df: pd.DataFrame,
    *,
    subset_label: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "model_label",
                "subject_label",
                "subject_model",
                "prompt_model",
                "provider",
                "subject_family",
                "setup_family",
            ],
            dropna=False,
        )
        .agg(
            sample_count=("id", "count"),
            memory_write_rate=("score_memory_write_numeric", "mean"),
            semantic_match_rate=("score_semantic_match_numeric", "mean"),
        )
        .reset_index()
    )
    grouped["subset"] = subset_label
    native_lookup = (
        grouped.loc[grouped["setup_family"] == grouped["subject_family"]]
        .set_index("subject_label")[["memory_write_rate", "semantic_match_rate"]]
        .to_dict(orient="index")
    )
    grouped["memory_write_delta_vs_native"] = grouped.apply(
        lambda row: row["memory_write_rate"]
        - native_lookup[row["subject_label"]]["memory_write_rate"],
        axis=1,
    )
    grouped["semantic_match_delta_vs_native"] = grouped.apply(
        lambda row: row["semantic_match_rate"]
        - native_lookup[row["subject_label"]]["semantic_match_rate"],
        axis=1,
    )
    grouped["subject_order"] = grouped["subject_label"].map(SUBJECT_ORDER).fillna(99)
    grouped["setup_order"] = grouped["setup_family"].map(SETUP_ORDER).fillna(99)
    grouped = grouped.sort_values(["subject_order", "setup_order", "model_label"]).drop(
        columns=["subject_order", "setup_order"]
    )
    return grouped


def format_rate(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def render_markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in [
        "memory_write_rate",
        "semantic_match_rate",
        "memory_write_delta_vs_native",
        "semantic_match_delta_vs_native",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_rate)
    columns = list(formatted.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def render_interpretation(overall: pd.DataFrame) -> str:
    lines = ["# Provider Setup Swap Interpretation", ""]
    for subject_label, subject_rows in overall.groupby("subject_label", sort=False):
        subject_rows = subject_rows.sort_values(
            ["semantic_match_rate", "memory_write_rate"],
            ascending=[False, False],
        )
        best = subject_rows.iloc[0]
        native = subject_rows.loc[
            subject_rows["setup_family"] == subject_rows["subject_family"]
        ].iloc[0]
        lines.append(f"## {subject_label}")
        lines.append(
            f"- native setup `{native['setup_family']}`: memory_write={format_rate(native['memory_write_rate'])}, "
            f"semantic_match={format_rate(native['semantic_match_rate'])}"
        )
        lines.append(
            f"- best semantic setup: `{best['setup_family']}` at {format_rate(best['semantic_match_rate'])}"
        )
        if best["setup_family"] == native["setup_family"]:
            lines.append("- native setup remains best on semantic match for this subject.")
        else:
            lines.append(
                f"- swapped setup `{best['setup_family']}` beats the native semantic rate by "
                f"{format_rate(best['semantic_match_rate'] - native['semantic_match_rate'])}."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run dir: {run_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "analysis" / "provider_setup_swap"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_lookup = model_lookup_by_label(config_payload)
    manifest = load_json(args.manifest.resolve())
    family_lookup = family_lookup_from_manifest(manifest)

    analysis_df = load_campaign_analysis_frame(str(run_dir / "logs"))
    ensure_required_columns(
        analysis_df,
        [
            "id",
            "model_label",
            "metadata_doc_id",
            "metadata_goal_id",
            "score_memory_write_numeric",
            "score_semantic_match_numeric",
        ],
    )
    if "sample_has_error" in analysis_df.columns:
        analysis_df = analysis_df.loc[~analysis_df["sample_has_error"]].copy()

    analysis_df["subject_model"] = analysis_df["model_label"].map(
        lambda label: model_lookup[str(label)]["subject_model"]
    )
    analysis_df["prompt_model"] = analysis_df["model_label"].map(
        lambda label: model_lookup[str(label)]["prompt_model"]
    )
    analysis_df["provider"] = analysis_df["model_label"].map(
        lambda label: model_lookup[str(label)]["provider"]
    )
    analysis_df["subject_label"] = analysis_df["model_label"].map(
        lambda label: model_lookup[str(label)]["subject_label"]
    )
    analysis_df["subject_family"] = analysis_df["model_label"].map(
        lambda label: model_lookup[str(label)]["subject_family"]
    )
    analysis_df["setup_family"] = analysis_df["model_label"].map(
        lambda label: model_lookup[str(label)]["setup_family"]
    )
    analysis_df["sample_family"] = analysis_df.apply(
        lambda row: family_lookup[
            (str(row["metadata_doc_id"]), str(row["metadata_goal_id"]))
        ],
        axis=1,
    )

    overall = summarize_subset(analysis_df, subset_label="overall")
    behavior = summarize_subset(
        analysis_df.loc[analysis_df["sample_family"] == "behavior"].copy(),
        subset_label="behavior",
    )
    agent = summarize_subset(
        analysis_df.loc[analysis_df["sample_family"] == "agent"].copy(),
        subset_label="agent",
    )

    for name, frame in (
        ("overall", overall),
        ("behavior", behavior),
        ("agent", agent),
    ):
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"
        frame.to_csv(csv_path, index=False)
        md_path.write_text(render_markdown_table(frame), encoding="utf-8")

    interpretation_path = output_dir / "interpretation.md"
    interpretation_path.write_text(render_interpretation(overall), encoding="utf-8")

    print(f"overall_csv={output_dir / 'overall.csv'}")
    print(f"behavior_csv={output_dir / 'behavior.csv'}")
    print(f"agent_csv={output_dir / 'agent.csv'}")
    print(f"interpretation_md={interpretation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
