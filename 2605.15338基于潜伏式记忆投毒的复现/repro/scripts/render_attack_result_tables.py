#!/usr/bin/env python3
"""Render tracker-ready attack result tables from a campaign summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FAMILY_LABELS = {
    "behavior": "Behaviour",
    "agent": "Action",
    "non-english": "Non-English",
    "non-english-with-memories": "Non-English With Memories",
    "non-english-without-memories": "Non-English Without Memories",
}

DEFENSE_ORDER = [
    "no-defense",
    "system-prompt-hardening",
    "untrusted-content-markers",
    "system-hardening-plus-markers",
    "gepa-prompt-hardening-suffix",
]

DEFENSE_DISPLAY = {
    "no-defense": "none",
    "system-prompt-hardening": "system hardening",
    "untrusted-content-markers": "untrusted markers",
    "system-hardening-plus-markers": "both defenses",
    "gepa-prompt-hardening-suffix": "GEPA hardening",
}

DEFENSE_ALIASES = {
    "both-defenses": "system-hardening-plus-markers",
    "gepa-prompt-hardening": "gepa-prompt-hardening-suffix",
}

ATTACK_ORDER = [
    "attack_from_literature",
    "mem0_external_prompt_leak_without_untrusted_markers",
    "mem0_external_prompt_leak",
    "universal_v2_optimized_without_markers",
    "universal_v2_optimized_with_markers",
    "evolved_v2",
]

ATTACK_DISPLAY = {
    "attack_from_literature": "literature",
    "mem0_external_prompt_leak_without_untrusted_markers": "actor-critic no markers [mem0 prompt leak]",
    "mem0_external_prompt_leak": "mem0 external leak [mem0 prompt leak]",
    "universal_v2_optimized_without_markers": "actor-critic no markers",
    "universal_v2_optimized_with_markers": "actor-critic with markers",
    "evolved_v2": "evolutionary",
}

MODEL_ORDER = [
    "gpt-5.4",
    "gpt-5.5",
    "claude-sonnet-4.6",
    "gemini-3.1-pro",
    "kimi-k2.6",
    "deepseek-v4-pro",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", help="Campaign summary.csv path.")
    parser.add_argument(
        "--include-placeholders",
        action="store_true",
        help="Render placeholder rows for defenses/attacks that are not present.",
    )
    parser.add_argument(
        "--collapse-non-english",
        action="store_true",
        help="Aggregate non-English with/without-memories into one weighted table.",
    )
    return parser.parse_args()


def family_for_dataset(dataset_label: str, *, collapse_non_english: bool = False) -> str:
    if dataset_label.startswith("behavior-"):
        return "behavior"
    if dataset_label.startswith("agent-"):
        return "agent"
    if dataset_label == "non-english-true-optimized-with-memories":
        if collapse_non_english:
            return "non-english"
        return "non-english-with-memories"
    if dataset_label == "non-english-true-optimized-without-memories":
        if collapse_non_english:
            return "non-english"
        return "non-english-without-memories"
    raise ValueError(f"Unsupported dataset label for family aggregation: {dataset_label}")


def format_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def parse_optional_rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return float(text)


def canonical_defense_label(defense_label: str) -> str:
    return DEFENSE_ALIASES.get(defense_label, defense_label)


def aggregates_from_rows(
    rows: Iterable[dict[str, object]],
    *,
    collapse_non_english: bool = False,
) -> dict[tuple[str, str, str, str], tuple[float | None, float | None]]:
    agg: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "n_write": 0.0,
            "write": 0.0,
            "n_semantic": 0.0,
            "semantic": 0.0,
        }
    )
    for row in rows:
        family = family_for_dataset(
            str(row["dataset_label"]),
            collapse_non_english=collapse_non_english,
        )
        model = str(row.get("model_label", "")).strip() or str(row.get("model", "")).strip()
        defense = canonical_defense_label(str(row["defense_label"]))
        attack = str(row["task_arg_attack"])
        sample_count = float(row["sample_count"])
        write_rate = parse_optional_rate(row.get("memory_write_rate"))
        semantic_rate = parse_optional_rate(row.get("semantic_match_rate"))
        key = (family, model, defense, attack)
        if write_rate is not None:
            agg[key]["n_write"] += sample_count
            agg[key]["write"] += sample_count * write_rate
        if semantic_rate is not None:
            agg[key]["n_semantic"] += sample_count
            agg[key]["semantic"] += sample_count * semantic_rate

    return {
        key: (
            value["write"] / value["n_write"] if value["n_write"] else None,
            value["semantic"] / value["n_semantic"] if value["n_semantic"] else None,
        )
        for key, value in agg.items()
        if value["n_write"] or value["n_semantic"]
    }


def load_aggregates(summary_csv: Path) -> dict[
    tuple[str, str, str], tuple[float | None, float | None]
]:
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    return aggregates_from_rows(rows)


def _ordered_labels(observed: set[str], preferred: list[str]) -> list[str]:
    ordered = [label for label in preferred if label in observed]
    ordered.extend(sorted(label for label in observed if label not in preferred))
    return ordered


def render_family_table(
    family: str,
    aggregates: dict[tuple[str, str, str, str], tuple[float, float]],
    *,
    include_placeholders: bool,
) -> str:
    family_models = {model for agg_family, model, _, _ in aggregates if agg_family == family}
    family_defenses = {defense for agg_family, _, defense, _ in aggregates if agg_family == family}
    family_attacks = {attack for agg_family, _, _, attack in aggregates if agg_family == family}
    defense_order = [label for label in DEFENSE_ORDER if label in family_defenses]
    if include_placeholders:
        for label in DEFENSE_ORDER:
            if label not in defense_order:
                defense_order.append(label)
    else:
        extras = sorted(label for label in family_defenses if label not in DEFENSE_ORDER)
        defense_order.extend(extras)

    attack_order = [label for label in ATTACK_ORDER if label in family_attacks]
    if include_placeholders:
        for label in ATTACK_ORDER:
            if label not in attack_order:
                attack_order.append(label)
    else:
        extras = sorted(label for label in family_attacks if label not in ATTACK_ORDER)
        attack_order.extend(extras)

    model_order = _ordered_labels(family_models, MODEL_ORDER)
    multi_model = len(model_order) > 1
    lines = [
        f"### {FAMILY_LABELS[family]}",
        "",
        "```text",
        (
            "Model                | Defense              | Attack                                        | Write  | Semantic ASR"
            if multi_model
            else "Defense              | Attack                                        | Write  | Semantic ASR"
        ),
        (
            "---------------------|----------------------|-----------------------------------------------|--------|-------------"
            if multi_model
            else "---------------------|-----------------------------------------------|--------|-------------"
        ),
    ]
    if multi_model:
        for model in model_order:
            for defense in defense_order:
                for attack in attack_order:
                    rates = aggregates.get((family, model, defense, attack))
                    if rates is None and not include_placeholders:
                        continue
                    write_rate = format_rate(rates[0] if rates else None)
                    semantic_rate = format_rate(rates[1] if rates else None)
                    lines.append(
                        f"{model:<21}| {DEFENSE_DISPLAY.get(defense, defense):<21}| "
                        f"{ATTACK_DISPLAY.get(attack, attack):<46}| {write_rate:<7}| {semantic_rate}"
                    )
    else:
        model = model_order[0] if model_order else ""
        for defense in defense_order:
            for attack in attack_order:
                rates = aggregates.get((family, model, defense, attack))
                if rates is None and not include_placeholders:
                    continue
                write_rate = format_rate(rates[0] if rates else None)
                semantic_rate = format_rate(rates[1] if rates else None)
                lines.append(
                    f"{DEFENSE_DISPLAY.get(defense, defense):<21}| "
                    f"{ATTACK_DISPLAY.get(attack, attack):<46}| {write_rate:<7}| {semantic_rate}"
                )
    lines.extend(["```", ""])
    return "\n".join(lines)


def render_tables_from_rows(
    rows: Iterable[dict[str, object]],
    *,
    include_placeholders: bool = False,
    collapse_non_english: bool = False,
) -> str:
    aggregates = aggregates_from_rows(rows, collapse_non_english=collapse_non_english)
    ordered_families = ["behavior", "agent"]
    if collapse_non_english:
        ordered_families.append("non-english")
    else:
        ordered_families.extend(
            ["non-english-with-memories", "non-english-without-memories"]
        )
    family_order = [
        family
        for family in ordered_families
        if any(agg_family == family for agg_family, _, _, _ in aggregates)
    ]
    return (
        "\n\n".join(
            render_family_table(
                family,
                aggregates,
                include_placeholders=include_placeholders,
            ).rstrip()
            for family in family_order
        ).rstrip()
        + "\n"
    )


def render_tables_from_summary_csv(
    summary_csv: Path,
    *,
    include_placeholders: bool = False,
    collapse_non_english: bool = False,
) -> str:
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    return render_tables_from_rows(
        rows,
        include_placeholders=include_placeholders,
        collapse_non_english=collapse_non_english,
    )


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    print(
        render_tables_from_summary_csv(
            summary_csv,
            include_placeholders=args.include_placeholders,
            collapse_non_english=args.collapse_non_english,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
