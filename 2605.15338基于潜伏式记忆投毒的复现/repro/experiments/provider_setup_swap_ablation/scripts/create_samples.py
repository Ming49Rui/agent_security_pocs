#!/usr/bin/env python3
"""Build and validate the provider-setup-swap true-optimized 100-sample dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sleeper_eval.dataset import is_supported_provider_doc_format

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260507
DEFAULT_BEHAVIOR_COUNT = 50
DEFAULT_AGENT_COUNT = 50
DEFAULT_OUTPUT_BASENAME = "merged_eval_in_provider_setup_swap_true_opt_100_seed_{seed}.json"

BEHAVIOR_DATASETS = ("datasets/released/behaviour_eval_in_with_true_optimized_goals.json",)
AGENT_DATASETS = ("datasets/released/agent_eval_in_with_true_optimized_goals.json",)


@dataclass(frozen=True)
class SourceSample:
    family: str
    source_dataset_file: str
    record: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return str(self.record["document"]["doc_id"])

    @property
    def goal_id(self) -> str:
        return str(self.record["goal"]["goal_id"])

    @property
    def source_sample_id(self) -> str:
        stem = Path(self.source_dataset_file).stem
        return f"{stem}__{self.doc_id}__{self.goal_id}"

    @property
    def memory_condition(self) -> str:
        preexisting = self.record.get("preexisting_memories") or {}
        if preexisting.get("goal_contradiction_memory") is not None:
            return "contradiction"
        memories = list(preexisting.get("memories") or [])
        return "with_memories" if memories else "without_memories"

    @property
    def doc_format(self) -> str:
        return str(
            (self.record["document"].get("annotations") or {}).get(
                "document_format", "unknown"
            )
        )

    @property
    def doc_domain(self) -> str:
        return str(self.record["document"]["domain"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--behavior-count", type=int, default=DEFAULT_BEHAVIOR_COUNT)
    parser.add_argument("--agent-count", type=int, default=DEFAULT_AGENT_COUNT)
    parser.add_argument(
        "--output",
        default="",
        help="Optional output dataset path. Defaults inside experiments/provider_setup_swap_ablation/data/.",
    )
    return parser


def resolve_repo_path(file_path: str | Path) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_records(file_path: str) -> list[dict[str, Any]]:
    path = resolve_repo_path(file_path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def source_samples_for_family(
    family: str,
    dataset_files: tuple[str, ...],
) -> list[SourceSample]:
    samples: list[SourceSample] = []
    for dataset_file in dataset_files:
        for record in load_records(dataset_file):
            sample = SourceSample(
                family=family,
                source_dataset_file=dataset_file,
                record=record,
            )
            if not is_supported_provider_doc_format(sample.doc_format):
                raise ValueError(
                    f"Unsupported doc_format {sample.doc_format!r} in {dataset_file} "
                    f"for source sample {sample.source_sample_id}."
                )
            samples.append(sample)
    samples.sort(key=lambda sample: sample.source_sample_id)
    return samples


def select_samples(
    samples: list[SourceSample],
    *,
    count: int,
    rng: random.Random,
) -> list[SourceSample]:
    if count <= 0:
        raise ValueError("Requested sample count must be positive.")
    if count > len(samples):
        raise ValueError(f"Requested {count} samples from a pool of {len(samples)}.")
    chosen = rng.sample(samples, count)
    chosen.sort(key=lambda sample: sample.source_sample_id)
    return chosen


def generated_sample_id(output_path: Path, sample: SourceSample) -> str:
    return f"{output_path.stem}__{sample.doc_id}__{sample.goal_id}"


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_samples(samples: list[SourceSample]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "source_dataset_counts": counter_dict(
            [sample.source_dataset_file for sample in samples]
        ),
        "memory_condition_counts": counter_dict(
            [sample.memory_condition for sample in samples]
        ),
        "doc_format_counts": counter_dict([sample.doc_format for sample in samples]),
        "doc_domain_counts": counter_dict([sample.doc_domain for sample in samples]),
        "unique_source_dataset_count": len(
            {sample.source_dataset_file for sample in samples}
        ),
        "unique_doc_format_count": len({sample.doc_format for sample in samples}),
        "unique_doc_domain_count": len({sample.doc_domain for sample in samples}),
    }


def build_validation_report(
    *,
    output_path: Path,
    expected_behavior_count: int,
    expected_agent_count: int,
    behavior_pool: list[SourceSample],
    agent_pool: list[SourceSample],
    selected_behavior: list[SourceSample],
    selected_agent: list[SourceSample],
    combined_selected: list[SourceSample],
) -> dict[str, Any]:
    allowed_behavior_sources = set(BEHAVIOR_DATASETS)
    allowed_agent_sources = set(AGENT_DATASETS)
    all_selected_source_ids = [sample.source_sample_id for sample in combined_selected]
    all_generated_ids = [generated_sample_id(output_path, sample) for sample in combined_selected]

    behavior_provenance_ok = all(
        sample.source_dataset_file in allowed_behavior_sources
        for sample in selected_behavior
    )
    agent_provenance_ok = all(
        sample.source_dataset_file in allowed_agent_sources for sample in selected_agent
    )
    duplicate_source_ids = [
        source_id
        for source_id, count in Counter(all_selected_source_ids).items()
        if count > 1
    ]
    duplicate_generated_ids = [
        sample_id for sample_id, count in Counter(all_generated_ids).items() if count > 1
    ]

    warnings: list[str] = []
    observations: list[str] = []
    for family, selected, pool in (
        ("behavior", selected_behavior, behavior_pool),
        ("agent", selected_agent, agent_pool),
    ):
        selected_summary = summarize_samples(selected)
        pool_summary = summarize_samples(pool)
        observations.append(
            f"{family} selection spans "
            f"{selected_summary['unique_source_dataset_count']}/{pool_summary['unique_source_dataset_count']} "
            "source files, "
            f"{selected_summary['unique_doc_format_count']}/{pool_summary['unique_doc_format_count']} "
            "doc formats, and "
            f"{selected_summary['unique_doc_domain_count']}/{pool_summary['unique_doc_domain_count']} "
            "doc domains."
        )
        if (
            pool_summary["unique_source_dataset_count"] > 1
            and selected_summary["unique_source_dataset_count"] == 1
        ):
            warnings.append(
                f"{family} selection only drew from one source dataset file despite multiple being available."
            )
        if (
            pool_summary["unique_doc_format_count"] > 1
            and selected_summary["unique_doc_format_count"] == 1
        ):
            warnings.append(
                f"{family} selection only covers one doc format despite broader source coverage."
            )
        if (
            pool_summary["unique_doc_domain_count"] > 1
            and selected_summary["unique_doc_domain_count"] == 1
        ):
            warnings.append(
                f"{family} selection only covers one doc domain despite broader source coverage."
            )

    return {
        "checks": {
            "total_sample_count_ok": len(combined_selected)
            == (expected_behavior_count + expected_agent_count),
            "behavior_sample_count_ok": len(selected_behavior) == expected_behavior_count,
            "agent_sample_count_ok": len(selected_agent) == expected_agent_count,
            "behavior_provenance_ok": behavior_provenance_ok,
            "agent_provenance_ok": agent_provenance_ok,
            "source_sample_ids_unique": not duplicate_source_ids,
            "generated_sample_ids_unique": not duplicate_generated_ids,
        },
        "duplicates": {
            "source_sample_ids": duplicate_source_ids,
            "generated_sample_ids": duplicate_generated_ids,
        },
        "selected_summary": {
            "combined": summarize_samples(combined_selected),
            "behavior": summarize_samples(selected_behavior),
            "agent": summarize_samples(selected_agent),
        },
        "source_pool_summary": {
            "behavior": summarize_samples(behavior_pool),
            "agent": summarize_samples(agent_pool),
        },
        "observations": observations,
        "warnings": warnings,
    }


def render_validation_markdown(report: dict[str, Any], *, output_path: Path) -> str:
    selected = report["selected_summary"]
    pools = report["source_pool_summary"]
    checks = report["checks"]
    lines = [
        "# Provider Setup Swap Dataset Validation",
        "",
        f"- dataset: `{output_path}`",
        f"- exact counts: behavior={selected['behavior']['sample_count']}, "
        f"agent={selected['agent']['sample_count']}, total={selected['combined']['sample_count']}",
        "",
        "## Checks",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- `{name}`: {'ok' if ok else 'FAILED'}")

    lines.extend(["", "## Selected Summary", ""])
    for family in ("combined", "behavior", "agent"):
        summary = selected[family]
        lines.append(f"### {family.title()}")
        lines.append(f"- source datasets: `{summary['source_dataset_counts']}`")
        lines.append(f"- memory conditions: `{summary['memory_condition_counts']}`")
        lines.append(f"- doc formats: `{summary['doc_format_counts']}`")
        lines.append(f"- doc domains: `{summary['doc_domain_counts']}`")
        lines.append("")

    lines.extend(["## Source Pool Coverage", ""])
    for family in ("behavior", "agent"):
        summary = pools[family]
        lines.append(f"### {family.title()}")
        lines.append(f"- source datasets: `{summary['source_dataset_counts']}`")
        lines.append(f"- memory conditions: `{summary['memory_condition_counts']}`")
        lines.append(f"- doc formats: `{summary['doc_format_counts']}`")
        lines.append(f"- doc domains: `{summary['doc_domain_counts']}`")
        lines.append("")

    lines.extend(["## Random-Ish Sanity", ""])
    for observation in report["observations"]:
        lines.append(f"- {observation}")
    if report["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = build_parser().parse_args()
    output_path = (
        resolve_repo_path(args.output)
        if args.output
        else EXPERIMENT_ROOT
        / "data"
        / DEFAULT_OUTPUT_BASENAME.format(seed=args.seed)
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    validation_json_path = output_path.with_suffix(".validation.json")
    validation_md_path = output_path.with_suffix(".validation.md")

    behavior_pool = source_samples_for_family("behavior", BEHAVIOR_DATASETS)
    agent_pool = source_samples_for_family("agent", AGENT_DATASETS)

    behavior_rng = random.Random(args.seed + 1)
    agent_rng = random.Random(args.seed + 2)
    shuffle_rng = random.Random(args.seed + 3)

    selected_behavior = select_samples(
        behavior_pool, count=args.behavior_count, rng=behavior_rng
    )
    selected_agent = select_samples(agent_pool, count=args.agent_count, rng=agent_rng)

    combined_selected = [*selected_behavior, *selected_agent]
    shuffle_rng.shuffle(combined_selected)

    dataset_records = [sample.record for sample in combined_selected]
    write_json(output_path, dataset_records)

    validation_report = build_validation_report(
        output_path=output_path,
        expected_behavior_count=args.behavior_count,
        expected_agent_count=args.agent_count,
        behavior_pool=behavior_pool,
        agent_pool=agent_pool,
        selected_behavior=selected_behavior,
        selected_agent=selected_agent,
        combined_selected=combined_selected,
    )
    write_json(validation_json_path, validation_report)
    validation_md_path.write_text(
        render_validation_markdown(validation_report, output_path=output_path),
        encoding="utf-8",
    )

    manifest = {
        "experiment": "provider_setup_swap_ablation",
        "seed": args.seed,
        "dataset_path": str(output_path.relative_to(REPO_ROOT)),
        "validation_json_path": str(validation_json_path.relative_to(REPO_ROOT)),
        "validation_md_path": str(validation_md_path.relative_to(REPO_ROOT)),
        "source_datasets": {
            "behavior": list(BEHAVIOR_DATASETS),
            "agent": list(AGENT_DATASETS),
        },
        "samples": [
            {
                "sample_id": generated_sample_id(output_path, sample),
                "source_sample_id": sample.source_sample_id,
                "family": sample.family,
                "source_dataset_file": sample.source_dataset_file,
                "doc_id": sample.doc_id,
                "goal_id": sample.goal_id,
                "memory_condition": sample.memory_condition,
                "doc_format": sample.doc_format,
                "doc_domain": sample.doc_domain,
            }
            for sample in sorted(
                combined_selected, key=lambda sample: generated_sample_id(output_path, sample)
            )
        ],
        "selected_summary": validation_report["selected_summary"],
        "source_pool_summary": validation_report["source_pool_summary"],
        "checks": validation_report["checks"],
        "warnings": validation_report["warnings"],
        "observations": validation_report["observations"],
    }
    write_json(manifest_path, manifest)

    print(f"dataset={output_path}")
    print(f"manifest={manifest_path}")
    print(f"validation_json={validation_json_path}")
    print(f"validation_md={validation_md_path}")
    print(f"behavior_selected={len(selected_behavior)}")
    print(f"agent_selected={len(selected_agent)}")
    print(f"combined_selected={len(combined_selected)}")
    for observation in validation_report["observations"]:
        print(f"observation={observation}")
    if validation_report["warnings"]:
        for warning in validation_report["warnings"]:
            print(f"warning={warning}")
    else:
        print("warning_count=0")

    if not all(validation_report["checks"].values()):
        raise SystemExit("Dataset validation failed. Inspect the validation report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
