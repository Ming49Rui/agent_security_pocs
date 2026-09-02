#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from sleeper_eval.dataset import is_supported_provider_doc_format

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "eval_dataset" / "datasets"
EXPERIMENT_DATA_DIR = EXPERIMENT_ROOT / "data"
SMOKE_DATASET_PATH = "smoke/merged_eval_in_provider_docrep_smoke.json"
DEFAULT_WITH_MEMORIES_DATASET = "in_dist/merged_eval_in_with_memories.json"
DEFAULT_WITHOUT_MEMORIES_DATASET = "in_dist/merged_eval_in_without_memories.json"
DEFAULT_SEED = 20260329
DEFAULT_WITH_COUNT = 25
DEFAULT_WITHOUT_COUNT = 25
FORMAT_ORDER = ["text", "html", "code", "email", "tweet", "pdf"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--with-memories-dataset",
        default=DEFAULT_WITH_MEMORIES_DATASET,
        help="Dataset file for the with-memories pool.",
    )
    parser.add_argument(
        "--without-memories-dataset",
        default=DEFAULT_WITHOUT_MEMORIES_DATASET,
        help="Dataset file for the without-memories pool.",
    )
    parser.add_argument(
        "--with-count",
        type=int,
        default=DEFAULT_WITH_COUNT,
        help="Number of with-memories samples to select.",
    )
    parser.add_argument(
        "--without-count",
        type=int,
        default=DEFAULT_WITHOUT_COUNT,
        help="Number of without-memories samples to select.",
    )
    return parser


def resolve_repo_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        resolved = path
    else:
        resolved = DATASET_ROOT / path
        if not resolved.exists():
            alt = REPO_ROOT / path
            if alt.exists():
                resolved = alt
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def load_records(file_path: str) -> list[dict[str, Any]]:
    return json.loads(resolve_repo_path(file_path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_id_for(dataset_file: str, record: dict[str, Any]) -> str:
    variant = resolve_repo_path(dataset_file).stem
    return f"{variant}__{record['document']['doc_id']}__{record['goal']['goal_id']}"


def has_memories(record: dict[str, Any]) -> bool:
    return bool((record.get("preexisting_memories") or {}).get("memories"))


def is_placeholder(record: dict[str, Any]) -> bool:
    text = record["document"].get("text")
    return isinstance(text, str) and text.endswith("_BINARY")


def document_format(record: dict[str, Any]) -> str:
    return (record["document"].get("annotations") or {}).get("document_format", "unknown")


def sort_records(records: list[dict[str, Any]], dataset_file: str) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: sample_id_for(dataset_file, record))


def supported_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if is_supported_provider_doc_format(document_format(record))
    ]


def group_records_by_domain(
    records: list[dict[str, Any]],
    *,
    dataset_file: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in sort_records(records, dataset_file):
        grouped.setdefault(record["document"]["domain"], []).append(record)
    return grouped


def select_records(
    records: list[dict[str, Any]],
    *,
    dataset_file: str,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped = group_records_by_domain(records, dataset_file=dataset_file)
    domains = sorted(grouped)
    ordered = sort_records(records, dataset_file)

    if count > len(ordered):
        raise ValueError(
            f"Requested {count} samples from {dataset_file}, but only found {len(ordered)}."
        )
    if count < len(domains):
        raise ValueError(
            f"Requested {count} samples from {dataset_file}, but need at least "
            f"{len(domains)} to cover every domain once."
        )

    selected_by_id: dict[str, dict[str, Any]] = {}
    for domain in domains:
        chosen = rng.choice(grouped[domain])
        selected_by_id[sample_id_for(dataset_file, chosen)] = chosen

    remaining = [
        record
        for record in ordered
        if sample_id_for(dataset_file, record) not in selected_by_id
    ]
    extra_count = count - len(selected_by_id)
    if extra_count > 0:
        for record in rng.sample(remaining, extra_count):
            selected_by_id[sample_id_for(dataset_file, record)] = record

    selected = list(selected_by_id.values())
    rng.shuffle(selected)
    return selected


def interleave_records(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for index in range(max(len(primary), len(secondary))):
        if index < len(primary):
            merged.append(primary[index])
        if index < len(secondary):
            merged.append(secondary[index])
    return merged


def metadata_richness(record: dict[str, Any]) -> int:
    document = record["document"]
    metadata = document.get("metadata") or {}
    richness = sum(1 for value in metadata.values() if value not in (None, "", [], {}))
    richness += int(bool(document.get("source_name")))
    richness += int(bool(document.get("source_path")))
    return richness


def select_preferred_format_record(
    records: list[dict[str, Any]],
    *,
    dataset_file: str,
) -> dict[str, Any]:
    def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
        return (
            -int(is_placeholder(record)),
            -metadata_richness(record),
            sample_id_for(dataset_file, record),
        )

    return min(records, key=sort_key)


def build_format_smoke_entries(
    with_records: list[dict[str, Any]],
    without_records: list[dict[str, Any]],
    *,
    with_dataset_file: str,
    without_dataset_file: str,
) -> list[dict[str, Any]]:
    with_by_format: dict[str, list[dict[str, Any]]] = {}
    without_by_format: dict[str, list[dict[str, Any]]] = {}
    for record in sort_records(with_records, with_dataset_file):
        with_by_format.setdefault(document_format(record), []).append(record)
    for record in sort_records(without_records, without_dataset_file):
        without_by_format.setdefault(document_format(record), []).append(record)

    entries: list[dict[str, Any]] = []
    for fmt in FORMAT_ORDER:
        if fmt in with_by_format:
            entries.append(
                {
                    "dataset_file": with_dataset_file,
                    "record": select_preferred_format_record(
                        with_by_format[fmt],
                        dataset_file=with_dataset_file,
                    ),
                }
            )
        if fmt in without_by_format:
            entries.append(
                {
                    "dataset_file": without_dataset_file,
                    "record": select_preferred_format_record(
                        without_by_format[fmt],
                        dataset_file=without_dataset_file,
                    ),
                }
            )
    return entries


def describe_record(
    record: dict[str, Any],
    *,
    dataset_file: str,
    smoke_ids: set[str],
    format_smoke_ids: set[str],
) -> dict[str, Any]:
    document = record["document"]
    goal = record["goal"]
    sample_id = sample_id_for(dataset_file, record)
    return {
        "sample_id": sample_id,
        "dataset_file": dataset_file,
        "memory_condition": "with_memories" if has_memories(record) else "without_memories",
        "doc_domain": document["domain"],
        "doc_id": document["doc_id"],
        "goal_id": goal["goal_id"],
        "goal_text": goal["goal_text"],
        "document_format": (document.get("annotations") or {}).get("document_format", "unknown"),
        "placeholder_backed": is_placeholder(record),
        "included_in_smoke_subset": sample_id in smoke_ids,
        "included_in_format_smoke_subset": sample_id in format_smoke_ids,
    }


def relative_repo_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> int:
    args = build_parser().parse_args()

    if args.with_count <= 0 or args.without_count <= 0:
        raise ValueError("with-count and without-count must both be positive.")

    rng = random.Random(args.seed)
    with_records = supported_records(load_records(args.with_memories_dataset))
    without_records = supported_records(load_records(args.without_memories_dataset))

    selected_with = select_records(
        with_records,
        dataset_file=args.with_memories_dataset,
        count=args.with_count,
        rng=rng,
    )
    selected_without = select_records(
        without_records,
        dataset_file=args.without_memories_dataset,
        count=args.without_count,
        rng=rng,
    )

    selected_with_entries = [
        {"dataset_file": args.with_memories_dataset, "record": record} for record in selected_with
    ]
    selected_without_entries = [
        {"dataset_file": args.without_memories_dataset, "record": record}
        for record in selected_without
    ]

    full_entries = interleave_records(selected_with_entries, selected_without_entries)
    format_smoke_entries = build_format_smoke_entries(
        with_records,
        without_records,
        with_dataset_file=args.with_memories_dataset,
        without_dataset_file=args.without_memories_dataset,
    )
    full_records = [entry["record"] for entry in full_entries]

    full_dataset_path = (
        EXPERIMENT_DATA_DIR / f"merged_eval_in_claude_prompt_validation_seed_{args.seed}.json"
    )
    manifest_path = EXPERIMENT_DATA_DIR / f"selection_seed_{args.seed}.json"

    write_json(full_dataset_path, full_records)
    format_smoke_dataset_path = resolve_repo_path(SMOKE_DATASET_PATH)
    if not format_smoke_dataset_path.exists():
        raise FileNotFoundError(
            f"Committed smoke dataset missing: {format_smoke_dataset_path}."
        )
    format_smoke_ids = {
        sample_id_for(entry["dataset_file"], entry["record"]) for entry in format_smoke_entries
    }

    sample_catalog: dict[str, dict[str, Any]] = {}
    for entry in selected_with_entries + selected_without_entries + format_smoke_entries:
        sample_catalog[sample_id_for(entry["dataset_file"], entry["record"])] = entry
    samples = [
        describe_record(
            entry["record"],
            dataset_file=entry["dataset_file"],
            smoke_ids=set(),
            format_smoke_ids=format_smoke_ids,
        )
        for entry in sample_catalog.values()
    ]

    full_dataset_id_source = relative_repo_path(full_dataset_path)
    full_sample_ids = [
        sample_id_for(full_dataset_id_source, entry["record"]) for entry in full_entries
    ]
    format_smoke_sample_ids = [
        sample_id_for(SMOKE_DATASET_PATH, entry["record"])
        for entry in format_smoke_entries
    ]

    manifest = {
        "experiment": "claude_prompt_validation",
        "seed": args.seed,
        "source_datasets": {
            "with_memories": args.with_memories_dataset,
            "without_memories": args.without_memories_dataset,
        },
        "counts": {
            "with_memories": args.with_count,
            "without_memories": args.without_count,
            "full": len(full_sample_ids),
            "format_smoke": len(format_smoke_sample_ids),
        },
        "generated_datasets": {
            "full": relative_repo_path(full_dataset_path),
            "format_smoke": SMOKE_DATASET_PATH,
        },
        "full_sample_ids": full_sample_ids,
        "format_smoke_sample_ids": format_smoke_sample_ids,
        "samples": sorted(samples, key=lambda sample: sample["sample_id"]),
    }
    write_json(manifest_path, manifest)

    print(f"Seed: {args.seed}")
    print(f"Manifest: {manifest_path}")
    print(f"Full dataset: {full_dataset_path}")
    print(f"Format smoke dataset: {format_smoke_dataset_path}")
    print(f"Selected samples: {len(full_sample_ids)}")
    print(
        "Memory split: "
        f"{args.with_count} with memories / {args.without_count} without memories"
    )
    print(f"Format smoke subset size: {len(format_smoke_sample_ids)}")
    print()
    print("Format smoke sample IDs:")
    for sample_id in format_smoke_sample_ids:
        print(f"- {sample_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
