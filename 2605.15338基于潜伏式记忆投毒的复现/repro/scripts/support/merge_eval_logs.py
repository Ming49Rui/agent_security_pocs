#!/usr/bin/env python3
"""Merge multiple eval log directories into one combined run directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

from inspect_ai.log import read_eval_log
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.analyze import load_analysis_frame, with_display_columns, write_analysis_outputs
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from analyze import load_analysis_frame, with_display_columns, write_analysis_outputs

try:
    from sleeper_eval.eval_campaign.run import load_campaign_analysis_frame
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    load_campaign_analysis_frame = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        help="Source campaign run dirs or logs/ dirs to merge.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Target combined run directory. Logs are written to <output-dir>/logs.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to materialize merged logs in the target directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing merged log if the same filename already exists.",
    )
    parser.add_argument(
        "--skip-duplicate-eval-id",
        action="store_true",
        help="Skip logs whose eval_id is already present in the merged output.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run scripts/analyze.py functionality on the merged logs after merging.",
    )
    return parser


def resolve_source_log_dir(source: str | Path) -> Path:
    path = Path(source).resolve()
    if path.is_dir() and path.name == "logs":
        return path
    logs_dir = path / "logs"
    if logs_dir.is_dir():
        return logs_dir
    raise ValueError(f"Could not resolve a logs directory from source: {source}")


def eval_files_in(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*.eval"))


def eval_id_for_log(path: Path) -> str:
    log = read_eval_log(str(path), header_only=True)
    return str(log.eval.eval_id)


def log_metadata(log: Any) -> dict[str, Any]:
    return (
        getattr(log, "metadata", None)
        or getattr(getattr(log, "eval", None), "metadata", None)
        or {}
    )


def task_label_key_for_log(log: Any) -> tuple[str, str, str, str]:
    metadata = log_metadata(log)
    return (
        str(metadata.get("campaign_dataset_label", "")),
        str(metadata.get("campaign_attack_label", "")),
        str(metadata.get("campaign_model_label", "")),
        str(metadata.get("campaign_defense_label", "")),
    )


def normalize_source_root(source: str | Path) -> Path | None:
    path = Path(source).resolve()
    if path.is_dir() and path.name == "logs":
        candidate = path.parent
    else:
        candidate = path
    if not candidate.is_dir():
        return None
    if (candidate / "logs").is_dir():
        return candidate
    return None


def source_config_path(source: str | Path) -> Path | None:
    root = normalize_source_root(source)
    if root is None:
        return None
    config_path = root / "config.yaml"
    return config_path if config_path.exists() else None


def source_manifest_path(source: str | Path) -> Path | None:
    root = normalize_source_root(source)
    if root is None:
        return None
    manifest_path = root / "run_manifest.yaml"
    return manifest_path if manifest_path.exists() else None


def summarize_source_config(source: str | Path) -> dict[str, Any] | None:
    config_path = source_config_path(source)
    if config_path is None:
        return None

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    manifest_path = source_manifest_path(source)
    manifest = (
        yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if manifest_path is not None
        else {}
    )

    def normalize_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str) and value == "none":
            return ""
        return value

    def index_by_label(items: list[dict[str, Any]], *, keys: list[str]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for item in items or []:
            label = str(item.get("label", ""))
            if not label:
                continue
            indexed[label] = {key: normalize_value(item.get(key)) for key in keys}
        return indexed

    return {
        "source": str(Path(source).resolve()),
        "config_path": str(config_path),
        "campaign_name": str(config.get("name", "")),
        "config_hash": str(manifest.get("config_hash", "")),
        "memory_backend": config.get("memory_backend"),
        "mention_memory_system": config.get("mention_memory_system"),
        "grader_model": config.get("grader_model"),
        "post_eval_scorers": tuple((config.get("scoring") or {}).get("post_eval_scorers", []) or []),
        "datasets": index_by_label(
            config.get("datasets") or [],
            keys=[
                "resolved_path",
                "evaluation_mode",
                "attack",
                "subcategory",
                "category",
                "doc_domain",
                "domain_seed",
            ],
        ),
        "attacks": index_by_label(config.get("attacks") or [], keys=["attack"]),
        "models": index_by_label(
            config.get("models") or [],
            keys=["model", "prompt_model", "provider"],
        ),
        "defenses": index_by_label(
            config.get("defenses") or [],
            keys=["defense", "suffix_sha256"],
        ),
    }


def compatibility_report(
    *,
    sources: list[str],
    source_log_dirs: list[Path],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress is not None:
        progress(f"Checking compatibility across {len(sources)} source(s)")
    summaries = [summary for source in sources if (summary := summarize_source_config(source)) is not None]
    warnings: list[str] = []
    errors: list[str] = []

    if len(summaries) != len(sources):
        warnings.append("Some sources do not have config.yaml; compatibility checks are partial.")

    def compare_scalar(field: str, *, strict: bool = True) -> None:
        values = {summary.get(field) for summary in summaries if field in summary}
        values.discard("")
        if len(values) <= 1:
            return
        message = f"Source configs disagree on {field}: {sorted(values)}"
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    compare_scalar("campaign_name", strict=False)
    compare_scalar("memory_backend")
    compare_scalar("mention_memory_system")
    compare_scalar("post_eval_scorers")

    scorer_sets = {
        tuple(summary.get("post_eval_scorers", []) or [])
        for summary in summaries
        if summary.get("post_eval_scorers") is not None
    }
    semantic_scorers_present = any(
        any("goal_match" in scorer for scorer in scorer_set)
        for scorer_set in scorer_sets
    )
    compare_scalar("grader_model", strict=semantic_scorers_present)
    if not semantic_scorers_present:
        compare_scalar("grader_model", strict=False)

    def compare_indexed(section: str) -> None:
        canonical: dict[str, dict[str, Any]] = {}
        for summary in summaries:
            for label, payload in (summary.get(section) or {}).items():
                incumbent = canonical.get(label)
                if incumbent is None:
                    canonical[label] = payload
                    continue
                if incumbent != payload:
                    errors.append(
                        f"Source configs disagree on {section}.{label}: "
                        f"{incumbent} != {payload}"
                    )

    for section in ("datasets", "attacks", "models", "defenses"):
        compare_indexed(section)

    seen_task_keys: dict[tuple[str, str, str, str], dict[str, str]] = {}
    campaign_metadata_missing = 0
    for log_dir in source_log_dirs:
        if progress is not None:
            progress(f"Scanning campaign task identities in {log_dir}")
        for log_path in eval_files_in(log_dir):
            log = read_eval_log(str(log_path), header_only=True)
            task_key = task_label_key_for_log(log)
            if not any(task_key):
                campaign_metadata_missing += 1
                continue
            incumbent = seen_task_keys.get(task_key)
            if incumbent is not None:
                if incumbent["source"] != str(log_dir):
                    errors.append(
                        "Overlapping task cell detected across sources: "
                        f"{task_key} appears in both {incumbent['log_path']} and {log_path}"
                    )
                    continue
                continue
            seen_task_keys[task_key] = {
                "source": str(log_dir),
                "log_path": str(log_path),
            }

    if campaign_metadata_missing:
        warnings.append(
            f"{campaign_metadata_missing} logs are missing campaign metadata; "
            "task-overlap checks were skipped for those logs."
        )

    return {
        "source_config_summaries": summaries,
        "warnings": warnings,
        "errors": errors,
    }


def should_use_campaign_analysis(log_dir: Path) -> bool:
    eval_files = eval_files_in(log_dir)
    if not eval_files or load_campaign_analysis_frame is None:
        return False
    for path in eval_files:
        log = read_eval_log(str(path), header_only=True)
        if not any(task_label_key_for_log(log)):
            return False
    return True


def materialize_log(source: Path, destination: Path, *, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir():
            raise IsADirectoryError(destination)
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def merge_logs(
    *,
    sources: list[str],
    output_dir: str | Path,
    mode: str = "copy",
    overwrite: bool = False,
    skip_duplicate_eval_id: bool = False,
    analyze: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if mode not in {"copy", "symlink"}:
        raise ValueError(f"Unsupported merge mode: {mode}")

    target_root = Path(output_dir).resolve()
    target_logs = target_root / "logs"
    target_analysis = target_root / "analysis"
    if progress is not None:
        progress(f"Preparing merged output at {target_root}")
    target_logs.mkdir(parents=True, exist_ok=True)

    existing_eval_ids: dict[str, str] = {}
    if progress is not None and target_logs.exists():
        progress(f"Scanning existing target logs in {target_logs}")
    for existing in eval_files_in(target_logs):
        try:
            existing_eval_ids[eval_id_for_log(existing)] = existing.name
        except Exception:
            continue

    source_log_dirs = [resolve_source_log_dir(source) for source in sources]
    compatibility = compatibility_report(
        sources=sources,
        source_log_dirs=source_log_dirs,
        progress=progress,
    )
    if progress is not None:
        for warning in compatibility["warnings"]:
            progress(f"Warning: {warning}")
    if compatibility["errors"]:
        raise ValueError("\n".join(["Merge compatibility check failed:", *compatibility["errors"]]))

    merged: list[str] = []
    skipped_existing_name: list[str] = []
    skipped_duplicate_eval_id_entries: list[dict[str, str]] = []
    source_log_dir_strings: list[str] = []

    for source_log_dir in source_log_dirs:
        source_log_dir_strings.append(str(source_log_dir))
        source_files = eval_files_in(source_log_dir)
        if progress is not None:
            progress(f"Materializing {len(source_files)} log(s) from {source_log_dir}")
        for source_file in source_files:
            destination = target_logs / source_file.name
            if destination.exists() and not overwrite:
                skipped_existing_name.append(source_file.name)
                continue

            source_eval_id = eval_id_for_log(source_file)
            if skip_duplicate_eval_id and source_eval_id in existing_eval_ids:
                skipped_duplicate_eval_id_entries.append(
                    {
                        "source_file": source_file.name,
                        "eval_id": source_eval_id,
                        "existing_file": existing_eval_ids[source_eval_id],
                    }
                )
                continue

            materialize_log(source_file, destination, mode=mode)
            existing_eval_ids[source_eval_id] = destination.name
            merged.append(destination.name)

    analysis_outputs: dict[str, str] = {}
    if analyze:
        if progress is not None:
            progress(f"Building analysis in {target_analysis}")
        target_analysis.mkdir(parents=True, exist_ok=True)
        if should_use_campaign_analysis(target_logs):
            analysis_df = load_campaign_analysis_frame(target_logs)
            analysis_mode = "campaign"
            if progress is not None:
                progress("Using campaign-aware analysis")
        else:
            analysis_df = with_display_columns(load_analysis_frame(str(target_logs)))
            analysis_mode = "generic"
            if progress is not None:
                progress("Using generic analysis")
        written = write_analysis_outputs(analysis_df, target_analysis)
        analysis_outputs = {label: str(path) for label, path in written.items()}
    else:
        analysis_mode = "none"

    manifest = {
        "sources": source_log_dir_strings,
        "output_dir": str(target_root),
        "logs_dir": str(target_logs),
        "mode": mode,
        "overwrite": overwrite,
        "skip_duplicate_eval_id": skip_duplicate_eval_id,
        "compatibility": compatibility,
        "merged_count": len(merged),
        "merged_logs": merged,
        "skipped_existing_name_count": len(skipped_existing_name),
        "skipped_existing_name": skipped_existing_name,
        "skipped_duplicate_eval_id_count": len(skipped_duplicate_eval_id_entries),
        "skipped_duplicate_eval_id": skipped_duplicate_eval_id_entries,
        "analysis_mode": analysis_mode,
        "analysis_outputs": analysis_outputs,
    }

    manifest_path = target_root / "merge_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    manifest = merge_logs(
        sources=args.sources,
        output_dir=args.output_dir,
        mode=args.mode,
        overwrite=args.overwrite,
        skip_duplicate_eval_id=args.skip_duplicate_eval_id,
        analyze=args.analyze,
        progress=print,
    )
    print(yaml.safe_dump(manifest, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
