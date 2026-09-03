"""Execution entrypoints for config-driven eval campaigns."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Any

from inspect_ai import eval_retry, eval_set, score as inspect_score
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log
import yaml

from sleeper_eval.task import sleeper_eval

from .config import CampaignConfig, load_campaign_config
from .mem0_qdrant import mem0_qdrant_runtime
from .mem0_replay import replay_mem0_manager_campaign, render_replay_plan
from .plan import render_plan
from .preflight import PreflightResult, preflight_campaign
from .scoring import resolve_scorers, scorer_names_for_log

try:
    from scripts.analyze import load_analysis_frame, with_display_columns, write_analysis_outputs
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from analyze import load_analysis_frame, with_display_columns, write_analysis_outputs


def configure_inspect_env(output_dir: Path) -> None:
    """Ensure Inspect writes logs/traces into the campaign output tree.

    Relying on ambient INSPECT_* env vars has repeatedly caused runs to fail when a
    previous shell session left behind a trace file path whose parent directory no
    longer exists. We normalize that here for every campaign run.
    """

    inspect_home = (output_dir / ".inspect_home").resolve()
    log_dir = (output_dir / "logs").resolve()
    trace_file = log_dir / "inspect-trace.log"

    inspect_home.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(inspect_home)
    os.environ["INSPECT_LOG_DIR"] = str(log_dir)
    os.environ["INSPECT_TRACE_FILE"] = str(trace_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Path to the YAML campaign config.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional override for output_dir in the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved execution plan and exit before running eval_set.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--retry-log",
        action="append",
        default=[],
        help="Retry an existing .eval log in-place using Inspect eval-retry. May be repeated.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Auto-discover retry-worthy logs for this config and run Inspect eval-retry on them.",
    )
    parser.add_argument(
        "--retry-max-connections",
        type=int,
        default=None,
        help="Optional max_connections override to use for eval-retry.",
    )
    parser.add_argument(
        "--retry-attempt-timeout",
        type=int,
        default=None,
        help="Optional attempt_timeout override to use for eval-retry.",
    )
    parser.add_argument(
        "--retry-max-retries",
        type=int,
        default=None,
        help="Optional max_retries override to use for eval-retry.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Rebuild analysis artifacts for this campaign from existing logs only.",
    )
    parser.add_argument(
        "--replay-mem0-manager-from",
        action="append",
        default=[],
        help=(
            "Replay mem0 manager extraction from existing .eval logs, logs/ dirs, or campaign "
            "run dirs instead of rerunning subject models. May be repeated."
        ),
    )
    return parser


def build_campaign_tasks(preflight: PreflightResult) -> list[Any]:
    tasks: list[Any] = []
    for task_plan in preflight.expanded_tasks:
        task = sleeper_eval(
            dataset_file=task_plan.dataset_file,
            defense=task_plan.defense,
            defense_suffix_override=task_plan.defense_suffix_override,
            attack=task_plan.attack,
            evaluation_mode=task_plan.evaluation_mode,
            memory_backend=task_plan.memory_backend,
            mem0_runtime=task_plan.mem0_runtime,
            mem0_provider=task_plan.mem0_provider,
            mem0_model=task_plan.mem0_model,
            mem0_prompt_variant=task_plan.mem0_prompt_variant,
            mem0_include_document_content=task_plan.mem0_include_document_content,
            mem0_thinking=task_plan.mem0_thinking,
            mem0_reasoning_effort=task_plan.mem0_reasoning_effort,
            mem0_qdrant_mode=task_plan.mem0_qdrant_mode,
            mem0_qdrant_url=task_plan.mem0_qdrant_url,
            mem0_qdrant_api_key_env=task_plan.mem0_qdrant_api_key_env,
            mem0_qdrant_collection_name=task_plan.mem0_qdrant_collection_name,
            mention_memory_system=task_plan.mention_memory_system,
            subcategory=task_plan.subcategory,
            category=task_plan.category,
            doc_domain=task_plan.doc_domain,
            domain_seed=task_plan.domain_seed,
            provider=task_plan.provider,
            subject_model=task_plan.subject_model,
            prompt_model=task_plan.prompt_model,
            sample_shuffle=task_plan.sample_shuffle,
            sample_shuffle_seed=task_plan.sample_shuffle_seed,
            reasoning_effort=task_plan.reasoning_effort,
            reasoning_tokens=task_plan.reasoning_tokens,
            batch=task_plan.batch,
            extra_headers=task_plan.extra_headers,
            extra_body=task_plan.extra_body,
        )
        task.metadata = {
            **(task.metadata or {}),
            "campaign_name": preflight.resolved_config["name"],
            "campaign_dataset_label": task_plan.dataset_label,
            "campaign_attack_label": task_plan.attack_label,
            "campaign_model_label": task_plan.model_label,
            "campaign_defense_label": task_plan.defense_label,
        }
        task.tags = sorted(
            {
                *(task.tags or []),
                "campaign",
                preflight.resolved_config["name"],
                task_plan.dataset_label,
                task_plan.attack_label,
                task_plan.model_label,
                task_plan.defense_label,
                task_plan.evaluation_mode,
            }
        )
        tasks.append(task)
    return tasks


def task_label_key_for_log(log: Any) -> tuple[str, str, str, str]:
    metadata = (
        getattr(log, "metadata", None)
        or getattr(getattr(log, "eval", None), "metadata", None)
        or {}
    )
    return (
        str(metadata.get("campaign_dataset_label", "")),
        str(metadata.get("campaign_attack_label", "")),
        str(metadata.get("campaign_model_label", "")),
        str(metadata.get("campaign_defense_label", "")),
    )


def normalize_log_location(location: str) -> str:
    if location.startswith("file://"):
        parsed = urlparse(location)
        return unquote(parsed.path)
    return location


def expected_task_label_keys(preflight: PreflightResult) -> set[tuple[str, str, str, str]]:
    return {
        (
            task.dataset_label,
            task.attack_label,
            task.model_label,
            task.defense_label,
        )
        for task in preflight.expanded_tasks
    }


def log_retry_details(log: Any) -> dict[str, Any]:
    samples = getattr(log, "samples", None) or []
    sample_error_count = 0
    invalidated_sample_count = 0
    for sample in samples:
        if getattr(sample, "error", None) is not None:
            sample_error_count += 1
        if getattr(sample, "invalidation", None) is not None:
            invalidated_sample_count += 1

    total_samples = getattr(getattr(log, "results", None), "total_samples", None)
    completed_samples = getattr(getattr(log, "results", None), "completed_samples", None)
    status = str(getattr(log, "status", ""))
    invalidated = bool(getattr(log, "invalidated", False))
    needs_retry = (
        status != "success"
        or invalidated
        or sample_error_count > 0
        or invalidated_sample_count > 0
        or (
            total_samples is not None
            and completed_samples is not None
            and completed_samples < total_samples
        )
    )

    return {
        "status": status,
        "invalidated": invalidated,
        "total_samples": total_samples,
        "completed_samples": completed_samples,
        "sample_error_count": sample_error_count,
        "invalidated_sample_count": invalidated_sample_count,
        "needs_retry": needs_retry,
    }


def latest_matching_logs_by_task_id(preflight: PreflightResult) -> dict[str, Any]:
    log_dir = Path(preflight.output_dir) / "logs"
    if not log_dir.exists():
        return {}

    expected_keys = expected_task_label_keys(preflight)
    latest_by_task_id: dict[str, tuple[str, Any]] = {}
    for log_info in list_eval_logs(str(log_dir)):
        log = read_eval_log(log_info.name)
        if task_label_key_for_log(log) not in expected_keys:
            continue
        task_id = str(log.eval.task_id or log.eval.eval_id)
        timestamp = (
            str(getattr(getattr(log, "stats", None), "completed_at", "") or "")
            or str(getattr(getattr(log, "eval", None), "created", "") or "")
            or Path(log_info.name).name
        )
        incumbent = latest_by_task_id.get(task_id)
        if incumbent is None or timestamp >= incumbent[0]:
            latest_by_task_id[task_id] = (timestamp, log)
    return {task_id: log for task_id, (_timestamp, log) in latest_by_task_id.items()}


def discover_retry_logs(preflight: PreflightResult) -> list[dict[str, Any]]:
    retry_candidates: list[dict[str, Any]] = []
    for log in latest_matching_logs_by_task_id(preflight).values():
        details = log_retry_details(log)
        if not details["needs_retry"]:
            continue
        retry_candidates.append(
            {
                "log_path": normalize_log_location(str(getattr(log, "location", ""))),
                "task_id": str(log.eval.task_id or log.eval.eval_id),
                "dataset_label": task_label_key_for_log(log)[0],
                "attack_label": task_label_key_for_log(log)[1],
                "model_label": task_label_key_for_log(log)[2],
                "defense_label": task_label_key_for_log(log)[3],
                **details,
            }
        )
    retry_candidates.sort(
        key=lambda item: (
            item["dataset_label"],
            item["attack_label"],
            item["model_label"],
            item["defense_label"],
        )
    )
    return retry_candidates


def render_retry_plan(
    *,
    retry_candidates: list[dict[str, Any]],
    retry_logs: list[str],
    max_connections: int | None,
    attempt_timeout: int | None,
    max_retries: int | None,
) -> str:
    lines = [
        "",
        "== Retry Plan ==",
        f"retry_logs={len(retry_logs)}",
        f"retry_max_connections={max_connections}",
        f"retry_attempt_timeout={attempt_timeout}",
        f"retry_max_retries={max_retries}",
    ]
    if retry_candidates:
        for item in retry_candidates:
            lines.append(
                "  - "
                + " ".join(
                    [
                        f"dataset={item['dataset_label']}",
                        f"attack={item['attack_label']}",
                        f"model={item['model_label']}",
                        f"defense={item['defense_label']}",
                        f"status={item['status']}",
                        f"completed={item['completed_samples']}/{item['total_samples']}",
                        f"sample_errors={item['sample_error_count']}",
                        f"invalidated_samples={item['invalidated_sample_count']}",
                        f"log={item['log_path']}",
                    ]
                )
            )
    else:
        lines.append("  - none")
    return "\n".join(lines)


def apply_post_eval_scoring(
    *,
    log_dir: Path,
    scorer_names: list[str],
    action: str,
) -> list[str]:
    if not scorer_names:
        return []

    scored_logs: list[str] = []
    for log_info in list_eval_logs(str(log_dir)):
        location = Path(log_info.name)
        log = read_eval_log(log_info.name)
        applicable_names = scorer_names_for_log(log, scorer_names)
        if not applicable_names:
            continue
        scored = inspect_score(
            log,
            scorers=resolve_scorers(applicable_names),
            action=action,
            copy=False,
        )
        write_eval_log(scored, location=location)
        scored_logs.append(str(location))
    return scored_logs


def apply_post_eval_scoring_to_logs(
    *,
    log_paths: list[Path],
    scorer_names: list[str],
    action: str,
) -> list[str]:
    if not scorer_names:
        return []

    scored_logs: list[str] = []
    for location in log_paths:
        log = read_eval_log(str(location))
        applicable_names = scorer_names_for_log(log, scorer_names)
        if not applicable_names:
            continue
        scored = inspect_score(
            log,
            scorers=resolve_scorers(applicable_names),
            action=action,
            copy=False,
        )
        write_eval_log(scored, location=location)
        scored_logs.append(str(location))
    return scored_logs


def log_lineage_by_eval_id(log_dir: Path) -> dict[str, dict[str, str]]:
    lineage: dict[str, dict[str, str]] = {}
    for log_info in list_eval_logs(str(log_dir)):
        # Header-only: lineage keys live in eval metadata; avoid loading all samples
        # per log (would roughly double peak RAM vs samples_df on large log dirs).
        log = read_eval_log(log_info.name, header_only=True)
        timestamp = (
            str(log.stats.completed_at or "")
            or str(log.eval.created or "")
            or Path(log_info.name).name
        )
        eval_id = str(log.eval.eval_id)
        dataset_label, attack_label, model_label, defense_label = task_label_key_for_log(log)
        lineage[eval_id] = {
            "task_id": str(log.eval.task_id or log.eval.eval_id),
            "timestamp": timestamp,
            "dataset_label": dataset_label,
            "attack_label": attack_label,
            "model_label": model_label,
            "defense_label": defense_label,
        }
    return lineage


def load_campaign_analysis_frame(log_dir: Path) -> Any:
    analysis_df = with_display_columns(load_analysis_frame(str(log_dir)))
    if analysis_df.empty or "eval_id" not in analysis_df.columns:
        return analysis_df

    lineage = log_lineage_by_eval_id(log_dir)
    if not lineage:
        return analysis_df

    display = analysis_df.copy()
    display["_eval_id_str"] = display["eval_id"].astype(str)
    display["_task_id"] = display["_eval_id_str"].map(
        lambda eval_id: lineage.get(eval_id, {}).get("task_id", "")
    )
    display["_timestamp"] = display["_eval_id_str"].map(
        lambda eval_id: lineage.get(eval_id, {}).get("timestamp", "")
    )

    if {
        "dataset_label",
        "task_arg_attack",
        "model_label",
        "defense_label",
        "id",
    }.issubset(display.columns):
        display = display.sort_values("_timestamp")
        display = display.drop_duplicates(
            subset=[
                "dataset_label",
                "task_arg_attack",
                "model_label",
                "defense_label",
                "id",
            ],
            keep="last",
        )

    return display.drop(columns=["_eval_id_str", "_task_id", "_timestamp"], errors="ignore")


def write_campaign_analysis(output_dir: Path) -> dict[str, Path]:
    analysis_df = load_campaign_analysis_frame(output_dir / "logs")
    return write_analysis_outputs(analysis_df, output_dir / "analysis")


def run_campaign(
    config: CampaignConfig,
    *,
    config_path: Path,
    preflight: PreflightResult | None = None,
) -> tuple[bool, list[Any], dict[str, Path]]:
    preflight_result = preflight or preflight_campaign(config, config_path=config_path)

    output_dir = Path(preflight_result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    configure_inspect_env(output_dir)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(preflight_result.resolved_config, sort_keys=False),
        encoding="utf-8",
    )

    with mem0_qdrant_runtime(config, output_dir=output_dir):
        tasks = build_campaign_tasks(preflight_result)
        bundle_dir = output_dir / "bundle" if config.eval.bundle else None
        success, logs = eval_set(
            tasks=tasks,
            log_dir=str(output_dir / "logs"),
            model_roles={"grader": config.grader_model},
            retry_attempts=config.retry.retry_attempts,
            retry_immediate=config.retry.retry_immediate,
            retry_wait=config.retry.retry_wait,
            retry_connections=config.retry.retry_connections,
            retry_on_error=config.retry.retry_on_error,
            bundle_dir=str(bundle_dir) if bundle_dir else None,
            bundle_overwrite=config.eval.bundle_overwrite,
            limit=config.eval.limit,
            max_tasks=config.eval.max_tasks,
            max_samples=config.eval.max_samples,
            max_connections=config.eval.max_connections,
            fail_on_error=config.eval.fail_on_error,
            max_retries=config.eval.max_retries,
            attempt_timeout=config.eval.attempt_timeout,
            score=(config.scoring.during_eval == "full"),
        )

        scored_logs = apply_post_eval_scoring(
            log_dir=output_dir / "logs",
            scorer_names=config.scoring.post_eval_scorers,
            action=config.scoring.post_eval_action,
        )

    written = write_campaign_analysis(output_dir)

    manifest = {
        "success": success,
        "task_count": len(tasks),
        "log_count": len(logs),
        "scored_log_count": len(scored_logs),
        "log_dir": str(output_dir / "logs"),
        "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "analysis_dir": str(output_dir / "analysis"),
        "config_hash": preflight_result.config_hash,
        "output_mode": preflight_result.output_mode,
        "datasets": preflight_result.resolved_config["datasets"],
        "scoring": preflight_result.resolved_config["scoring"],
        "analysis_outputs": {name: str(path) for name, path in written.items()},
    }
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return success, logs, written


def retry_campaign_logs(
    config: CampaignConfig,
    *,
    config_path: Path,
    retry_logs: list[str],
    preflight: PreflightResult | None = None,
    max_connections: int | None = None,
    attempt_timeout: int | None = None,
    max_retries: int | None = None,
) -> tuple[bool, list[Any], dict[str, Path]]:
    preflight_result = preflight or preflight_campaign(config, config_path=config_path)

    output_dir = Path(preflight_result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    configure_inspect_env(output_dir)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(preflight_result.resolved_config, sort_keys=False),
        encoding="utf-8",
    )

    if not retry_logs:
        written = write_campaign_analysis(output_dir)
        manifest = {
            "success": True,
            "retry_mode": True,
            "retried_source_logs": [],
            "retried_log_count": 0,
            "scored_log_count": 0,
            "log_dir": str(output_dir / "logs"),
            "analysis_dir": str(output_dir / "analysis"),
            "config_hash": preflight_result.config_hash,
            "output_mode": preflight_result.output_mode,
            "datasets": preflight_result.resolved_config["datasets"],
            "scoring": preflight_result.resolved_config["scoring"],
            "analysis_outputs": {name: str(path) for name, path in written.items()},
        }
        (output_dir / "run_manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        return True, [], written

    with mem0_qdrant_runtime(config, output_dir=output_dir):
        retried_logs = eval_retry(
            retry_logs,
            log_dir=str(output_dir / "logs"),
            max_connections=max_connections,
            attempt_timeout=attempt_timeout,
            max_retries=max_retries,
            score=(config.scoring.during_eval == "full"),
        )
        retry_log_paths = [
            Path(normalize_log_location(str(log.location)))
            for log in retried_logs
            if getattr(log, "location", None)
        ]

        scored_logs = apply_post_eval_scoring_to_logs(
            log_paths=retry_log_paths,
            scorer_names=config.scoring.post_eval_scorers,
            action=config.scoring.post_eval_action,
        )

    written = write_campaign_analysis(output_dir)

    manifest = {
        "success": all(getattr(log, "status", None) == "success" for log in retried_logs),
        "retry_mode": True,
        "retried_source_logs": retry_logs,
        "retried_log_count": len(retried_logs),
        "scored_log_count": len(scored_logs),
        "log_dir": str(output_dir / "logs"),
        "analysis_dir": str(output_dir / "analysis"),
        "config_hash": preflight_result.config_hash,
        "output_mode": preflight_result.output_mode,
        "datasets": preflight_result.resolved_config["datasets"],
        "scoring": preflight_result.resolved_config["scoring"],
        "analysis_outputs": {name: str(path) for name, path in written.items()},
    }
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest["success"], retried_logs, written


def analyze_campaign(
    config: CampaignConfig,
    *,
    config_path: Path,
    preflight: PreflightResult | None = None,
) -> dict[str, Path]:
    preflight_result = preflight or preflight_campaign(config, config_path=config_path)

    output_dir = Path(preflight_result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    configure_inspect_env(output_dir)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(preflight_result.resolved_config, sort_keys=False),
        encoding="utf-8",
    )

    written = write_campaign_analysis(output_dir)
    manifest = {
        "success": True,
        "analyze_only": True,
        "log_dir": str(output_dir / "logs"),
        "analysis_dir": str(output_dir / "analysis"),
        "config_hash": preflight_result.config_hash,
        "output_mode": preflight_result.output_mode,
        "datasets": preflight_result.resolved_config["datasets"],
        "scoring": preflight_result.resolved_config["scoring"],
        "analysis_outputs": {name: str(path) for name, path in written.items()},
    }
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config_file).resolve()
    config = load_campaign_config(config_path)
    if args.output_dir:
        config = config.model_copy(update={"output_dir": args.output_dir})
    if args.analyze_only and (args.retry or args.retry_log or args.replay_mem0_manager_from):
        raise SystemExit(
            "--analyze-only cannot be combined with --retry, --retry-log, or --replay-mem0-manager-from."
        )
    if args.replay_mem0_manager_from and (args.retry or args.retry_log):
        raise SystemExit(
            "--replay-mem0-manager-from cannot be combined with --retry or --retry-log."
        )

    preflight = preflight_campaign(config, config_path=config_path)
    print(render_plan(preflight))
    if args.replay_mem0_manager_from:
        print(
            render_replay_plan(
                sources=args.replay_mem0_manager_from,
                config=config,
                output_dir=Path(config.output_dir).resolve(),
            )
        )

    auto_retry_candidates: list[dict[str, Any]] = []
    retry_logs: list[str] = []
    if args.retry:
        auto_retry_candidates = discover_retry_logs(preflight)
        retry_logs.extend(item["log_path"] for item in auto_retry_candidates if item["log_path"])
    if args.retry_log:
        retry_logs.extend(str(Path(log).resolve()) for log in args.retry_log)
    retry_logs = list(dict.fromkeys(retry_logs))
    if args.retry or args.retry_log:
        explicit_retry_items = [
            {
                "log_path": str(Path(log).resolve()),
                "dataset_label": "(explicit)",
                "attack_label": "(explicit)",
                "model_label": "(explicit)",
                "defense_label": "(explicit)",
                "status": "manual",
                "completed_samples": "?",
                "total_samples": "?",
                "sample_errors": "?",
                "sample_error_count": "?",
                "invalidated_sample_count": "?",
            }
            for log in args.retry_log
            if str(Path(log).resolve()) not in {item["log_path"] for item in auto_retry_candidates}
        ]
        print(
            render_retry_plan(
                retry_candidates=[*auto_retry_candidates, *explicit_retry_items],
                retry_logs=retry_logs,
                max_connections=args.retry_max_connections,
                attempt_timeout=args.retry_attempt_timeout,
                max_retries=args.retry_max_retries,
            )
        )

    if args.dry_run:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Refusing to start a real campaign without --yes in non-interactive mode. "
                "Use --dry-run to inspect the plan first."
            )
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if args.analyze_only:
        written = analyze_campaign(config, config_path=config_path, preflight=preflight)
        success = True
    elif args.replay_mem0_manager_from:
        with mem0_qdrant_runtime(config, output_dir=Path(config.output_dir).resolve()):
            summary = replay_mem0_manager_campaign(
                config=config,
                config_path=config_path,
                source_logs=args.replay_mem0_manager_from,
                output_dir=Path(config.output_dir).resolve(),
            )
        written = summary.analysis_outputs
        success = True
    elif args.retry or args.retry_log:
        success, _logs, written = retry_campaign_logs(
            config,
            config_path=config_path,
            retry_logs=retry_logs,
            preflight=preflight,
            max_connections=args.retry_max_connections,
            attempt_timeout=args.retry_attempt_timeout,
            max_retries=args.retry_max_retries,
        )
    else:
        success, _logs, written = run_campaign(config, config_path=config_path, preflight=preflight)
    print(
        "\n".join(
            [
                "",
                "== Eval Campaign ==",
                f"name={config.name}",
                f"output_dir={config.output_dir}",
                f"success={success}",
                *(f"{label}={path}" for label, path in written.items()),
            ]
        )
    )
    return 0 if success else 1
