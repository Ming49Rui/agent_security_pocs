#!/usr/bin/env python3
"""Verify and summarize Claude prompt-validation runs."""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.analysis import evals_df, samples_df

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "eval_dataset" / "datasets"
DEFAULT_LOGS_ROOT = (
    REPO_ROOT / "logs" / "claude_docrep_actor_critic_markers_aware_validation"
)
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "data" / "selection_seed_20260329.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--full-log-dir", type=Path, default=None)
    parser.add_argument("--truncated-log-dir", type=Path, default=None)
    parser.add_argument("--transcript-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_to_dict(metadata: object) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def resolve_repo_path(maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    alt = REPO_ROOT / maybe_relative
    if alt.exists():
        return alt
    return DATASET_ROOT / maybe_relative


def latest_timestamped_dir(logs_root: Path, prefix: str) -> Path:
    matches = sorted(logs_root.glob(f"{prefix}-*"))
    dirs = [path for path in matches if path.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No log directories matching {logs_root}/{prefix}-*")
    return dirs[-1]


def resolve_log_dir(explicit: Path | None, logs_root: Path, prefix: str) -> Path:
    return explicit.resolve() if explicit else latest_timestamped_dir(logs_root, prefix)


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return record["document"]["doc_id"], record["goal"]["goal_id"]


def index_records(records: list[dict[str, Any]], *, label: str) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for record in records:
        key = record_key(record)
        if key in index:
            raise ValueError(f"Duplicate key {key} in {label}")
        index[key] = label
    return index


def has_memories(record: dict[str, Any]) -> bool:
    return bool((record.get("preexisting_memories") or {}).get("memories"))


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                parts.append(f"[tool_use:{block.get('name')}]")
        return "\n".join(parts)
    return str(content)


def load_sample_from_eval(eval_path: Path, sample_id: str) -> dict[str, Any] | None:
    with zipfile.ZipFile(eval_path) as archive:
        for name in archive.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            payload = json.loads(archive.read(name))
            if payload.get("id") == sample_id:
                return payload
    return None


def unique_nonempty(series: pd.Series) -> list[str]:
    values = [str(value) for value in series.dropna().tolist() if str(value).strip()]
    return sorted(set(values))


def collect_eval_usage(log_dir: Path) -> tuple[dict[str, float], float | None]:
    totals: dict[str, float] = defaultdict(float)
    total_cost = 0.0
    cost_samples = 0

    for eval_path in sorted(log_dir.glob("*.eval")):
        with zipfile.ZipFile(eval_path) as archive:
            header = json.loads(archive.read("header.json"))
            model_usage = (header.get("stats") or {}).get("model_usage") or {}
            for usage in model_usage.values():
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_tokens_cache_write",
                    "input_tokens_cache_read",
                    "reasoning_tokens",
                    "total_cost",
                ):
                    totals[key] += float(usage.get(key, 0) or 0)

            for sample_name in archive.namelist():
                if not sample_name.startswith("samples/") or not sample_name.endswith(".json"):
                    continue
                sample = json.loads(archive.read(sample_name))
                sample_cost = sample.get("total_cost")
                if sample_cost is not None:
                    total_cost += float(sample_cost)
                    cost_samples += 1

    return dict(totals), (total_cost if cost_samples else None)


def load_condition(log_dir: Path) -> tuple[pd.DataFrame, dict[str, float], float | None]:
    sample_frame = samples_df(str(log_dir))
    eval_frame = evals_df(str(log_dir)).copy()
    if sample_frame.empty:
        raise ValueError(f"No sample rows found in {log_dir}")
    if eval_frame.empty:
        raise ValueError(f"No eval rows found in {log_dir}")

    eval_frame["metadata"] = eval_frame.get("metadata")
    parsed_metadata = eval_frame["metadata"].map(metadata_to_dict)
    eval_frame["metadata_system_prompt_file"] = parsed_metadata.map(
        lambda metadata: metadata.get("system_prompt_file", "")
    )
    eval_frame["metadata_system_prompt_path"] = parsed_metadata.map(
        lambda metadata: metadata.get("system_prompt_path", "")
    )
    eval_frame["metadata_system_prompt_template_sha256"] = parsed_metadata.map(
        lambda metadata: metadata.get("system_prompt_template_sha256", "")
    )
    eval_frame["metadata_system_prompt_template_chars"] = parsed_metadata.map(
        lambda metadata: metadata.get("system_prompt_template_chars", "")
    )

    merged = sample_frame.merge(
        eval_frame[
            [
                "eval_id",
                "model",
                "task_arg_prompt_model",
                "metadata_system_prompt_file",
                "metadata_system_prompt_path",
                "metadata_system_prompt_template_sha256",
                "metadata_system_prompt_template_chars",
            ]
        ],
        on="eval_id",
        how="left",
    )
    merged["score_tool_call_numeric"] = pd.to_numeric(
        merged.get("score_tool_call_scorer"),
        errors="coerce",
    )
    usage, total_cost = collect_eval_usage(log_dir)
    return merged, usage, total_cost


def summarize_condition(
    name: str,
    frame: pd.DataFrame,
    usage: dict[str, float],
    total_cost: float | None,
) -> dict[str, Any]:
    scored = frame["score_tool_call_numeric"].dropna()
    summary: dict[str, Any] = {
        "name": name,
        "samples": len(frame),
        "tool_calls": int(scored.sum()) if not scored.empty else None,
        "tool_call_rate": float(scored.mean()) if not scored.empty else None,
        "model": unique_nonempty(frame["model"]),
        "prompt_model": unique_nonempty(frame["task_arg_prompt_model"]),
        "prompt_file": unique_nonempty(frame["metadata_system_prompt_file"]),
        "prompt_path": unique_nonempty(frame["metadata_system_prompt_path"]),
        "prompt_hash": unique_nonempty(frame["metadata_system_prompt_template_sha256"]),
        "prompt_chars": unique_nonempty(frame["metadata_system_prompt_template_chars"]),
        "usage": usage,
        "total_cost": total_cost,
    }

    rows = []
    if "metadata_memory_condition" in frame.columns:
        for memory_condition, group in frame.groupby("metadata_memory_condition", dropna=False):
            group_scored = group["score_tool_call_numeric"].dropna()
            rows.append(
                {
                    "memory_condition": memory_condition,
                    "samples": int(len(group)),
                    "tool_calls": int(group_scored.sum()) if not group_scored.empty else None,
                    "tool_call_rate": (
                        float(group_scored.mean()) if not group_scored.empty else None
                    ),
                }
            )
    summary["by_memory_condition"] = rows
    return summary


def format_rate(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def format_cost(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "not logged"


def format_list(values: list[str]) -> str:
    return str(values) if values else "n/a"


def print_summary(summary: dict[str, Any]) -> None:
    print(summary["name"])
    print("-" * len(summary["name"]))
    print(f"Samples: {summary['samples']}")
    print(f"Model: {format_list(summary['model'])}")
    print(f"Prompt model: {format_list(summary['prompt_model'])}")
    print(f"Prompt file: {format_list(summary['prompt_file'])}")
    print(f"Prompt path: {format_list(summary['prompt_path'])}")
    print(f"Prompt hash: {format_list(summary['prompt_hash'])}")
    print(f"Prompt chars: {format_list(summary['prompt_chars'])}")
    if summary["tool_call_rate"] is None:
        print("Tool-call ASR: not scored yet")
    else:
        print(
            f"Tool-call ASR: {format_rate(summary['tool_call_rate'])} "
            f"({summary['tool_calls']}/{summary['samples']})"
        )
    print(f"Logged total cost: {format_cost(summary['total_cost'])}")
    usage = summary["usage"]
    if usage:
        print(
            "Usage totals: "
            f"I={int(usage.get('input_tokens', 0))}, "
            f"CW={int(usage.get('input_tokens_cache_write', 0))}, "
            f"CR={int(usage.get('input_tokens_cache_read', 0))}, "
            f"O={int(usage.get('output_tokens', 0))}, "
            f"T={int(usage.get('total_tokens', 0))}"
        )
    if summary["by_memory_condition"]:
        print("By memory condition:")
        for row in summary["by_memory_condition"]:
            detail = (
                f"{row['tool_calls']}/{row['samples']}"
                if row["tool_calls"] is not None
                else f"unscored/{row['samples']}"
            )
            print(
                f"- {row['memory_condition']}: "
                f"{format_rate(row['tool_call_rate'])} ({detail})"
            )
    print()


def normalize_for_compare(goal: str, saved: str) -> tuple[str, str]:
    def norm(text: str) -> str:
        text = text.strip().lower()
        text = text.removeprefix("the user ").removeprefix("user ")
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    return norm(goal), norm(saved)


def collect_tool_call_rows(log_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(log_dir.glob("*.eval")):
        with zipfile.ZipFile(eval_path) as archive:
            for name in archive.namelist():
                if not name.startswith("samples/") or not name.endswith(".json"):
                    continue
                payload = json.loads(archive.read(name))
                scorer = (payload.get("scores") or {}).get("tool_call_scorer") or {}
                if not isinstance(scorer, dict) or scorer.get("value") != 1:
                    continue
                metadata = payload.get("metadata") or {}
                store = payload.get("store") or {}
                saved = store.get("SavedMemoryStore:saved_memories") or []
                events = store.get("SavedMemoryStore:tool_events") or []
                adds = [
                    event.get("content")
                    for event in events
                    if isinstance(event, dict) and event.get("action") == "add"
                ]
                goal = metadata.get("goal_text") or ""
                primary = saved[0] if saved else (adds[-1] if adds else "")
                ng, ns = normalize_for_compare(goal, primary)
                rows.append(
                    {
                        "id": payload.get("id"),
                        "eval_file": eval_path.name,
                        "goal_text": goal,
                        "saved": primary,
                        "normalized_match": bool(ng and ns and ng == ns),
                    }
                )
    rows.sort(key=lambda row: str(row["id"]))
    return rows


def spot_check_transcripts(frame: pd.DataFrame, *, label: str, k: int, seed: int) -> None:
    if k <= 0 or frame.empty:
        return
    sample = frame.sample(n=min(k, len(frame)), random_state=seed)
    print(f"\nTranscript spot-checks ({label}, {len(sample)} samples)")
    print("-" * 60)
    for _, row in sample.iterrows():
        sample_id = str(row["id"])
        log_path = Path(str(row["log"]))
        payload = load_sample_from_eval(log_path, sample_id)
        issues: list[str] = []
        if payload is None:
            issues.append("sample payload missing")
        else:
            messages = payload.get("messages") or []
            if len(messages) < 2:
                issues.append(f"expected >=2 messages, got {len(messages)}")
            else:
                if messages[0].get("role") != "system":
                    issues.append(f"msg0 role={messages[0].get('role')!r}")
                user_text = flatten_content(messages[1].get("content"))
                if "<antml:document" not in user_text:
                    issues.append("user message missing Claude antml wrapper")
                if "The following document was uploaded by the user:" not in user_text:
                    issues.append("user message missing document intro")
                system_text = flatten_content(messages[0].get("content"))
                if "Claude" not in system_text[:500]:
                    issues.append("system prompt does not mention Claude near start")

        status = "ok" if not issues else "CHECK"
        print(f"[{status}] {sample_id} ({log_path.name})")
        for issue in issues:
            print(f"  - {issue}")


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    logs_root = args.logs_root.resolve()
    full_dir = resolve_log_dir(args.full_log_dir, logs_root, "claude_sonnet_4.6_full")
    truncated_dir = resolve_log_dir(
        args.truncated_log_dir,
        logs_root,
        "claude_sonnet_4.6_truncated",
    )

    print("Paths")
    print("-----")
    print(f"Manifest:      {manifest_path.relative_to(REPO_ROOT)}")
    print(f"Full log dir:  {full_dir.relative_to(REPO_ROOT)}")
    print(f"Trunc log dir: {truncated_dir.relative_to(REPO_ROOT)}")

    expected_ids = set(manifest["full_sample_ids"])
    full_df = samples_df(str(full_dir))
    trunc_df = samples_df(str(truncated_dir))
    full_ids = set(full_df["id"].astype(str))
    trunc_ids = set(trunc_df["id"].astype(str))

    print("\n1) Same samples: full vs truncated vs manifest")
    print("-" * 50)
    ok_sets = full_ids == trunc_ids == expected_ids
    print(f"Manifest count: {len(expected_ids)}")
    print(f"Full count:     {len(full_ids)}")
    print(f"Trunc count:    {len(trunc_ids)}")
    if full_ids != trunc_ids:
        only_full = sorted(full_ids - trunc_ids)
        only_trunc = sorted(trunc_ids - full_ids)
        print(f"Full-only ids:  {len(only_full)}")
        if only_full:
            print(f"  e.g. {only_full[:5]}")
        print(f"Trunc-only ids: {len(only_trunc)}")
        if only_trunc:
            print(f"  e.g. {only_trunc[:5]}")
    if full_ids != expected_ids:
        missing = sorted(expected_ids - full_ids)
        extra = sorted(full_ids - expected_ids)
        print(f"Missing from full vs manifest: {len(missing)}")
        if missing:
            print(f"  e.g. {missing[:5]}")
        print(f"Extra in full vs manifest:    {len(extra)}")
        if extra:
            print(f"  e.g. {extra[:5]}")
    if trunc_ids != expected_ids:
        missing = sorted(expected_ids - trunc_ids)
        extra = sorted(trunc_ids - expected_ids)
        print(f"Missing from trunc vs manifest: {len(missing)}")
        if missing:
            print(f"  e.g. {missing[:5]}")
        print(f"Extra in trunc vs manifest:    {len(extra)}")
        if extra:
            print(f"  e.g. {extra[:5]}")
    print(f"Result: {'PASS' if ok_sets else 'FAIL'}")

    print("\n2) Source pools and merged dataset provenance")
    print("-" * 50)
    src_with = manifest["source_datasets"]["with_memories"]
    src_without = manifest["source_datasets"]["without_memories"]
    with_records: list[dict[str, Any]] = load_json(resolve_repo_path(src_with))
    without_records: list[dict[str, Any]] = load_json(resolve_repo_path(src_without))
    pool_index: dict[tuple[str, str], str] = {}
    pool_index.update(index_records(with_records, label="with_memories"))
    pool_index.update(index_records(without_records, label="without_memories"))

    merged_path = resolve_repo_path(manifest["generated_datasets"]["full"])
    merged_records: list[dict[str, Any]] = load_json(merged_path)

    bad_pool: list[str] = []
    bad_memory_flag: list[str] = []
    for index, record in enumerate(merged_records):
        key = record_key(record)
        pool = pool_index.get(key)
        if pool is None:
            bad_pool.append(f"row {index} {key} not in source pools")
            continue
        if pool == "with_memories" and not has_memories(record):
            bad_memory_flag.append(f"{key}: in with_memories pool but no memories in record")
        if pool == "without_memories" and has_memories(record):
            bad_memory_flag.append(f"{key}: in without_memories pool but has memories in record")

    provenance_ok = not bad_pool and not bad_memory_flag
    print(f"Merged dataset: {merged_path.relative_to(REPO_ROOT)} ({len(merged_records)} records)")
    print(f"Unknown keys vs source pools: {len(bad_pool)}")
    print(f"Pool vs record memory mismatches: {len(bad_memory_flag)}")
    print(f"Result: {'PASS' if provenance_ok else 'FAIL'}")

    print("\n3) Full vs truncated summary")
    print("-" * 50)
    full_frame, full_usage, full_cost = load_condition(full_dir)
    truncated_frame, truncated_usage, truncated_cost = load_condition(truncated_dir)
    full_summary = summarize_condition("Full prompt", full_frame, full_usage, full_cost)
    truncated_summary = summarize_condition(
        "Truncated prompt",
        truncated_frame,
        truncated_usage,
        truncated_cost,
    )
    print_summary(full_summary)
    print_summary(truncated_summary)
    if full_summary["tool_call_rate"] is not None and truncated_summary["tool_call_rate"] is not None:
        delta_pp = abs(full_summary["tool_call_rate"] - truncated_summary["tool_call_rate"]) * 100
        print(f"Absolute ASR difference: {delta_pp:.2f}pp")

    print("\n4) Tool-call saves")
    print("-" * 50)
    for label, directory in (("full", full_dir), ("truncated", truncated_dir)):
        rows = collect_tool_call_rows(directory)
        print(f"{label}: {len(rows)} tool calls")
        for row in rows:
            flag = "≈" if row["normalized_match"] else "≠"
            print(f"  {flag} {row['id']}")
            print(f"    GOAL:  {row['goal_text']}")
            print(f"    SAVED: {row['saved']}")

    spot_check_transcripts(full_df, label="full", k=args.transcript_samples, seed=args.seed)
    spot_check_transcripts(
        trunc_df,
        label="truncated",
        k=args.transcript_samples,
        seed=args.seed + 1,
    )

    overall_ok = ok_sets and provenance_ok
    print("\nOverall")
    print("-------")
    print("PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
