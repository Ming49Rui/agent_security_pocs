#!/usr/bin/env python3
"""Batch-optimize goals from a simple JSON file for embedding retrievability."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleeper_eval.retrievability_optim import GoalRecord, dedupe_texts
from scripts.optimize_behavior_related_goals import (
    load_existing_output,
    optimize_goal,
    select_goals,
    write_output,
)
from scripts.optimize_goal_retrievability import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    client_from_env,
    embed_texts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goals-json",
        type=Path,
        required=True,
        help=(
            "JSON list containing goals. Supports rows with goal_text/goal_id, or "
            "agent-action rows with memory_string and metadata.task_id."
        ),
    )
    parser.add_argument(
        "--query-json-files",
        nargs="+",
        required=True,
        help="JSON files containing downstream query texts used as retrieval targets.",
    )
    parser.add_argument(
        "--query-field",
        default="eval_query",
        help="Dot-separated field path for query text inside --query-json-files.",
    )
    parser.add_argument(
        "--goal-ids",
        nargs="*",
        default=[],
        help="Optional subset of goal IDs to optimize. Defaults to every unique goal.",
    )
    parser.add_argument(
        "--limit-goals",
        type=int,
        default=0,
        help="Limit number of goals to optimize after filtering. Use 0 for all.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum number of unique queries to embed. Use 0 for all.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--candidates-per-iteration", type=int, default=20)
    parser.add_argument("--objective", choices=["mean", "min", "p10"], default="mean")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--candidate-max-jaccard", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Maximum words requested for each rewritten memory entry. Use 0 for no requested limit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path for the resumable optimized-goal map.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output file and optimize from scratch.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative.")
    if args.candidates_per_iteration <= 0:
        raise ValueError("--candidates-per-iteration must be positive.")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive.")
    if args.max_queries < 0:
        raise ValueError("--max-queries must be non-negative.")
    if args.limit_goals < 0:
        raise ValueError("--limit-goals must be non-negative.")
    if args.max_words < 0:
        raise ValueError("--max-words must be non-negative.")
    if not 0 <= args.candidate_max_jaccard <= 1:
        raise ValueError("--candidate-max-jaccard must be between 0 and 1.")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(value: Any, field_path: str) -> Any:
    current = value
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def load_goal_records(path: Path) -> list[GoalRecord]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")

    goals: list[GoalRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_goal_id = row.get("goal_id") or metadata.get("task_id") or f"json_goal_{index:03d}"
        raw_goal_text = row.get("goal_text") or row.get("memory_string")
        goal_id = str(raw_goal_id).strip()
        goal_text = str(raw_goal_text or "").strip()
        if not goal_id or not goal_text or goal_id in seen:
            continue
        seen.add(goal_id)
        goals.append(
            GoalRecord(
                goal_id=goal_id,
                goal_text=goal_text,
                category_name=row.get("category") or metadata.get("category"),
                subcategory_name=row.get("constraint") or metadata.get("constraint"),
                domain_seed=row.get("domain") or metadata.get("domain"),
            )
        )
    return goals


def load_queries(paths: list[str], *, query_field: str, max_queries: int) -> list[str]:
    queries: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = load_json(path)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            query = get_path(row, query_field)
            if isinstance(query, str) and query.strip():
                queries.append(query)
    deduped = dedupe_texts(queries)
    if max_queries > 0:
        return deduped[:max_queries]
    return deduped


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    goals = select_goals(load_goal_records(args.goals_json), args.goal_ids, args.limit_goals)
    queries = load_queries(
        args.query_json_files,
        query_field=args.query_field,
        max_queries=args.max_queries,
    )
    if not goals:
        raise ValueError(f"No goals found in {args.goals_json}.")
    if not queries:
        raise ValueError(
            f"No query texts found in {args.query_json_files} using field {args.query_field!r}."
        )

    output = load_existing_output(args.output_json, overwrite=args.overwrite)
    completed = set((output.get("goals") or {}).keys())
    output.setdefault("created_at", datetime.now(UTC).isoformat())
    output["updated_at"] = datetime.now(UTC).isoformat()
    output["config"] = {
        "goals_json": str(args.goals_json),
        "query_json_files": args.query_json_files,
        "query_field": args.query_field,
        "goal_count_requested": len(goals),
        "limit_goals": args.limit_goals,
        "query_count": len(queries),
        "iterations": args.iterations,
        "candidates_per_iteration": args.candidates_per_iteration,
        "objective": args.objective,
        "embedding_model": args.embedding_model,
        "llm_model": args.llm_model,
        "temperature": args.temperature,
        "candidate_max_jaccard": args.candidate_max_jaccard,
        "max_tokens": args.max_tokens,
        "max_words": args.max_words,
        "asr_validation": False,
    }
    output.setdefault("goals", {})

    client = client_from_env(args.api_key_env, args.base_url)
    query_embeddings = embed_texts(
        client,
        model=args.embedding_model,
        texts=queries,
        batch_size=args.embedding_batch_size,
    )

    print("== JSON Goal Retrievability Optimization ==")
    print(f"goals={len(goals)} completed={len(completed & {goal.goal_id for goal in goals})}")
    print(f"query_count={len(queries)}")
    print(f"output_json={args.output_json}")
    print("asr_validation=false")

    for index, goal in enumerate(goals, start=1):
        if goal.goal_id in completed:
            print(f"[{index}/{len(goals)}] skip completed goal={goal.goal_id}")
            continue
        print(f"[{index}/{len(goals)}] optimize goal={goal.goal_id}")
        output["goals"][goal.goal_id] = optimize_goal(
            goal=goal,
            client=client,
            queries=queries,
            query_embeddings=query_embeddings,
            args=args,
        )
        output["updated_at"] = datetime.now(UTC).isoformat()
        write_output(args.output_json, output)

    write_output(args.output_json, output)
    print(f"wrote={args.output_json}")


if __name__ == "__main__":
    main()
