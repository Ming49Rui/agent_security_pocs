#!/usr/bin/env python3
"""Batch semantic scoring for goal-match judge prompts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from inspect_ai.log import ProvenanceData, edit_score, read_eval_log, write_eval_log
from inspect_ai.scorer import ScoreEdit
from openai import OpenAI

from sleeper_eval.batch_goal_match import (
    build_openai_batch_request,
    deterministic_score_from_manifest_dict,
    deterministic_score_to_manifest_dict,
    extract_openai_batch_output_text,
    group_requests_by_sample,
    prepare_goal_match_batch,
    request_from_manifest_dict,
    request_to_manifest_dict,
    score_from_judge_outputs,
)


def normalize_openai_judge_model(model: str) -> str:
    """Accept either raw OpenAI model ids or inspect-style openai/<model> ids."""

    if model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the OpenAI Batch API to score goal-match semantics in Inspect eval logs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit semantic grading jobs for an eval log.")
    submit.add_argument("eval_log", help="Path to the .eval log to score.")
    submit.add_argument("--judge-model", required=True, help="Judge model to use for /v1/responses.")
    submit.add_argument(
        "--manifest",
        help="Path to write the batch manifest JSON. Defaults next to the eval log.",
    )
    submit.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    submit.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    submit.add_argument(
        "--scorer",
        default=None,
        help="Semantic scorer name to populate. Defaults from the log backend.",
    )
    submit.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples that already have the target semantic score.",
    )

    status = subparsers.add_parser("status", help="Check the status of a submitted batch job.")
    status.add_argument("manifest", help="Manifest JSON emitted by the submit command.")

    apply = subparsers.add_parser("apply", help="Apply completed batch results back to an eval log.")
    apply.add_argument("manifest", help="Manifest JSON emitted by the submit command.")
    apply.add_argument(
        "--output-file",
        default=None,
        help="Write the updated eval log to this path. Defaults to a sibling *.batch-scored.eval file.",
    )
    apply.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original eval log in-place.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "submit":
        submit_batch(args)
    elif args.command == "status":
        print_batch_status(args)
    else:
        apply_batch_results(args)


def submit_batch(args: argparse.Namespace) -> None:
    log_path = Path(args.eval_log)
    log = read_eval_log(str(log_path))
    prepared = prepare_goal_match_batch(
        log,
        scorer_name=args.scorer,
        skip_existing=args.skip_existing,
    )

    if not prepared.requests:
        print("No batch requests needed.")
        print(f"Deterministic scores ready: {len(prepared.deterministic_scores)}")
        return

    client = openai_client(args.api_key_env, args.base_url)
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(log_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    judge_model = normalize_openai_judge_model(args.judge_model)

    with NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
        temp_path = Path(handle.name)
        for request in prepared.requests:
            handle.write(
                json.dumps(build_openai_batch_request(request, judge_model), ensure_ascii=True)
            )
            handle.write("\n")

    try:
        with temp_path.open("rb") as request_file:
            uploaded_file = client.files.create(file=request_file, purpose="batch")
        batch = client.batches.create(
            completion_window="24h",
            endpoint="/v1/responses",
            input_file_id=uploaded_file.id,
            metadata={
                "script": "batch_score_goal_match.py",
                "eval_log": str(log_path),
                "scorer_name": prepared.scorer_name,
            },
        )
    finally:
        temp_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "provider": "openai",
        "api_key_env": args.api_key_env,
        "base_url": args.base_url,
        "batch_id": batch.id,
        "input_file_id": uploaded_file.id,
        "eval_log": str(log_path),
        "scorer_name": prepared.scorer_name,
        "judge_model": judge_model,
        "submitted_at": datetime.now(UTC).isoformat(),
        "requests": [request_to_manifest_dict(request) for request in prepared.requests],
        "deterministic_scores": [
            deterministic_score_to_manifest_dict(score)
            for score in prepared.deterministic_scores
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Submitted batch {batch.id}")
    print(f"Manifest: {manifest_path}")
    print(f"Requests: {len(prepared.requests)}")
    print(f"Deterministic scores: {len(prepared.deterministic_scores)}")


def print_batch_status(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    client = openai_client(str(manifest["api_key_env"]), manifest.get("base_url"))
    batch = client.batches.retrieve(str(manifest["batch_id"]))

    print(f"Batch id: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Input file: {batch.input_file_id}")
    print(f"Output file: {batch.output_file_id}")
    print(f"Error file: {batch.error_file_id}")
    print(f"Requests submitted: {len(manifest['requests'])}")
    if getattr(batch, "errors", None):
        print(f"Errors: {batch.errors}")


def apply_batch_results(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    client = openai_client(str(manifest["api_key_env"]), manifest.get("base_url"))
    batch = client.batches.retrieve(str(manifest["batch_id"]))

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch.id} is not completed yet. Current status: {batch.status}."
        )
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} has no output file.")

    output_text = client.files.content(batch.output_file_id).text
    judge_outputs = parse_batch_output_text(output_text)

    requests = [request_from_manifest_dict(payload) for payload in manifest["requests"]]
    deterministic_scores = [
        deterministic_score_from_manifest_dict(payload)
        for payload in manifest["deterministic_scores"]
    ]

    sample_scores = list(deterministic_scores)
    for request_group in group_requests_by_sample(requests).values():
        sample_scores.append(score_from_judge_outputs(request_group, judge_outputs))

    log_path = Path(str(manifest["eval_log"]))
    log = read_eval_log(str(log_path))
    provenance = ProvenanceData(
        author="scripts/batch_score_goal_match.py",
        reason="Applied OpenAI Batch API semantic grading results.",
        metadata={
            "batch_id": batch.id,
            "judge_model": manifest["judge_model"],
            "provider": "openai",
        },
    )

    for score in sample_scores:
        edit_score(
            log,
            sample_id=score.sample_id,
            epoch=score.epoch,
            score_name=score.scorer_name,
            edit=ScoreEdit(
                value=score.value,
                answer=score.answer,
                explanation=score.explanation,
                metadata=score.metadata,
                provenance=provenance,
            ),
            recompute_metrics=False,
        )

    output_path = resolve_output_path(
        original_path=log_path,
        output_file=args.output_file,
        overwrite=args.overwrite,
    )
    write_eval_log(log, str(output_path))

    print(f"Wrote updated log: {output_path}")
    print(f"Applied scores: {len(sample_scores)}")


def openai_client(api_key_env: str, base_url: str | None) -> OpenAI:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set.")
    return OpenAI(api_key=api_key, base_url=base_url)


def default_manifest_path(log_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return log_path.with_name(f"{log_path.stem}.goal-match-batch.{timestamp}.json")


def resolve_output_path(original_path: Path, output_file: str | None, overwrite: bool) -> Path:
    if output_file:
        return Path(output_file)
    if overwrite:
        return original_path
    if original_path.suffix == ".eval":
        return original_path.with_name(f"{original_path.stem}.batch-scored.eval")
    return original_path.with_name(f"{original_path.name}.batch-scored")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_batch_output_text(text: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = payload.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError("Batch output line is missing a valid custom_id.")
        outputs[custom_id] = extract_openai_batch_output_text(payload)
    return outputs


if __name__ == "__main__":
    main()
