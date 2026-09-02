#!/usr/bin/env python3
"""Replay prompt-only mem0 manager extraction through the Gemini Batch API."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import Any

from google import genai
from google.genai import types
from inspect_ai.log import read_eval_log, write_eval_log
import yaml

from sleeper_eval.batch_replay_mem0_gemini import (
    apply_gemini_replay_results_to_log,
    build_file_batch_entry,
    build_gemini_replay_requests,
    build_inline_batch_entry,
    parse_gemini_batch_jsonl_output,
    parse_gemini_inline_batch_output,
    prepared_request_to_dict,
)
from sleeper_eval.eval_campaign.config import load_campaign_config
from sleeper_eval.eval_campaign.mem0_replay import load_replay_analysis_frame, resolve_replay_source_logs

try:
    from scripts.analyze import write_analysis_outputs
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from analyze import write_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit Gemini batch manager replay jobs.")
    submit.add_argument("config_file", help="Transcript-only or prompt-only campaign config.")
    submit.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source .eval log, logs dir, or campaign run dir. May be repeated.",
    )
    submit.add_argument(
        "--output-dir",
        required=True,
        help="Directory where replayed logs and analysis will be written during apply.",
    )
    submit.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults under output-dir.",
    )
    submit.add_argument(
        "--api-key-env",
        default="GOOGLE_API_KEY",
        help="Environment variable containing the Gemini API key.",
    )
    mode = submit.add_mutually_exclusive_group()
    mode.add_argument("--inline", action="store_true", help="Use inline requests for small smokes.")
    mode.add_argument(
        "--file",
        action="store_true",
        help="Use uploaded JSONL file requests. This is the default.",
    )
    submit.add_argument(
        "--display-name",
        default="mem0-gemini-batch-replay",
        help="Gemini batch display name.",
    )
    submit.add_argument(
        "--split-source-log",
        action="store_true",
        help="Submit one Gemini batch per source .eval log instead of one combined batch.",
    )
    submit.add_argument(
        "--max-active-batches",
        type=int,
        default=None,
        help=(
            "When used with --split-source-log, submit at most this many child batches "
            "at a time, then poll/apply them before submitting more."
        ),
    )
    submit.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds for rolling split submission.",
    )
    submit.add_argument(
        "--apply-on-success",
        action="store_true",
        help="Automatically apply successful batches during rolling split submission.",
    )
    submit.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop rolling split submission if any submitted batch fails.",
    )

    status = subparsers.add_parser("status", help="Check Gemini batch replay status.")
    status.add_argument("manifest", help="Manifest JSON from submit.")

    wait = subparsers.add_parser(
        "wait",
        help="Poll one or more Gemini batch manifests until all reach a terminal state.",
    )
    wait.add_argument(
        "manifests",
        nargs="+",
        help="One or more manifest JSON files from submit.",
    )
    wait.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    wait.add_argument(
        "--apply-on-success",
        action="store_true",
        help="Automatically apply successful batches once they complete.",
    )
    wait.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Exit early if any batch enters FAILED, CANCELLED, or EXPIRED.",
    )

    apply = subparsers.add_parser("apply", help="Apply completed Gemini batch results.")
    apply.add_argument("manifest", help="Manifest JSON from submit.")

    return parser.parse_args()


def _client(api_key_env: str) -> genai.Client:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing Gemini API key env var: {api_key_env}")
    return genai.Client(api_key=api_key)


def _default_manifest_path(output_dir: Path) -> Path:
    return output_dir / "gemini_batch_manifest.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest payload: {path}")
    return manifest


def _is_group_manifest(manifest: dict[str, Any]) -> bool:
    manifest_paths = manifest.get("manifest_paths")
    return isinstance(manifest_paths, list)


def _expand_manifest_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        manifest = _load_manifest(path)
        if _is_group_manifest(manifest):
            parent = path.parent
            child_paths = [
                (parent / str(item)).resolve()
                for item in manifest.get("manifest_paths", [])
            ]
            for child in _expand_manifest_paths(child_paths):
                if child not in seen:
                    expanded.append(child)
                    seen.add(child)
            continue
        resolved = path.resolve()
        if resolved not in seen:
            expanded.append(resolved)
            seen.add(resolved)
    return expanded


def _batch_terminal_state(state_name: str) -> bool:
    return state_name in {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }


def _manifest_label(manifest_path: Path, manifest: dict[str, Any]) -> str:
    output_dir = str(manifest.get("output_dir") or "")
    if output_dir:
        return f"{manifest_path.name} ({Path(output_dir).name})"
    return manifest_path.name


def _write_group_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    config_file: str,
    request_mode: str,
    source_logs: list[Path],
    child_manifest_paths: list[Path],
    total_requests_submitted: int,
    rolling_limit: int | None,
) -> None:
    group_manifest = {
        "schema_version": 1,
        "provider": "gemini",
        "group_manifest": True,
        "config_file": str(Path(config_file).resolve()),
        "output_dir": str(output_dir),
        "request_mode": request_mode,
        "split_strategy": "source_log",
        "source_logs": [str(path) for path in source_logs],
        "manifest_paths": [str(path.relative_to(manifest_path.parent)) for path in child_manifest_paths],
        "submitted_at": datetime.now(UTC).isoformat(),
        "total_source_logs": len(source_logs),
        "submitted_batches": len(child_manifest_paths),
        "submitted_requests": total_requests_submitted,
        "rolling_limit": rolling_limit,
    }
    manifest_path.write_text(json.dumps(group_manifest, indent=2, sort_keys=True), encoding="utf-8")


def _batch_status_record(manifest_path: Path) -> tuple[dict[str, Any], Any]:
    manifest = _load_manifest(manifest_path)
    if _is_group_manifest(manifest):
        raise ValueError(f"Expected batch manifest, got group manifest: {manifest_path}")
    client = _client(str(manifest["api_key_env"]))
    batch = client.batches.get(name=str(manifest["batch_name"]))
    return manifest, batch


def _validate_config_for_gemini_batch(config: Any) -> None:
    if config.memory_backend != "mem0":
        raise ValueError("Gemini batch replay requires memory_backend=mem0.")
    if config.mem0_provider != "gemini":
        raise ValueError("Gemini batch replay currently supports only mem0_provider=gemini.")
    if config.mem0_runtime not in {"prompt_only", "transcript_only"}:
        raise ValueError(
            "Gemini batch replay requires mem0_runtime=prompt_only or transcript_only."
        )


def submit_batch(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config_file)
    _validate_config_for_gemini_batch(config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve() if args.manifest else _default_manifest_path(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    source_logs = resolve_replay_source_logs(args.source)
    request_mode = "inline" if args.inline else "file"
    client = _client(args.api_key_env)

    def build_payloads_for_source(source_log_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        requests_payload: list[dict[str, Any]] = []
        inline_entries: list[dict[str, Any]] = []
        file_entries: list[dict[str, Any]] = []
        log = read_eval_log(str(source_log_path))
        for request in build_gemini_replay_requests(
            log,
            source_log_path=source_log_path,
            mem0_prompt_variant=config.mem0_prompt_variant,
            mem0_include_document_content=config.mem0_include_document_content,
        ):
            requests_payload.append(
                {
                    "key": request.key,
                    "source_log_path": request.source_log_path,
                    "sample_id": request.sample_id,
                    "epoch": request.epoch,
                    "prepared_request": prepared_request_to_dict(request.prepared_request),
                }
            )
            inline_entries.append(
                build_inline_batch_entry(
                    request,
                    mem0_thinking=config.mem0_thinking,
                    mem0_reasoning_effort=config.mem0_reasoning_effort,
                )
            )
            file_entries.append(
                build_file_batch_entry(
                    request,
                    mem0_thinking=config.mem0_thinking,
                    mem0_reasoning_effort=config.mem0_reasoning_effort,
                )
            )
        return requests_payload, inline_entries, file_entries

    def submit_entries(
        *,
        requests_payload: list[dict[str, Any]],
        inline_entries: list[dict[str, Any]],
        file_entries: list[dict[str, Any]],
        manifest_dest: Path,
        source_log_subset: list[Path],
        display_name: str,
    ) -> tuple[str, str | None]:
        if not requests_payload:
            raise ValueError("No replayable samples found in the provided source logs.")
        uploaded_file_name: str | None = None
        if request_mode == "inline":
            batch = client.batches.create(
                model=config.mem0_model,
                src=inline_entries,
                config={"display_name": display_name},
            )
        else:
            with NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
                temp_path = Path(handle.name)
                for entry in file_entries:
                    handle.write(json.dumps(entry, ensure_ascii=False))
                    handle.write("\n")
            try:
                uploaded_file = client.files.upload(
                    file=temp_path,
                    config=types.UploadFileConfig(
                        display_name=display_name,
                        mime_type="jsonl",
                    ),
                )
                uploaded_file_name = uploaded_file.name
                batch = client.batches.create(
                    model=config.mem0_model,
                    src=uploaded_file.name,
                    config={"display_name": display_name},
                )
            finally:
                temp_path.unlink(missing_ok=True)

        manifest = {
            "schema_version": 1,
            "provider": "gemini",
            "api_key_env": args.api_key_env,
            "config_file": str(Path(args.config_file).resolve()),
            "output_dir": str(output_dir),
            "source_logs": [str(path) for path in source_log_subset],
            "request_mode": request_mode,
            "batch_name": batch.name,
            "uploaded_file_name": uploaded_file_name,
            "mem0_provider": config.mem0_provider,
            "mem0_model": config.mem0_model,
            "mem0_prompt_variant": config.mem0_prompt_variant,
            "mem0_include_document_content": config.mem0_include_document_content,
            "mem0_thinking": config.mem0_thinking,
            "mem0_reasoning_effort": config.mem0_reasoning_effort,
            "mem0_qdrant_mode": config.mem0_qdrant_mode,
            "mem0_qdrant_url": config.mem0_qdrant_url,
            "mem0_qdrant_api_key_env": config.mem0_qdrant_api_key_env,
            "mem0_qdrant_collection_name": config.mem0_qdrant_collection_name,
            "source_mem0_runtime": config.mem0_runtime,
            "replay_mem0_runtime": "prompt_only",
            "requests": requests_payload,
            "submitted_at": datetime.now(UTC).isoformat(),
            "resolved_config": config.model_dump(mode="json"),
        }
        manifest_dest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return batch.name, uploaded_file_name

    if args.split_source_log:
        if args.max_active_batches is not None and args.max_active_batches <= 0:
            raise ValueError("--max-active-batches must be greater than 0.")
        manifests_dir = output_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        child_manifest_paths: list[Path] = []
        total_requests = 0
        submitted_count = 0
        active_manifests: list[Path] = []
        next_index = 0

        while next_index < len(source_logs) or active_manifests:
            while next_index < len(source_logs) and (
                args.max_active_batches is None or len(active_manifests) < args.max_active_batches
            ):
                source_log_path = source_logs[next_index]
                idx = next_index + 1
                requests_payload, inline_entries, file_entries = build_payloads_for_source(source_log_path)
                total_requests += len(requests_payload)
                child_manifest = manifests_dir / f"{source_log_path.stem}.json"
                batch_name, _uploaded_file_name = submit_entries(
                    requests_payload=requests_payload,
                    inline_entries=inline_entries,
                    file_entries=file_entries,
                    manifest_dest=child_manifest,
                    source_log_subset=[source_log_path],
                    display_name=f"{args.display_name}-{idx:02d}",
                )
                child_manifest_paths.append(child_manifest)
                active_manifests.append(child_manifest)
                submitted_count += 1
                next_index += 1
                _write_group_manifest(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    config_file=args.config_file,
                    request_mode=request_mode,
                    source_logs=source_logs,
                    child_manifest_paths=child_manifest_paths,
                    total_requests_submitted=total_requests,
                    rolling_limit=args.max_active_batches,
                )
                print(
                    f"Submitted Gemini batch {batch_name} for {source_log_path.name} "
                    f"({len(requests_payload)} requests)"
                )

            if not active_manifests:
                break

            print(
                f"[{datetime.now(UTC).isoformat()}] rolling poll {len(active_manifests)} active batch(es) "
                f"(submitted {submitted_count}/{len(source_logs)})"
            )
            still_active: list[Path] = []
            saw_failure = False
            any_progress = False

            for child_manifest in active_manifests:
                manifest, batch = _batch_status_record(child_manifest)
                state_name = batch.state.name
                output_file = getattr(getattr(batch, "dest", None), "file_name", None)
                print(
                    f"  - {_manifest_label(child_manifest, manifest)}: {state_name} "
                    f"(requests={len(manifest['requests'])}, output_file={output_file})"
                )

                if not _batch_terminal_state(state_name):
                    still_active.append(child_manifest)
                    continue

                any_progress = True
                if state_name != "JOB_STATE_SUCCEEDED":
                    saw_failure = True
                    continue

                if args.apply_on_success and not manifest.get("applied_at"):
                    print(f"    applying successful batch from {child_manifest}")
                    apply_batch(argparse.Namespace(manifest=str(child_manifest)))

            if saw_failure and args.stop_on_failure:
                raise RuntimeError(
                    "At least one Gemini batch entered a failure terminal state during rolling submission."
                )

            active_manifests = still_active
            if active_manifests:
                if any_progress:
                    continue
                time.sleep(args.poll_seconds)

        _write_group_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
            config_file=args.config_file,
            request_mode=request_mode,
            source_logs=source_logs,
            child_manifest_paths=child_manifest_paths,
            total_requests_submitted=total_requests,
            rolling_limit=args.max_active_batches,
        )
        print(f"Group manifest: {manifest_path}")
        print(f"Source logs: {len(source_logs)}")
        print(f"Submitted batches: {len(child_manifest_paths)}")
        print(f"Submitted requests: {total_requests}")
        print(f"Mode: {request_mode}")
        print("Split strategy: source_log")
        if args.max_active_batches is not None:
            print(f"Rolling limit: {args.max_active_batches}")
        return

    requests_payload: list[dict[str, Any]] = []
    inline_entries: list[dict[str, Any]] = []
    file_entries: list[dict[str, Any]] = []
    for source_log_path in source_logs:
        source_requests, source_inline, source_file = build_payloads_for_source(source_log_path)
        requests_payload.extend(source_requests)
        inline_entries.extend(source_inline)
        file_entries.extend(source_file)

    batch_name, _uploaded_file_name = submit_entries(
        requests_payload=requests_payload,
        inline_entries=inline_entries,
        file_entries=file_entries,
        manifest_dest=manifest_path,
        source_log_subset=source_logs,
        display_name=args.display_name,
    )
    print(f"Submitted Gemini batch {batch_name}")
    print(f"Manifest: {manifest_path}")
    print(f"Source logs: {len(source_logs)}")
    print(f"Requests: {len(requests_payload)}")
    print(f"Mode: {request_mode}")


def print_status(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    if _is_group_manifest(manifest):
        child_paths = _expand_manifest_paths([manifest_path])
        print(f"Group manifest: {manifest_path}")
        print(f"Batches: {len(child_paths)}")
        for child_path in child_paths:
            child_manifest, batch = _batch_status_record(child_path)
            print(
                f"- {child_path.name}: {batch.state.name} "
                f"(requests={len(child_manifest['requests'])}, "
                f"output_file={getattr(getattr(batch, 'dest', None), 'file_name', None)})"
            )
        return
    _manifest, batch = _batch_status_record(manifest_path)
    print(f"Batch name: {batch.name}")
    print(f"State: {batch.state.name}")
    print(f"Model: {batch.model}")
    print(f"Requests: {len(manifest['requests'])}")
    print(f"Output file: {getattr(getattr(batch, 'dest', None), 'file_name', None)}")


def wait_for_batches(args: argparse.Namespace) -> None:
    manifest_paths = _expand_manifest_paths([Path(path).resolve() for path in args.manifests])
    applied: set[Path] = set()
    last_state_by_path: dict[Path, str] = {}

    while True:
        print(f"[{datetime.now(UTC).isoformat()}] polling {len(manifest_paths)} batch(es)")
        all_terminal = True
        saw_failure = False

        for manifest_path in manifest_paths:
            manifest, batch = _batch_status_record(manifest_path)
            state_name = batch.state.name
            label = _manifest_label(manifest_path, manifest)
            changed_marker = ""
            if last_state_by_path.get(manifest_path) != state_name:
                changed_marker = " *"
                last_state_by_path[manifest_path] = state_name

            output_file = getattr(getattr(batch, "dest", None), "file_name", None)
            print(
                f"  - {label}: {state_name}{changed_marker} "
                f"(requests={len(manifest['requests'])}, output_file={output_file})"
            )

            if not _batch_terminal_state(state_name):
                all_terminal = False
                continue

            if state_name != "JOB_STATE_SUCCEEDED":
                saw_failure = True
                continue

            if args.apply_on_success and manifest_path not in applied:
                if manifest.get("applied_at"):
                    print(f"    already applied: {manifest_path}")
                else:
                    print(f"    applying successful batch from {manifest_path}")
                    apply_batch(argparse.Namespace(manifest=str(manifest_path)))
                applied.add(manifest_path)

        if args.stop_on_failure and saw_failure:
            raise RuntimeError("At least one Gemini batch entered a failure terminal state.")
        if all_terminal:
            print("All batches reached terminal states.")
            return
        time.sleep(args.poll_seconds)


def apply_batch(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    if _is_group_manifest(manifest):
        child_paths = _expand_manifest_paths([manifest_path])
        for child_path in child_paths:
            apply_batch(argparse.Namespace(manifest=str(child_path)))
        return
    client = _client(str(manifest["api_key_env"]))
    batch = client.batches.get(name=str(manifest["batch_name"]))
    if batch.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch {batch.name} is not completed successfully. State: {batch.state.name}"
        )

    if manifest["request_mode"] == "inline":
        inlined_responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
        results_by_key = parse_gemini_inline_batch_output(list(inlined_responses))
    else:
        result_file_name = getattr(getattr(batch, "dest", None), "file_name", None)
        if not result_file_name:
            raise RuntimeError(f"Batch {batch.name} has no result file.")
        result_text = client.files.download(file=result_file_name).decode("utf-8")
        results_by_key = parse_gemini_batch_jsonl_output(result_text)

    output_dir = Path(str(manifest["output_dir"]))
    log_dir = output_dir / "logs"
    analysis_dir = output_dir / "analysis"
    log_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    requests_by_log: dict[str, list[dict[str, Any]]] = {}
    for request in manifest["requests"]:
        requests_by_log.setdefault(str(request["source_log_path"]), []).append(request)

    output_logs: list[str] = []
    for source_log, request_payloads in requests_by_log.items():
        source_path = Path(source_log)
        log = read_eval_log(str(source_path))
        replayed = apply_gemini_replay_results_to_log(
            log,
            source_log_path=source_path,
            requests=request_payloads,
            results_by_key=results_by_key,
            mem0_provider=str(manifest["mem0_provider"]),
            mem0_model=str(manifest["mem0_model"]),
            mem0_prompt_variant=str(manifest["mem0_prompt_variant"]),
            mem0_include_document_content=bool(manifest["mem0_include_document_content"]),
            mem0_thinking=str(manifest["mem0_thinking"]),
            mem0_reasoning_effort=str(manifest["mem0_reasoning_effort"]),
            mem0_qdrant_mode=str(manifest["mem0_qdrant_mode"]),
            mem0_qdrant_url=str(manifest["mem0_qdrant_url"]),
            mem0_qdrant_api_key_env=str(manifest["mem0_qdrant_api_key_env"]),
            mem0_qdrant_collection_name=str(manifest["mem0_qdrant_collection_name"]),
        )
        target_path = log_dir / source_path.name
        write_eval_log(replayed, location=target_path)
        output_logs.append(str(target_path))

    analysis_outputs = write_analysis_outputs(load_replay_analysis_frame(log_dir), analysis_dir)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(manifest["resolved_config"], sort_keys=False),
        encoding="utf-8",
    )

    manifest["applied_at"] = datetime.now(UTC).isoformat()
    manifest["output_logs"] = output_logs
    manifest["analysis_outputs"] = {name: str(path) for name, path in analysis_outputs.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Applied Gemini batch replay to {len(output_logs)} log(s)")
    print(f"Logs: {log_dir}")
    print(f"Analysis: {analysis_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "submit":
        submit_batch(args)
    elif args.command == "status":
        print_status(args)
    elif args.command == "wait":
        wait_for_batches(args)
    else:
        apply_batch(args)


if __name__ == "__main__":
    main()
