"""Helpers for Gemini Batch replay of prompt-only mem0 manager extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from inspect_ai.log import recompute_metrics

from sleeper_eval.eval_campaign.mem0_replay import (
    _first_system_message_from_raw_store,
    _seed_memories_for_sample,
    _replay_provenance,
    apply_mem0_write_scores,
    build_replay_state,
    replay_target_mem0_runtime,
    stamp_replayed_log_metadata,
)
from sleeper_eval.memory_backend import (
    Mem0PromptOnlyPreparedRequest,
    SavedMemoryStore,
    _inspect_model_name_for_mem0_manager,
    _mem0_prompt_only_response_is_malformed,
    _serialize_mem0_prompt_only_response,
    build_mem0_prompt_only_raw_backend_result,
    parse_mem0_prompt_only_response,
    prepare_mem0_prompt_only_request,
)


@dataclass(frozen=True)
class GeminiReplayRequest:
    key: str
    source_log_path: str
    sample_id: str
    epoch: int
    prepared_request: Mem0PromptOnlyPreparedRequest


def prepared_request_to_dict(prepared: Mem0PromptOnlyPreparedRequest) -> dict[str, Any]:
    return {
        "input_messages": prepared.input_messages,
        "parsed_messages": prepared.parsed_messages,
        "seeded_existing_memories": prepared.seeded_existing_memories,
        "existing_memories_prompt_count": prepared.existing_memories_prompt_count,
        "extraction_system_prompt": prepared.extraction_system_prompt,
        "extraction_system_prompt_path": prepared.extraction_system_prompt_path,
        "extraction_user_prompt": prepared.extraction_user_prompt,
    }


def prepared_request_from_dict(payload: dict[str, Any]) -> Mem0PromptOnlyPreparedRequest:
    return Mem0PromptOnlyPreparedRequest(
        input_messages=[
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in payload["input_messages"]
        ],
        parsed_messages=str(payload["parsed_messages"]),
        seeded_existing_memories=[str(item) for item in payload["seeded_existing_memories"]],
        existing_memories_prompt_count=int(payload["existing_memories_prompt_count"]),
        extraction_system_prompt=str(payload["extraction_system_prompt"]),
        extraction_system_prompt_path=str(payload["extraction_system_prompt_path"]),
        extraction_user_prompt=str(payload["extraction_user_prompt"]),
    )


def gemini_batch_request_key(source_log_path: Path, sample_id: Any, epoch: Any) -> str:
    return f"{source_log_path.name}::{sample_id}::{epoch}"


def _thinking_config_payload(mem0_thinking: str, mem0_reasoning_effort: str) -> dict[str, Any] | None:
    if mem0_thinking == "disabled":
        return {"thinking_budget": 0}
    level = "HIGH" if mem0_reasoning_effort in {"high", "max"} else None
    if level is None:
        return None
    return {"thinking_level": level}


def build_gemini_generate_content_request(
    prepared: Mem0PromptOnlyPreparedRequest,
    *,
    mem0_thinking: str,
    mem0_reasoning_effort: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": prepared.extraction_system_prompt}]},
        "response_mime_type": "application/json",
    }
    thinking_config = _thinking_config_payload(mem0_thinking, mem0_reasoning_effort)
    if thinking_config is not None:
        config["thinking_config"] = thinking_config
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prepared.extraction_user_prompt}],
            }
        ],
        "config": config,
    }


def build_inline_batch_entry(
    request: GeminiReplayRequest,
    *,
    mem0_thinking: str,
    mem0_reasoning_effort: str,
) -> dict[str, Any]:
    payload = build_gemini_generate_content_request(
        request.prepared_request,
        mem0_thinking=mem0_thinking,
        mem0_reasoning_effort=mem0_reasoning_effort,
    )
    payload["metadata"] = {"key": request.key}
    return payload


def build_file_batch_entry(
    request: GeminiReplayRequest,
    *,
    mem0_thinking: str,
    mem0_reasoning_effort: str,
) -> dict[str, Any]:
    thinking_config = _thinking_config_payload(mem0_thinking, mem0_reasoning_effort)
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
    }
    if thinking_config is not None:
        generation_config["thinkingConfig"] = thinking_config
    return {
        "key": request.key,
        "request": {
            "systemInstruction": {
                "parts": [{"text": request.prepared_request.extraction_system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": request.prepared_request.extraction_user_prompt}],
                }
            ],
            "generationConfig": generation_config,
        },
    }


def build_gemini_replay_requests(
    log: Any,
    *,
    source_log_path: Path,
    mem0_prompt_variant: str,
    mem0_include_document_content: bool,
) -> list[GeminiReplayRequest]:
    requests: list[GeminiReplayRequest] = []
    for sample in getattr(log, "samples", None) or []:
        if getattr(sample, "output", None) is None:
            raise ValueError(
                f"Replay source sample {sample.id!r} has no output in {source_log_path}."
            )
        state = build_replay_state(log, sample)
        system_prompt = _first_system_message_from_raw_store(sample)
        seeded_memories = _seed_memories_for_sample(sample)
        state.store_as(SavedMemoryStore).seeded_existing_memories = list(seeded_memories)
        prepared = prepare_mem0_prompt_only_request(
            state,
            mem0_prompt_variant=mem0_prompt_variant,
            system_prompt=system_prompt,
            include_document_content=mem0_include_document_content,
        )
        requests.append(
            GeminiReplayRequest(
                key=gemini_batch_request_key(source_log_path, sample.id, sample.epoch),
                source_log_path=str(source_log_path),
                sample_id=str(sample.id),
                epoch=int(sample.epoch),
                prepared_request=prepared,
            )
        )
    return requests


def extract_gemini_response_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(response, dict):
        direct = response.get("text")
        if isinstance(direct, str) and direct.strip():
            return direct
        candidates = response.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                texts = [
                    str(part.get("text")).strip()
                    for part in parts
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                ]
                joined = "\n".join(text for text in texts if text)
                if joined:
                    return joined
    return None


def parse_gemini_batch_jsonl_output(text: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        key = str(payload.get("key") or "")
        if not key:
            raise ValueError(f"Gemini batch output line missing key: {line}")
        response_text = extract_gemini_response_text(payload.get("response"))
        parsed[key] = {
            "response_text": response_text,
            "error": payload.get("error"),
        }
    return parsed


def parse_gemini_inline_batch_output(inlined_responses: list[Any]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for item in inlined_responses:
        metadata = getattr(item, "metadata", None)
        if metadata is None and isinstance(item, dict):
            metadata = item.get("metadata")
        key = ""
        if isinstance(metadata, dict):
            key = str(metadata.get("key") or "")
        if not key:
            raise ValueError("Gemini inline batch response missing metadata.key.")
        response = getattr(item, "response", None)
        if response is None and isinstance(item, dict):
            response = item.get("response")
        error = getattr(item, "error", None)
        if error is None and isinstance(item, dict):
            error = item.get("error")
        parsed[key] = {
            "response_text": extract_gemini_response_text(response),
            "error": error,
        }
    return parsed


def apply_gemini_replay_results_to_log(
    log: Any,
    *,
    source_log_path: Path,
    requests: list[dict[str, Any]],
    results_by_key: dict[str, dict[str, Any]],
    mem0_provider: str,
    mem0_model: str,
    mem0_prompt_variant: str,
    mem0_include_document_content: bool,
    mem0_thinking: str,
    mem0_reasoning_effort: str,
    mem0_qdrant_mode: str,
    mem0_qdrant_url: str,
    mem0_qdrant_api_key_env: str,
    mem0_qdrant_collection_name: str,
) -> Any:
    sample_index = {
        (str(sample.id), int(sample.epoch)): sample
        for sample in getattr(log, "samples", None) or []
    }
    target_runtime = replay_target_mem0_runtime("prompt_only")
    llm_config = {"model": mem0_model.strip()} if mem0_model.strip() else {}
    inspect_model_name = _inspect_model_name_for_mem0_manager(mem0_provider, mem0_model)

    for request_payload in requests:
        key = str(request_payload["key"])
        sample_key = (str(request_payload["sample_id"]), int(request_payload["epoch"]))
        sample = sample_index.get(sample_key)
        if sample is None:
            raise ValueError(f"Could not locate sample for replay request {key}.")
        result_payload = results_by_key.get(key)
        if result_payload is None:
            raise ValueError(f"No Gemini batch result found for request key {key}.")
        if result_payload.get("error") is not None:
            raise RuntimeError(f"Gemini batch request {key} failed: {result_payload['error']}")

        prepared = prepared_request_from_dict(dict(request_payload["prepared_request"]))
        response_text = result_payload.get("response_text")
        if not isinstance(response_text, str):
            raise RuntimeError(f"Gemini batch request {key} returned no response text.")

        state = build_replay_state(log, sample)
        store = state.store_as(SavedMemoryStore)
        store.seeded_existing_memories = list(prepared.seeded_existing_memories)
        saved_memories, _items = parse_mem0_prompt_only_response(response_text)
        serialized_response = _serialize_mem0_prompt_only_response(response_text)
        malformed_json = _mem0_prompt_only_response_is_malformed(serialized_response)
        raw_backend_result = build_mem0_prompt_only_raw_backend_result(
            prepared=prepared,
            user_id=store.mem0_scope_id or f"sleeper-eval-{state.uuid}",
            mem0_provider=mem0_provider,
            mem0_model=mem0_model,
            inspect_model_name=inspect_model_name,
            mem0_thinking=mem0_thinking,
            mem0_reasoning_effort=mem0_reasoning_effort,
            mem0_prompt_variant=mem0_prompt_variant,
            llm_config=llm_config,
            response=response_text,
            response_attempts=[
                {
                    "attempt": 1,
                    "raw_completion": response_text,
                    "response": serialized_response,
                    "response_type": type(serialized_response).__name__,
                    "malformed_json": malformed_json,
                    "stop_reason": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "total_tokens": None,
                }
            ],
            structured_output_used=True,
            manager_latency_summary={"batch_mode": True},
            llm_latency_events=[],
        )
        store.backend_name = "mem0"
        store.runtime_name = target_runtime
        store.provider_id = store.provider_id or "replay"
        store.saved_memories = list(saved_memories)
        store.raw_backend_result = raw_backend_result
        sample.store = dict(state.store.items())

    provenance = _replay_provenance(
        source_log_path=source_log_path,
        mem0_provider=mem0_provider,
        mem0_model=mem0_model,
        mem0_runtime=target_runtime,
    )
    apply_mem0_write_scores(log, provenance=provenance)
    recompute_metrics(log)
    stamp_replayed_log_metadata(
        log,
        source_log_path=source_log_path,
        mem0_runtime=target_runtime,
        mem0_provider=mem0_provider,
        mem0_model=mem0_model,
        mem0_include_document_content=mem0_include_document_content,
        mem0_thinking=mem0_thinking,
        mem0_reasoning_effort=mem0_reasoning_effort,
        mem0_qdrant_mode=mem0_qdrant_mode,
        mem0_qdrant_url=mem0_qdrant_url,
        mem0_qdrant_api_key_env=mem0_qdrant_api_key_env,
        mem0_qdrant_collection_name=mem0_qdrant_collection_name,
        apply_semantic_scoring=False,
        replay_max_concurrency=1,
    )
    return log
