"""Memory backend abstractions for sleeper memory evaluation."""

from __future__ import annotations
import ast
from contextvars import ContextVar
import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Protocol
import uuid

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.solver import TaskState
from inspect_ai.util._json import JSONSchema
from inspect_ai.util import StoreModel, store_as
from pydantic import Field

VALID_MEMORY_BACKENDS = {"tool", "mem0"}
VALID_MEM0_RUNTIMES = {"local", "sdk", "prompt_only", "transcript_only"}
VALID_MEM0_PROVIDERS = {"openai", "openrouter", "anthropic", "deepseek", "gemini"}
VALID_MEM0_PROMPT_VARIANTS = {"mem0_default", "lenient_assistant"}
VALID_MEM0_THINKING = {"enabled", "disabled"}
VALID_MEM0_REASONING_EFFORTS = {"high", "max"}
VALID_MEM0_QDRANT_MODES = {"local", "server", "managed"}
MEM0_MANAGER_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "mem0_manager"
MEM0_MANAGER_SYSTEM_PROMPT_FILES = {
    "mem0_default": "mem0_default_system_prompt.md",
    "lenient_assistant": "mem0_lenient_assistant_system_prompt.md",
}
DEFAULT_MEMORY_BACKEND_NAME = "tool"
DEFAULT_MEM0_RUNTIME = "local"
DEFAULT_MEM0_PROVIDER = "openai"
DEFAULT_MEM0_MODEL = ""
DEFAULT_MEM0_PROMPT_VARIANT = "mem0_default"
DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT = True
DEFAULT_MEM0_THINKING = "enabled"
DEFAULT_MEM0_REASONING_EFFORT = "high"
DEFAULT_MEM0_ROOT_DIR = ".inspect_home/mem0-sdk"
DEFAULT_MEM0_QDRANT_MODE = "local"
DEFAULT_MEM0_QDRANT_URL = ""
DEFAULT_MEM0_QDRANT_API_KEY_ENV = "QDRANT_API_KEY"
DEFAULT_MEM0_QDRANT_COLLECTION_NAME = ""
DEFAULT_MEM0_MANAGER_TEMPERATURE = 0.1
DEFAULT_MEM0_MANAGER_TOP_P = 0.1
DEFAULT_MEM0_MANAGER_MAX_TOKENS = 20480
DEFAULT_MEM0_PROMPT_ONLY_MAX_ATTEMPTS = 3
MANAGED_MEM0_QDRANT_URL_ENV = "SLEEPER_EVAL_MEM0_QDRANT_URL"
MEM0_LATENCY_LOG_ENV = "SLEEPER_EVAL_MEM0_LATENCY_LOG"
MEM0_MANAGER_DEBUG_FILE_ENV = "SLEEPER_EVAL_DEBUG_MEM0_MANAGER_FILE"
logger = logging.getLogger(__name__)
_MEM0_LATENCY_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "sleeper_eval_mem0_latency_context",
    default=None,
)


class SavedMemoryStore(StoreModel):
    """Per-sample store of new memories written by the active backend."""

    backend_name: str = DEFAULT_MEMORY_BACKEND_NAME
    runtime_name: str | None = None
    tool_name: str | None = None
    provider_id: str | None = None
    current_memories: list[str] = Field(default_factory=list)
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    saved_memories: list[str] = Field(default_factory=list)
    mem0_scope_id: str | None = None
    seeded_existing_memories: list[str] = Field(default_factory=list)
    raw_backend_result: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryBackendResult:
    """Normalized memory-write result for scoring and analysis."""

    backend_name: str
    saved_memories: list[str]
    runtime_name: str | None = None
    tool_name: str | None = None
    provider_id: str | None = None
    raw_backend_result: dict[str, Any] | None = None

    @property
    def write_count(self) -> int:
        return len(self.saved_memories)

    @property
    def wrote_memory(self) -> bool:
        return self.write_count > 0


@dataclass(frozen=True)
class Mem0PromptOnlyPreparedRequest:
    """Normalized prompt-only extraction payload shared by sync and batch paths."""

    input_messages: list[dict[str, str]]
    parsed_messages: str
    seeded_existing_memories: list[str]
    existing_memories_prompt_count: int
    extraction_system_prompt: str
    extraction_system_prompt_path: str
    extraction_user_prompt: str


class MemoryBackend(Protocol):
    """Protocol for memory backends used during eval runs."""

    @property
    def backend_name(self) -> str: ...

    @property
    def runtime_name(self) -> str | None: ...

    def record_memory(self, memory_text: str) -> str:
        """Persist one memory and return the stored text."""
        ...

    async def seed_existing_memories(self, state: TaskState, memories: list[str]) -> None:
        """Seed preexisting memories into any backend-managed store before generation."""
        ...

    async def persist_from_state(self, state: TaskState) -> MemoryBackendResult:
        """Persist any autonomous memories for a completed task state."""
        ...

    def result_for_state(self, state: TaskState) -> MemoryBackendResult:
        """Return normalized write results for a completed task state."""
        ...


def _normalize_system_prompt(system_prompt: str | None) -> str | None:
    if system_prompt is None:
        return None
    cleaned = system_prompt.strip()
    return cleaned or None


def _query_text_from_state(state: TaskState) -> str:
    query_text = state.metadata.get("query_text") if isinstance(state.metadata, dict) else None
    if isinstance(query_text, str) and query_text.strip():
        return query_text.strip()
    return state.input_text.strip()


def _clean_memory_text(memory_text: str) -> str:
    return memory_text.strip()


def _debug_print_mem0_input(messages: list[dict[str, str]]) -> None:
    if os.getenv("SLEEPER_EVAL_DEBUG_MEM0_INPUT", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    print(
        "[mem0-debug] input_messages="
        + json.dumps(messages, ensure_ascii=True, sort_keys=True)
    )


def _debug_print_mem0_manager_prompt_payload(
    *,
    runtime_name: str,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    response: Any | None = None,
) -> None:
    enabled = os.getenv("SLEEPER_EVAL_DEBUG_MEM0_MANAGER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    debug_file = os.getenv(MEM0_MANAGER_DEBUG_FILE_ENV, "").strip()
    if not enabled and not debug_file:
        return

    lines: list[str] = []
    if system_prompt is not None:
        lines.extend(
            [
                f"[mem0-debug] runtime={runtime_name} extraction_system_prompt_begin",
                system_prompt,
                f"[mem0-debug] runtime={runtime_name} extraction_system_prompt_end",
                "",
            ]
        )
    if user_prompt is not None:
        lines.extend(
            [
                f"[mem0-debug] runtime={runtime_name} extraction_user_prompt_begin",
                user_prompt,
                f"[mem0-debug] runtime={runtime_name} extraction_user_prompt_end",
            ]
        )
    if response is not None:
        lines.append("")
        lines.append(f"[mem0-debug] runtime={runtime_name} extraction_response_begin")
        if isinstance(response, str):
            lines.append(response)
        else:
            lines.append(json.dumps(response, ensure_ascii=False, default=str))
        lines.append(f"[mem0-debug] runtime={runtime_name} extraction_response_end")

    payload = "\n".join(lines) + "\n"
    if enabled:
        print(payload, end="")
    if debug_file:
        debug_path = Path(debug_file).expanduser()
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)


def _extract_json_candidate(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _pythonize_json_literals(text: str) -> str:
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)
    return text


def _normalize_mem0_json_response(response: str) -> str | None:
    candidate = _extract_json_candidate(response)
    if not candidate:
        return None

    parse_attempts = (
        candidate,
        re.sub(r",(\s*[}\]])", r"\1", candidate),
    )
    for attempt in parse_attempts:
        try:
            parsed = json.loads(attempt, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)

    try:
        parsed = ast.literal_eval(_pythonize_json_literals(candidate))
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return None


def _response_format_expects_json_object(response_format: Any) -> bool:
    if not isinstance(response_format, Mapping):
        return False
    return response_format.get("type") == "json_object"


def _append_mem0_latency_jsonl(payload: dict[str, Any]) -> None:
    path = os.getenv(MEM0_LATENCY_LOG_ENV, "").strip()
    if not path:
        return
    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _wrap_mem0_json_generation(
    memory_client: Any,
    *,
    record_event: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    llm = getattr(memory_client, "llm", None)
    if llm is None or getattr(llm, "_sleeper_mem0_json_wrapper_installed", False):
        return

    original_generate_response = getattr(llm, "generate_response", None)
    if not callable(original_generate_response):
        return

    original_get_common_params = getattr(llm, "_get_common_params", None)
    if callable(original_get_common_params) and not getattr(
        llm, "_sleeper_mem0_openai_token_patch_installed", False
    ):
        def wrapped_get_common_params(**kwargs: Any) -> dict[str, Any]:
            params = original_get_common_params(**kwargs)
            model_name = str(getattr(getattr(llm, "config", None), "model", "") or "").lower()
            base_model = model_name.rsplit("/", 1)[-1]
            if base_model.startswith("gpt-5"):
                max_tokens = params.pop("max_tokens", None)
                if max_tokens is not None:
                    params["max_completion_tokens"] = max_tokens
            return params

        llm._get_common_params = wrapped_get_common_params
        setattr(llm, "_sleeper_mem0_openai_token_patch_installed", True)

    def wrapped_generate_response(*args: Any, **kwargs: Any) -> Any:
        response_format = kwargs.get("response_format")
        tools = kwargs.get("tools")
        expects_json_object = _response_format_expects_json_object(response_format) and not tools

        def _event_base(*, attempt: int | None = None) -> dict[str, Any]:
            context = _MEM0_LATENCY_CONTEXT.get() or {}
            return {
                "timestamp": time.time(),
                "scope_id": context.get("scope_id"),
                "sample_id": context.get("sample_id"),
                "phase": context.get("phase"),
                "provider": context.get("provider"),
                "model": context.get("model"),
                "expects_json_object": expects_json_object,
                "tools_present": bool(tools),
                "attempt": attempt,
            }

        if not expects_json_object:
            started = time.perf_counter()
            response = original_generate_response(*args, **kwargs)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            if record_event is not None:
                record_event(
                    {
                        **_event_base(),
                        "duration_ms": duration_ms,
                        "response_type": type(response).__name__,
                        "normalized_json": False,
                    }
                )
            return response

        last_response: str | None = None
        for attempt in range(3):
            started = time.perf_counter()
            response = original_generate_response(*args, **kwargs)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            if not isinstance(response, str):
                if record_event is not None:
                    record_event(
                        {
                            **_event_base(attempt=attempt),
                            "duration_ms": duration_ms,
                            "response_type": type(response).__name__,
                            "normalized_json": False,
                            "returned_non_string": True,
                        }
                    )
                return response
            last_response = response
            normalized = _normalize_mem0_json_response(response)
            if record_event is not None:
                record_event(
                    {
                        **_event_base(attempt=attempt),
                        "duration_ms": duration_ms,
                        "response_type": "str",
                        "normalized_json": normalized is not None,
                        "raw_preview": response[:200],
                    }
                )
            if normalized is not None:
                if attempt:
                    logger.warning(
                        "Recovered malformed mem0 JSON extraction response on retry %s.",
                        attempt,
                    )
                return normalized

        logger.warning(
            "Unable to normalize mem0 JSON extraction response after retries; "
            "returning an empty memory payload. Raw preview: %r",
            (last_response or "")[:400],
        )
        return '{"memory":[]}'

    llm.generate_response = wrapped_generate_response
    setattr(llm, "_sleeper_mem0_json_wrapper_installed", True)


def _bool_env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mem0_telemetry_enabled_from_env() -> bool:
    return _bool_env_enabled("MEM0_TELEMETRY", default=True)


def _sanitize_nested_config(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if "api_key" in lowered or lowered.endswith("token"):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = _sanitize_nested_config(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_nested_config(item) for item in value]
    return value


def _embedder_details(memory_client: Any) -> tuple[str, str]:
    embedder = getattr(memory_client, "embedding_model", None)
    if embedder is None:
        return "(unknown)", "(unknown)"

    config = getattr(embedder, "config", None)
    provider = getattr(config, "provider", None)
    model = getattr(config, "model", None)

    if not provider:
        module_name = getattr(embedder.__class__, "__module__", "")
        provider = module_name.rsplit(".", maxsplit=1)[-1] if module_name else "(unknown)"
    if not model:
        model = getattr(embedder, "model", None) or "(unknown)"
    return str(provider), str(model)


def _disable_mem0_bm25(memory_client: Any) -> None:
    """Disable mem0 Qdrant BM25/hybrid behavior for eval runs.

    The mem0 Qdrant adapter enables a BM25 sparse-vector path by default.
    That triggers fastembed model downloads (`Qdrant/bm25`) which are
    unnecessary for this harness and create noisy failures in constrained
    environments. Dense semantic retrieval is sufficient for these evals.
    """

    vector_store = getattr(memory_client, "vector_store", None)
    if vector_store is not None and hasattr(vector_store, "_has_bm25_slot"):
        try:
            vector_store._has_bm25_slot = False
            vector_store._bm25_encoder = False
        except Exception:
            pass
    try:
        entity_store = getattr(memory_client, "entity_store", None)
    except Exception:
        entity_store = None
    if entity_store is not None and hasattr(entity_store, "_has_bm25_slot"):
        try:
            entity_store._has_bm25_slot = False
            entity_store._bm25_encoder = False
        except Exception:
            pass


def _deepseek_config_param_names() -> set[str]:
    """Best-effort introspection of the installed mem0 DeepSeek config surface."""

    try:
        module = importlib.import_module("mem0.configs.llms.deepseek")
        deepseek_config = getattr(module, "DeepSeekConfig")
        return set(inspect.signature(deepseek_config).parameters)
    except Exception:
        # mem0 1.0.6 shape observed locally
        return {
            "model",
            "temperature",
            "api_key",
            "max_tokens",
            "top_p",
            "top_k",
            "enable_vision",
            "vision_details",
            "http_client_proxies",
            "deepseek_base_url",
        }


def validate_mem0_prompt_variant(mem0_prompt_variant: str) -> str:
    if mem0_prompt_variant not in VALID_MEM0_PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown mem0 prompt variant '{mem0_prompt_variant}'. "
            f"Available: {sorted(VALID_MEM0_PROMPT_VARIANTS)}"
        )
    return mem0_prompt_variant


def mem0_manager_prompt_path(mem0_prompt_variant: str) -> Path:
    variant = validate_mem0_prompt_variant(mem0_prompt_variant)
    filename = MEM0_MANAGER_SYSTEM_PROMPT_FILES[variant]
    return MEM0_MANAGER_PROMPT_DIR / filename


def _load_repo_mem0_system_prompt(*, mem0_prompt_variant: str) -> str:
    prompt_path = mem0_manager_prompt_path(mem0_prompt_variant)
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing mem0 manager prompt file for variant '{mem0_prompt_variant}': {prompt_path}"
        ) from exc


def _sample_scope_id(state: TaskState) -> str:
    store = state.store_as(SavedMemoryStore)
    if store.mem0_scope_id:
        return store.mem0_scope_id
    store.mem0_scope_id = f"sleeper-eval-{state.uuid}"
    return store.mem0_scope_id


def _write_store_result(
    store: SavedMemoryStore,
    *,
    backend_name: str,
    saved_memories: list[str],
    runtime_name: str | None = None,
    tool_name: str | None = None,
    provider_id: str | None = None,
    raw_backend_result: dict[str, Any] | None = None,
) -> MemoryBackendResult:
    store.backend_name = backend_name
    store.runtime_name = runtime_name
    if tool_name is not None:
        store.tool_name = tool_name
    if provider_id is not None:
        store.provider_id = provider_id
    store.saved_memories = list(saved_memories)
    store.raw_backend_result = raw_backend_result
    return MemoryBackendResult(
        backend_name=backend_name,
        saved_memories=list(saved_memories),
        runtime_name=runtime_name,
        tool_name=store.tool_name,
        provider_id=store.provider_id,
        raw_backend_result=raw_backend_result,
    )


def _result_from_store(store: SavedMemoryStore) -> MemoryBackendResult:
    return MemoryBackendResult(
        backend_name=store.backend_name,
        saved_memories=list(store.saved_memories),
        runtime_name=store.runtime_name,
        tool_name=store.tool_name,
        provider_id=store.provider_id,
        raw_backend_result=store.raw_backend_result,
    )


def result_from_store_dict(store: Mapping[str, Any] | None) -> MemoryBackendResult:
    """Reconstruct a normalized backend result from an eval log store payload."""

    if store is None:
        return MemoryBackendResult(
            backend_name=DEFAULT_MEMORY_BACKEND_NAME,
            saved_memories=[],
        )

    backend_name = store.get("SavedMemoryStore:backend_name", DEFAULT_MEMORY_BACKEND_NAME)
    runtime_name = store.get("SavedMemoryStore:runtime_name")
    saved_memories = store.get("SavedMemoryStore:saved_memories") or []
    raw_backend_result = store.get("SavedMemoryStore:raw_backend_result")

    normalized_memories = [
        memory.strip()
        for memory in saved_memories
        if isinstance(memory, str) and memory.strip()
    ]

    return MemoryBackendResult(
        backend_name=str(backend_name),
        runtime_name=runtime_name if isinstance(runtime_name, str) else None,
        saved_memories=normalized_memories,
        raw_backend_result=raw_backend_result if isinstance(raw_backend_result, dict) else None,
    )


class ToolMemoryBackend:
    """Current tool-based memory backend."""

    backend_name = "tool"
    runtime_name = "tool"

    def record_memory(self, memory_text: str) -> str:
        cleaned = _clean_memory_text(memory_text)
        store = store_as(SavedMemoryStore)
        store.backend_name = self.backend_name
        store.runtime_name = self.runtime_name
        store.saved_memories.append(cleaned)
        return cleaned

    async def seed_existing_memories(self, state: TaskState, memories: list[str]) -> None:
        del state, memories

    async def persist_from_state(self, state: TaskState) -> MemoryBackendResult:
        return self.result_for_state(state)

    def result_for_state(self, state: TaskState) -> MemoryBackendResult:
        return _result_from_store(state.store_as(SavedMemoryStore))


def _state_messages_as_mem0_input(
    state: TaskState,
    system_prompt: str | None = None,
    include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
) -> list[dict[str, str]]:
    if not include_document_content:
        messages: list[dict[str, str]] = []
        normalized_system_prompt = _normalize_system_prompt(system_prompt)
        if normalized_system_prompt is not None:
            messages.append({"role": "system", "content": normalized_system_prompt})
        query_text = _query_text_from_state(state)
        if query_text:
            messages.append({"role": "user", "content": query_text})
        completion = state.output.completion.strip()
        if completion:
            messages.append({"role": "assistant", "content": completion})
        return messages

    messages: list[dict[str, str]] = []
    normalized_system_prompt = _normalize_system_prompt(system_prompt)
    if normalized_system_prompt is not None:
        messages.append({"role": "system", "content": normalized_system_prompt})
    for message in state.messages:
        text = getattr(message, "text", "").strip()
        if not text:
            continue
        if isinstance(message, ChatMessageSystem):
            role = "system"
        elif isinstance(message, ChatMessageAssistant):
            role = "assistant"
        elif isinstance(message, ChatMessageUser):
            role = "user"
        else:
            role = getattr(message, "role", "user")
        messages.append({"role": role, "content": text})

    completion = state.output.completion.strip()
    if completion:
        if not messages or messages[-1]["role"] != "assistant" or messages[-1]["content"] != completion:
            messages.append({"role": "assistant", "content": completion})
    return messages


def _extract_local_memories(state: TaskState) -> list[str]:
    completion = state.output.completion.strip()
    if not completion:
        return []

    memories: list[str] = []
    for raw_line in completion.splitlines():
        line = raw_line.strip(" -*\t")
        lowered = line.lower()
        if not line:
            continue
        if "user " in lowered or "user's" in lowered or lowered.startswith("user"):
            memories.append(line)

    deduped: list[str] = []
    seen: set[str] = set()
    for memory in memories:
        key = memory.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(memory)
    return deduped


class Mem0LocalBackend:
    """Deterministic local mem0 stand-in used for tests and offline development."""

    backend_name = "mem0"
    runtime_name = "local"

    def __init__(
        self,
        system_prompt: str | None = None,
        include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
    ) -> None:
        self.system_prompt = _normalize_system_prompt(system_prompt)
        self.include_document_content = include_document_content

    def record_memory(self, memory_text: str) -> str:
        raise RuntimeError("mem0 backends do not accept direct record_memory calls.")

    async def seed_existing_memories(self, state: TaskState, memories: list[str]) -> None:
        store = state.store_as(SavedMemoryStore)
        store.seeded_existing_memories = [
            cleaned for memory in memories if (cleaned := _clean_memory_text(memory))
        ]

    async def persist_from_state(self, state: TaskState) -> MemoryBackendResult:
        store = state.store_as(SavedMemoryStore)
        saved_memories = _extract_local_memories(state)
        return _write_store_result(
            store,
            backend_name=self.backend_name,
            runtime_name=self.runtime_name,
            saved_memories=saved_memories,
            raw_backend_result={
                "mode": "local",
                "input_messages": _state_messages_as_mem0_input(
                    state,
                    system_prompt=self.system_prompt,
                    include_document_content=self.include_document_content,
                ),
            },
        )

    def result_for_state(self, state: TaskState) -> MemoryBackendResult:
        return _result_from_store(state.store_as(SavedMemoryStore))


class Mem0TranscriptOnlyBackend:
    """Preserve the real C2 prompt surface while deferring manager execution."""

    backend_name = "mem0"
    runtime_name = "transcript_only"

    def __init__(
        self,
        system_prompt: str | None = None,
        include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
    ) -> None:
        self.system_prompt = _normalize_system_prompt(system_prompt)
        self.include_document_content = include_document_content

    def record_memory(self, memory_text: str) -> str:
        raise RuntimeError("mem0 backends do not accept direct record_memory calls.")

    async def seed_existing_memories(self, state: TaskState, memories: list[str]) -> None:
        store = state.store_as(SavedMemoryStore)
        store.seeded_existing_memories = [
            cleaned for memory in memories if (cleaned := _clean_memory_text(memory))
        ]

    async def persist_from_state(self, state: TaskState) -> MemoryBackendResult:
        store = state.store_as(SavedMemoryStore)
        input_messages = _state_messages_as_mem0_input(
            state,
            system_prompt=self.system_prompt,
            include_document_content=self.include_document_content,
        )
        _debug_print_mem0_input(input_messages)
        raw_backend_result = {
            "mode": "transcript_only",
            "input_messages": input_messages,
            "seeded_existing_memories": list(store.seeded_existing_memories),
            "query_text": _query_text_from_state(state),
            "assistant_completion": state.output.completion.strip(),
        }
        return _write_store_result(
            store,
            backend_name=self.backend_name,
            runtime_name=self.runtime_name,
            saved_memories=[],
            raw_backend_result=raw_backend_result,
        )

    def result_for_state(self, state: TaskState) -> MemoryBackendResult:
        return _result_from_store(state.store_as(SavedMemoryStore))


def _ensure_mem0_provider_credentials(mem0_provider: str) -> None:
    if mem0_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openai_api_key and not openrouter_api_key:
            raise RuntimeError(
                "mem0 backend requested with provider 'openai' but neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set."
            )
    elif mem0_provider == "openrouter":
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError(
                "mem0 backend requested with provider 'openrouter' but OPENROUTER_API_KEY is not set."
            )
    elif mem0_provider == "anthropic":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise RuntimeError(
                "mem0 backend requested with provider 'anthropic' but ANTHROPIC_API_KEY is not set."
            )
    elif mem0_provider == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise RuntimeError(
                "mem0 backend requested with provider 'deepseek' but DEEPSEEK_API_KEY is not set."
            )
    elif mem0_provider == "gemini":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError(
                "mem0 backend requested with provider 'gemini' but GOOGLE_API_KEY is not set."
            )


def _mem0_llm_config_dict(*, mem0_provider: str, mem0_model: str) -> dict[str, Any]:
    llm_config: dict[str, Any] = {}
    if mem0_model:
        llm_config["model"] = mem0_model.strip()
    if mem0_provider == "deepseek":
        deepseek_config_params = _deepseek_config_param_names()
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_API_BASE")
        if deepseek_base_url and "deepseek_base_url" in deepseek_config_params:
            llm_config["deepseek_base_url"] = deepseek_base_url
    return llm_config


def _effective_mem0_llm_provider(mem0_provider: str) -> str:
    if mem0_provider == "openrouter":
        return "openai"
    return mem0_provider


def _default_mem0_manager_model(mem0_provider: str) -> str:
    defaults = {
        "openai": "gpt-5-mini",
        "openrouter": "openai/gpt-5-mini",
        "anthropic": "claude-3-5-sonnet-20240620",
        "deepseek": "deepseek-chat",
        "gemini": "gemini-2.0-flash",
    }
    return defaults[mem0_provider]


def _inspect_model_name_for_mem0_manager(mem0_provider: str, mem0_model: str) -> str:
    model_name = mem0_model.strip() or _default_mem0_manager_model(mem0_provider)
    if mem0_provider == "deepseek":
        return f"openai-api/deepseek/{model_name}"
    prefix = {
        "openai": "openai",
        "openrouter": "openrouter",
        "anthropic": "anthropic",
        "gemini": "google",
    }[mem0_provider]
    return f"{prefix}/{model_name}"


def _mem0_prompt_only_supports_structured_output(mem0_provider: str) -> bool:
    return mem0_provider in {"openai", "openrouter", "anthropic", "gemini"}


def _mem0_prompt_only_response_schema(mem0_provider: str) -> ResponseSchema | None:
    if not _mem0_prompt_only_supports_structured_output(mem0_provider):
        return None
    strict = True if mem0_provider in {"openai", "openrouter"} else None
    return ResponseSchema(
        name="mem0_saved_memories",
        json_schema=JSONSchema(
            type="object",
            properties={
                "memory": JSONSchema(
                    type="array",
                    items=JSONSchema(
                        type="object",
                        properties={
                            "text": JSONSchema(type="string"),
                        },
                        required=["text"],
                        additionalProperties=False,
                    ),
                )
            },
            required=["memory"],
            additionalProperties=False,
        ),
        description=(
            "Return extracted persistent memories as a JSON object with a top-level "
            "'memory' array of objects containing a required 'text' field."
        ),
        strict=strict,
    )


def _mem0_prompt_only_generate_config(
    *,
    mem0_provider: str,
    mem0_thinking: str,
    mem0_reasoning_effort: str,
) -> tuple[GenerateConfig, bool]:
    response_schema = _mem0_prompt_only_response_schema(mem0_provider)
    temperature: float | None = DEFAULT_MEM0_MANAGER_TEMPERATURE
    top_p: float | None = None
    if mem0_provider in {"openai", "openrouter"} and mem0_thinking == "enabled":
        temperature = None
    config = GenerateConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=DEFAULT_MEM0_MANAGER_MAX_TOKENS,
        response_schema=response_schema,
    )
    if mem0_thinking == "enabled":
        config.reasoning_effort = mem0_reasoning_effort
    return config, response_schema is not None


def _serialize_mem0_prompt_only_response(raw_response: str) -> Any:
    if not raw_response.strip():
        return raw_response
    normalized = _normalize_mem0_json_response(raw_response)
    if normalized is None:
        return raw_response
    try:
        return json.loads(normalized, strict=False)
    except json.JSONDecodeError:
        return raw_response


def _mem0_prompt_only_response_is_malformed(response: Any) -> bool:
    if isinstance(response, dict):
        memory_items = response.get("memory")
        return not isinstance(memory_items, list)
    if not isinstance(response, str) or not response.strip():
        return True
    normalized = _normalize_mem0_json_response(response)
    if normalized is None:
        return True
    try:
        parsed = json.loads(normalized, strict=False)
    except json.JSONDecodeError:
        return True
    if not isinstance(parsed, dict):
        return True
    memory_items = parsed.get("memory")
    return not isinstance(memory_items, list)


def _model_output_stop_reason(output: Any) -> str | None:
    choices = getattr(output, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    stop_reason = getattr(choices[0], "stop_reason", None)
    if isinstance(stop_reason, str) and stop_reason.strip():
        return stop_reason.strip()
    return None


def _load_mem0_prompt_only_components() -> tuple[
    Callable[..., str],
    Callable[[Any], str],
]:
    try:
        prompts_module = importlib.import_module("mem0.configs.prompts")
        utils_module = importlib.import_module("mem0.memory.utils")
    except ImportError as exc:
        raise RuntimeError(
            "mem0 prompt-only backend requested but the 'mem0' package is not installed."
        ) from exc

    prompt_builder = getattr(prompts_module, "generate_additive_extraction_prompt")
    parse_messages = getattr(utils_module, "parse_messages")
    return prompt_builder, parse_messages


def prepare_mem0_prompt_only_request(
    state: TaskState,
    *,
    mem0_prompt_variant: str = DEFAULT_MEM0_PROMPT_VARIANT,
    system_prompt: str | None = None,
    include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
) -> Mem0PromptOnlyPreparedRequest:
    """Build the prompt-only manager request from a completed task state."""

    prompt_variant = validate_mem0_prompt_variant(mem0_prompt_variant)
    messages = _state_messages_as_mem0_input(
        state,
        system_prompt=system_prompt,
        include_document_content=include_document_content,
    )
    _debug_print_mem0_input(messages)
    prompt_builder, parse_messages = _load_mem0_prompt_only_components()
    store = state.store_as(SavedMemoryStore)
    parsed_messages = parse_messages(messages)
    existing_memories = [
        {"id": str(idx), "text": memory_text}
        for idx, memory_text in enumerate(store.seeded_existing_memories)
    ]
    extraction_user_prompt = prompt_builder(
        summary="",
        recently_extracted_memories=[],
        existing_memories=existing_memories,
        new_messages=parsed_messages,
        last_k_messages=[],
    )
    extraction_system_prompt = _load_repo_mem0_system_prompt(
        mem0_prompt_variant=prompt_variant
    )
    prepared = Mem0PromptOnlyPreparedRequest(
        input_messages=messages,
        parsed_messages=parsed_messages,
        seeded_existing_memories=list(store.seeded_existing_memories),
        existing_memories_prompt_count=len(existing_memories),
        extraction_system_prompt=extraction_system_prompt,
        extraction_system_prompt_path=str(mem0_manager_prompt_path(prompt_variant)),
        extraction_user_prompt=extraction_user_prompt,
    )
    _debug_print_mem0_manager_prompt_payload(
        runtime_name="prompt_only",
        system_prompt=prepared.extraction_system_prompt,
        user_prompt=prepared.extraction_user_prompt,
    )
    return prepared


def parse_mem0_prompt_only_response(response: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse saved memories from a prompt-only manager response payload."""

    extracted_memories: list[dict[str, Any]] = []
    if isinstance(response, str) and response.strip():
        try:
            extracted_memories = json.loads(response, strict=False).get("memory", [])
        except json.JSONDecodeError:
            extracted_json = _extract_json_candidate(response)
            try:
                extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
            except Exception:
                extracted_memories = []
    elif isinstance(response, dict):
        memory_items = response.get("memory", [])
        if isinstance(memory_items, list):
            extracted_memories = [item for item in memory_items if isinstance(item, dict)]

    saved_memories: list[str] = []
    for item in extracted_memories:
        candidate = item.get("text") or item.get("memory") or item.get("content")
        if isinstance(candidate, str) and candidate.strip():
            saved_memories.append(candidate.strip())
    return saved_memories, extracted_memories


def build_mem0_prompt_only_raw_backend_result(
    *,
    prepared: Mem0PromptOnlyPreparedRequest,
    user_id: str,
    mem0_provider: str,
    mem0_model: str,
    inspect_model_name: str,
    mem0_thinking: str,
    mem0_reasoning_effort: str,
    mem0_prompt_variant: str,
    llm_config: dict[str, Any],
    response: Any,
    response_attempts: list[dict[str, Any]],
    structured_output_used: bool,
    manager_latency_summary: dict[str, Any],
    llm_latency_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "prompt_only",
        "user_id": user_id,
        "mem0_provider": mem0_provider,
        "mem0_model": mem0_model,
        "inspect_model_name": inspect_model_name,
        "mem0_requested_thinking": mem0_thinking,
        "mem0_requested_reasoning_effort": mem0_reasoning_effort,
        "mem0_prompt_variant": mem0_prompt_variant,
        "input_messages": prepared.input_messages,
        "parsed_messages": prepared.parsed_messages,
        "seeded_existing_memories": prepared.seeded_existing_memories,
        "existing_memories_prompt_count": prepared.existing_memories_prompt_count,
        "extraction_system_prompt": prepared.extraction_system_prompt,
        "extraction_system_prompt_path": prepared.extraction_system_prompt_path,
        "extraction_user_prompt": prepared.extraction_user_prompt,
        "structured_output_used": structured_output_used,
        "llm_config": {
            "provider": mem0_provider,
            "config": llm_config,
        },
        "response": response,
        "response_attempts": response_attempts,
        "mem0_manager_latency": manager_latency_summary,
        "mem0_manager_llm_calls": llm_latency_events,
    }


class Mem0PromptOnlyBackend:
    """Prompt-only external manager using mem0's additive extraction prompt without retrieval plumbing."""

    backend_name = "mem0"
    runtime_name = "prompt_only"

    def __init__(
        self,
        *,
        mem0_provider: str = DEFAULT_MEM0_PROVIDER,
        mem0_model: str = DEFAULT_MEM0_MODEL,
        mem0_prompt_variant: str = DEFAULT_MEM0_PROMPT_VARIANT,
        mem0_thinking: str = DEFAULT_MEM0_THINKING,
        mem0_reasoning_effort: str = DEFAULT_MEM0_REASONING_EFFORT,
        system_prompt: str | None = None,
        include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
    ) -> None:
        self.mem0_provider = mem0_provider
        self.mem0_model = mem0_model.strip()
        self.mem0_prompt_variant = validate_mem0_prompt_variant(mem0_prompt_variant)
        self.mem0_thinking = validate_mem0_thinking(mem0_thinking)
        self.mem0_reasoning_effort = validate_mem0_reasoning_effort(mem0_reasoning_effort)
        self.system_prompt = _normalize_system_prompt(system_prompt)
        self.include_document_content = include_document_content

    def record_memory(self, memory_text: str) -> str:
        raise RuntimeError("mem0 backends do not accept direct record_memory calls.")

    async def seed_existing_memories(self, state: TaskState, memories: list[str]) -> None:
        store = state.store_as(SavedMemoryStore)
        store.seeded_existing_memories = [
            cleaned for memory in memories if (cleaned := _clean_memory_text(memory))
        ]

    async def persist_from_state(self, state: TaskState) -> MemoryBackendResult:
        user_id = _sample_scope_id(state)
        store = state.store_as(SavedMemoryStore)
        _ensure_mem0_provider_credentials(self.mem0_provider)
        prepared = prepare_mem0_prompt_only_request(
            state,
            mem0_prompt_variant=self.mem0_prompt_variant,
            system_prompt=self.system_prompt,
            include_document_content=self.include_document_content,
        )
        inspect_model_name = _inspect_model_name_for_mem0_manager(
            self.mem0_provider,
            self.mem0_model,
        )
        generate_config, structured_output_used = _mem0_prompt_only_generate_config(
            mem0_provider=self.mem0_provider,
            mem0_thinking=self.mem0_thinking,
            mem0_reasoning_effort=self.mem0_reasoning_effort,
        )
        llm_config = {
            "model": self.mem0_model.strip() or _default_mem0_manager_model(self.mem0_provider),
            "inspect_model": inspect_model_name,
            "temperature": generate_config.temperature,
            "top_p": generate_config.top_p,
            "max_tokens": DEFAULT_MEM0_MANAGER_MAX_TOKENS,
            "reasoning_effort": (
                self.mem0_reasoning_effort if self.mem0_thinking == "enabled" else None
            ),
            "structured_output": structured_output_used,
        }

        llm_latency_events: list[dict[str, Any]] = []
        response_attempts: list[dict[str, Any]] = []
        response: Any = ""
        persist_started = time.perf_counter()
        async with get_model(inspect_model_name, memoize=False) as model:
            for attempt in range(1, DEFAULT_MEM0_PROMPT_ONLY_MAX_ATTEMPTS + 1):
                started = time.perf_counter()
                output = await model.generate(
                    [
                        ChatMessageSystem(content=prepared.extraction_system_prompt),
                        ChatMessageUser(content=prepared.extraction_user_prompt),
                    ],
                    config=generate_config,
                )
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                response = _serialize_mem0_prompt_only_response(output.completion)
                malformed_json = _mem0_prompt_only_response_is_malformed(response)
                usage = output.usage
                stop_reason = _model_output_stop_reason(output)
                llm_latency_events.append(
                    {
                        "timestamp": time.time(),
                        "scope_id": user_id,
                        "sample_id": state.sample_id,
                        "phase": "persist_from_state",
                        "attempt": attempt,
                        "provider": self.mem0_provider,
                        "model": self.mem0_model,
                        "inspect_model": inspect_model_name,
                        "duration_ms": duration_ms,
                        "response_type": type(response).__name__,
                        "normalized_json": isinstance(response, dict),
                        "structured_output_used": structured_output_used,
                        "malformed_json": malformed_json,
                        "stop_reason": stop_reason,
                        "input_tokens": usage.input_tokens if usage is not None else None,
                        "output_tokens": usage.output_tokens if usage is not None else None,
                        "reasoning_tokens": usage.reasoning_tokens if usage is not None else None,
                        "total_tokens": usage.total_tokens if usage is not None else None,
                    }
                )
                response_attempts.append(
                    {
                        "attempt": attempt,
                        "raw_completion": output.completion,
                        "response": response,
                        "response_type": type(response).__name__,
                        "malformed_json": malformed_json,
                        "stop_reason": stop_reason,
                        "input_tokens": usage.input_tokens if usage is not None else None,
                        "output_tokens": usage.output_tokens if usage is not None else None,
                        "reasoning_tokens": usage.reasoning_tokens if usage is not None else None,
                        "total_tokens": usage.total_tokens if usage is not None else None,
                    }
                )
                if not malformed_json:
                    break
                logger.warning(
                    "Malformed mem0 prompt-only response from %s on attempt %s/%s for sample %s; retrying.",
                    inspect_model_name,
                    attempt,
                    DEFAULT_MEM0_PROMPT_ONLY_MAX_ATTEMPTS,
                    state.sample_id,
                )
        _debug_print_mem0_manager_prompt_payload(
            runtime_name=self.runtime_name,
            response=response,
        )
        persist_latency_ms = round((time.perf_counter() - persist_started) * 1000, 3)
        saved_memories, _extracted_memories = parse_mem0_prompt_only_response(response)

        manager_latency_summary = {
            "persist_latency_ms": persist_latency_ms,
            "llm_call_count": len(llm_latency_events),
            "llm_latency_ms_total": round(
                sum(
                    float(event.get("duration_ms", 0.0))
                    for event in llm_latency_events
                    if isinstance(event.get("duration_ms"), (int, float))
                ),
                3,
            ),
            "llm_latency_ms_max": round(
                max(
                    (
                        float(event.get("duration_ms", 0.0))
                        for event in llm_latency_events
                        if isinstance(event.get("duration_ms"), (int, float))
                    ),
                    default=0.0,
                ),
                3,
            ),
        }
        raw_backend_result = build_mem0_prompt_only_raw_backend_result(
            prepared=prepared,
            user_id=user_id,
            mem0_provider=self.mem0_provider,
            mem0_model=self.mem0_model,
            inspect_model_name=inspect_model_name,
            mem0_thinking=self.mem0_thinking,
            mem0_reasoning_effort=self.mem0_reasoning_effort,
            mem0_prompt_variant=self.mem0_prompt_variant,
            llm_config=llm_config,
            response=response,
            response_attempts=response_attempts,
            structured_output_used=structured_output_used,
            manager_latency_summary=manager_latency_summary,
            llm_latency_events=llm_latency_events,
        )
        _append_mem0_latency_jsonl(
            {
                "sample_id": state.sample_id,
                "dataset_label": state.metadata.get("dataset_label")
                if isinstance(state.metadata, dict)
                else None,
                "backend": "mem0",
                "provider": self.mem0_provider,
                "model": self.mem0_model,
                "runtime": self.runtime_name,
                "saved_memory_count": len(saved_memories),
                **manager_latency_summary,
            }
        )
        return _write_store_result(
            store,
            backend_name=self.backend_name,
            runtime_name=self.runtime_name,
            saved_memories=saved_memories,
            raw_backend_result=raw_backend_result,
        )

    def result_for_state(self, state: TaskState) -> MemoryBackendResult:
        return _result_from_store(state.store_as(SavedMemoryStore))


class Mem0SdkBackend:
    """Mem0 OSS adapter using provider config rather than the managed API."""

    backend_name = "mem0"
    runtime_name = "sdk"

    def __init__(
        self,
        mem0_provider: str = DEFAULT_MEM0_PROVIDER,
        mem0_model: str = DEFAULT_MEM0_MODEL,
        mem0_thinking: str = DEFAULT_MEM0_THINKING,
        mem0_reasoning_effort: str = DEFAULT_MEM0_REASONING_EFFORT,
        mem0_qdrant_mode: str = DEFAULT_MEM0_QDRANT_MODE,
        mem0_qdrant_url: str = DEFAULT_MEM0_QDRANT_URL,
        mem0_qdrant_api_key_env: str = DEFAULT_MEM0_QDRANT_API_KEY_ENV,
        mem0_qdrant_collection_name: str = DEFAULT_MEM0_QDRANT_COLLECTION_NAME,
        system_prompt: str | None = None,
        include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
    ) -> None:
        self.mem0_provider = mem0_provider
        self.mem0_model = mem0_model.strip()
        self.mem0_thinking = validate_mem0_thinking(mem0_thinking)
        self.mem0_reasoning_effort = validate_mem0_reasoning_effort(mem0_reasoning_effort)
        self.mem0_qdrant_mode = validate_mem0_qdrant_mode(mem0_qdrant_mode)
        self.mem0_qdrant_url = mem0_qdrant_url.strip()
        self.mem0_qdrant_api_key_env = mem0_qdrant_api_key_env.strip()
        self.mem0_qdrant_collection_name = mem0_qdrant_collection_name.strip() or "mem0"
        self.system_prompt = _normalize_system_prompt(system_prompt)
        self.include_document_content = include_document_content
        self._memory_client: Any | None = None
        self._storage_root = self._build_storage_root()
        self._telemetry_env_forced = False
        self._effective_embedder_provider = "(unknown)"
        self._effective_embedder_model = "(unknown)"
        self._latency_events_by_scope: dict[str, list[dict[str, Any]]] = {}

    def record_memory(self, memory_text: str) -> str:
        raise RuntimeError("mem0 backends do not accept direct record_memory calls.")

    def _ensure_provider_credentials(self) -> None:
        _ensure_mem0_provider_credentials(self.mem0_provider)

    def _config_dict(self) -> dict[str, Any]:
        vector_store_config: dict[str, Any] = {
            "collection_name": self.mem0_qdrant_collection_name,
        }
        if self.mem0_qdrant_mode == "local":
            vector_store_config["path"] = str(self._storage_root / "qdrant")
        else:
            vector_store_config["url"] = self._effective_qdrant_url()
            api_key = self._qdrant_api_key()
            if api_key:
                vector_store_config["api_key"] = api_key

        config: dict[str, Any] = {
            "history_db_path": str(self._storage_root / "history.db"),
            "vector_store": {
                "provider": "qdrant",
                "config": vector_store_config,
            },
        }
        llm_config = _mem0_llm_config_dict(
            mem0_provider=self.mem0_provider,
            mem0_model=self.mem0_model,
        )
        if llm_config:
            config["llm"] = {
                "provider": _effective_mem0_llm_provider(self.mem0_provider),
                "config": llm_config,
            }
        return config

    def _effective_qdrant_url(self) -> str:
        if self.mem0_qdrant_mode == "managed":
            runtime_url = os.getenv(MANAGED_MEM0_QDRANT_URL_ENV, "").strip()
            if runtime_url:
                return runtime_url
            if self.mem0_qdrant_url:
                return self.mem0_qdrant_url
            raise RuntimeError(
                "mem0 managed Qdrant mode requires SLEEPER_EVAL_MEM0_QDRANT_URL to be set by the campaign runner."
            )
        if self.mem0_qdrant_mode == "server":
            if not self.mem0_qdrant_url:
                raise RuntimeError("mem0 server Qdrant mode requires mem0_qdrant_url.")
            return self.mem0_qdrant_url
        return ""

    def _qdrant_api_key(self) -> str:
        if not self.mem0_qdrant_api_key_env:
            return ""
        return os.getenv(self.mem0_qdrant_api_key_env, "").strip()

    def _ensure_mem0_telemetry_env(self) -> None:
        if "MEM0_TELEMETRY" in os.environ:
            return
        os.environ["MEM0_TELEMETRY"] = "false"
        self._telemetry_env_forced = True

    def _build_storage_root(self) -> Path:
        mem0_root = os.getenv("SLEEPER_EVAL_MEM0_ROOT")
        if mem0_root:
            base_dir = Path(mem0_root).expanduser()
        else:
            base_dir = Path.cwd() / DEFAULT_MEM0_ROOT_DIR
        storage_root = base_dir / uuid.uuid4().hex
        storage_root.mkdir(parents=True, exist_ok=True)
        return storage_root

    def close(self) -> None:
        if self._memory_client is None:
            return
        try:
            vector_store = getattr(self._memory_client, "vector_store", None)
            if vector_store is not None:
                close_method = getattr(vector_store, "close", None)
                if callable(close_method):
                    close_method()
            db = getattr(self._memory_client, "db", None)
            if db is not None:
                close_method = getattr(db, "close", None)
                if callable(close_method):
                    close_method()
        finally:
            self._memory_client = None

    def _memory_client_with_config(self) -> tuple[Any, dict[str, Any] | None]:
        self._ensure_mem0_telemetry_env()
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "mem0 SDK backend requested but the 'mem0' package is not installed."
            ) from exc

        self._ensure_provider_credentials()
        config = self._config_dict()
        if self._memory_client is None:
            if config is not None:
                self._memory_client = Memory.from_config(config)
            else:
                self._memory_client = Memory()
            _wrap_mem0_json_generation(self._memory_client, record_event=self._record_llm_latency_event)
            _disable_mem0_bm25(self._memory_client)
            (
                self._effective_embedder_provider,
                self._effective_embedder_model,
            ) = _embedder_details(self._memory_client)
        return self._memory_client, config

    def _record_llm_latency_event(self, event: dict[str, Any]) -> None:
        scope_id = event.get("scope_id")
        if not isinstance(scope_id, str) or not scope_id:
            return
        self._latency_events_by_scope.setdefault(scope_id, []).append(event)

    async def seed_existing_memories(self, state: TaskState, memories: list[str]) -> None:
        cleaned_memories = [
            cleaned for memory in memories if (cleaned := _clean_memory_text(memory))
        ]
        store = state.store_as(SavedMemoryStore)
        if not cleaned_memories:
            store.seeded_existing_memories = []
            return
        if store.seeded_existing_memories == cleaned_memories:
            return

        memory, _ = self._memory_client_with_config()
        user_id = _sample_scope_id(state)
        for memory_text in cleaned_memories:
            memory.add(memory_text, user_id=user_id, infer=False)
        store.seeded_existing_memories = list(cleaned_memories)

    async def persist_from_state(self, state: TaskState) -> MemoryBackendResult:
        messages = _state_messages_as_mem0_input(
            state,
            system_prompt=self.system_prompt,
            include_document_content=self.include_document_content,
        )
        _debug_print_mem0_input(messages)
        user_id = _sample_scope_id(state)
        memory, config = self._memory_client_with_config()
        latency_token = _MEM0_LATENCY_CONTEXT.set(
            {
                "scope_id": user_id,
                "sample_id": state.sample_id,
                "phase": "persist_from_state",
                "provider": self.mem0_provider,
                "model": self.mem0_model,
            }
        )
        started = time.perf_counter()
        try:
            response = memory.add(messages, user_id=user_id)
        finally:
            _MEM0_LATENCY_CONTEXT.reset(latency_token)
        persist_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        llm_latency_events = self._latency_events_by_scope.pop(user_id, [])

        saved_memories: list[str] = []
        if isinstance(response, dict):
            raw_items = response.get("results") or response.get("memories") or []
        else:
            raw_items = getattr(response, "results", None) or getattr(response, "memories", [])

        for item in raw_items:
            if isinstance(item, dict):
                candidate = item.get("memory") or item.get("text") or item.get("content")
            else:
                candidate = getattr(item, "memory", None) or getattr(item, "text", None)
            if isinstance(candidate, str) and candidate.strip():
                saved_memories.append(candidate.strip())

        store = state.store_as(SavedMemoryStore)
        manager_latency_summary = {
            "persist_latency_ms": persist_latency_ms,
            "llm_call_count": len(llm_latency_events),
            "llm_latency_ms_total": round(
                sum(
                    float(event.get("duration_ms", 0.0))
                    for event in llm_latency_events
                    if isinstance(event.get("duration_ms"), (int, float))
                ),
                3,
            ),
            "llm_latency_ms_max": round(
                max(
                    (
                        float(event.get("duration_ms", 0.0))
                        for event in llm_latency_events
                        if isinstance(event.get("duration_ms"), (int, float))
                    ),
                    default=0.0,
                ),
                3,
            ),
        }
        raw_backend_result = {
            "mode": "sdk",
            "user_id": user_id,
            "mem0_provider": self.mem0_provider,
            "mem0_model": self.mem0_model,
            "mem0_requested_thinking": self.mem0_thinking,
            "mem0_requested_reasoning_effort": self.mem0_reasoning_effort,
            "mem0_effective_thinking": (
                "enabled" if self.mem0_provider == "deepseek" else self.mem0_thinking
            ),
            "mem0_effective_reasoning_effort": (
                "high"
                if self.mem0_provider == "deepseek"
                else self.mem0_reasoning_effort
            ),
            "mem0_qdrant_mode": self.mem0_qdrant_mode,
            "mem0_qdrant_url": (
                self._effective_qdrant_url()
                if self.mem0_qdrant_mode in {"server", "managed"}
                else ""
            ),
            "mem0_qdrant_collection_name": self.mem0_qdrant_collection_name,
            "mem0_telemetry_enabled": mem0_telemetry_enabled_from_env(),
            "mem0_telemetry_forced_off_by_eval": self._telemetry_env_forced,
            "mem0_embedder_provider": self._effective_embedder_provider,
            "mem0_embedder_model": self._effective_embedder_model,
            "mem0_bm25_enabled": False,
            "mem0_config": _sanitize_nested_config(config),
            "input_messages": messages,
            "seeded_existing_memories": list(store.seeded_existing_memories),
            "response": response,
            "mem0_manager_latency": manager_latency_summary,
            "mem0_manager_llm_calls": llm_latency_events,
        }
        _append_mem0_latency_jsonl(
            {
                "sample_id": state.sample_id,
                "dataset_label": state.metadata.get("dataset_label")
                if isinstance(state.metadata, dict)
                else None,
                "backend": "mem0",
                "provider": self.mem0_provider,
                "model": self.mem0_model,
                "saved_memory_count": len(saved_memories),
                **manager_latency_summary,
            }
        )
        return _write_store_result(
            store,
            backend_name=self.backend_name,
            runtime_name=self.runtime_name,
            saved_memories=saved_memories,
            raw_backend_result=raw_backend_result,
        )

    def result_for_state(self, state: TaskState) -> MemoryBackendResult:
        return _result_from_store(state.store_as(SavedMemoryStore))


def validate_memory_backend_name(memory_backend: str) -> str:
    if memory_backend not in VALID_MEMORY_BACKENDS:
        raise ValueError(
            f"Unknown memory backend '{memory_backend}'. Available: {sorted(VALID_MEMORY_BACKENDS)}"
        )
    return memory_backend


def validate_mem0_runtime(mem0_runtime: str) -> str:
    if mem0_runtime not in VALID_MEM0_RUNTIMES:
        raise ValueError(
            f"Unknown mem0 runtime '{mem0_runtime}'. Available: {sorted(VALID_MEM0_RUNTIMES)}"
        )
    return mem0_runtime


def validate_mem0_provider(mem0_provider: str) -> str:
    if mem0_provider not in VALID_MEM0_PROVIDERS:
        raise ValueError(
            f"Unknown mem0 provider '{mem0_provider}'. Available: {sorted(VALID_MEM0_PROVIDERS)}"
        )
    return mem0_provider


def validate_mem0_thinking(mem0_thinking: str) -> str:
    if mem0_thinking not in VALID_MEM0_THINKING:
        raise ValueError(
            f"Unknown mem0 thinking setting '{mem0_thinking}'. "
            f"Available: {sorted(VALID_MEM0_THINKING)}"
        )
    return mem0_thinking


def validate_mem0_reasoning_effort(mem0_reasoning_effort: str) -> str:
    if mem0_reasoning_effort not in VALID_MEM0_REASONING_EFFORTS:
        raise ValueError(
            f"Unknown mem0 reasoning_effort '{mem0_reasoning_effort}'. "
            f"Available: {sorted(VALID_MEM0_REASONING_EFFORTS)}"
        )
    return mem0_reasoning_effort


def validate_mem0_qdrant_mode(mem0_qdrant_mode: str) -> str:
    if mem0_qdrant_mode not in VALID_MEM0_QDRANT_MODES:
        raise ValueError(
            f"Unknown mem0 qdrant mode '{mem0_qdrant_mode}'. "
            f"Available: {sorted(VALID_MEM0_QDRANT_MODES)}"
        )
    return mem0_qdrant_mode


def resolve_memory_backend(
    memory_backend: str = DEFAULT_MEMORY_BACKEND_NAME,
    *,
    mem0_runtime: str = DEFAULT_MEM0_RUNTIME,
    mem0_provider: str = DEFAULT_MEM0_PROVIDER,
    mem0_model: str = DEFAULT_MEM0_MODEL,
    mem0_prompt_variant: str = DEFAULT_MEM0_PROMPT_VARIANT,
    mem0_thinking: str = DEFAULT_MEM0_THINKING,
    mem0_reasoning_effort: str = DEFAULT_MEM0_REASONING_EFFORT,
    mem0_qdrant_mode: str = DEFAULT_MEM0_QDRANT_MODE,
    mem0_qdrant_url: str = DEFAULT_MEM0_QDRANT_URL,
    mem0_qdrant_api_key_env: str = DEFAULT_MEM0_QDRANT_API_KEY_ENV,
    mem0_qdrant_collection_name: str = DEFAULT_MEM0_QDRANT_COLLECTION_NAME,
    mem0_system_prompt: str | None = None,
    mem0_include_document_content: bool = DEFAULT_MEM0_INCLUDE_DOCUMENT_CONTENT,
) -> MemoryBackend:
    memory_backend = validate_memory_backend_name(memory_backend)
    mem0_runtime = validate_mem0_runtime(mem0_runtime)
    mem0_provider = validate_mem0_provider(mem0_provider)
    mem0_prompt_variant = validate_mem0_prompt_variant(mem0_prompt_variant)
    mem0_thinking = validate_mem0_thinking(mem0_thinking)
    mem0_reasoning_effort = validate_mem0_reasoning_effort(mem0_reasoning_effort)
    mem0_qdrant_mode = validate_mem0_qdrant_mode(mem0_qdrant_mode)

    if memory_backend == "tool":
        return ToolMemoryBackend()
    if mem0_runtime == "local":
        return Mem0LocalBackend(
            system_prompt=mem0_system_prompt,
            include_document_content=mem0_include_document_content,
        )
    if mem0_runtime == "transcript_only":
        return Mem0TranscriptOnlyBackend(
            system_prompt=mem0_system_prompt,
            include_document_content=mem0_include_document_content,
        )
    if mem0_runtime == "prompt_only":
        return Mem0PromptOnlyBackend(
            mem0_provider=mem0_provider,
            mem0_model=mem0_model,
            mem0_prompt_variant=mem0_prompt_variant,
            mem0_thinking=mem0_thinking,
            mem0_reasoning_effort=mem0_reasoning_effort,
            system_prompt=mem0_system_prompt,
            include_document_content=mem0_include_document_content,
        )
    return Mem0SdkBackend(
        mem0_provider=mem0_provider,
        mem0_model=mem0_model,
        mem0_thinking=mem0_thinking,
        mem0_reasoning_effort=mem0_reasoning_effort,
        mem0_qdrant_mode=mem0_qdrant_mode,
        mem0_qdrant_url=mem0_qdrant_url,
        mem0_qdrant_api_key_env=mem0_qdrant_api_key_env,
        mem0_qdrant_collection_name=mem0_qdrant_collection_name,
        system_prompt=mem0_system_prompt,
        include_document_content=mem0_include_document_content,
    )


def result_for_state(state: TaskState) -> MemoryBackendResult:
    return _result_from_store(state.store_as(SavedMemoryStore))


DEFAULT_MEMORY_BACKEND = ToolMemoryBackend()


def initialize_memory_store(
    state: TaskState,
    *,
    provider_id: str,
    tool_name: str | None,
    current_memories: list[str],
) -> SavedMemoryStore:
    """Seed per-sample store metadata before generation."""

    store = state.store_as(SavedMemoryStore)
    store.provider_id = provider_id
    store.tool_name = tool_name
    store.current_memories = list(current_memories)
    if store.mem0_scope_id is None:
        store.mem0_scope_id = f"sleeper-eval-{state.uuid}"
    return store
