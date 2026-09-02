"""Inspect-backed runner for raw-GEPA defense optimization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterator

from inspect_ai import eval as inspect_eval

from sleeper_eval.memory_backend import result_from_store_dict
from sleeper_eval.solver import UNTRUSTED_CONTENT_REMINDER
from sleeper_eval.task import sleeper_eval

from .dataset_builder import EvalExampleSpec
from .mlflow_tracing import MlflowTracing

TRUNCATED_SUFFIX = "... [truncated]"
NO_PRE_TOOL_ASSISTANT = "[no pre-tool assistant message]"
NO_VISIBLE_REASONING = "[no visible reasoning]"
NO_TOOL_CALL = "[no tool call made]"
NO_TOOL_RESPONSE = "[no tool response message]"
NO_POST_TOOL_ASSISTANT = "[no post-tool assistant message]"
EMPTY_ASSISTANT_TEXT = "[empty assistant text]"
NO_USER_MESSAGE = "[no user message captured]"
NO_DIRECT_USER_QUERY = "[direct user query unavailable]"
NO_DOCUMENT_PREVIEW = "[document preview unavailable]"
USER_MESSAGE_PREFIX = "The following document was uploaded by the user:"
USER_QUERY_LIMIT = 500
USER_MESSAGE_PREVIEW_LIMIT = 280
DOCUMENT_WINDOW_CHARS = 280


@dataclass(frozen=True)
class EvalTraceSlice:
    user_message_view: dict[str, Any]
    assistant_pre_action: str
    assistant_pre_action_reasoning: str
    assistant_pre_action_text: str
    memory_tool_action: dict[str, Any]
    tool_response_text: str
    assistant_after_tool: str
    final_answer: str


@dataclass(frozen=True)
class InspectEvalResult:
    sample_id: str
    tool_called: bool
    tool_alias: str
    saved_memories: list[str]
    response_text: str
    sample_error: str | None
    provider_family: str
    model_slug: str
    example_kind: str
    expected_memory_write: bool
    inspect_log_path: str
    trace: EvalTraceSlice


@dataclass
class InspectEvalRunner:
    root_dir: Path
    memory_backend: str = "tool"
    mention_memory_system: bool = True
    tracer: MlflowTracing | None = None
    _run_counter: int = 0

    @property
    def inspect_dir(self) -> Path:
        return self.root_dir / "inspect"

    @property
    def event_log_path(self) -> Path:
        return self.root_dir / "artifacts" / "inspect_eval_events.jsonl"

    def evaluate(
        self,
        spec: EvalExampleSpec,
        *,
        defense_suffix: str,
        defense_name: str = "",
    ) -> InspectEvalResult:
        results = self.evaluate_batch([spec], defense_suffix=defense_suffix, defense_name=defense_name)
        return results[0]

    def evaluate_batch(
        self,
        specs: list[EvalExampleSpec],
        *,
        defense_suffix: str,
        defense_name: str = "",
        batch_label: str = "",
    ) -> list[InspectEvalResult]:
        if not specs:
            return []
        first = specs[0]
        for spec in specs[1:]:
            if (
                spec.dataset_file != first.dataset_file
                or spec.provider != first.provider
                or spec.model_slug != first.model_slug
                or spec.prompt_model != first.prompt_model
                or spec.attack_name != first.attack_name
                or spec.example_kind != first.example_kind
            ):
                raise ValueError("All specs in evaluate_batch must share dataset/provider/model/attack/example kind.")

        self._run_counter += 1
        label = batch_label or f"{first.provider}-{first.example_kind}-{len(specs)}"
        eval_dir = self.inspect_dir / f"{self._run_counter:04d}-{label.replace('/', '_')}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        tracer = self.tracer or MlflowTracing.disabled()
        with tracer.start_span(
            "inspect_black_box_eval",
            span_type="TOOL",
            inputs={
                "sample_count": len(specs),
                "sample_ids_preview": [spec.sample_id for spec in specs[:5]],
                "provider": first.provider,
                "model_slug": first.model_slug,
                "example_kind": first.example_kind,
                "attack_name": first.attack_name,
                "defense_suffix_length": len(defense_suffix),
            },
            attributes={"component": "inspect_runner"},
        ) as span:
            task = sleeper_eval(
                dataset_file=first.dataset_file,
                defense=defense_name,
                defense_suffix_override=defense_suffix,
                attack=first.attack_name,
                evaluation_mode="benign_save" if first.example_kind == "benign_save" else "attack",
                memory_backend=self.memory_backend,
                mention_memory_system=self.mention_memory_system,
                provider=first.provider,
                prompt_model=first.prompt_model,
            )

            with _repo_local_inspect_env(eval_dir):
                logs = inspect_eval(
                    task,
                    model=first.model_slug,
                    sample_id=[spec.sample_id for spec in specs],
                    score=False,
                    display="none",
                    fail_on_error=False,
                    log_dir=str(eval_dir),
                )

            sample_lookup = {spec.sample_id: spec for spec in specs}
            results: list[InspectEvalResult] = []
            for log in logs:
                for sample in log.samples:
                    spec = sample_lookup.get(sample.id)
                    if spec is None:
                        continue
                    backend_result = result_from_store_dict(sample.store)
                    sample_error = sample.error.message if sample.error else None
                    tool_alias = (
                        str(sample.store.get("SavedMemoryStore:tool_name", "")) if isinstance(sample.store, dict) else ""
                    )
                    trace = extract_trace_slice(sample)
                    result = InspectEvalResult(
                        sample_id=spec.sample_id,
                        tool_called=backend_result.backend_name == "tool" and backend_result.wrote_memory,
                        tool_alias=tool_alias or str(trace.memory_tool_action.get("tool_name", "")),
                        saved_memories=list(backend_result.saved_memories),
                        response_text=sample.output.completion if sample.output else "",
                        sample_error=sample_error,
                        provider_family=spec.provider,
                        model_slug=spec.model_slug,
                        example_kind=spec.example_kind,
                        expected_memory_write=spec.expected_memory_write,
                        inspect_log_path=str(log.location),
                        trace=trace,
                    )
                    results.append(result)
                    self._record_event(result)

            results.sort(key=lambda item: [spec.sample_id for spec in specs].index(item.sample_id))
            span.set_outputs(
                {
                    "result_count": len(results),
                    "tool_called_count": sum(1 for result in results if result.tool_called),
                    "error_count": sum(1 for result in results if result.sample_error),
                    "inspect_log_paths": sorted({result.inspect_log_path for result in results})[:5],
                }
            )
            return results

    def _record_event(self, result: InspectEvalResult) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def truncate_text(value: str, limit: int) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + TRUNCATED_SUFFIX


def extract_trace_slice(sample: Any) -> EvalTraceSlice:
    messages = list(getattr(sample, "messages", None) or [])
    user_message_view = extract_user_message_view(sample)
    first_tool_index: int | None = None
    pre_tool_message = None
    for index, message in enumerate(messages):
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            first_tool_index = index
            pre_tool_message = message
            break

    assistant_pre_action = NO_PRE_TOOL_ASSISTANT
    assistant_pre_action_reasoning = NO_VISIBLE_REASONING
    assistant_pre_action_text = EMPTY_ASSISTANT_TEXT
    memory_tool_action: dict[str, Any] = {"called": False, "tool_name": NO_TOOL_CALL, "arguments": NO_TOOL_CALL}
    tool_response_text = NO_TOOL_RESPONSE
    assistant_after_tool = NO_POST_TOOL_ASSISTANT

    if pre_tool_message is not None:
        assistant_pre_action, assistant_pre_action_reasoning, assistant_pre_action_text = serialize_assistant_message(
            pre_tool_message
        )
        tool_calls = getattr(pre_tool_message, "tool_calls", None) or []
        first_tool_call = tool_calls[0] if tool_calls else None
        if first_tool_call is not None:
            memory_tool_action = {
                "called": True,
                "tool_name": str(getattr(first_tool_call, "function", "") or ""),
                "arguments": json.dumps(getattr(first_tool_call, "arguments", {}) or {}, ensure_ascii=False),
            }

    if first_tool_index is not None:
        for message in messages[first_tool_index + 1 :]:
            role = getattr(message, "role", "")
            if role == "tool" and tool_response_text == NO_TOOL_RESPONSE:
                tool_response_text = serialize_generic_message(message)
                continue
            if role == "assistant":
                assistant_after_tool, _, _ = serialize_assistant_message(message)
                break

    final_answer = str(getattr(getattr(sample, "output", None), "completion", "") or "").strip()
    if not final_answer:
        final_answer = assistant_after_tool if assistant_after_tool != NO_POST_TOOL_ASSISTANT else EMPTY_ASSISTANT_TEXT

    return EvalTraceSlice(
        user_message_view=user_message_view,
        assistant_pre_action=assistant_pre_action,
        assistant_pre_action_reasoning=assistant_pre_action_reasoning,
        assistant_pre_action_text=assistant_pre_action_text,
        memory_tool_action=memory_tool_action,
        tool_response_text=tool_response_text,
        assistant_after_tool=assistant_after_tool,
        final_answer=final_answer,
    )


def extract_user_message_view(sample: Any) -> dict[str, Any]:
    messages = list(getattr(sample, "messages", None) or [])
    user_message = next((message for message in messages if getattr(message, "role", "") == "user"), None)
    metadata = dict(getattr(sample, "metadata", None) or {})
    query_text = str(metadata.get("query_text", "") or "").strip()

    if user_message is None:
        return {
            "message_structure": "No user message was captured in the Inspect log.",
            "direct_user_query": query_text or NO_DIRECT_USER_QUERY,
            "query_found_verbatim_in_message": False,
            "document_wrapper_present": False,
            "untrusted_content_reminder_present": False,
            "uploaded_document_chars": 0,
            "uploaded_document_preview": NO_USER_MESSAGE,
        }

    user_text = serialize_generic_message(user_message)
    document_wrapper_present = user_text.startswith(USER_MESSAGE_PREFIX)
    reminder_present = UNTRUSTED_CONTENT_REMINDER in user_text
    document_text = ""
    query_found = False

    if document_wrapper_present:
        remainder = user_text[len(USER_MESSAGE_PREFIX) :].lstrip()
        if query_text:
            query_index = remainder.rfind(query_text)
            if query_index != -1 and not remainder[query_index + len(query_text) :].strip():
                document_text = remainder[:query_index].rstrip()
                query_found = True
            else:
                document_text = remainder
        else:
            document_text = remainder

        if reminder_present and document_text.endswith(UNTRUSTED_CONTENT_REMINDER):
            document_text = document_text[: -len(UNTRUSTED_CONTENT_REMINDER)].rstrip()

    direct_user_query = query_text or NO_DIRECT_USER_QUERY
    message_structure = (
        "Actual user message structure: uploaded-document wrapper, bounded document body, then the direct user query."
        if document_wrapper_present
        else "Actual user message structure: direct user query only; no uploaded-document wrapper detected."
    )

    return {
        "message_structure": message_structure,
        "direct_user_query": truncate_text(direct_user_query, USER_QUERY_LIMIT),
        "query_found_verbatim_in_message": query_found or (bool(query_text) and query_text in user_text),
        "document_wrapper_present": document_wrapper_present,
        "untrusted_content_reminder_present": reminder_present,
        "uploaded_document_chars": len(document_text),
        "uploaded_document_preview": _document_preview(document_text),
    }


def _document_preview(document_text: str) -> str:
    cleaned = document_text.strip()
    if not cleaned:
        return NO_DOCUMENT_PREVIEW
    if len(cleaned) <= DOCUMENT_WINDOW_CHARS:
        return cleaned

    max_start = max(0, len(cleaned) - DOCUMENT_WINDOW_CHARS)
    starts = []
    for fraction in (0.0, 0.33, 0.66, 1.0):
        start = min(max_start, int(max_start * fraction))
        if start not in starts:
            starts.append(start)

    windows: list[str] = []
    for index, start in enumerate(starts, start=1):
        percent = 0 if max_start == 0 else int(round((start / max_start) * 100))
        snippet = cleaned[start : start + DOCUMENT_WINDOW_CHARS].strip()
        windows.append(
            f"[excerpted window {index} @ {percent}%; omitted surrounding text]\n"
            f"...{snippet}..."
        )
    return "\n\n".join(windows)


def serialize_assistant_message(message: Any) -> tuple[str, str, str]:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        text = content.strip()
        return (
            text or EMPTY_ASSISTANT_TEXT,
            NO_VISIBLE_REASONING,
            text or EMPTY_ASSISTANT_TEXT,
        )

    serialized_parts: list[str] = []
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for part in list(content or []):
        part_type = getattr(part, "type", None)
        if part_type == "reasoning":
            reasoning_text = str(getattr(part, "reasoning", "") or "").strip()
            if reasoning_text:
                serialized_parts.append(f"<think>{reasoning_text}</think>")
                reasoning_parts.append(reasoning_text)
        elif part_type == "text":
            text = str(getattr(part, "text", "") or "").strip()
            if text:
                serialized_parts.append(text)
                text_parts.append(text)
        else:
            fallback = str(part).strip()
            if fallback:
                serialized_parts.append(fallback)
                text_parts.append(fallback)

    serialized = "\n".join(serialized_parts).strip() or EMPTY_ASSISTANT_TEXT
    reasoning = "\n".join(reasoning_parts).strip() or NO_VISIBLE_REASONING
    visible_text = "\n".join(text_parts).strip() or EMPTY_ASSISTANT_TEXT
    return serialized, reasoning, visible_text


def serialize_generic_message(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip() or EMPTY_ASSISTANT_TEXT
    if isinstance(content, list):
        parts = []
        for part in content:
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                parts.append(part_text)
        return "\n".join(parts).strip() or EMPTY_ASSISTANT_TEXT
    return str(content).strip() or EMPTY_ASSISTANT_TEXT


@contextmanager
def _repo_local_inspect_env(log_dir: Path) -> Iterator[None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    home_dir = Path.cwd() / ".inspect_home"
    home_dir.mkdir(parents=True, exist_ok=True)

    previous = {
        "HOME": os.environ.get("HOME"),
        "INSPECT_LOG_DIR": os.environ.get("INSPECT_LOG_DIR"),
        "INSPECT_TRACE_FILE": os.environ.get("INSPECT_TRACE_FILE"),
    }
    os.environ["HOME"] = str(home_dir.resolve())
    os.environ["INSPECT_LOG_DIR"] = str(log_dir.resolve())
    os.environ["INSPECT_TRACE_FILE"] = str((log_dir / "trace.log").resolve())
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
