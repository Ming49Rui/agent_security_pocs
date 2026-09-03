from __future__ import annotations

import hashlib
from pathlib import Path

from inspect_ai.dataset._dataset import MemoryDataset
from inspect_ai.dataset import Sample

from sleeper_eval.task import sleeper_eval


def _stub_dataset() -> list[Sample]:
    return [Sample(input="prompt", target="goal", metadata={})]


def _stub_memory_dataset() -> MemoryDataset:
    return MemoryDataset(
        samples=[
            Sample(input="prompt-1", target="goal-1", metadata={}, id="a"),
            Sample(input="prompt-2", target="goal-2", metadata={}, id="b"),
            Sample(input="prompt-3", target="goal-3", metadata={}, id="c"),
            Sample(input="prompt-4", target="goal-4", metadata={}, id="d"),
        ],
        name="stub",
    )


def _set_model_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")


def test_task_uses_tool_solver_path_by_default(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval()
    prompt_template = (
        Path("sleeper_eval/prompts/provider/generic.md").read_text(encoding="utf-8").strip()
    )

    assert len(task.solver) == 3
    assert len(task.scorer) == 2
    assert task.metadata["memory_backend"] == "tool"
    assert task.metadata["mem0_runtime"] == "tool"
    assert task.metadata["provider"] == "generic"
    assert task.metadata["model_api"] == ""
    assert task.metadata["model_route_type"] == "unknown"
    assert task.metadata["model_vendor"] == ""
    assert task.metadata["system_prompt_file"] == "generic.md"
    assert task.metadata["system_prompt_path"] == "sleeper_eval/prompts/provider/generic.md"
    assert task.metadata["system_prompt_template_sha256"] == hashlib.sha256(
        prompt_template.encode("utf-8")
    ).hexdigest()
    assert task.metadata["system_prompt_template_chars"] == len(prompt_template)
    assert task.metadata["memory_tool_name"] == "save_memory"
    assert task.metadata["memory_placement"] == "system_prompt_end"
    assert task.scorer[0].__registry_info__.name == "tool_call_scorer"
    assert task.scorer[1].__registry_info__.name == "goal_match_scorer"


def test_task_uses_mem0_solver_path_without_tool_solver(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval(
        memory_backend="mem0",
        mem0_runtime="local",
        mem0_provider="anthropic",
        mem0_model="openai/gpt-4.1-mini",
        mem0_include_document_content=False,
        mem0_thinking="enabled",
        mem0_reasoning_effort="high",
    )

    assert len(task.solver) == 3
    assert len(task.scorer) == 2
    assert task.metadata["memory_backend"] == "mem0"
    assert task.metadata["mem0_runtime"] == "local"
    assert task.metadata["mem0_provider"] == "anthropic"
    assert task.metadata["mem0_model"] == "openai/gpt-4.1-mini"
    assert task.metadata["mem0_prompt_variant"] == "mem0_default"
    assert task.metadata["mem0_include_document_content"] is False
    assert task.metadata["mem0_thinking"] == "enabled"
    assert task.metadata["mem0_reasoning_effort"] == "high"
    assert task.metadata["mem0_qdrant_mode"] == "local"
    assert task.metadata["mem0_qdrant_url"] == ""
    assert task.metadata["mem0_qdrant_collection_name"] == ""
    assert task.scorer[0].__registry_info__.name == "mem0_write_scorer"
    assert task.scorer[1].__registry_info__.name == "mem0_goal_match_scorer"


def test_task_uses_deepseek_mem0_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval(memory_backend="mem0")

    assert task.metadata["mem0_provider"] == "deepseek"
    assert task.metadata["mem0_model"] == "deepseek-v4-pro"
    assert task.metadata["mem0_thinking"] == "enabled"
    assert task.metadata["mem0_reasoning_effort"] == "high"
    assert task.metadata["mem0_qdrant_mode"] == "local"


def test_task_records_mem0_qdrant_metadata(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval(
        memory_backend="mem0",
        mem0_runtime="sdk",
        mem0_qdrant_mode="server",
        mem0_qdrant_url="http://127.0.0.1:6333",
        mem0_qdrant_collection_name="campaign-mem0",
    )

    assert task.metadata["mem0_qdrant_mode"] == "server"
    assert task.metadata["mem0_qdrant_url"] == "http://127.0.0.1:6333"
    assert task.metadata["mem0_qdrant_collection_name"] == "campaign-mem0"
    assert task.metadata["mem0_telemetry_enabled"] is False
    assert task.metadata["mem0_embedder_provider"] == "openai"
    assert task.metadata["mem0_embedder_model"] == "text-embedding-3-small"
    assert task.metadata["mem0_bm25_enabled"] is False


def test_task_records_mem0_prompt_variant(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval(
        memory_backend="mem0",
        mem0_runtime="prompt_only",
        mem0_prompt_variant="lenient_assistant",
    )

    assert task.metadata["mem0_runtime"] == "prompt_only"
    assert task.metadata["mem0_prompt_variant"] == "lenient_assistant"


def test_task_accepts_transcript_only_mem0_runtime(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval(
        memory_backend="mem0",
        mem0_runtime="transcript_only",
        mem0_provider="gemini",
        mem0_model="gemini-3.1-flash-lite-preview",
    )

    assert task.metadata["mem0_runtime"] == "transcript_only"


def test_task_uses_benign_save_mode_with_separate_semantic_scorer(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    task = sleeper_eval(
        evaluation_mode="benign_save",
        prompt_model="openrouter/moonshotai/kimi-k2.5",
    )

    assert task.metadata["evaluation_mode"] == "benign_save"
    assert task.model is not None
    assert task.scorer[0].__registry_info__.name == "tool_call_scorer"
    assert task.scorer[1].__registry_info__.name == "benign_save_goal_match_scorer"


def test_task_metadata_changes_with_provider(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    claude_task = sleeper_eval(
        provider="claude",
        prompt_model="openrouter/anthropic/claude-haiku-4.5",
    )
    gpt_task = sleeper_eval(
        provider="gpt",
        prompt_model="openrouter/openai/gpt-5-mini",
    )
    gemini_task = sleeper_eval(
        provider="gemini",
        prompt_model="openrouter/google/gemini-2.5-flash-lite",
    )

    assert claude_task.metadata["memory_tool_name"] == "memory_user_edits"
    assert claude_task.metadata["memory_placement"] == "user_memories_tag"
    assert claude_task.metadata["model_api"] == "openrouter"
    assert claude_task.metadata["model_route_type"] == "openrouter"
    assert claude_task.metadata["model_vendor"] == "anthropic"
    assert gpt_task.metadata["memory_tool_name"] == "bio"
    assert gpt_task.metadata["memory_placement"] == "model_set_context"
    assert gpt_task.metadata["model_api"] == "openrouter"
    assert gpt_task.metadata["model_route_type"] == "openrouter"
    assert gpt_task.metadata["model_vendor"] == "openai"
    assert gemini_task.metadata["memory_tool_name"] == "save_memory"
    assert gemini_task.metadata["memory_placement"] == "system_prompt_end"
    assert gemini_task.metadata["model_api"] == "openrouter"
    assert gemini_task.metadata["model_route_type"] == "openrouter"
    assert gemini_task.metadata["model_vendor"] == "google"


def test_task_metadata_records_native_model_route(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    task = sleeper_eval(
        prompt_model="openai/gpt-5.4",
    )

    assert task.metadata["provider"] == "gpt"
    assert task.metadata["model_api"] == "openai"
    assert task.metadata["model_route_type"] == "native"
    assert task.metadata["model_vendor"] == "openai"
    assert task.metadata["system_prompt_file"] == "gpt.md"


def test_task_metadata_records_vertex_model_route(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    task = sleeper_eval(
        prompt_model="google/vertex/gemini-3.1-pro-preview",
    )

    assert task.metadata["provider"] == "gemini"
    assert task.metadata["model_api"] == "google"
    assert task.metadata["model_route_type"] == "vertex"
    assert task.metadata["model_vendor"] == "google"
    assert task.metadata["system_prompt_file"] == "gemini.md"


def test_task_metadata_records_prompt_override_details(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    task = sleeper_eval(
        provider="claude",
        prompt_model="anthropic/claude-sonnet-4-6",
        system_prompt_file="claude_truncated.md",
    )
    prompt_template = (
        Path("sleeper_eval/prompts/provider/claude_truncated.md")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert task.metadata["system_prompt_file"] == "claude_truncated.md"
    assert (
        task.metadata["system_prompt_path"]
        == "sleeper_eval/prompts/provider/claude_truncated.md"
    )
    assert task.metadata["system_prompt_template_sha256"] == hashlib.sha256(
        prompt_template.encode("utf-8")
    ).hexdigest()
    assert task.metadata["system_prompt_template_chars"] == len(prompt_template)


def test_task_builtin_gepa_defense_resolves_effective_suffix(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())

    task = sleeper_eval(defense="gepa_prompt_hardening_suffix")

    assert (
        "### The Core Rule for Memory Writes"
        in task.metadata["defense_suffix_override"]
    )
    assert task.metadata["gepa_prompt_hardening_suffix"] is True


def test_task_records_explicit_reasoning_settings(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    task = sleeper_eval(
        prompt_model="openai/gpt-5.4",
        reasoning_effort="medium",
        reasoning_tokens=2048,
    )

    assert task.config.reasoning_effort == "medium"
    assert task.config.reasoning_tokens == 2048
    assert task.metadata["requested_reasoning_effort"] == "medium"
    assert task.metadata["requested_reasoning_tokens"] == 2048


def test_task_records_batch_settings(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_dataset())
    _set_model_env(monkeypatch)

    task = sleeper_eval(
        prompt_model="google/gemini-3.1-pro-preview",
        batch={
            "size": 1000,
            "max_size": 1000,
            "send_delay": 60,
            "tick": 60,
            "max_batches": 2,
        },
    )

    assert task.config.batch is not None
    assert task.config.batch.size == 1000
    assert task.config.batch.max_size == 1000
    assert task.config.batch.send_delay == 60
    assert task.config.batch.tick == 60
    assert task.config.batch.max_batches == 2
    assert task.metadata["requested_batch"] == {
        "size": 1000,
        "max_size": 1000,
        "send_delay": 60.0,
        "tick": 60.0,
        "max_batches": 2,
    }


def test_task_can_shuffle_dataset_with_seed(monkeypatch) -> None:
    monkeypatch.setattr("sleeper_eval.task.load_dataset", lambda _: _stub_memory_dataset())

    task = sleeper_eval(
        sample_shuffle=True,
        sample_shuffle_seed=7,
    )

    ordered_ids = [sample.id for sample in task.dataset]

    assert ordered_ids != ["a", "b", "c", "d"]
    assert task.metadata["requested_sample_shuffle"] is True
    assert task.metadata["requested_sample_shuffle_seed"] == 7
