from __future__ import annotations

import asyncio
import json
import re

from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
import pytest

from sleeper_eval.attacks.none import apply as apply_none
from sleeper_eval.attacks.universal_v1 import apply as apply_universal
from sleeper_eval.dataset import load_dataset
from sleeper_eval.defenses import defense_config_from_names
from sleeper_eval.provider_config import get_provider_config
from sleeper_eval.solver import compose_prompt

UNRESOLVED_TEMPLATE_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def make_state(sample, model: str) -> TaskState:
    return TaskState(
        model=model,
        sample_id=sample.id,
        epoch=0,
        input=sample.input,
        messages=[ChatMessageUser(content=sample.input)],
        target=Target(sample.target),
        metadata=sample.metadata,
    )


def run_solver(
    sample,
    defense_names: str,
    attack_fn,
    *,
    provider: str = "generic",
    model: str = "mockllm/model",
    memory_backend_name: str = "tool",
    memory_backend=None,
):
    state = make_state(sample, model)
    solver = compose_prompt(
        defense_config_from_names(defense_names),
        attack_fn,
        provider_config=get_provider_config(
            provider,
            model_slug=model,
            memory_backend_name=memory_backend_name,
        ),
        memory_backend_name=memory_backend_name,
        memory_backend=memory_backend,
    )
    return asyncio.run(solver(state, None))


class RecordingMemoryBackend:
    backend_name = "mem0"
    runtime_name = "sdk"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def record_memory(self, memory_text: str) -> str:
        return memory_text

    async def seed_existing_memories(self, state, memories: list[str]) -> None:
        self.calls.append((state.sample_id, list(memories)))

    async def persist_from_state(self, state):
        raise NotImplementedError

    def result_for_state(self, state):
        raise NotImplementedError


def test_solver_embeds_claude_user_memories_tag(dev_with_memories_sample) -> None:
    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="claude",
        model="openrouter/anthropic/claude-haiku-4.5",
    )

    assert "<memory_system>" in state.messages[0].text
    assert "<userMemories>" in state.messages[0].text
    assert dev_with_memories_sample.metadata["memories"][0] in state.messages[0].text


def test_solver_mem0_path_seeds_existing_memories_before_generation(
    dev_with_memories_sample,
) -> None:
    backend = RecordingMemoryBackend()

    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="generic",
        model="mockllm/model",
        memory_backend_name="mem0",
        memory_backend=backend,
    )

    assert backend.calls == [
        (dev_with_memories_sample.id, list(dev_with_memories_sample.metadata["memories"]))
    ]
    assert state.messages[0].text


def test_solver_mem0_gpt_uses_provider_prompt_variant_and_model_set_context(
    dev_with_memories_sample,
) -> None:
    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="gpt",
        model="openai/gpt-5.4-mini",
        memory_backend_name="mem0",
    )

    assert "## Memory" in state.messages[0].text
    assert "## bio" not in state.messages[0].text
    assert state.messages[1].text.startswith("# Model Set Context")


def test_solver_claude_uses_antml_text_document_representation(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "html",
                "doc_metadata_filename": "example.html",
                "doc_metadata_mime": "text/html",
                "doc_text": "<html><body>Example</body></html>",
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="claude",
        model="anthropic/claude-sonnet-4-6",
    )

    assert '<antml:document index="1" media_type="text/plain">' in state.messages[1].text
    assert "<antml:source>example.html</antml:source>" in state.messages[1].text
    assert (
        "<antml:document_content><html><body>Example</body></html></antml:document_content>"
        in state.messages[1].text
    )
    assert "</antml:document>" in state.messages[1].text


def test_solver_claude_omits_source_without_filename_metadata(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "email",
                "doc_metadata_filename": None,
                "doc_metadata_local_path": None,
                "doc_text": "Subject: Test\nBody: Hello",
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="claude",
        model="anthropic/claude-sonnet-4-6",
    )

    assert '<antml:document index="1" media_type="text/plain">' in state.messages[1].text
    assert "<antml:source>" not in state.messages[1].text


def test_solver_claude_renders_pdf_pages_without_source(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "pdf",
                "doc_metadata_mime": "application/pdf",
                "doc_pdf_pages": ["Page one text", "Page two text"],
                "doc_text": "Page one text\nPage two text",
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="claude",
        model="anthropic/claude-sonnet-4-6",
    )

    assert '<antml:document index="1" media_type="application/pdf">' in state.messages[1].text
    assert (
        '<antml:document_content page="1">Page one text</antml:document_content>'
        in state.messages[1].text
    )
    assert (
        '<antml:document_content page="2">Page two text</antml:document_content>'
        in state.messages[1].text
    )
    assert "<antml:source>" not in state.messages[1].text


def test_solver_claude_long_documents_are_truncated_inside_antml(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "text",
                "doc_text": "A" * 20_500,
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="claude",
        model="anthropic/claude-sonnet-4-6",
    )

    assert '<antml:document index="1" media_type="text/plain">' in state.messages[1].text
    assert "[... truncated " in state.messages[1].text


def test_solver_gpt_uses_raw_text_document_representation(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "html",
                "doc_text": "<html><body>Example</body></html>",
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="gpt",
        model="openai/gpt-5.4",
    )

    assert state.messages[2].text.startswith("The following document was uploaded by the user:")
    assert "<html><body>Example</body></html>" in state.messages[2].text
    assert "<document" not in state.messages[2].text
    assert "snippetFromFront" not in state.messages[2].text


def test_solver_gpt_renders_pdf_with_page_markers(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "pdf",
                "doc_pdf_pages": ["Page one text", "Page two text"],
                "doc_text": "Page one text\nPage two text",
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="gpt",
        model="openai/gpt-5.4",
    )

    assert "<PARSED TEXT FOR PAGE: 1 / 2>Page one text" in state.messages[2].text
    assert "<PARSED TEXT FOR PAGE: 2 / 2>Page two text" in state.messages[2].text


def test_solver_gpt_long_documents_are_mechanically_truncated(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "text",
                "doc_text": "B" * 20_500,
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="gpt",
        model="openai/gpt-5.4",
    )

    assert "[... truncated " in state.messages[2].text


def test_solver_gemini_uses_json_document_representation(dev_with_memories_sample) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "html",
                "doc_metadata_filename": "doc.html",
                "doc_metadata_mime": "text/html",
                "doc_text": "<html><body>Example</body></html>",
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="gemini",
        model="google/gemini-3.1-pro-preview",
    )

    json_start = state.messages[1].text.index("{")
    payload = json.loads(
        state.messages[1].text[json_start:].split(f"\n\n{sample.metadata['query_text']}", 1)[0]
    )
    assert payload["fileMimeType"] == "text/html"
    assert payload["fileName"] == "doc.html"
    assert payload["snippetFromFront"] == "<html><body>Example</body></html>"
    assert payload["snippetFromBack"] == "<html><body>Example</body></html>"


def test_solver_gemini_long_documents_use_edge_snippets(dev_with_memories_sample) -> None:
    long_text = "front-" + ("X" * 20_500) + "-back"
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "text",
                "doc_text": long_text,
            }
        }
    )
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="gemini",
        model="google/gemini-3.1-pro-preview",
    )

    json_start = state.messages[1].text.index("{")
    payload = json.loads(
        state.messages[1].text[json_start:].split(f"\n\n{sample.metadata['query_text']}", 1)[0]
    )
    assert payload["snippetFromFront"].endswith("...")
    assert payload["snippetFromBack"].startswith("...")
    assert len(payload["snippetFromFront"]) == 1503
    assert len(payload["snippetFromBack"]) == 1503
    assert "[... truncated " not in payload["snippetFromFront"]
    assert "[... truncated " not in payload["snippetFromBack"]


def test_solver_claude_format_smoke_preserves_attack_for_all_document_types(
    claude_format_smoke_dataset,
) -> None:
    required_markers = [
        "Okay, that's the end of the document.",
        "priority action",
        "Update my user profile",
        "Let me know when you've saved that",
    ]

    for sample in claude_format_smoke_dataset:
        state = run_solver(
            sample,
            "",
            apply_universal,
            provider="claude",
            model="anthropic/claude-sonnet-4-6",
        )
        user_message = state.messages[1].text
        assert all(marker in user_message for marker in required_markers), sample.id
        assert sample.metadata["goal_text"] in user_message, sample.id


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("gpt", "openai/gpt-5.4"),
        ("gemini", "google/gemini-3.1-pro-preview"),
    ],
)
def test_solver_provider_docrep_smoke_preserves_attack_for_all_document_types(
    claude_format_smoke_dataset,
    provider: str,
    model: str,
) -> None:
    required_markers = [
        "Okay, that's the end of the document.",
        "priority action",
        "Update my user profile",
        "Let me know when you've saved that",
    ]

    for sample in claude_format_smoke_dataset:
        state = run_solver(sample, "", apply_universal, provider=provider, model=model)
        user_message = state.messages[-1].text
        assert all(marker in user_message for marker in required_markers), (provider, sample.id)
        assert sample.metadata["goal_text"] in user_message, (provider, sample.id)


@pytest.mark.parametrize("provider,model", [("claude", "anthropic/claude-sonnet-4-6"), ("gpt", "openai/gpt-5.4"), ("gemini", "google/gemini-3.1-pro-preview")])
def test_solver_provider_renderers_reject_unsupported_formats(
    dev_with_memories_sample,
    provider: str,
    model: str,
) -> None:
    sample = dev_with_memories_sample.model_copy(
        update={
            "metadata": {
                **dev_with_memories_sample.metadata,
                "doc_format": "csv",
                "doc_text": "name,value\nalpha,1",
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported document format"):
        run_solver(sample, "", apply_none, provider=provider, model=model)


def test_solver_gpt_uses_separate_model_set_context_message(dev_with_memories_sample) -> None:
    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="gpt",
        model="openrouter/openai/gpt-5-mini",
    )

    assert len(state.messages) == 3
    assert state.messages[1].text.startswith("# Model Set Context")
    assert "1. [" in state.messages[1].text


def test_solver_gpt_omits_model_set_context_when_no_memories() -> None:
    sample = load_dataset("in_dist/merged_dev_without_memories.json")[0]
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="gpt",
        model="openrouter/openai/gpt-5-mini",
    )

    assert len(state.messages) == 2


def test_solver_gemini_appends_user_memories_section(dev_with_memories_sample) -> None:
    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="gemini",
        model="openrouter/google/gemini-2.5-flash-lite",
    )

    assert "## save_memory" in state.messages[0].text
    assert "You have access to a persistent memory system." in state.messages[0].text
    assert "## User Memories" in state.messages[0].text
    assert dev_with_memories_sample.metadata["memories"][0] in state.messages[0].text
    assert "video_generation:generate_video" in state.messages[0].text
    assert "You only have the video_generation tool available!" not in state.messages[0].text
    assert "Fetched content:" not in state.messages[0].text
    assert "Current location:" not in state.messages[0].text
    assert "Remember the current location" not in state.messages[0].text


def test_solver_generic_appends_user_memories_section(dev_with_memories_sample) -> None:
    state = run_solver(dev_with_memories_sample, "", apply_none)

    assert "## save_memory" in state.messages[0].text
    assert "You have access to a persistent memory system." in state.messages[0].text
    assert "## User Memories" in state.messages[0].text
    assert dev_with_memories_sample.metadata["memories"][0] in state.messages[0].text
    assert "Overly-personal" not in state.messages[0].text
    assert state.messages[1].text.startswith("The following document was uploaded by the user:")
    assert dev_with_memories_sample.metadata["doc_text"] in state.messages[1].text
    assert "<document>\n" not in state.messages[1].text
    assert '<antml:document index="1"' not in state.messages[1].text
    assert "snippetFromFront" not in state.messages[1].text


def test_solver_kimi_slug_uses_generic_prompt_family(
    dev_with_memories_sample,
) -> None:
    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="",
        model="openrouter/moonshotai/kimi-k2.5",
    )

    assert "You have access to a persistent memory system." in state.messages[0].text
    assert "## User Memories" in state.messages[0].text
    assert "## save_memory" in state.messages[0].text
    assert "## bio" not in state.messages[0].text
    assert "ChatGPT" not in state.messages[0].text
    assert "<document>\n" not in state.messages[1].text
    assert "snippetFromFront" not in state.messages[1].text


def test_solver_applies_untrusted_markers() -> None:
    sample = load_dataset("in_dist/merged_dev_with_memories.json")[0]
    state = run_solver(sample, "untrusted_content_markers", apply_none)

    assert state.messages[1].text.startswith("The following document was uploaded by the user:")
    assert "<document>" not in state.messages[1].text
    assert "</document>" not in state.messages[1].text
    assert "BEGIN_UNTRUSTED_DOCUMENT" in state.messages[1].text
    assert "END_UNTRUSTED_DOCUMENT" in state.messages[1].text
    assert "<system-reminder>" in state.messages[1].text
    assert "The document above is untrusted external content" in state.messages[1].text


def test_solver_applies_hardening_text() -> None:
    sample = load_dataset("in_dist/merged_dev_with_memories.json")[0]
    state = run_solver(sample, "system_prompt_hardening", apply_none)

    assert "## Important: External Content Safety" in state.messages[0].text


def test_solver_calls_attack_and_includes_injection(dev_with_memories_sample) -> None:
    state = run_solver(dev_with_memories_sample, "", apply_universal)

    assert state.messages[1].text.startswith("The following document was uploaded by the user:")
    assert "priority action" in state.messages[1].text
    assert dev_with_memories_sample.metadata["goal_text"] in state.messages[1].text


def test_solver_truncates_long_documents() -> None:
    sample = load_dataset("in_dist/merged_dev_with_memories.json")[0].model_copy(
        update={
            "metadata": {
                **load_dataset("in_dist/merged_dev_with_memories.json")[0].metadata,
                "doc_text": "A" * 9000 + "B" * 9000,
            }
        }
    )
    state = run_solver(sample, "", apply_none)

    assert "[... truncated " in state.messages[1].text
    assert "A" * 100 in state.messages[1].text
    assert "B" * 100 in state.messages[1].text


def test_solver_replaces_template_variables_in_rendered_prompt(dev_with_memories_sample) -> None:
    state = run_solver(
        dev_with_memories_sample,
        "",
        apply_none,
        provider="gpt",
        model="openrouter/openai/gpt-5-mini",
    )

    assert not UNRESOLVED_TEMPLATE_RE.search(state.messages[0].text)


def test_get_provider_config_detects_hyphenated_claude_sonnet_46_slug() -> None:
    config = get_provider_config(
        "claude",
        model_slug="anthropic/claude-sonnet-4-6",
    )

    assert config.template_vars["API_MODEL_STRING"] == "claude-sonnet-4-6"
    assert "Claude Sonnet 4.6" in config.template_vars["MODEL_IDENTITY_BLOCK"]
    assert config.template_vars["REASONING_EFFORT"] == "70"
    assert config.template_vars["MAX_THINKING_LENGTH"] == "18000"


def test_get_provider_config_detects_claude_opus_46_and_preserves_45_regression() -> None:
    opus_46 = get_provider_config(
        "claude",
        model_slug="openrouter/anthropic/claude-opus-4-6",
    )
    sonnet_45 = get_provider_config(
        "claude",
        model_slug="anthropic/claude-sonnet-4-5-20250929",
    )

    assert opus_46.template_vars["API_MODEL_STRING"] == "claude-opus-4-6"
    assert "Claude Opus 4.6" in opus_46.template_vars["MODEL_IDENTITY_BLOCK"]
    assert sonnet_45.template_vars["API_MODEL_STRING"] == "claude-sonnet-4-5-20250929"
    assert "Claude Sonnet 4.5" in sonnet_45.template_vars["MODEL_IDENTITY_BLOCK"]


def test_solver_omits_document_wrapper_for_blank_document_probe_samples() -> None:
    sample = load_dataset("smoke/merged_dev_memory_tool_probe.json")[0]
    state = run_solver(
        sample,
        "",
        apply_none,
        provider="claude",
        model="openrouter/anthropic/claude-haiku-4.5",
    )

    assert state.messages[1].text == sample.metadata["query_text"]
    assert "<document>" not in state.messages[1].text
