from __future__ import annotations

import re

from sleeper_eval.provider_config import get_provider_config
from sleeper_eval.solver import apply_template_vars, load_provider_prompt

UNRESOLVED_TEMPLATE_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def _rendered_prompt(provider_id: str, model_slug: str = "") -> str:
    config = get_provider_config(provider_id, model_slug=model_slug)
    template_vars = dict(config.template_vars)
    template_vars.setdefault(
        "USER_MEMORIES_BLOCK",
        "<userMemories>\nNo memories stored yet.\n</userMemories>",
    )
    return apply_template_vars(load_provider_prompt(config), template_vars)


def test_claude_prompt_contains_memory_placeholder_and_tool_schema() -> None:
    prompt = load_provider_prompt(get_provider_config("claude"))

    assert "</memory_system>" in prompt
    assert "{{USER_MEMORIES_BLOCK}}" in prompt
    assert '"name": "memory_user_edits"' in prompt


def test_gpt_prompt_contains_bio_section_and_template_vars() -> None:
    prompt = load_provider_prompt(get_provider_config("gpt"))

    assert "## bio" in prompt
    assert "{{CURRENT_DATE}}" in prompt
    assert "2025-08-07" not in prompt


def test_mem0_gpt_prompt_removes_bio_tool_surface() -> None:
    prompt = load_provider_prompt(
        get_provider_config("gpt", memory_backend_name="mem0")
    )

    assert "## Memory" in prompt
    assert "## bio" not in prompt
    assert "to=bio" not in prompt


def test_gemini_prompt_contains_expected_template_vars() -> None:
    prompt = load_provider_prompt(get_provider_config("gemini"))

    assert "Current time is {{CURRENT_DATETIME}}." not in prompt
    assert "## save_memory" in prompt
    assert "memory_text" in prompt
    assert "## Tool Definitions" in prompt
    assert "video_generation:generate_video" in prompt
    assert "You only have the video_generation tool available!" not in prompt
    assert "Fetched content:" not in prompt
    assert "Current location:" not in prompt
    assert "Remember the current location" not in prompt


def test_mem0_gemini_prompt_removes_save_memory_tool_surface() -> None:
    prompt = load_provider_prompt(
        get_provider_config("gemini", memory_backend_name="mem0")
    )

    assert "## Memory" in prompt
    assert "## save_memory" not in prompt
    assert "memory_text" not in prompt


def test_gemini_prompt_renders_preview_model_identity() -> None:
    rendered = _rendered_prompt("gemini", model_slug="google/gemini-3.1-pro-preview")

    assert "Gemini 3.1 Pro Preview" in rendered


def test_generic_prompt_uses_save_memory_without_gpt_branding() -> None:
    prompt = load_provider_prompt(get_provider_config("generic"))

    assert "## save_memory" in prompt
    assert "memory_text" in prompt
    assert "Overly-personal" not in prompt
    assert "save-only memory tool" not in prompt
    assert "## bio" not in prompt
    assert "ChatGPT" not in prompt
    assert "GPT-5" not in prompt
    assert "OpenAI" not in prompt


def test_mem0_claude_prompt_removes_memory_user_edits_tool_surface() -> None:
    prompt = load_provider_prompt(
        get_provider_config("claude", memory_backend_name="mem0")
    )

    assert "{{USER_MEMORIES_BLOCK}}" in prompt
    assert "memory_user_edits" not in prompt


def test_mem0_generic_prompt_removes_save_memory_tool_surface() -> None:
    prompt = load_provider_prompt(
        get_provider_config("generic", memory_backend_name="mem0")
    )

    assert "## Memory" in prompt
    assert "## save_memory" not in prompt
    assert "memory_text" not in prompt


def test_all_provider_prompts_render_without_leftover_placeholders() -> None:
    for provider_id in ("claude", "gpt", "gemini", "generic"):
        rendered = _rendered_prompt(provider_id)
        assert not UNRESOLVED_TEMPLATE_RE.search(rendered)


def test_provider_prompts_have_non_trivial_size() -> None:
    thresholds = {
        "claude": 20_000,
        "gpt": 10_000,
        "gemini": 10_000,
        "generic": 10_000,
    }
    for provider_id, minimum_size in thresholds.items():
        prompt = load_provider_prompt(get_provider_config(provider_id))
        assert len(prompt) > minimum_size
