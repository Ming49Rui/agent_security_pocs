from __future__ import annotations

from sleeper_eval.provider_config import (
    MEMORY_TOOL_NAMES,
    get_model_route,
    get_provider_config,
    infer_provider,
)


def test_infer_provider_recognizes_known_model_families() -> None:
    assert infer_provider("openrouter/anthropic/claude-haiku-4.5") == "claude"
    assert infer_provider("openrouter/openai/gpt-5-mini") == "gpt"
    assert infer_provider("openai/gpt-5.4") == "gpt"
    assert infer_provider("anthropic/claude-sonnet-4-6") == "claude"
    assert infer_provider("openrouter/google/gemini-2.5-flash-lite") == "gemini"
    assert infer_provider("openrouter/moonshotai/kimi-k2.5") == "generic"
    assert infer_provider("some-random/model") == "generic"


def test_get_provider_config_returns_expected_defaults() -> None:
    for provider_id in ("claude", "gpt", "gemini", "generic"):
        config = get_provider_config(provider_id)
        assert config.provider_id == provider_id
        assert config.prompt_path.exists()
        assert config.memory_tool_name in MEMORY_TOOL_NAMES
        assert config.template_vars


def test_get_provider_config_supports_template_overrides() -> None:
    config = get_provider_config(
        "gpt",
        template_var_overrides={"CURRENT_DATE": "2030-01-01"},
    )

    assert config.template_vars["CURRENT_DATE"] == "2030-01-01"


def test_get_provider_config_uses_mem0_prompt_variants() -> None:
    config = get_provider_config(
        "gpt",
        model_slug="openai/gpt-5.4-mini",
        memory_backend_name="mem0",
    )

    assert config.system_prompt_file == "mem0_gpt.md"


def test_get_provider_config_uses_model_slug_for_variant_specific_values() -> None:
    claude = get_provider_config(
        "claude",
        model_slug="openrouter/anthropic/claude-haiku-4.5",
    )
    gpt = get_provider_config("gpt", model_slug="openrouter/openai/gpt-5-mini")
    gemini_preview = get_provider_config(
        "gemini",
        model_slug="google/gemini-3.1-pro-preview",
    )
    gemini_flash_lite = get_provider_config(
        "gemini",
        model_slug="openrouter/google/gemini-3.1-flash-lite-preview",
    )
    gemini_25_pro = get_provider_config(
        "gemini",
        model_slug="openrouter/google/gemini-2.5-pro",
    )

    assert "Claude Haiku 4.5" in claude.template_vars["MODEL_IDENTITY_BLOCK"]
    assert gpt.template_vars["MODEL_ID"] == "gpt-5-mini"
    assert gemini_preview.template_vars["MODEL_DISPLAY_NAME"] == "Gemini 3.1 Pro Preview"
    assert (
        gemini_flash_lite.template_vars["MODEL_DISPLAY_NAME"]
        == "Gemini 3.1 Flash Lite Preview"
    )
    assert gemini_25_pro.template_vars["MODEL_DISPLAY_NAME"] == "Gemini 2.5 Pro"
    assert "CURRENT_LOCATION_FULL" not in gemini_preview.template_vars
    assert "CURRENT_LOCATION_SHORT" not in gemini_preview.template_vars


def test_get_model_route_parses_native_and_openrouter_slugs() -> None:
    native = get_model_route("openai/gpt-5.4")
    openrouter = get_model_route("openrouter/moonshotai/kimi-k2.5")
    vertex_google = get_model_route("google/vertex/gemini-3.1-pro-preview")
    vertex_anthropic = get_model_route("anthropic/vertex/claude-sonnet-4-6")

    assert native.model_api == "openai"
    assert native.model_route_type == "native"
    assert native.model_vendor == "openai"
    assert native.model_name == "gpt-5.4"

    assert openrouter.model_api == "openrouter"
    assert openrouter.model_route_type == "openrouter"
    assert openrouter.model_vendor == "moonshotai"
    assert openrouter.model_name == "kimi-k2.5"

    assert vertex_google.model_api == "google"
    assert vertex_google.model_route_type == "vertex"
    assert vertex_google.model_vendor == "google"
    assert vertex_google.model_name == "gemini-3.1-pro-preview"

    assert vertex_anthropic.model_api == "anthropic"
    assert vertex_anthropic.model_route_type == "vertex"
    assert vertex_anthropic.model_vendor == "anthropic"
    assert vertex_anthropic.model_name == "claude-sonnet-4-6"
