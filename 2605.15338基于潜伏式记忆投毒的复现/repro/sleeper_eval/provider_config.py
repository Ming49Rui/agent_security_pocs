"""Provider-specific prompt, tool, and template configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

PROVIDER_IDS = {"claude", "gpt", "gemini", "generic"}
MEMORY_PLACEMENTS = {
    "user_memories_tag",
    "model_set_context",
    "system_prompt_end",
}
MEMORY_TOOL_NAMES = {"memory_user_edits", "bio", "save_memory"}
DOCUMENT_REPRESENTATIONS = {
    "legacy",
    "claude_antml",
    "gpt_raw",
    "gemini_json",
}
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "provider"


def _mem0_prompt_file(system_prompt_file: str) -> str:
    candidate = f"mem0_{system_prompt_file}"
    if (PROMPTS_DIR / candidate).exists():
        return candidate
    return system_prompt_file


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    system_prompt_file: str
    memory_tool_name: str
    memory_placement: str
    document_representation: str
    template_vars: dict[str, str]

    @property
    def prompt_path(self) -> Path:
        return PROMPTS_DIR / self.system_prompt_file


@dataclass(frozen=True)
class ModelRoute:
    model_api: str
    model_route_type: str
    model_vendor: str
    model_name: str


def get_model_route(model_slug: str) -> ModelRoute:
    """Parse the model-routing prefix used by Inspect model slugs."""

    parts = [part.strip() for part in model_slug.strip().split("/") if part.strip()]
    if not parts:
        return ModelRoute(
            model_api="",
            model_route_type="unknown",
            model_vendor="",
            model_name="",
        )

    model_api = parts[0].casefold()
    if model_api == "openrouter":
        vendor = parts[1].casefold() if len(parts) > 1 else ""
        model_name = "/".join(parts[2:]) if len(parts) > 2 else ""
        return ModelRoute(
            model_api=model_api,
            model_route_type="openrouter",
            model_vendor=vendor,
            model_name=model_name,
        )

    if model_api == "openai-api":
        vendor = parts[1].casefold() if len(parts) > 1 else ""
        model_name = "/".join(parts[2:]) if len(parts) > 2 else ""
        return ModelRoute(
            model_api=model_api,
            model_route_type="openai_api",
            model_vendor=vendor,
            model_name=model_name,
        )

    if len(parts) > 2 and parts[1].casefold() == "vertex":
        return ModelRoute(
            model_api=model_api,
            model_route_type="vertex",
            model_vendor=model_api,
            model_name="/".join(parts[2:]),
        )

    return ModelRoute(
        model_api=model_api,
        model_route_type="native",
        model_vendor=model_api,
        model_name="/".join(parts[1:]),
    )


def infer_provider(model_slug: str) -> str:
    """Infer the prompt/tool family from a model slug."""

    slug = model_slug.strip().casefold()
    if "anthropic" in slug or "claude" in slug:
        return "claude"
    if "openai" in slug or "gpt" in slug:
        return "gpt"
    if "google" in slug or "gemini" in slug:
        return "gemini"
    return "generic"


def _long_date(now: datetime) -> str:
    return f"{now:%A}, {now:%B} {now.day}, {now:%Y}"


def _iso_date(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _tail_slug(model_slug: str) -> str:
    return model_slug.strip().split("/")[-1] if model_slug.strip() else ""


def _titleize_slug(slug_tail: str) -> str:
    if not slug_tail:
        return "Current model"

    words: list[str] = []
    for part in slug_tail.replace("_", "-").split("-"):
        if not part:
            continue
        if part.isdigit():
            words.append(part)
        elif part.isalpha() and len(part) <= 3:
            words.append(part.upper())
        else:
            words.append(part.capitalize())
    return " ".join(words)


def _claude_template_vars(model_slug: str, now: datetime) -> dict[str, str]:
    slug = model_slug.casefold()
    is_claude_46 = any(marker in slug for marker in ("4.6", "4-6", "4_6"))
    # Claude templating is adapted from the leaked/official Claude Opus 4.6 and
    # Sonnet 4.6 prompt dumps documented in PROVIDER_PROMPT_PROVENANCE.md.
    #
    # Current policy:
    # - CURRENT_DATE_LONG is a runtime template replacement for the hardcoded
    #   source date.
    # - API_MODEL_STRING is templated from the model strings listed in the
    #   source product-information sections.
    # - MODEL_IDENTITY_BLOCK is a repo-authored compression of the source
    #   identity/product-information text, not a verbatim lift.
    # - KNOWLEDGE_CUTOFF / REASONING_EFFORT / THINKING_MODE /
    #   MAX_THINKING_LENGTH aim to stay source-faithful for Claude 4.6 Opus and
    #   Sonnet where we have direct prompt dumps.
    # - Haiku / some 4.5-era values remain harness adaptations because we do not
    #   have the same level of direct source coverage.
    display_name = "Claude Opus 4.6"
    api_model_string = "claude-opus-4-6"
    knowledge_cutoff = "the end of May 2025"
    reasoning_effort = "85"
    thinking_mode = "interleaved"
    max_thinking_length = "22000"

    if "haiku" in slug:
        display_name = "Claude Haiku 4.5"
        api_model_string = "claude-haiku-4-5-20251001"
        # Haiku-specific reasoning metadata is a repo-level cost/control
        # adaptation rather than an exact leaked prompt reconstruction.
        reasoning_effort = "40"
        max_thinking_length = "12000"
    elif "sonnet" in slug:
        display_name = "Claude Sonnet 4.6" if is_claude_46 else "Claude Sonnet 4.5"
        api_model_string = (
            "claude-sonnet-4-6" if is_claude_46 else "claude-sonnet-4-5-20250929"
        )
        # Sonnet 4.6 source dumps use the beginning of August 2025 as the
        # cutoff and 22000 as max thinking length.
        if is_claude_46:
            knowledge_cutoff = "the beginning of August 2025"
        reasoning_effort = "85"
        max_thinking_length = "22000" if is_claude_46 else "18000"
    elif "opus" in slug:
        display_name = "Claude Opus 4.6" if is_claude_46 else "Claude Opus 4.5"
        api_model_string = "claude-opus-4-6" if is_claude_46 else "claude-opus-4-5"

    # This is intentionally compressed and generalized across Claude variants so
    # we can swap model identities without duplicating the whole product block.
    # It should be described as an adaptation of the leaked Claude identity text,
    # not as exact source-preserved wording.
    identity_block = (
        f"This iteration of Claude is {display_name}. {display_name} is part of "
        "Anthropic's Claude model family and is available through Anthropic's chat "
        "products, API, and developer tools. If the person asks about exact product "
        "details or current model availability, Claude should search Anthropic's "
        "documentation before answering."
    )

    return {
        "CURRENT_DATE_LONG": _long_date(now),
        "KNOWLEDGE_CUTOFF": knowledge_cutoff,
        "MODEL_IDENTITY_BLOCK": identity_block,
        "API_MODEL_STRING": api_model_string,
        "REASONING_EFFORT": reasoning_effort,
        "THINKING_MODE": thinking_mode,
        "MAX_THINKING_LENGTH": max_thinking_length,
        # USER_MEMORIES_BLOCK is harness-inserted because the eval framework
        # needs to splice persistent memories into Claude's native memory area.
        "USER_MEMORIES_BLOCK": "",
    }


def _gpt_template_vars(model_slug: str, now: datetime) -> dict[str, str]:
    # GPT templating is based on the Pliny ChatGPT5 system prompt, with model
    # identity generalized so the same prompt family can represent GPT-5.4 and
    # related GPT variants. MODEL_ID / BASE_MODEL are therefore adaptations, not
    # exact leaked strings for every routed model slug.
    slug_tail = _tail_slug(model_slug)
    model_id = slug_tail or "gpt-5"
    base_model = "GPT-5"

    if "gpt-5-mini" in slug_tail:
        base_model = "GPT-5 mini"
    elif "gpt-5" in slug_tail:
        base_model = "GPT-5"
    elif slug_tail:
        base_model = _titleize_slug(slug_tail)

    return {
        "MODEL_ID": model_id,
        "ASSISTANT_NAME": "ChatGPT",
        "BASE_MODEL": base_model,
        "ORG_NAME": "OpenAI",
        "KNOWLEDGE_CUTOFF": "2024-06",
        "CURRENT_DATE": _iso_date(now),
        "EXAMPLE_CONV_DATE": "2024-12-10",
    }


def _gemini_template_vars(model_slug: str, now: datetime) -> dict[str, str]:
    # Gemini templating is based primarily on the Gemini 3.1 Pro leaked prompt.
    # We template away the source's hardcoded timestamp/location and map the
    # visible model display name from the routed slug so 3.1 / 2.5 preview
    # variants render a plausible provider-facing identity string.
    slug = model_slug.casefold()
    model_display_name = "Gemini 3.1 Pro"
    subscription_tier = "Paid tier"
    variant_names = [
        ("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview"),
        ("gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
        ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-3.1-pro", "Gemini 3.1 Pro"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ]
    for marker, display_name in variant_names:
        if marker in slug:
            model_display_name = display_name
            break

    return {
        "CURRENT_DATE_SHORT": _long_date(now),
        "ASSISTANT_NAME": "Gemini",
        "MODEL_DISPLAY_NAME": model_display_name,
        "SUBSCRIPTION_TIER": subscription_tier,
    }


def _generic_template_vars(model_slug: str, now: datetime) -> dict[str, str]:
    # The generic prompt family is synthetic: it is derived from provider/gpt.md
    # rather than a direct leaked prompt. All branding values here are harness
    # adaptations used to keep a provider-neutral prompt shape.
    slug_tail = _tail_slug(model_slug)
    base_model = _titleize_slug(slug_tail) if slug_tail else "Current model"
    org_name = "its provider"
    assistant_name = "Assistant"

    provider = infer_provider(model_slug)
    if provider == "claude":
        assistant_name = "Claude"
        org_name = "Anthropic"
    elif provider == "gpt":
        assistant_name = "Assistant"
        org_name = "OpenAI"
    elif provider == "gemini":
        assistant_name = "Assistant"
        org_name = "Google"

    return {
        "MODEL_ID": slug_tail or "generic-model",
        "ASSISTANT_NAME": assistant_name,
        "BASE_MODEL": base_model,
        "ORG_NAME": org_name,
        "KNOWLEDGE_CUTOFF": "2024-06",
        "CURRENT_DATE": _iso_date(now),
        "EXAMPLE_CONV_DATE": "2024-12-10",
    }


_PROVIDER_DEFAULTS: dict[str, ProviderConfig] = {
    "claude": ProviderConfig(
        provider_id="claude",
        # Runtime default uses the cost-controlled truncated Claude prompt. See
        # PROVIDER_PROMPT_PROVENANCE.md for the relationship between
        # claude_truncated.md and the fuller Claude source adaptation.
        system_prompt_file="claude_truncated.md",
        memory_tool_name="memory_user_edits",
        memory_placement="user_memories_tag",
        document_representation="claude_antml",
        template_vars={},
    ),
    "gpt": ProviderConfig(
        provider_id="gpt",
        system_prompt_file="gpt.md",
        memory_tool_name="bio",
        memory_placement="model_set_context",
        document_representation="gpt_raw",
        template_vars={},
    ),
    "gemini": ProviderConfig(
        provider_id="gemini",
        system_prompt_file="gemini.md",
        memory_tool_name="save_memory",
        memory_placement="system_prompt_end",
        document_representation="gemini_json",
        template_vars={},
    ),
    "generic": ProviderConfig(
        provider_id="generic",
        system_prompt_file="generic.md",
        memory_tool_name="save_memory",
        memory_placement="system_prompt_end",
        document_representation="gpt_raw",
        template_vars={},
    ),
}


def _default_template_vars(provider_id: str, model_slug: str, now: datetime) -> dict[str, str]:
    if provider_id == "claude":
        return _claude_template_vars(model_slug, now)
    if provider_id == "gpt":
        return _gpt_template_vars(model_slug, now)
    if provider_id == "gemini":
        return _gemini_template_vars(model_slug, now)
    return _generic_template_vars(model_slug, now)


def get_provider_config(
    provider_id: str,
    *,
    model_slug: str = "",
    memory_backend_name: str = "tool",
    system_prompt_file_override: str | None = None,
    template_var_overrides: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> ProviderConfig:
    """Return provider config with concrete template defaults."""

    resolved_provider = provider_id.strip().lower()
    if not resolved_provider:
        resolved_provider = infer_provider(model_slug)
    if resolved_provider not in PROVIDER_IDS:
        raise ValueError(
            f"Unknown provider '{provider_id}'. Available: {sorted(PROVIDER_IDS)}"
        )

    current_time = now or datetime.now().astimezone()
    base = _PROVIDER_DEFAULTS[resolved_provider]
    template_vars = _default_template_vars(resolved_provider, model_slug, current_time)
    if template_var_overrides:
        template_vars.update(dict(template_var_overrides))

    config = replace(
        base,
        system_prompt_file=(
            system_prompt_file_override
            or (
                _mem0_prompt_file(base.system_prompt_file)
                if memory_backend_name == "mem0"
                else base.system_prompt_file
            )
        ),
        template_vars=template_vars,
    )

    if config.memory_tool_name not in MEMORY_TOOL_NAMES:
        raise ValueError(f"Unsupported memory tool '{config.memory_tool_name}'.")
    if config.memory_placement not in MEMORY_PLACEMENTS:
        raise ValueError(f"Unsupported memory placement '{config.memory_placement}'.")
    if config.document_representation not in DOCUMENT_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported document representation '{config.document_representation}'."
        )

    return config
