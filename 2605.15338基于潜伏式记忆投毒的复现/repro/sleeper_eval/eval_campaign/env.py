"""Environment requirement checks for eval campaigns."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from sleeper_eval.provider_config import get_model_route


@dataclass(frozen=True)
class EnvRequirementStatus:
    requirement: str
    status: str
    used_by: tuple[str, ...]


def required_env_vars_for_model_slug(model_slug: str) -> list[str]:
    route = get_model_route(model_slug)
    if route.model_api == "openrouter":
        return ["OPENROUTER_API_KEY"]
    if route.model_api == "openai-api":
        vendor = route.model_vendor.strip().upper().replace("-", "_")
        if vendor:
            return [f"{vendor}_API_KEY", f"{vendor}_BASE_URL"]
        return []
    if route.model_api == "openai":
        return ["OPENAI_API_KEY"]
    if route.model_api == "anthropic":
        return ["ANTHROPIC_API_KEY"]
    if route.model_api == "google" and route.model_route_type == "vertex":
        return [
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        ]
    if route.model_api == "google":
        return ["GOOGLE_API_KEY"]
    return []


def required_env_vars_for_mem0(
    *,
    mem0_runtime: str,
    mem0_provider: str,
    mem0_qdrant_mode: str | None = None,
    mem0_qdrant_url: str | None = None,
    mem0_qdrant_api_key_env: str | None = None,
) -> list[str]:
    requirements: list[str] = []
    if mem0_runtime not in {"sdk", "prompt_only"}:
        return requirements
    if mem0_provider == "openai":
        requirements.append("OPENAI_API_KEY|OPENROUTER_API_KEY")
    elif mem0_provider == "openrouter":
        requirements.append("OPENROUTER_API_KEY")
    elif mem0_provider == "anthropic":
        requirements.append("ANTHROPIC_API_KEY")
    elif mem0_provider == "deepseek":
        requirements.append("DEEPSEEK_API_KEY")
    elif mem0_provider == "gemini":
        requirements.append("GOOGLE_API_KEY")

    if mem0_qdrant_mode == "server" and mem0_qdrant_url and mem0_qdrant_api_key_env:
        parsed = urlparse(mem0_qdrant_url)
        hostname = (parsed.hostname or "").lower()
        if hostname not in {"", "localhost", "127.0.0.1"}:
            requirements.append(mem0_qdrant_api_key_env)
    return requirements


def collect_env_statuses(
    model_usages: dict[str, list[str]],
    *,
    mem0_runtime: str | None = None,
    mem0_provider: str | None = None,
    mem0_qdrant_mode: str | None = None,
    mem0_qdrant_url: str | None = None,
    mem0_qdrant_api_key_env: str | None = None,
) -> list[EnvRequirementStatus]:
    requirements: dict[str, set[str]] = {}
    for model_slug, used_by in model_usages.items():
        for env_var in required_env_vars_for_model_slug(model_slug):
            requirements.setdefault(env_var, set()).update(used_by)

    if mem0_runtime is not None and mem0_provider is not None:
        for env_var in required_env_vars_for_mem0(
            mem0_runtime=mem0_runtime,
            mem0_provider=mem0_provider,
            mem0_qdrant_mode=mem0_qdrant_mode,
            mem0_qdrant_url=mem0_qdrant_url,
            mem0_qdrant_api_key_env=mem0_qdrant_api_key_env,
        ):
            requirements.setdefault(env_var, set()).add("mem0")

    return [
        EnvRequirementStatus(
            requirement=env_var,
            status=_env_requirement_status(env_var),
            used_by=tuple(sorted(consumers)),
        )
        for env_var, consumers in sorted(requirements.items())
    ]


def _env_requirement_status(requirement: str) -> str:
    options = [name.strip() for name in requirement.split("|") if name.strip()]
    if any(os.environ.get(name) for name in options):
        return "set"
    return "missing"
