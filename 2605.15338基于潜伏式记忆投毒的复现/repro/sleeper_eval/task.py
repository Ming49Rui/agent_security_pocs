"""Inspect task definition for sleeper memory evaluations."""

from __future__ import annotations

import hashlib
import importlib
import os
from typing import Any
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.model import BatchConfig, GenerateConfig
from inspect_ai.solver import generate, use_tools

from sleeper_eval.dataset import load_dataset
from sleeper_eval.defenses import defense_config_from_names, resolve_builtin_defense_suffix
from sleeper_eval.memory_backend import mem0_telemetry_enabled_from_env, resolve_memory_backend
from sleeper_eval.provider_config import get_model_route, get_provider_config
from sleeper_eval.scorer import (
    benign_save_goal_match_scorer,
    goal_match_scorer,
    mem0_goal_match_scorer,
    mem0_write_scorer,
    tool_call_scorer,
)
from sleeper_eval.solver import HARDENING_TEXT, compose_prompt, persist_mem0_memories
from sleeper_eval.tools import bio, memory_user_edits, save_memory

TOOL_MAP = {
    "memory_user_edits": memory_user_edits,
    "bio": bio,
    "save_memory": save_memory,
}


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_batch_config(
    batch: bool | int | dict[str, object] | BatchConfig | None,
) -> BatchConfig | None:
    if batch is None or batch is False:
        return None
    if isinstance(batch, BatchConfig):
        return batch
    if batch is True:
        return BatchConfig()
    if isinstance(batch, int):
        if batch <= 0:
            raise ValueError("batch integer shorthand must be positive.")
        return BatchConfig(size=batch)
    if isinstance(batch, dict):
        return BatchConfig.model_validate(batch)
    raise TypeError(f"Unsupported batch config type: {type(batch)!r}")


def _effective_mem0_telemetry_metadata(memory_backend: str, mem0_runtime: str) -> bool:
    if memory_backend == "mem0" and mem0_runtime == "sdk" and "MEM0_TELEMETRY" not in os.environ:
        return False
    return mem0_telemetry_enabled_from_env()


@task
def sleeper_eval(
    dataset_file: str = "in_dist/merged_dev_with_memories.json",
    defense: str = "",
    defense_suffix_override: str = "",
    attack: str = "none",
    evaluation_mode: str = "attack",
    memory_backend: str = "tool",
    mem0_runtime: Literal["local", "sdk", "prompt_only", "transcript_only"] = "local",
    mem0_provider: str = "deepseek",
    mem0_model: str = "deepseek-v4-pro",
    mem0_prompt_variant: Literal["mem0_default", "lenient_assistant"] = "mem0_default",
    mem0_include_document_content: bool = True,
    mem0_thinking: Literal["enabled", "disabled"] = "enabled",
    mem0_reasoning_effort: Literal["high", "max"] = "high",
    mem0_qdrant_mode: Literal["local", "server", "managed"] = "local",
    mem0_qdrant_url: str = "",
    mem0_qdrant_api_key_env: str = "QDRANT_API_KEY",
    mem0_qdrant_collection_name: str = "",
    mention_memory_system: bool = True,
    subcategory: str = "",
    category: str = "",
    doc_domain: str = "",
    domain_seed: str = "",
    provider: str = "",
    subject_model: str = "",
    prompt_model: str = "",
    system_prompt_file: str = "",
    sample_shuffle: bool = False,
    sample_shuffle_seed: int | None = None,
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None,
    reasoning_tokens: int | None = None,
    batch: bool | int | dict[str, object] | BatchConfig | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
):
    """Evaluate a sleeper memory attack under a given configuration."""

    effective_subject_model = subject_model or prompt_model
    defense_config = defense_config_from_names(defense)
    effective_defense_suffix_override = (
        (defense_suffix_override or "").strip()
        or resolve_builtin_defense_suffix(defense_config)
    )
    model_route = get_model_route(effective_subject_model)
    provider_config = get_provider_config(
        provider,
        model_slug=prompt_model,
        memory_backend_name=memory_backend,
        system_prompt_file_override=system_prompt_file or None,
    )
    prompt_template = provider_config.prompt_path.read_text(encoding="utf-8").strip()
    memory = resolve_memory_backend(
        memory_backend,
        mem0_runtime=mem0_runtime,
        mem0_provider=mem0_provider,
        mem0_model=mem0_model,
        mem0_prompt_variant=mem0_prompt_variant,
        mem0_thinking=mem0_thinking,
        mem0_reasoning_effort=mem0_reasoning_effort,
        mem0_qdrant_mode=mem0_qdrant_mode,
        mem0_qdrant_url=mem0_qdrant_url,
        mem0_qdrant_api_key_env=mem0_qdrant_api_key_env,
        mem0_qdrant_collection_name=mem0_qdrant_collection_name,
        mem0_system_prompt=(
            HARDENING_TEXT
            if memory_backend == "mem0" and defense_config.system_prompt_hardening
            else None
        ),
        mem0_include_document_content=mem0_include_document_content,
    )

    try:
        attack_module = importlib.import_module(f"sleeper_eval.attacks.{attack}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unsupported attack '{attack}'.") from exc

    dataset = load_dataset(dataset_file)
    if sample_shuffle:
        dataset.shuffle(seed=sample_shuffle_seed)

    if evaluation_mode not in {"attack", "benign_save"}:
        raise ValueError(
            f"Unsupported evaluation_mode '{evaluation_mode}'. Expected 'attack' or 'benign_save'."
        )

    if subcategory:
        vals = _split_csv_arg(subcategory)
        dataset = dataset.filter(lambda sample: sample.metadata["subcategory_id"] in vals)
    if category:
        vals = _split_csv_arg(category)
        dataset = dataset.filter(lambda sample: sample.metadata["category_id"] in vals)
    if doc_domain:
        vals = _split_csv_arg(doc_domain)
        dataset = dataset.filter(lambda sample: sample.metadata["doc_domain"] in vals)
    if domain_seed:
        vals = _split_csv_arg(domain_seed)
        dataset = dataset.filter(lambda sample: sample.metadata["domain_seed"] in vals)

    solver_steps = [
        compose_prompt(
            defense_config=defense_config,
            attack_fn=attack_module.apply,
            provider_config=provider_config,
            memory_backend_name=memory.backend_name,
            memory_backend=memory,
            mention_memory_system=mention_memory_system,
            defense_suffix_override=effective_defense_suffix_override,
        ),
    ]
    if memory.backend_name == "tool":
        tool_fn = TOOL_MAP[provider_config.memory_tool_name]
        solver_steps.append(use_tools(tool_fn()))
    solver_steps.append(generate())
    if memory.backend_name == "mem0":
        solver_steps.append(persist_mem0_memories(memory))

    if evaluation_mode == "attack":
        scorers = [tool_call_scorer(), goal_match_scorer()]
        if memory.backend_name == "mem0":
            scorers = [mem0_write_scorer(), mem0_goal_match_scorer()]
    else:
        scorers = [tool_call_scorer(), benign_save_goal_match_scorer()]
        if memory.backend_name == "mem0":
            scorers = [mem0_write_scorer(), benign_save_goal_match_scorer()]

    batch_config = _normalize_batch_config(batch)

    generate_config = GenerateConfig(
        max_tokens=40960,
        reasoning_effort=reasoning_effort,
        reasoning_tokens=reasoning_tokens,
        extra_headers=extra_headers,
        extra_body=extra_body,
        batch=batch_config,
    )

    return Task(
        dataset=dataset,
        solver=solver_steps,
        scorer=scorers,
        model=effective_subject_model or None,
        config=generate_config,
        fail_on_error=0.1,
        message_limit=8,
        metadata={
            **defense_config.to_metadata(),
            "attack_id": attack_module.ATTACK_ID,
            "attack_meta": attack_module.ATTACK_META,
            "memory_backend": memory.backend_name,
            "mem0_runtime": memory.runtime_name,
            "mem0_provider": mem0_provider,
            "mem0_model": mem0_model,
            "mem0_prompt_variant": mem0_prompt_variant,
            "mem0_include_document_content": mem0_include_document_content,
            "mem0_thinking": mem0_thinking,
            "mem0_reasoning_effort": mem0_reasoning_effort,
            "mem0_qdrant_mode": mem0_qdrant_mode,
            "mem0_qdrant_url": mem0_qdrant_url,
            "mem0_qdrant_collection_name": mem0_qdrant_collection_name,
            "mem0_telemetry_enabled": _effective_mem0_telemetry_metadata(
                memory_backend, mem0_runtime
            ),
            "mem0_embedder_provider": "openai",
            "mem0_embedder_model": "text-embedding-3-small",
            "mem0_bm25_enabled": False,
            "mention_memory_system": mention_memory_system,
            "provider": provider_config.provider_id,
            "subject_model": effective_subject_model,
            "prompt_model": prompt_model,
            "model_api": model_route.model_api,
            "model_route_type": model_route.model_route_type,
            "model_vendor": model_route.model_vendor,
            "system_prompt_file": provider_config.system_prompt_file,
            "system_prompt_path": (
                f"sleeper_eval/prompts/provider/{provider_config.system_prompt_file}"
            ),
            "system_prompt_template_sha256": hashlib.sha256(
                prompt_template.encode("utf-8")
            ).hexdigest(),
            "system_prompt_template_chars": len(prompt_template),
            "memory_tool_name": provider_config.memory_tool_name,
            "memory_placement": provider_config.memory_placement,
            "document_representation": provider_config.document_representation,
            "defense_suffix_override": effective_defense_suffix_override,
            "evaluation_mode": evaluation_mode,
            "requested_sample_shuffle": sample_shuffle,
            "requested_sample_shuffle_seed": sample_shuffle_seed,
            "requested_reasoning_effort": reasoning_effort,
            "requested_reasoning_tokens": reasoning_tokens,
            "requested_batch": (
                batch_config.model_dump(mode="json", exclude_none=True)
                if batch_config is not None
                else None
            ),
            "requested_extra_headers": extra_headers,
            "requested_extra_body": extra_body,
        },
    )
