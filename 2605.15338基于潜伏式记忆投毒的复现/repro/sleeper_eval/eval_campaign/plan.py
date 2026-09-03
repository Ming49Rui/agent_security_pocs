"""Formatting helpers for resolved campaign plans."""

from __future__ import annotations

import json

from .preflight import PreflightResult


def render_plan(result: PreflightResult) -> str:
    resolved = result.resolved_config
    lines: list[str] = [
        "== Eval Campaign Plan ==",
        f"config={result.config_path}",
        f"output_dir={result.output_dir}",
        f"output_mode={result.output_mode}",
        f"config_hash={result.config_hash}",
        "",
        "campaign:",
        f"  - name={resolved['name']}",
        f"  - memory_backend={resolved['memory_backend']}",
        f"  - mem0_runtime={resolved['mem0_runtime']}",
        f"  - mem0_provider={resolved['mem0_provider']}",
        f"  - mem0_model={resolved['mem0_model']}",
        f"  - mem0_prompt_variant={resolved['mem0_prompt_variant']}",
        f"  - mem0_include_document_content={resolved['mem0_include_document_content']}",
        f"  - mem0_thinking={resolved['mem0_thinking']}",
        f"  - mem0_reasoning_effort={resolved['mem0_reasoning_effort']}",
        f"  - mem0_qdrant_mode={resolved['mem0_qdrant_mode']}",
        "  - mem0_qdrant_url="
        + (
            resolved["mem0_qdrant_url"]
            or ("(managed/runtime)" if resolved["mem0_qdrant_mode"] == "managed" else "(none)")
        ),
        f"  - mem0_qdrant_collection_name={resolved['mem0_qdrant_collection_name']}",
        f"  - mem0_qdrant_managed_image={resolved['mem0_qdrant_managed_image']}",
        f"  - mention_memory_system={resolved['mention_memory_system']}",
        f"  - grader_model={resolved['grader_model']}",
        "",
        f"datasets={len(result.datasets)}",
    ]
    for dataset in result.datasets:
        filter_parts = [
            f"subcategory={dataset.subcategory}" if dataset.subcategory else "",
            f"category={dataset.category}" if dataset.category else "",
            f"doc_domain={dataset.doc_domain}" if dataset.doc_domain else "",
            f"domain_seed={dataset.domain_seed}" if dataset.domain_seed else "",
        ]
        filters = [part for part in filter_parts if part]
        suffix = f", filters={{{', '.join(filters)}}}" if filters else ""
        lines.append(
            f"  - {dataset.label}: {dataset.resolved_path} "
            f"(samples={dataset.sample_count}, mode={dataset.evaluation_mode}, "
            f"fixed_attack={dataset.attack or '(global/default)'}{suffix})"
        )

    lines.extend(
        [
            "",
            f"attacks={len(result.attacks)}",
        ]
    )
    if result.attacks:
        for attack in result.attacks:
            subset = (
                f" defense_labels={list(attack.defense_labels)}"
                if attack.defense_labels
                else ""
            )
            lines.append(f"  - {attack.label}: attack={attack.attack}{subset}")
    else:
        lines.append("  - none (datasets use fixed attacks or default to none)")

    lines.extend(
        [
            "",
            f"models={len(result.models)}",
        ]
    )
    for model in result.models:
        reasoning_parts = []
        if model.reasoning_effort is not None:
            reasoning_parts.append(f"reasoning_effort={model.reasoning_effort}")
        if model.reasoning_tokens is not None:
            reasoning_parts.append(f"reasoning_tokens={model.reasoning_tokens}")
        if model.batch is not None:
            batch_value = (
                "enabled"
                if not model.batch
                else json.dumps(model.batch, sort_keys=True, separators=(",", ":"))
            )
            reasoning_parts.append(f"batch={batch_value}")
        reasoning_suffix = (
            f", {', '.join(reasoning_parts)}" if reasoning_parts else ""
        )
        lines.append(
            f"  - {model.label}: model={model.model} prompt_model={model.prompt_model} "
            f"(provider={model.provider}, api={model.model_api or '(unknown)'}, "
            f"route={model.route_type}, vendor={model.vendor or '(unknown)'}, "
            f"prompt={model.prompt_file}, tool={model.memory_tool_name}{reasoning_suffix})"
        )

    lines.extend(
        [
            "",
            f"defenses={len(result.defenses)}",
        ]
    )
    for defense in result.defenses:
        lines.append(
            f"  - {defense.label}: defense={defense.defense!r} "
            f"(suffix_len={defense.suffix_length}, suffix_sha256={defense.suffix_sha256[:12]})"
        )

    lines.extend(["", "required_env_vars:"])
    if result.env_statuses:
        for status in result.env_statuses:
            lines.append(
                f"  - {status.requirement}: {status.status} ({', '.join(status.used_by)})"
            )
    else:
        lines.append("  - none")

    retry = resolved["retry"]
    eval_config = resolved["eval"]
    scoring = resolved["scoring"]
    lines.extend(
        [
            "",
            "retry:",
            f"  - retry_attempts={retry['retry_attempts']}",
            f"  - retry_immediate={retry['retry_immediate']}",
            f"  - retry_wait={retry['retry_wait']}",
            f"  - retry_connections={retry['retry_connections']}",
            f"  - retry_on_error={retry['retry_on_error']}",
            "",
            "eval:",
            f"  - limit={eval_config['limit']}",
            f"  - sample_shuffle={eval_config['sample_shuffle']}",
            f"  - sample_shuffle_seed={eval_config['sample_shuffle_seed']}",
            f"  - max_tasks={eval_config['max_tasks']}",
            f"  - max_samples={eval_config['max_samples']}",
            f"  - max_connections={eval_config['max_connections']}",
            f"  - fail_on_error={eval_config['fail_on_error']}",
            f"  - bundle={eval_config['bundle']}",
            f"  - bundle_overwrite={eval_config['bundle_overwrite']}",
            f"  - max_retries={eval_config['max_retries']}",
            f"  - attempt_timeout={eval_config['attempt_timeout']}",
            "",
            "scoring:",
            f"  - during_eval={scoring['during_eval']}",
            f"  - post_eval_scorers={scoring['post_eval_scorers']}",
            f"  - post_eval_action={scoring['post_eval_action']}",
            "",
            f"expanded_tasks={len(result.expanded_tasks)}",
        ]
    )
    for task in result.expanded_tasks:
        lines.append(
            f"  - dataset={task.dataset_label} attack={task.attack_label}:{task.attack} "
            f"model={task.model_label} defense={task.defense_label} mode={task.evaluation_mode}"
        )

    return "\n".join(lines)
