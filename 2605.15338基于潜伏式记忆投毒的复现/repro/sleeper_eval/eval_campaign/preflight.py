"""Preflight validation and resolved campaign planning."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from sleeper_eval.defenses import defense_config_from_names, resolve_builtin_defense_suffix
from sleeper_eval.memory_backend import (
    validate_mem0_prompt_variant,
    validate_mem0_provider,
    validate_mem0_qdrant_mode,
    validate_mem0_reasoning_effort,
    validate_mem0_runtime,
    validate_mem0_thinking,
)
from sleeper_eval.provider_config import get_model_route, get_provider_config, infer_provider

from .config import CampaignConfig, resolve_dataset_file, resolve_defense_suffix
from .env import EnvRequirementStatus, collect_env_statuses
from .scoring import known_scorer_names, scorer_is_compatible, scorer_uses_grader


@dataclass(frozen=True)
class ResolvedDataset:
    label: str
    dataset_file: str
    resolved_path: str
    sample_count: int
    evaluation_mode: str
    attack: str
    subcategory: str
    category: str
    doc_domain: str
    domain_seed: str


@dataclass(frozen=True)
class ResolvedAttack:
    label: str
    attack: str
    defense_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedModel:
    label: str
    model: str
    prompt_model: str
    provider: str
    inferred_provider: str
    model_api: str
    route_type: str
    vendor: str
    prompt_family: str
    prompt_file: str
    memory_tool_name: str
    reasoning_effort: str | None
    reasoning_tokens: int | None
    batch: dict[str, Any] | None
    extra_headers: dict[str, str] | None
    extra_body: dict[str, Any] | None


@dataclass(frozen=True)
class ResolvedDefense:
    label: str
    defense: str
    suffix_length: int
    suffix_sha256: str


@dataclass(frozen=True)
class ExpandedTaskPlan:
    dataset_label: str
    attack_label: str
    model_label: str
    defense_label: str
    dataset_file: str
    evaluation_mode: str
    attack: str
    defense: str
    defense_suffix_override: str
    provider: str
    subject_model: str
    prompt_model: str
    sample_shuffle: bool
    sample_shuffle_seed: int | None
    reasoning_effort: str | None
    reasoning_tokens: int | None
    batch: dict[str, Any] | None
    extra_headers: dict[str, str] | None
    extra_body: dict[str, Any] | None
    memory_backend: str
    mem0_runtime: str
    mem0_provider: str
    mem0_model: str
    mem0_prompt_variant: str
    mem0_include_document_content: bool
    mem0_thinking: str
    mem0_reasoning_effort: str
    mem0_qdrant_mode: str
    mem0_qdrant_url: str
    mem0_qdrant_api_key_env: str
    mem0_qdrant_collection_name: str
    mention_memory_system: bool
    subcategory: str
    category: str
    doc_domain: str
    domain_seed: str


@dataclass(frozen=True)
class PreflightResult:
    config_path: str
    output_dir: str
    output_mode: str
    config_hash: str
    resolved_config: dict[str, Any]
    datasets: tuple[ResolvedDataset, ...]
    attacks: tuple[ResolvedAttack, ...]
    models: tuple[ResolvedModel, ...]
    defenses: tuple[ResolvedDefense, ...]
    env_statuses: tuple[EnvRequirementStatus, ...]
    expanded_tasks: tuple[ExpandedTaskPlan, ...]


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _record_matches_filters(record: dict[str, Any], dataset: Any) -> bool:
    goal = record.get("goal", {}) or {}
    document = record.get("document", {}) or {}

    if dataset.subcategory:
        if str(goal.get("subcategory_id", "")) not in _split_csv_arg(dataset.subcategory):
            return False
    if dataset.category:
        if str(goal.get("category_id", "")) not in _split_csv_arg(dataset.category):
            return False
    if dataset.doc_domain:
        if str(document.get("domain", "")) not in _split_csv_arg(dataset.doc_domain):
            return False
    if dataset.domain_seed:
        if str(goal.get("domain_seed", "")) not in _split_csv_arg(dataset.domain_seed):
            return False
    return True


def _count_dataset_records(path: Path, dataset: Any) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Dataset {path} must contain a top-level JSON list.")
    filtered = [record for record in payload if _record_matches_filters(record, dataset)]
    if not filtered:
        raise ValueError(f"Dataset {path} is empty.")
    return len(filtered)


def _validate_attack_name(attack: str) -> None:
    try:
        importlib.import_module(f"sleeper_eval.attacks.{attack}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unsupported attack '{attack}'.") from exc


def _validate_scoring_compatibility(config: CampaignConfig) -> None:
    known = set(known_scorer_names())
    unknown = [name for name in config.scoring.post_eval_scorers if name not in known]
    if unknown:
        raise ValueError(f"Unknown scorers requested: {sorted(unknown)}.")

    dataset_modes = {dataset.evaluation_mode for dataset in config.datasets}
    for scorer_name in config.scoring.post_eval_scorers:
        compatible_with_any = any(
            scorer_is_compatible(
                scorer_name,
                memory_backend=config.memory_backend,
                evaluation_mode=evaluation_mode,
            )
            for evaluation_mode in dataset_modes
        )
        if not compatible_with_any:
            raise ValueError(
                f"Requested scorer '{scorer_name}' is incompatible with "
                f"memory_backend={config.memory_backend!r} and dataset modes {sorted(dataset_modes)}."
            )


def _validate_mem0_config(config: CampaignConfig) -> None:
    if config.memory_backend != "mem0":
        return

    validate_mem0_runtime(config.mem0_runtime)
    validate_mem0_provider(config.mem0_provider)
    validate_mem0_prompt_variant(config.mem0_prompt_variant)
    validate_mem0_thinking(config.mem0_thinking)
    validate_mem0_reasoning_effort(config.mem0_reasoning_effort)
    validate_mem0_qdrant_mode(config.mem0_qdrant_mode)

    if config.mem0_runtime == "sdk":
        prereq_errors = _mem0_sdk_prereq_errors(config)
        if prereq_errors:
            raise EnvironmentError(
                "mem0 SDK prerequisites are not satisfied:\n- " + "\n- ".join(prereq_errors)
            )

        if config.mem0_qdrant_mode == "local":
            if config.eval.max_tasks is not None and config.eval.max_tasks > 1:
                raise ValueError(
                    "mem0 SDK runs with mem0_qdrant_mode='local' require eval.max_tasks <= 1 because embedded local Qdrant uses filesystem locks."
                )
            if config.eval.max_connections is not None and config.eval.max_connections > 1:
                raise ValueError(
                    "mem0 SDK runs with mem0_qdrant_mode='local' require eval.max_connections <= 1 because embedded local Qdrant uses filesystem locks."
                )


def _mem0_sdk_prereq_errors(config: CampaignConfig) -> list[str]:
    errors: list[str] = []
    if importlib.util.find_spec("mem0") is None:
        errors.append(
            "The 'mem0' package is not installed. Install the repo mem0 eval extras first."
        )
    if importlib.util.find_spec("spacy") is None:
        errors.append(
            "spaCy is not installed. Install the repo mem0 eval extras or `mem0ai[nlp,extras]`."
        )
    if importlib.util.find_spec("fastembed") is None:
        errors.append(
            "fastembed is not installed. Install the repo mem0 eval extras or `mem0ai[nlp,extras]`."
        )
    if importlib.util.find_spec("en_core_web_sm") is None:
        errors.append(
            "The spaCy model `en_core_web_sm` is not installed. Run `uv run python -m spacy download en_core_web_sm`."
        )
    if config.mem0_qdrant_mode == "server" and not config.mem0_qdrant_url.strip():
        errors.append("mem0_qdrant_mode='server' requires mem0_qdrant_url.")
    if config.mem0_qdrant_mode == "managed":
        docker_error = _docker_daemon_error()
        if docker_error is not None:
            errors.append(docker_error)
    remote_api_env = _required_remote_qdrant_api_key_env(config)
    if remote_api_env and not os.getenv(remote_api_env):
        errors.append(
            f"{remote_api_env} is required for the configured remote Qdrant server."
        )
    return errors


def _docker_daemon_error() -> str | None:
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "Docker is not installed or not on PATH, but mem0_qdrant_mode='managed' requires Docker."

    if completed.returncode == 0:
        return None
    message = completed.stderr.strip() or completed.stdout.strip() or "unknown Docker error"
    return f"Docker daemon is unavailable, but mem0_qdrant_mode='managed' requires it: {message}"


def _required_remote_qdrant_api_key_env(config: CampaignConfig) -> str | None:
    if config.mem0_qdrant_mode != "server":
        return None
    parsed = urlparse(config.mem0_qdrant_url.strip())
    hostname = (parsed.hostname or "").lower()
    if hostname in {"", "localhost", "127.0.0.1"}:
        return None
    return config.mem0_qdrant_api_key_env.strip() or None


def _slugify_mem0_collection_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-_").lower()
    return slug or "mem0"


def _default_mem0_qdrant_collection_name(
    config: CampaignConfig,
    *,
    config_hash_seed: str,
) -> str:
    output_suffix = Path(config.output_dir).name or "run"
    return _slugify_mem0_collection_name(
        f"{config.name}-{output_suffix}-{config_hash_seed[:8]}"
    )


def _effective_defense_suffix_text(defense: Any, *, config_dir: Path) -> str:
    suffix_text = resolve_defense_suffix(defense, config_dir=config_dir)
    if suffix_text.strip():
        return suffix_text.strip()
    return resolve_builtin_defense_suffix(defense_config_from_names(defense.defense))


def _resolved_config_dict(
    config: CampaignConfig,
    *,
    config_path: Path,
    datasets: list[ResolvedDataset],
    attacks: list[ResolvedAttack],
    defenses: list[ResolvedDefense],
    mem0_qdrant_collection_name: str,
) -> dict[str, Any]:
    defense_index = {defense.label: defense for defense in defenses}
    return {
        "name": config.name,
        "config_path": str(config_path),
        "output_dir": config.output_dir,
        "memory_backend": config.memory_backend,
        "mem0_runtime": config.mem0_runtime,
        "mem0_provider": config.mem0_provider,
        "mem0_model": config.mem0_model,
        "mem0_prompt_variant": config.mem0_prompt_variant,
        "mem0_include_document_content": config.mem0_include_document_content,
        "mem0_thinking": config.mem0_thinking,
        "mem0_reasoning_effort": config.mem0_reasoning_effort,
        "mem0_qdrant_mode": config.mem0_qdrant_mode,
        "mem0_qdrant_url": config.mem0_qdrant_url,
        "mem0_qdrant_api_key_env": config.mem0_qdrant_api_key_env,
        "mem0_qdrant_collection_name": mem0_qdrant_collection_name,
        "mem0_qdrant_managed_image": config.mem0_qdrant_managed_image,
        "mem0_qdrant_managed_storage_dir": config.mem0_qdrant_managed_storage_dir,
        "mention_memory_system": config.mention_memory_system,
        "grader_model": config.grader_model,
        "retry": config.retry.model_dump(mode="json"),
        "eval": config.eval.model_dump(mode="json"),
        "scoring": config.scoring.model_dump(mode="json"),
        "datasets": [asdict(dataset) for dataset in datasets],
        "attacks": [asdict(attack) for attack in attacks],
        "models": [
            {
                "label": model.label,
                "model": model.model,
                "prompt_model": model.resolved_prompt_model,
                "provider": model.provider,
                "reasoning_effort": model.reasoning_effort,
                "reasoning_tokens": model.reasoning_tokens,
                "batch": model.resolved_batch,
            }
            for model in config.models
        ],
        "defenses": [
            {
                "label": defense.label,
                "defense": defense.defense,
                "suffix_length": defense_index[defense.label].suffix_length,
                "suffix_sha256": defense_index[defense.label].suffix_sha256,
            }
            for defense in config.defenses
        ],
    }


def _config_hash(resolved_config: dict[str, Any]) -> str:
    payload = json.dumps(
        _normalize_resolved_config_for_hash(resolved_config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_resolved_config_for_hash(resolved_config: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(resolved_config))

    eval_config = normalized.get("eval")
    if isinstance(eval_config, dict):
        if eval_config.get("sample_shuffle") is False:
            eval_config.pop("sample_shuffle", None)
        if eval_config.get("sample_shuffle_seed") is None:
            eval_config.pop("sample_shuffle_seed", None)

    attacks = normalized.get("attacks")
    if isinstance(attacks, list):
        for attack in attacks:
            if (
                isinstance(attack, dict)
                and attack.get("defense_labels") in (None, [], ())
            ):
                attack.pop("defense_labels", None)

    return normalized


def _run_requires_grader(config: CampaignConfig) -> bool:
    if config.scoring.during_eval == "full":
        return True
    return any(scorer_uses_grader(name) for name in config.scoring.post_eval_scorers)


def _existing_output_mode(output_dir: Path, *, config_hash: str) -> str:
    if not output_dir.exists():
        return "fresh"
    if not any(output_dir.iterdir()):
        return "fresh"

    manifest_path = output_dir / "run_manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        existing_hash = str(manifest.get("config_hash", ""))
        if existing_hash == config_hash:
            return "resume"
        config_path = output_dir / "config.yaml"
        if config_path.exists():
            existing_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            existing_hash = _config_hash(existing_config)
            if existing_hash == config_hash:
                return "resume"
        raise ValueError(
            f"Output dir {output_dir} already contains a different campaign "
            f"(existing config_hash={existing_hash}, current config_hash={config_hash})."
        )

    config_path = output_dir / "config.yaml"
    if config_path.exists():
        existing_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        existing_hash = _config_hash(existing_config)
        if existing_hash == config_hash:
            return "resume"
        raise ValueError(
            f"Output dir {output_dir} already contains a different config.yaml "
            f"(existing config_hash={existing_hash}, current config_hash={config_hash})."
        )

    raise ValueError(
        f"Output dir {output_dir} exists but does not contain a compatible run_manifest.yaml. "
        "Use a fresh output dir for a new campaign."
    )


def preflight_campaign(config: CampaignConfig, *, config_path: Path) -> PreflightResult:
    _validate_scoring_compatibility(config)
    _validate_mem0_config(config)

    resolved_datasets: list[ResolvedDataset] = []
    for dataset in config.datasets:
        resolved_path = resolve_dataset_file(dataset.dataset_file, config_dir=config_path.parent)
        sample_count = _count_dataset_records(resolved_path, dataset)
        resolved_datasets.append(
            ResolvedDataset(
                label=dataset.label,
                dataset_file=dataset.dataset_file,
                resolved_path=str(resolved_path),
                sample_count=sample_count,
                evaluation_mode=dataset.evaluation_mode,
                attack=dataset.attack,
                subcategory=dataset.subcategory,
                category=dataset.category,
                doc_domain=dataset.doc_domain,
                domain_seed=dataset.domain_seed,
            )
        )

    resolved_attacks: list[ResolvedAttack] = []
    defense_labels = {defense.label for defense in config.defenses}
    for attack in config.attacks:
        _validate_attack_name(attack.attack)
        unknown_defense_labels = sorted(set(attack.defense_labels) - defense_labels)
        if unknown_defense_labels:
            raise ValueError(
                f"Attack label '{attack.label}' references unknown defense labels: {unknown_defense_labels}."
            )
        resolved_attacks.append(
            ResolvedAttack(
                label=attack.label,
                attack=attack.attack,
                defense_labels=tuple(attack.defense_labels),
            )
        )

    resolved_models: list[ResolvedModel] = []
    model_usages: dict[str, list[str]] = {}
    for model in config.models:
        prompt_model = model.resolved_prompt_model
        inferred_provider = infer_provider(prompt_model)
        if model.provider != inferred_provider:
            raise ValueError(
                f"Model label '{model.label}' has provider={model.provider!r}, "
                f"but prompt_model={prompt_model!r} resolves to provider family {inferred_provider!r}."
            )
        route = get_model_route(prompt_model)
        model_route = get_model_route(model.model)
        provider_config = get_provider_config(
            model.provider,
            model_slug=prompt_model,
            memory_backend_name=config.memory_backend,
        )
        resolved_models.append(
            ResolvedModel(
                label=model.label,
                model=model.model,
                prompt_model=prompt_model,
                provider=model.provider,
                inferred_provider=inferred_provider,
                model_api=model_route.model_api,
                route_type=model_route.model_route_type,
                vendor=model_route.model_vendor,
                prompt_family=provider_config.provider_id,
                prompt_file=provider_config.system_prompt_file,
                memory_tool_name=provider_config.memory_tool_name,
                reasoning_effort=model.reasoning_effort,
                reasoning_tokens=model.reasoning_tokens,
                batch=model.resolved_batch,
                extra_headers=model.resolved_extra_headers,
                extra_body=model.resolved_extra_body,
            )
        )
        model_usages.setdefault(model.model, []).append(f"subject:{model.label}")

    resolved_defenses: list[ResolvedDefense] = []
    for defense in config.defenses:
        defense_config_from_names(defense.defense)
        suffix_text = _effective_defense_suffix_text(defense, config_dir=config_path.parent)
        if defense.defense_suffix_file and not suffix_text:
            raise ValueError(f"Defense suffix file for '{defense.label}' is empty.")
        resolved_defenses.append(
            ResolvedDefense(
                label=defense.label,
                defense=defense.defense,
                suffix_length=len(suffix_text),
                suffix_sha256=hashlib.sha256(suffix_text.encode("utf-8")).hexdigest(),
            )
        )

    if _run_requires_grader(config):
        model_usages.setdefault(config.grader_model, []).append("grader")
    env_statuses = collect_env_statuses(
        model_usages,
        mem0_runtime=config.mem0_runtime if config.memory_backend == "mem0" else None,
        mem0_provider=config.mem0_provider if config.memory_backend == "mem0" else None,
        mem0_qdrant_mode=config.mem0_qdrant_mode if config.memory_backend == "mem0" else None,
        mem0_qdrant_url=config.mem0_qdrant_url if config.memory_backend == "mem0" else None,
        mem0_qdrant_api_key_env=(
            config.mem0_qdrant_api_key_env if config.memory_backend == "mem0" else None
        ),
    )
    missing = [status for status in env_statuses if status.status == "missing"]
    if missing:
        details = "; ".join(
            f"{status.requirement} for {', '.join(status.used_by)}"
            for status in missing
        )
        raise EnvironmentError(f"Missing required environment variables: {details}")

    defense_suffixes = {
        defense.label: _effective_defense_suffix_text(defense, config_dir=config_path.parent)
        for defense in config.defenses
    }
    defense_by_label = {defense.label: defense for defense in config.defenses}
    config_hash_seed = _config_hash(
        {
            "name": config.name,
            "output_dir": config.output_dir,
            "memory_backend": config.memory_backend,
            "mem0_runtime": config.mem0_runtime,
            "mem0_provider": config.mem0_provider,
            "mem0_model": config.mem0_model,
            "mem0_prompt_variant": config.mem0_prompt_variant,
            "datasets": [asdict(dataset) for dataset in resolved_datasets],
            "models": [
                {
                    "label": model.label,
                    "model": model.model,
                    "prompt_model": model.resolved_prompt_model,
                    "provider": model.provider,
                    "extra_headers": model.resolved_extra_headers,
                    "extra_body": model.resolved_extra_body,
                }
                for model in config.models
            ],
            "defenses": [
                {
                    "label": defense.label,
                    "defense": defense.defense,
                }
                for defense in config.defenses
            ],
        }
    )
    mem0_qdrant_collection_name = (
        config.mem0_qdrant_collection_name.strip()
        or _default_mem0_qdrant_collection_name(config, config_hash_seed=config_hash_seed)
    )
    expanded_tasks_list: list[ExpandedTaskPlan] = []
    for dataset in resolved_datasets:
        dataset_attacks: list[ResolvedAttack]
        if dataset.attack:
            _validate_attack_name(dataset.attack)
            dataset_attacks = [ResolvedAttack(label=dataset.attack, attack=dataset.attack)]
        elif dataset.evaluation_mode == "benign_save":
            dataset_attacks = [ResolvedAttack(label="none", attack="none")]
        elif resolved_attacks:
            dataset_attacks = resolved_attacks
        else:
            _validate_attack_name("none")
            dataset_attacks = [ResolvedAttack(label="none", attack="none")]

        for attack in dataset_attacks:
            if attack.defense_labels:
                dataset_defenses = [defense_by_label[label] for label in attack.defense_labels]
            else:
                dataset_defenses = config.defenses
            for model in resolved_models:
                for defense in dataset_defenses:
                    expanded_tasks_list.append(
                        ExpandedTaskPlan(
                            dataset_label=dataset.label,
                            attack_label=attack.label,
                            model_label=model.label,
                            defense_label=defense.label,
                            dataset_file=dataset.resolved_path,
                            evaluation_mode=dataset.evaluation_mode,
                            attack=attack.attack,
                            defense=defense.defense,
                            defense_suffix_override=defense_suffixes[defense.label],
                            provider=model.provider,
                            subject_model=model.model,
                            prompt_model=model.prompt_model,
                            reasoning_effort=model.reasoning_effort,
                            reasoning_tokens=model.reasoning_tokens,
                            batch=model.batch,
                            extra_headers=model.extra_headers,
                            extra_body=model.extra_body,
                            sample_shuffle=config.eval.sample_shuffle,
                            sample_shuffle_seed=config.eval.sample_shuffle_seed,
                            memory_backend=config.memory_backend,
                            mem0_runtime=config.mem0_runtime,
                            mem0_provider=config.mem0_provider,
                            mem0_model=config.mem0_model,
                            mem0_prompt_variant=config.mem0_prompt_variant,
                            mem0_include_document_content=config.mem0_include_document_content,
                            mem0_thinking=config.mem0_thinking,
                            mem0_reasoning_effort=config.mem0_reasoning_effort,
                            mem0_qdrant_mode=config.mem0_qdrant_mode,
                            mem0_qdrant_url=config.mem0_qdrant_url,
                            mem0_qdrant_api_key_env=config.mem0_qdrant_api_key_env,
                            mem0_qdrant_collection_name=mem0_qdrant_collection_name,
                            mention_memory_system=config.mention_memory_system,
                            subcategory=dataset.subcategory,
                            category=dataset.category,
                            doc_domain=dataset.doc_domain,
                            domain_seed=dataset.domain_seed,
                        )
                    )
    expanded_tasks = tuple(expanded_tasks_list)

    resolved_config = _resolved_config_dict(
        config,
        config_path=config_path,
        datasets=resolved_datasets,
        attacks=resolved_attacks,
        defenses=resolved_defenses,
        mem0_qdrant_collection_name=mem0_qdrant_collection_name,
    )
    config_hash = _config_hash(resolved_config)
    output_mode = _existing_output_mode(Path(config.output_dir), config_hash=config_hash)

    return PreflightResult(
        config_path=str(config_path),
        output_dir=config.output_dir,
        output_mode=output_mode,
        config_hash=config_hash,
        resolved_config=resolved_config,
        datasets=tuple(resolved_datasets),
        attacks=tuple(resolved_attacks),
        models=tuple(resolved_models),
        defenses=tuple(resolved_defenses),
        env_statuses=tuple(env_statuses),
        expanded_tasks=expanded_tasks,
    )
