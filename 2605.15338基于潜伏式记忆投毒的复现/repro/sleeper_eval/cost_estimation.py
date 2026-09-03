"""Estimate in-distribution eval costs from smoke logs and dataset prompts."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sleeper_eval.attacks.universal_v1 import apply as universal_attack
from sleeper_eval.dataset import SampleMeta, load_dataset, resolve_dataset_path
from sleeper_eval.defenses import defense_config_from_names
from sleeper_eval.provider_config import get_provider_config
from sleeper_eval.solver import (
    build_system_prompt,
    build_user_message,
    format_memories_model_set_context,
)

DEFAULT_DATASET_FILES = [
    "in_dist/merged_dev_with_memories.json",
    "in_dist/merged_dev_without_memories.json",
    "in_dist/merged_eval_in_with_memories.json",
    "in_dist/merged_eval_in_without_memories.json",
]
DEFAULT_DEFENSES = ["", "untrusted_content_markers", "system_prompt_hardening"]
DEFAULT_GRADER_MODEL = "openai/gpt-5.4-mini"
DEFAULT_TARGET_MODELS = [
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4-6",
    "openrouter/google/gemini-3.1-pro-preview",
]
DEFAULT_LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_CHARS_PER_TOKEN = 4.0

NONE_LABEL = "(none)"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    input_tokens_cache_read: float = 0.0
    input_tokens_cache_write: float = 0.0
    reasoning_tokens: float = 0.0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> TokenUsage:
        data = payload or {}
        return cls(
            input_tokens=float(data.get("input_tokens", 0) or 0),
            output_tokens=float(data.get("output_tokens", 0) or 0),
            input_tokens_cache_read=float(data.get("input_tokens_cache_read", 0) or 0),
            input_tokens_cache_write=float(data.get("input_tokens_cache_write", 0) or 0),
            reasoning_tokens=float(data.get("reasoning_tokens", 0) or 0),
        )

    @property
    def total_tokens(self) -> float:
        return (
            self.input_tokens
            + self.output_tokens
            + self.input_tokens_cache_read
            + self.input_tokens_cache_write
            + self.reasoning_tokens
        )

    def scale_all(self, factor: float) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens * factor,
            output_tokens=self.output_tokens * factor,
            input_tokens_cache_read=self.input_tokens_cache_read * factor,
            input_tokens_cache_write=self.input_tokens_cache_write * factor,
            reasoning_tokens=self.reasoning_tokens * factor,
        )

    def scale_inputs(self, factor: float) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens * factor,
            output_tokens=self.output_tokens,
            input_tokens_cache_read=self.input_tokens_cache_read * factor,
            input_tokens_cache_write=self.input_tokens_cache_write * factor,
            reasoning_tokens=self.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        return payload

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            input_tokens_cache_read=self.input_tokens_cache_read + other.input_tokens_cache_read,
            input_tokens_cache_write=self.input_tokens_cache_write + other.input_tokens_cache_write,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class PricingRates:
    model: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None = None
    cache_write_per_million: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostBreakdown:
    input_cost: float
    cache_read_cost: float
    cache_write_cost: float
    output_cost: float
    total_cost: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SmokeSource:
    target_model: str
    observed_model: str
    pattern: str
    note: str = ""


@dataclass(frozen=True)
class SmokeObservation:
    sample_id: str
    memory_condition: str
    prompt_tokens_estimate: int
    transcript_tokens_estimate: int
    subject_usage: TokenUsage
    grader_usage: TokenUsage


@dataclass(frozen=True)
class UsageBaseline:
    sample_count: int
    avg_prompt_tokens_estimate: float
    avg_transcript_tokens_estimate: float
    avg_subject_usage: TokenUsage
    avg_grader_usage: TokenUsage

    def to_dict(self) -> dict[str, Any]:
        observed_input_like = (
            self.avg_subject_usage.input_tokens
            + self.avg_subject_usage.input_tokens_cache_read
            + self.avg_subject_usage.input_tokens_cache_write
        )
        return {
            "sample_count": self.sample_count,
            "avg_prompt_tokens_estimate": self.avg_prompt_tokens_estimate,
            "avg_transcript_tokens_estimate": self.avg_transcript_tokens_estimate,
            "avg_observed_input_like_tokens": observed_input_like,
            "avg_prompt_estimate_gap_vs_observed_input": (
                self.avg_prompt_tokens_estimate - observed_input_like
            ),
            "avg_transcript_estimate_gap_vs_observed_total": (
                self.avg_transcript_tokens_estimate - self.avg_subject_usage.total_tokens
            ),
            "avg_subject_usage": self.avg_subject_usage.to_dict(),
            "avg_grader_usage": self.avg_grader_usage.to_dict(),
        }


@dataclass(frozen=True)
class SmokeProfile:
    log_dir: str
    subject_model: str
    grader_model: str
    aggregate_usage_by_model: dict[str, TokenUsage]
    aggregate_role_usage: dict[str, TokenUsage]
    baselines_by_condition: dict[str, UsageBaseline]
    overall_baseline: UsageBaseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_dir": self.log_dir,
            "subject_model": self.subject_model,
            "grader_model": self.grader_model,
            "aggregate_usage_by_model": {
                model: usage.to_dict() for model, usage in self.aggregate_usage_by_model.items()
            },
            "aggregate_role_usage": {
                role: usage.to_dict() for role, usage in self.aggregate_role_usage.items()
            },
            "baselines_by_condition": {
                condition: baseline.to_dict()
                for condition, baseline in self.baselines_by_condition.items()
            },
            "overall_baseline": self.overall_baseline.to_dict(),
        }


SMOKE_SOURCES = {
    "openai/gpt-5.4": SmokeSource(
        target_model="openai/gpt-5.4",
        observed_model="openai/gpt-5.4",
        pattern="actual-smoke-gpt54-*",
    ),
    "anthropic/claude-sonnet-4-6": SmokeSource(
        target_model="anthropic/claude-sonnet-4-6",
        observed_model="anthropic/claude-sonnet-4-6",
        pattern="actual-smoke-claude-sonnet46-*",
    ),
    "openrouter/google/gemini-3.1-pro-preview": SmokeSource(
        target_model="openrouter/google/gemini-3.1-pro-preview",
        observed_model="openrouter/google/gemini-3.1-flash-lite-preview",
        pattern="actual-smoke-gemini31-flash-lite-*",
        note=(
            "Using Gemini 3.1 Flash Lite smoke usage as a proxy for Gemini 3.1 Pro prompt/output "
            "shape because no Gemini 3.1 Pro smoke log was found in the repo."
        ),
    ),
}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _average_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    items = list(usages)
    if not items:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_mean(item.input_tokens for item in items),
        output_tokens=_mean(item.output_tokens for item in items),
        input_tokens_cache_read=_mean(item.input_tokens_cache_read for item in items),
        input_tokens_cache_write=_mean(item.input_tokens_cache_write for item in items),
        reasoning_tokens=_mean(item.reasoning_tokens for item in items),
    )


def format_defense_label(defense_name: str) -> str:
    return defense_name if defense_name else NONE_LABEL


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def split_semicolon_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def count_text_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than 0.")
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


def extract_text_segments(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return parts
    return []


def build_initial_prompt_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user"}:
            break
        parts.extend(extract_text_segments(message.get("content")))
    return "\n\n".join(part for part in parts if part.strip())


def build_full_transcript_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.extend(extract_text_segments(message.get("content")))
    return "\n\n".join(part for part in parts if part.strip())


def locate_latest_smoke_log_dir(logs_root: Path, pattern: str) -> Path:
    matches = sorted(path for path in logs_root.glob(pattern) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No smoke log directories matched '{pattern}' under {logs_root}.")
    return matches[-1]


def load_logs_json_summary(log_dir: Path) -> tuple[dict[str, TokenUsage], dict[str, TokenUsage]]:
    path = log_dir / "logs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing logs summary file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        raise ValueError(f"No eval entries found in {path}")
    aggregate_models: dict[str, TokenUsage] = defaultdict(TokenUsage)
    aggregate_roles: dict[str, TokenUsage] = defaultdict(TokenUsage)
    for entry in payload.values():
        stats = entry.get("stats", {})
        for model, usage in (stats.get("model_usage") or {}).items():
            aggregate_models[model] = aggregate_models[model] + TokenUsage.from_payload(usage)
        for role, usage in (stats.get("role_usage") or {}).items():
            aggregate_roles[role] = aggregate_roles[role] + TokenUsage.from_payload(usage)
    return dict(aggregate_models), dict(aggregate_roles)


def iter_eval_sample_payloads(log_dir: Path) -> Iterable[dict[str, Any]]:
    eval_paths = sorted(log_dir.glob("*.eval"))
    if not eval_paths:
        raise FileNotFoundError(f"No .eval files found in {log_dir}")
    for eval_path in eval_paths:
        with zipfile.ZipFile(eval_path) as archive:
            sample_names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            )
            for sample_name in sample_names:
                yield json.loads(archive.read(sample_name))


def get_grader_usage(
    sample_payload: Mapping[str, Any],
    grader_model: str,
) -> TokenUsage:
    model_usage = sample_payload.get("model_usage") or {}
    if grader_model in model_usage:
        return TokenUsage.from_payload(model_usage[grader_model])
    role_usage = sample_payload.get("role_usage") or {}
    return TokenUsage.from_payload(role_usage.get("grader"))


def build_smoke_profile(
    log_dir: Path,
    *,
    subject_model: str,
    grader_model: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> SmokeProfile:
    aggregate_usage_by_model, aggregate_role_usage = load_logs_json_summary(log_dir)
    observations: list[SmokeObservation] = []
    for sample in iter_eval_sample_payloads(log_dir):
        metadata = sample.get("metadata") or {}
        messages = sample.get("messages") or []
        prompt_tokens = count_text_tokens(
            build_initial_prompt_text(messages),
            chars_per_token=chars_per_token,
        )
        transcript_tokens = count_text_tokens(
            build_full_transcript_text(messages),
            chars_per_token=chars_per_token,
        )
        model_usage = sample.get("model_usage") or {}
        observations.append(
            SmokeObservation(
                sample_id=sample["id"],
                memory_condition=metadata.get("memory_condition", "unknown"),
                prompt_tokens_estimate=prompt_tokens,
                transcript_tokens_estimate=transcript_tokens,
                subject_usage=TokenUsage.from_payload(model_usage.get(subject_model)),
                grader_usage=get_grader_usage(sample, grader_model),
            )
        )

    if not observations:
        raise ValueError(f"No sample observations were extracted from {log_dir}")

    def make_baseline(items: list[SmokeObservation]) -> UsageBaseline:
        return UsageBaseline(
            sample_count=len(items),
            avg_prompt_tokens_estimate=_mean(item.prompt_tokens_estimate for item in items),
            avg_transcript_tokens_estimate=_mean(item.transcript_tokens_estimate for item in items),
            avg_subject_usage=_average_usage(item.subject_usage for item in items),
            avg_grader_usage=_average_usage(item.grader_usage for item in items),
        )

    by_condition: dict[str, UsageBaseline] = {}
    grouped: dict[str, list[SmokeObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.memory_condition].append(observation)
    for condition, items in grouped.items():
        by_condition[condition] = make_baseline(items)

    return SmokeProfile(
        log_dir=str(log_dir),
        subject_model=subject_model,
        grader_model=grader_model,
        aggregate_usage_by_model=aggregate_usage_by_model,
        aggregate_role_usage=aggregate_role_usage,
        baselines_by_condition=by_condition,
        overall_baseline=make_baseline(observations),
    )


def build_prompt_messages(
    meta: SampleMeta,
    *,
    model_slug: str,
    defense_name: str,
) -> list[str]:
    defense_config = defense_config_from_names(defense_name)
    provider_config = get_provider_config("", model_slug=model_slug)
    attacked_document = universal_attack(meta.doc_text, meta.goal_text, meta.model_dump())
    messages = [
        build_system_prompt(
            meta,
            defense_config,
            provider_config=provider_config,
            memory_backend_name="tool",
            mention_memory_system=True,
        )
    ]
    if provider_config.memory_placement == "model_set_context" and meta.memories:
        entry_date = provider_config.template_vars.get("CURRENT_DATE", "2026-03-25")
        messages.append(format_memories_model_set_context(meta.memories, entry_date=entry_date))
    messages.append(
        build_user_message(
            meta=meta,
            provider_config=provider_config,
            query_text=meta.query_text,
            document_text=attacked_document,
            defense_config=defense_config,
        )
    )
    return messages


def count_prompt_tokens_for_sample(
    meta: SampleMeta,
    *,
    model_slug: str,
    defense_name: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    return count_text_tokens(
        "\n\n".join(build_prompt_messages(meta, model_slug=model_slug, defense_name=defense_name)),
        chars_per_token=chars_per_token,
    )


def get_pricing_rates(
    model_slug: str,
    *,
    anthropic_cache_write_per_million: float = 3.75,
) -> PricingRates:
    if model_slug == "openai/gpt-5.4":
        return PricingRates(
            model=model_slug,
            input_per_million=2.50,
            cache_read_per_million=0.25,
            output_per_million=15.00,
            notes=("Using standard GPT-5.4 short-context pricing.",),
        )
    if model_slug == "openai/gpt-5.4-mini":
        return PricingRates(
            model=model_slug,
            input_per_million=0.75,
            cache_read_per_million=0.075,
            output_per_million=4.50,
            notes=("Using standard GPT-5.4 mini pricing.",),
        )
    if model_slug == "anthropic/claude-sonnet-4-6":
        return PricingRates(
            model=model_slug,
            input_per_million=3.00,
            cache_write_per_million=anthropic_cache_write_per_million,
            cache_read_per_million=0.30,
            output_per_million=15.00,
            notes=(
                "Anthropic cache writes are priced using the 5-minute cache-write rate by default.",
            ),
        )
    if model_slug == "openrouter/google/gemini-3.1-pro-preview":
        return PricingRates(
            model=model_slug,
            input_per_million=2.00,
            cache_read_per_million=2.00,
            output_per_million=12.00,
            notes=(
                "OpenRouter Gemini cache-read tokens are assumed to bill at the standard input rate.",
            ),
        )
    raise ValueError(f"No pricing table configured for model '{model_slug}'.")


def price_usage(usage: TokenUsage, pricing: PricingRates) -> CostBreakdown:
    input_cost = usage.input_tokens * pricing.input_per_million / 1_000_000
    cache_read_rate = pricing.cache_read_per_million or pricing.input_per_million
    cache_write_rate = pricing.cache_write_per_million or pricing.input_per_million
    cache_read_cost = usage.input_tokens_cache_read * cache_read_rate / 1_000_000
    cache_write_cost = usage.input_tokens_cache_write * cache_write_rate / 1_000_000
    output_cost = usage.output_tokens * pricing.output_per_million / 1_000_000
    total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
    return CostBreakdown(
        input_cost=input_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        output_cost=output_cost,
        total_cost=total_cost,
    )


def estimate_sample_usage(
    baseline: UsageBaseline,
    *,
    prompt_tokens_estimate: int,
) -> tuple[TokenUsage, TokenUsage]:
    ratio = 1.0
    if baseline.avg_prompt_tokens_estimate > 0:
        ratio = prompt_tokens_estimate / baseline.avg_prompt_tokens_estimate
    subject_usage = baseline.avg_subject_usage.scale_inputs(ratio)
    grader_usage = baseline.avg_grader_usage
    return subject_usage, grader_usage


def validate_dataset_files(dataset_files: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    missing: list[str] = []
    for dataset_file in dataset_files:
        try:
            resolved.append(resolve_dataset_path(dataset_file))
        except FileNotFoundError:
            missing.append(str(dataset_file))
    if missing:
        listed = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Missing required in-dist dataset files. The estimator expects local dataset JSONs.\n"
            f"{listed}"
        )
    return resolved


def build_model_estimate(
    *,
    target_model: str,
    grader_model: str,
    smoke_profile: SmokeProfile,
    dataset_files: list[str | Path],
    defenses: list[str],
    attack_multiplier: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    anthropic_cache_write_per_million: float = 3.75,
) -> dict[str, Any]:
    subject_pricing = get_pricing_rates(
        target_model,
        anthropic_cache_write_per_million=anthropic_cache_write_per_million,
    )
    grader_pricing = get_pricing_rates(grader_model)

    task_breakdowns: list[dict[str, Any]] = []
    defense_rollup_usage: dict[str, dict[str, TokenUsage]] = defaultdict(
        lambda: {"subject": TokenUsage(), "grader": TokenUsage()}
    )
    dataset_rollup_usage: dict[str, dict[str, TokenUsage]] = defaultdict(
        lambda: {"subject": TokenUsage(), "grader": TokenUsage()}
    )
    total_subject_usage = TokenUsage()
    total_grader_usage = TokenUsage()
    total_sample_evaluations = 0

    for dataset_file in dataset_files:
        dataset = load_dataset(dataset_file)
        dataset_path = resolve_dataset_path(dataset_file)
        for defense_name in defenses:
            task_subject = TokenUsage()
            task_grader = TokenUsage()
            condition_counts: dict[str, int] = defaultdict(int)

            for sample in dataset:
                meta = SampleMeta.model_validate(sample.metadata)
                prompt_tokens = count_prompt_tokens_for_sample(
                    meta,
                    model_slug=target_model,
                    defense_name=defense_name,
                    chars_per_token=chars_per_token,
                )
                baseline = smoke_profile.baselines_by_condition.get(
                    meta.memory_condition,
                    smoke_profile.overall_baseline,
                )
                estimated_subject, estimated_grader = estimate_sample_usage(
                    baseline,
                    prompt_tokens_estimate=prompt_tokens,
                )
                task_subject = task_subject + estimated_subject
                task_grader = task_grader + estimated_grader
                condition_counts[meta.memory_condition] += 1

            task_subject = task_subject.scale_all(attack_multiplier)
            task_grader = task_grader.scale_all(attack_multiplier)
            total_subject_usage = total_subject_usage + task_subject
            total_grader_usage = total_grader_usage + task_grader
            total_sample_evaluations += len(dataset) * attack_multiplier
            defense_rollup_usage[defense_name]["subject"] = (
                defense_rollup_usage[defense_name]["subject"] + task_subject
            )
            defense_rollup_usage[defense_name]["grader"] = (
                defense_rollup_usage[defense_name]["grader"] + task_grader
            )
            dataset_key = str(dataset_path)
            dataset_rollup_usage[dataset_key]["subject"] = (
                dataset_rollup_usage[dataset_key]["subject"] + task_subject
            )
            dataset_rollup_usage[dataset_key]["grader"] = (
                dataset_rollup_usage[dataset_key]["grader"] + task_grader
            )

            subject_cost = price_usage(task_subject, subject_pricing)
            grader_cost = price_usage(task_grader, grader_pricing)
            task_breakdowns.append(
                {
                    "dataset_file": str(dataset_path),
                    "dataset_name": dataset_path.name,
                    "defense": defense_name,
                    "defense_label": format_defense_label(defense_name),
                    "sample_count": len(dataset),
                    "condition_counts": dict(condition_counts),
                    "subject_usage": task_subject.to_dict(),
                    "grader_usage": task_grader.to_dict(),
                    "subject_cost": subject_cost.to_dict(),
                    "grader_cost": grader_cost.to_dict(),
                    "total_cost": subject_cost.total_cost + grader_cost.total_cost,
                }
            )

    subject_cost_total = price_usage(total_subject_usage, subject_pricing)
    grader_cost_total = price_usage(total_grader_usage, grader_pricing)
    return {
        "model": target_model,
        "grader_model": grader_model,
        "smoke_profile": smoke_profile.to_dict(),
        "subject_pricing": asdict(subject_pricing),
        "grader_pricing": asdict(grader_pricing),
        "attack_multiplier": attack_multiplier,
        "sample_evaluations": total_sample_evaluations,
        "subject_usage": total_subject_usage.to_dict(),
        "grader_usage": total_grader_usage.to_dict(),
        "subject_cost": subject_cost_total.to_dict(),
        "grader_cost": grader_cost_total.to_dict(),
        "total_cost": subject_cost_total.total_cost + grader_cost_total.total_cost,
        "cost_per_sample_evaluation": (
            (subject_cost_total.total_cost + grader_cost_total.total_cost) / total_sample_evaluations
            if total_sample_evaluations
            else 0.0
        ),
        "tasks": task_breakdowns,
        "per_defense": {
            format_defense_label(defense): {
                "subject_usage": usage["subject"].to_dict(),
                "grader_usage": usage["grader"].to_dict(),
                "subject_cost": price_usage(usage["subject"], subject_pricing).to_dict(),
                "grader_cost": price_usage(usage["grader"], grader_pricing).to_dict(),
                "total_cost": (
                    price_usage(usage["subject"], subject_pricing).total_cost
                    + price_usage(usage["grader"], grader_pricing).total_cost
                ),
            }
            for defense, usage in defense_rollup_usage.items()
        },
        "per_dataset": {
            dataset_name: {
                "subject_usage": usage["subject"].to_dict(),
                "grader_usage": usage["grader"].to_dict(),
                "subject_cost": price_usage(usage["subject"], subject_pricing).to_dict(),
                "grader_cost": price_usage(usage["grader"], grader_pricing).to_dict(),
                "total_cost": (
                    price_usage(usage["subject"], subject_pricing).total_cost
                    + price_usage(usage["grader"], grader_pricing).total_cost
                ),
            }
            for dataset_name, usage in dataset_rollup_usage.items()
        },
    }


def generate_cost_report(
    *,
    models: list[str],
    dataset_files: list[str | Path],
    defenses: list[str],
    attack_multiplier: int,
    logs_root: Path = DEFAULT_LOGS_ROOT,
    grader_model: str = DEFAULT_GRADER_MODEL,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    anthropic_cache_write_per_million: float = 3.75,
    smoke_log_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    resolved_datasets = validate_dataset_files(dataset_files)
    smoke_log_overrides = smoke_log_overrides or {}
    estimates: list[dict[str, Any]] = []
    assumptions = [
        "Provider-reported Inspect usage is the billing baseline; prompt-size scaling uses a character-based estimate rather than tokenizer-exact counts.",
        f"Token estimates use approximately 1 token ~= {chars_per_token:g} characters.",
        "The estimator assumes one attack by default and scales linearly with --attack-multiplier.",
        "Only the tool backend is modeled by default.",
        "OpenAI cached input uses the cached-input price tier.",
        "Anthropic cache writes default to the 5-minute write rate unless overridden.",
        "OpenRouter Gemini cache-read tokens, when present, are billed at the standard input rate.",
    ]

    for model in models:
        smoke_source = SMOKE_SOURCES[model]
        override = smoke_log_overrides.get(model)
        smoke_log_dir = Path(override) if override is not None else locate_latest_smoke_log_dir(
            logs_root,
            smoke_source.pattern,
        )
        smoke_profile = build_smoke_profile(
            smoke_log_dir,
            subject_model=smoke_source.observed_model,
            grader_model=grader_model,
            chars_per_token=chars_per_token,
        )
        estimate = build_model_estimate(
            target_model=model,
            grader_model=grader_model,
            smoke_profile=smoke_profile,
            dataset_files=[str(path) for path in resolved_datasets],
            defenses=defenses,
            attack_multiplier=attack_multiplier,
            chars_per_token=chars_per_token,
            anthropic_cache_write_per_million=anthropic_cache_write_per_million,
        )
        notes = list(estimate["subject_pricing"]["notes"]) + list(estimate["grader_pricing"]["notes"])
        if smoke_source.note:
            notes.append(smoke_source.note)
        estimate["notes"] = notes
        estimates.append(estimate)

    return {
        "config": {
            "models": models,
            "dataset_files": [str(path) for path in resolved_datasets],
            "defenses": defenses,
            "defense_labels": [format_defense_label(item) for item in defenses],
            "attack_multiplier": attack_multiplier,
            "grader_model": grader_model,
            "chars_per_token": chars_per_token,
        },
        "assumptions": assumptions,
        "estimates": estimates,
    }


def format_currency(amount: float) -> str:
    return f"${amount:,.4f}"


def format_report(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    config = report["config"]
    lines.append("In-Dist Cost Estimate")
    lines.append(f"Dataset files: {len(config['dataset_files'])}")
    lines.append(
        "Defenses: " + ", ".join(format_defense_label(defense) for defense in config["defenses"])
    )
    lines.append(f"Attack multiplier: {config['attack_multiplier']}")
    lines.append(f"Grader: {config['grader_model']}")
    lines.append("")
    lines.append("Summary")
    for estimate in report["estimates"]:
        lines.append(
            f"- {estimate['model']}: "
            f"subject={format_currency(estimate['subject_cost']['total_cost'])}, "
            f"grader={format_currency(estimate['grader_cost']['total_cost'])}, "
            f"total={format_currency(estimate['total_cost'])}, "
            f"sample-evals={estimate['sample_evaluations']}, "
            f"avg/sample-eval={format_currency(estimate['cost_per_sample_evaluation'])}"
        )
    for estimate in report["estimates"]:
        lines.append("")
        lines.append(f"Details: {estimate['model']}")
        lines.append(
            f"Smoke source: {estimate['smoke_profile']['log_dir']} "
            f"(observed model: {estimate['smoke_profile']['subject_model']})"
        )
        overall = estimate["smoke_profile"]["overall_baseline"]
        lines.append(
            "Smoke baseline: "
            f"{overall['sample_count']} samples, "
            f"avg prompt estimate={overall['avg_prompt_tokens_estimate']:.1f}, "
            f"avg transcript estimate={overall['avg_transcript_tokens_estimate']:.1f}"
        )
        lines.append(
            "Observed smoke usage: "
            f"input={overall['avg_subject_usage']['input_tokens']:.1f}, "
            f"cache_read={overall['avg_subject_usage']['input_tokens_cache_read']:.1f}, "
            f"cache_write={overall['avg_subject_usage']['input_tokens_cache_write']:.1f}, "
            f"output={overall['avg_subject_usage']['output_tokens']:.1f}"
        )
        lines.append(
            "Estimate gap: "
            f"prompt-vs-observed-input={overall['avg_prompt_estimate_gap_vs_observed_input']:.1f}, "
            f"transcript-vs-observed-total={overall['avg_transcript_estimate_gap_vs_observed_total']:.1f}"
        )
        lines.append("Per defense:")
        for defense_label, defense_report in estimate["per_defense"].items():
            lines.append(f"  {defense_label}: {format_currency(defense_report['total_cost'])}")
        lines.append("Per dataset:")
        for dataset_name, dataset_report in estimate["per_dataset"].items():
            lines.append(f"  {Path(dataset_name).name}: {format_currency(dataset_report['total_cost'])}")
        if estimate["notes"]:
            lines.append("Notes:")
            for note in estimate["notes"]:
                lines.append(f"  - {note}")
    lines.append("")
    lines.append("Global assumptions:")
    for note in report["assumptions"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_TARGET_MODELS),
        help="Comma-separated models to estimate.",
    )
    parser.add_argument(
        "--dataset-files",
        default=",".join(DEFAULT_DATASET_FILES),
        help="Comma-separated dataset files or absolute paths.",
    )
    parser.add_argument(
        "--defenses",
        default=";".join(DEFAULT_DEFENSES),
        help="Semicolon-separated defenses. Use an empty first entry for no defense.",
    )
    parser.add_argument(
        "--attack-multiplier",
        type=int,
        default=1,
        help="Linear multiplier for attack variants/runs.",
    )
    parser.add_argument(
        "--grader-model",
        default=DEFAULT_GRADER_MODEL,
        help="Native grader model for goal-match scoring.",
    )
    parser.add_argument(
        "--logs-root",
        default=str(DEFAULT_LOGS_ROOT),
        help="Logs directory that contains actual-smoke folders.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help="Approximate token counts using this many characters per token (default: 4).",
    )
    parser.add_argument(
        "--anthropic-cache-write-rate",
        type=float,
        default=3.75,
        help="Anthropic cache-write $/MTok assumption (defaults to 5-minute writes).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    parser.add_argument("--gpt-smoke-log-dir", default="", help="Override GPT smoke log dir.")
    parser.add_argument("--claude-smoke-log-dir", default="", help="Override Claude smoke log dir.")
    parser.add_argument("--gemini-smoke-log-dir", default="", help="Override Gemini smoke log dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defenses = split_semicolon_arg(args.defenses)
    models = split_csv_arg(args.models)
    datasets = split_csv_arg(args.dataset_files)
    if not defenses:
        raise SystemExit("At least one defense entry is required.")
    if not models:
        raise SystemExit("At least one target model is required.")
    if not datasets:
        raise SystemExit("At least one dataset file is required.")

    overrides = {
        "openai/gpt-5.4": args.gpt_smoke_log_dir,
        "anthropic/claude-sonnet-4-6": args.claude_smoke_log_dir,
        "openrouter/google/gemini-3.1-pro-preview": args.gemini_smoke_log_dir,
    }
    try:
        report = generate_cost_report(
            models=models,
            dataset_files=datasets,
            defenses=defenses,
            attack_multiplier=args.attack_multiplier,
            logs_root=Path(args.logs_root),
            grader_model=args.grader_model,
            chars_per_token=args.chars_per_token,
            anthropic_cache_write_per_million=args.anthropic_cache_write_rate,
            smoke_log_overrides={key: value for key, value in overrides.items() if value},
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0
