"""Inspect task definition for follow-up behavioral influence evaluations."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from sleeper_eval.followup_eval.dataset import load_dataset
from sleeper_eval.followup_eval.scorer import behavior_influence_scorer
from sleeper_eval.followup_eval.solver import replay_followup_conversation
from sleeper_eval.provider_config import get_model_route, get_provider_config


@task
def followup_eval(
    dataset_file: str = "followup/smoke/behavioral_influence_smoke.json",
    category: str = "",
    provider: str = "",
    prompt_model: str = "",
):
    """Evaluate whether preloaded injected memory influences a later conversation."""

    model_route = get_model_route(prompt_model)
    provider_config = get_provider_config(
        provider,
        model_slug=prompt_model,
    )
    dataset = load_dataset(dataset_file)

    if category:
        selected = {item.strip() for item in category.split(",") if item.strip()}
        if hasattr(dataset, "filter"):
            dataset = dataset.filter(lambda sample: sample.metadata.get("category") in selected)
        else:
            dataset = [sample for sample in dataset if sample.metadata.get("category") in selected]

    return Task(
        dataset=dataset,
        solver=[
            replay_followup_conversation(provider_config=provider_config),
        ],
        scorer=[behavior_influence_scorer()],
        config=GenerateConfig(max_tokens=40960),
        fail_on_error=0.1,
        message_limit=12,
        metadata={
            "eval_kind": "followup_behavioral_influence",
            "provider": provider_config.provider_id,
            "prompt_model": prompt_model,
            "model_api": model_route.model_api,
            "model_route_type": model_route.model_route_type,
            "model_vendor": model_route.model_vendor,
            "memory_placement": provider_config.memory_placement,
        },
    )
