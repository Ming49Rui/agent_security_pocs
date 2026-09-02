from __future__ import annotations

import importlib

from inspect_ai.dataset import Sample

followup_task_module = importlib.import_module("sleeper_eval.followup_eval.task")
followup_eval = followup_task_module.followup_eval


def _stub_dataset() -> list[Sample]:
    return [Sample(input="prompt", target="target", metadata={"category": "prefs"})]


def test_followup_task_uses_dedicated_solver_and_scorer(monkeypatch) -> None:
    monkeypatch.setattr(followup_task_module, "load_dataset", lambda _: _stub_dataset())

    task = followup_eval(prompt_model="openai/gpt-5.4")

    assert len(task.solver) == 1
    assert len(task.scorer) == 1
    assert task.metadata["eval_kind"] == "followup_behavioral_influence"
    assert task.metadata["provider"] == "gpt"
    assert task.metadata["model_api"] == "openai"
    assert task.metadata["model_route_type"] == "native"
    assert task.scorer[0].__registry_info__.name == "behavior_influence_scorer"


def test_followup_task_filters_by_category(monkeypatch) -> None:
    monkeypatch.setattr(
        followup_task_module,
        "load_dataset",
        lambda _: [
            Sample(input="a", target="a", metadata={"category": "prefs"}),
            Sample(input="b", target="b", metadata={"category": "travel"}),
        ],
    )

    task = followup_eval(category="travel")

    assert len(task.dataset) == 1
    assert task.dataset[0].metadata["category"] == "travel"
