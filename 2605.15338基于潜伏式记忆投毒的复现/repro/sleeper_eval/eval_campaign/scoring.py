"""Scorer registry and compatibility rules for eval campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inspect_ai.log import EvalLog

from sleeper_eval.scorer import (
    benign_save_goal_match_scorer,
    goal_match_scorer,
    mem0_goal_match_scorer,
    mem0_write_scorer,
    tool_call_scorer,
)


@dataclass(frozen=True)
class ScorerCompatibility:
    name: str
    memory_backends: frozenset[str]
    evaluation_modes: frozenset[str]


SCORER_REGISTRY = {
    "tool_call_scorer": tool_call_scorer,
    "goal_match_scorer": goal_match_scorer,
    "benign_save_goal_match_scorer": benign_save_goal_match_scorer,
    "mem0_write_scorer": mem0_write_scorer,
    "mem0_goal_match_scorer": mem0_goal_match_scorer,
}

SCORER_COMPATIBILITY = {
    "tool_call_scorer": ScorerCompatibility(
        name="tool_call_scorer",
        memory_backends=frozenset({"tool"}),
        evaluation_modes=frozenset({"attack", "benign_save"}),
    ),
    "goal_match_scorer": ScorerCompatibility(
        name="goal_match_scorer",
        memory_backends=frozenset({"tool"}),
        evaluation_modes=frozenset({"attack"}),
    ),
    "benign_save_goal_match_scorer": ScorerCompatibility(
        name="benign_save_goal_match_scorer",
        memory_backends=frozenset({"tool", "mem0"}),
        evaluation_modes=frozenset({"benign_save"}),
    ),
    "mem0_write_scorer": ScorerCompatibility(
        name="mem0_write_scorer",
        memory_backends=frozenset({"mem0"}),
        evaluation_modes=frozenset({"attack", "benign_save"}),
    ),
    "mem0_goal_match_scorer": ScorerCompatibility(
        name="mem0_goal_match_scorer",
        memory_backends=frozenset({"mem0"}),
        evaluation_modes=frozenset({"attack"}),
    ),
}

GRADER_SCORERS = frozenset(
    {
        "goal_match_scorer",
        "benign_save_goal_match_scorer",
        "mem0_goal_match_scorer",
    }
)


def known_scorer_names() -> list[str]:
    return sorted(SCORER_REGISTRY)


def resolve_scorers(names: list[str]) -> list[Any]:
    scorers: list[Any] = []
    for name in names:
        factory = SCORER_REGISTRY.get(name)
        if factory is None:
            known = ", ".join(known_scorer_names())
            raise ValueError(f"Unknown scorer '{name}'. Known scorers: {known}.")
        scorers.append(factory())
    return scorers


def scorer_is_compatible(
    scorer_name: str,
    *,
    memory_backend: str,
    evaluation_mode: str,
) -> bool:
    compatibility = SCORER_COMPATIBILITY[scorer_name]
    return (
        memory_backend in compatibility.memory_backends
        and evaluation_mode in compatibility.evaluation_modes
    )


def scorer_names_for_log(log: EvalLog, requested_scorers: list[str]) -> list[str]:
    task_args = log.eval.task_args or {}
    memory_backend = str(task_args.get("memory_backend", "tool"))
    evaluation_mode = str(task_args.get("evaluation_mode", "attack"))
    return [
        scorer_name
        for scorer_name in requested_scorers
        if scorer_is_compatible(
            scorer_name,
            memory_backend=memory_backend,
            evaluation_mode=evaluation_mode,
        )
    ]


def scorer_uses_grader(scorer_name: str) -> bool:
    return scorer_name in GRADER_SCORERS
