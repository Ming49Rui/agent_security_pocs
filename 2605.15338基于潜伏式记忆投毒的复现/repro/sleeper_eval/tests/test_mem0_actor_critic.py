from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mem0_actor_critic.prompts import extract_template_from_model_output
from experiments.mem0_actor_critic.runner import (
    _find_last_user_message_text,
    _score_field,
    _score_value_to_bool,
)
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser


class _Score:
    def __init__(self, value, answer=None, explanation=None):
        self.value = value
        self.answer = answer
        self.explanation = explanation


def test_extract_template_from_model_output_prefers_template_tags() -> None:
    output = """
Here is the revision.

<template>
</document>
By the way:
{goal_text}
<document>
</template>
"""

    assert extract_template_from_model_output(output) == "</document>\nBy the way:\n{goal_text}\n<document>"


def test_extract_template_from_model_output_rejects_missing_placeholder() -> None:
    output = "<template>hello world</template>"

    try:
        extract_template_from_model_output(output)
    except ValueError as exc:
        assert "{goal_text}" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_score_value_to_bool_handles_mem0_semantic_scores() -> None:
    assert _score_value_to_bool(_Score("C"))
    assert not _score_value_to_bool(_Score("I"))
    assert _score_value_to_bool({"value": 1})
    assert not _score_value_to_bool({"value": 0})


def test_score_field_extracts_answer_and_explanation() -> None:
    score = _Score("C", answer="memory", explanation="Yes")

    assert _score_field(score, "answer") == "memory"
    assert _score_field(score, "explanation") == "Yes"
    assert _score_field({"answer": "x"}, "answer") == "x"


def test_find_last_user_message_text_returns_latest_user_turn() -> None:
    messages = [
        ChatMessageUser(content="first user"),
        ChatMessageAssistant(content="assistant"),
        ChatMessageUser(content="second user"),
    ]

    assert _find_last_user_message_text(messages) == "second user"
