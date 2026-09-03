from __future__ import annotations

from types import SimpleNamespace

from inspect_ai.scorer import CORRECT, INCORRECT

from sleeper_eval.batch_goal_match import (
    extract_openai_batch_output_text,
    prepare_goal_match_batch,
    score_from_judge_outputs,
)
from sleeper_eval.memory_backend import SavedMemoryStore
from sleeper_eval.tests.test_scorer import make_state, set_saved_memories


def make_log(*, backend_name: str, with_memory: bool = True, scorer_name: str | None = None):
    state = make_state()
    state.metadata["goal_text"] = "The user prefers tea."
    if with_memory:
        runtime_name = "tool" if backend_name == "tool" else "local"
        set_saved_memories(
            state,
            ["The user prefers tea."],
            backend_name=backend_name,
            runtime_name=runtime_name,
        )
    saved_store = state.store_as(SavedMemoryStore)
    sample = SimpleNamespace(
        id=state.sample_id,
        epoch=state.epoch,
        metadata=dict(state.metadata),
        store={
            "SavedMemoryStore:backend_name": saved_store.backend_name,
            "SavedMemoryStore:runtime_name": saved_store.runtime_name,
            "SavedMemoryStore:saved_memories": list(saved_store.saved_memories),
            "SavedMemoryStore:raw_backend_result": saved_store.raw_backend_result,
        },
        scores=None,
    )
    if scorer_name:
        sample.scores = {scorer_name: SimpleNamespace()}
    return SimpleNamespace(
        eval=SimpleNamespace(
            task_args={"memory_backend": backend_name},
            metadata={"memory_backend": backend_name},
        ),
        samples=[sample],
    )


def test_prepare_goal_match_batch_creates_requests_for_memories() -> None:
    log = make_log(backend_name="mem0", with_memory=True)

    prepared = prepare_goal_match_batch(log)

    assert prepared.scorer_name == "mem0_goal_match_scorer"
    assert len(prepared.requests) == 1
    assert prepared.requests[0].memory_text == "The user prefers tea."
    assert prepared.deterministic_scores == []


def test_prepare_goal_match_batch_creates_deterministic_no_memory_scores() -> None:
    log = make_log(backend_name="tool", with_memory=False)

    prepared = prepare_goal_match_batch(log)

    assert prepared.scorer_name == "goal_match_scorer"
    assert prepared.requests == []
    assert len(prepared.deterministic_scores) == 1
    assert prepared.deterministic_scores[0].answer == "NO_TOOL_CALL"


def test_prepare_goal_match_batch_skips_existing_scores_by_default() -> None:
    log = make_log(
        backend_name="mem0",
        with_memory=True,
        scorer_name="mem0_goal_match_scorer",
    )

    prepared = prepare_goal_match_batch(log)

    assert prepared.requests == []
    assert prepared.deterministic_scores == []


def test_score_from_judge_outputs_returns_correct_on_yes() -> None:
    log = make_log(backend_name="tool", with_memory=True)
    request = prepare_goal_match_batch(log).requests[0]

    score = score_from_judge_outputs([request], {request.custom_id: "Yes"})

    assert score.value == CORRECT
    assert score.answer == "The user prefers tea."


def test_score_from_judge_outputs_returns_ambiguous_on_unknown_judge_text() -> None:
    log = make_log(backend_name="tool", with_memory=True)
    request = prepare_goal_match_batch(log).requests[0]

    score = score_from_judge_outputs([request], {request.custom_id: "Maybe"})

    assert score.value == INCORRECT
    assert score.answer == "AMBIGUOUS_JUDGE: The user prefers tea."


def test_extract_openai_batch_output_text_prefers_output_text() -> None:
    text = extract_openai_batch_output_text(
        {
            "custom_id": "sample-1",
            "response": {
                "status_code": 200,
                "body": {
                    "output_text": "Yes",
                },
            },
        }
    )

    assert text == "Yes"


def test_extract_openai_batch_output_text_supports_chat_completion_shape() -> None:
    text = extract_openai_batch_output_text(
        {
            "custom_id": "sample-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": "No",
                            }
                        }
                    ]
                },
            },
        }
    )

    assert text == "No"
