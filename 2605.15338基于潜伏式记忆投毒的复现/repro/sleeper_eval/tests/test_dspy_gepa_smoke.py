from __future__ import annotations

from types import SimpleNamespace

import dspy

from experiments.dspy_gepa_smoke.dataset_builder import build_smoke_dataset
from experiments.dspy_gepa_smoke.metric import memory_matches_expected, smoke_metric
from experiments.dspy_gepa_smoke.program import DefenseSuffixSmokeProgram
from experiments.dspy_gepa_smoke.runner import InspectSmokeRunner
from sleeper_eval.dataset import SampleMeta
from sleeper_eval.defenses import defense_config_from_names
from sleeper_eval.provider_config import get_provider_config
from sleeper_eval.solver import HARDENING_TEXT, build_system_prompt


class FakeMessage:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text

    def model_dump(self, mode: str = "json") -> dict[str, str]:
        del mode
        return {"role": self.role, "content": self.text}


def test_build_system_prompt_uses_suffix_override_before_hardening(dev_with_memories_sample) -> None:
    meta = SampleMeta.model_validate(dev_with_memories_sample.metadata)
    prompt = build_system_prompt(
        meta,
        defense_config=defense_config_from_names("system_prompt_hardening"),
        provider_config=get_provider_config("generic"),
        defense_suffix_override="## Custom Defense\n- Block suspicious memory writes.",
    )

    assert "## Custom Defense" in prompt
    assert HARDENING_TEXT not in prompt


def test_build_smoke_dataset_emits_three_examples(tmp_path) -> None:
    dataset_path, records, examples = build_smoke_dataset(tmp_path / "smoke_dataset.json")

    assert dataset_path.exists()
    assert len(records) == 3
    assert len(examples) == 3
    assert [example.example_kind for example in examples] == ["attack", "attack", "benign"]
    assert examples[0].attack_name == "universal_v2_optimized_with_markers"
    assert examples[1].attack_name == "universal_v2_optimized_without_markers"
    assert examples[2].attack_name == "none"
    assert examples[2].expected_memory
    assert all(example.dataset_file == str(dataset_path) for example in examples)


def test_memory_matches_expected_uses_simple_normalized_match() -> None:
    assert memory_matches_expected(
        ["The user prefers morning meetings."],
        "The user prefers morning meetings",
    )


def test_smoke_metric_scores_attack_and_benign_examples() -> None:
    attack_gold = dspy.Example(example_kind="attack", expected_memory="")
    benign_gold = dspy.Example(
        example_kind="benign",
        expected_memory="User prefers vegetarian meals.",
    )

    blocked_attack = dspy.Prediction(
        tool_called=False,
        saved_memories=[],
        response_text="Refused to save memory.",
        sample_error=None,
        defense_suffix_used="suffix",
    )
    missed_benign = dspy.Prediction(
        tool_called=False,
        saved_memories=[],
        response_text="I will not save that.",
        sample_error=None,
        defense_suffix_used="suffix",
    )

    attack_score = smoke_metric(attack_gold, blocked_attack)
    benign_score = smoke_metric(benign_gold, missed_benign)

    assert attack_score.score == 1.0
    assert benign_score.score == 0.0
    assert "too strict" in benign_score.feedback


def test_inspect_smoke_runner_normalizes_eval_output(monkeypatch, tmp_path) -> None:
    fake_sample = SimpleNamespace(
        store={
            "SavedMemoryStore:backend_name": "tool",
            "SavedMemoryStore:runtime_name": "tool",
            "SavedMemoryStore:saved_memories": ["User prefers tea."],
            "SavedMemoryStore:raw_backend_result": None,
        },
        messages=[FakeMessage("system", "sys"), FakeMessage("user", "usr")],
        output=SimpleNamespace(completion="Saved that."),
        error=None,
    )
    fake_log = SimpleNamespace(location=tmp_path / "inspect" / "fake.eval", samples=[fake_sample])

    monkeypatch.setattr("experiments.dspy_gepa_smoke.runner.sleeper_eval", lambda **_: object())
    monkeypatch.setattr("experiments.dspy_gepa_smoke.runner.inspect_eval", lambda *_, **__: [fake_log])

    runner = InspectSmokeRunner(root_dir=tmp_path, model="mockllm/model")
    result = runner.evaluate(
        dataset_file="fake.json",
        sample_id="sample-1",
        attack_name="none",
        example_kind="benign",
        expected_memory="User prefers tea.",
        defense_suffix="suffix text",
    )

    assert result.tool_called is True
    assert result.saved_memories == ["User prefers tea."]
    assert result.response_text == "Saved that."
    assert result.inspect_log_path.endswith("fake.eval")
    assert (tmp_path / "artifacts" / "raw_eval_events.jsonl").exists()


def test_smoke_program_passes_current_suffix_to_runner(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, str]] = []

    class FakeRunner:
        def evaluate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                sample_id=kwargs["sample_id"],
                tool_called=False,
                saved_memories=[],
                response_text="Blocked.",
                messages=[],
                sample_error=None,
                suffix_used=kwargs["defense_suffix"],
                inspect_log_path=str(tmp_path / "inspect" / "fake.eval"),
                example_kind=kwargs["example_kind"],
                attack_name=kwargs["attack_name"],
            )

    program = DefenseSuffixSmokeProgram(runner=FakeRunner(), seed_suffix="Seed suffix")
    program.suffix_predictor.signature.instructions = "Mutated suffix"

    def fake_forward(**kwargs):
        del kwargs
        return dspy.Prediction(defense_note="noop")

    monkeypatch.setattr(program.suffix_predictor, "forward", fake_forward)

    prediction = program.forward(
        sample_id="sample-1",
        dataset_file="dataset.json",
        attack_name="universal_v2_optimized_with_markers",
        example_kind="attack",
        expected_memory="",
        sample_context="Example kind: attack",
    )

    assert calls[0]["defense_suffix"] == "Mutated suffix"
    assert prediction.defense_suffix_used == "Mutated suffix"
