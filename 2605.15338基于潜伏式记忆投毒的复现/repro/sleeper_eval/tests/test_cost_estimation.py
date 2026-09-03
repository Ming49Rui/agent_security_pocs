from __future__ import annotations

import json
import zipfile
from pathlib import Path

from sleeper_eval import cost_estimation as ce


def make_record(*, with_memories: bool, doc_id: str) -> dict[str, object]:
    memories = ["User likes espresso."] if with_memories else []
    return {
        "document": {
            "domain": "legal",
            "doc_id": doc_id,
            "text": "This is a short document body.",
            "metadata": {},
            "annotations": {"document_format": "text"},
        },
        "goal": {
            "goal_text": "User prefers morning meetings",
            "goal_id": f"goal-{doc_id}",
            "category_id": "A",
            "subcategory_id": "A1",
            "domain_seed": "seed_001",
        },
        "query": {"query": "Summarize this document", "style": "formal"},
        "preexisting_memories": {"memories": memories, "goal_contradiction_memory": None},
    }


def write_dataset(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def write_smoke_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_name = "2026-03-26T00-00-00+00-00_sleeper-eval_fake.eval"
    aggregate_subject = {
        "input_tokens": 300,
        "output_tokens": 80,
        "input_tokens_cache_read": 40,
        "reasoning_tokens": 0,
    }
    aggregate_grader = {
        "input_tokens": 20,
        "output_tokens": 4,
        "reasoning_tokens": 0,
    }
    logs_json = {
        eval_name: {
            "stats": {
                "model_usage": {
                    "openai/gpt-5.4": aggregate_subject,
                    "openai/gpt-5.4-mini": aggregate_grader,
                },
                "role_usage": {"grader": aggregate_grader},
            }
        }
    }
    (log_dir / "logs.json").write_text(json.dumps(logs_json), encoding="utf-8")
    sample_payloads = [
        {
            "id": "sample-with",
            "metadata": {"memory_condition": "with_memories"},
            "messages": [
                {"role": "system", "content": "sys with"},
                {"role": "user", "content": "user with"},
                {"role": "assistant", "content": "assistant with"},
            ],
            "model_usage": {
                "openai/gpt-5.4": {
                    "input_tokens": 200,
                    "output_tokens": 50,
                    "input_tokens_cache_read": 30,
                    "reasoning_tokens": 0,
                },
                "openai/gpt-5.4-mini": {
                    "input_tokens": 12,
                    "output_tokens": 2,
                    "reasoning_tokens": 0,
                },
            },
            "role_usage": {"grader": {"input_tokens": 12, "output_tokens": 2, "reasoning_tokens": 0}},
        },
        {
            "id": "sample-without",
            "metadata": {"memory_condition": "without_memories"},
            "messages": [
                {"role": "system", "content": "sys without"},
                {"role": "user", "content": "user without"},
                {"role": "assistant", "content": "assistant without"},
            ],
            "model_usage": {
                "openai/gpt-5.4": {
                    "input_tokens": 100,
                    "output_tokens": 30,
                    "input_tokens_cache_read": 10,
                    "reasoning_tokens": 0,
                },
                "openai/gpt-5.4-mini": {
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "reasoning_tokens": 0,
                },
            },
            "role_usage": {"grader": {"input_tokens": 8, "output_tokens": 2, "reasoning_tokens": 0}},
        },
    ]
    with zipfile.ZipFile(log_dir / eval_name, "w") as archive:
        for index, sample in enumerate(sample_payloads, start=1):
            archive.writestr(f"samples/{index}.json", json.dumps(sample))


def test_price_usage_handles_cache_categories() -> None:
    usage = ce.TokenUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        input_tokens_cache_read=500_000,
        input_tokens_cache_write=200_000,
    )

    openai_cost = ce.price_usage(usage, ce.get_pricing_rates("openai/gpt-5.4"))
    anthropic_cost = ce.price_usage(
        usage,
        ce.get_pricing_rates("anthropic/claude-sonnet-4-6"),
    )
    gemini_cost = ce.price_usage(
        usage,
        ce.get_pricing_rates("openrouter/google/gemini-3.1-pro-preview"),
    )

    assert round(openai_cost.total_cost, 3) == 4.625
    assert round(anthropic_cost.total_cost, 3) == 5.4
    assert round(gemini_cost.total_cost, 3) == 4.6


def test_build_smoke_profile_groups_by_memory_condition(tmp_path: Path, monkeypatch) -> None:
    log_dir = tmp_path / "actual-smoke-gpt54-20260326-000000"
    write_smoke_log_dir(log_dir)
    monkeypatch.setattr(
        ce,
        "count_text_tokens",
        lambda text, chars_per_token=ce.DEFAULT_CHARS_PER_TOKEN: len(text),
    )

    profile = ce.build_smoke_profile(
        log_dir,
        subject_model="openai/gpt-5.4",
        grader_model="openai/gpt-5.4-mini",
    )

    assert set(profile.baselines_by_condition) == {"with_memories", "without_memories"}
    assert profile.baselines_by_condition["with_memories"].sample_count == 1
    assert profile.baselines_by_condition["without_memories"].avg_subject_usage.input_tokens == 100
    assert profile.aggregate_usage_by_model["openai/gpt-5.4"].input_tokens == 300


def test_generate_cost_report_with_temp_datasets(tmp_path: Path, monkeypatch) -> None:
    with_memories = tmp_path / "merged_dev_with_memories.json"
    without_memories = tmp_path / "merged_eval_in_without_memories.json"
    write_dataset(with_memories, [make_record(with_memories=True, doc_id="doc-1")])
    write_dataset(without_memories, [make_record(with_memories=False, doc_id="doc-2")])

    log_dir = tmp_path / "actual-smoke-gpt54-20260326-000000"
    write_smoke_log_dir(log_dir)
    monkeypatch.setattr(
        ce,
        "count_text_tokens",
        lambda text, chars_per_token=ce.DEFAULT_CHARS_PER_TOKEN: len(text),
    )

    report = ce.generate_cost_report(
        models=["openai/gpt-5.4"],
        dataset_files=[str(with_memories), str(without_memories)],
        defenses=["", "system_prompt_hardening"],
        attack_multiplier=2,
        logs_root=tmp_path,
        grader_model="openai/gpt-5.4-mini",
        smoke_log_overrides={"openai/gpt-5.4": log_dir},
    )

    estimate = report["estimates"][0]
    assert estimate["sample_evaluations"] == 8
    assert set(estimate["per_defense"]) == {"(none)", "system_prompt_hardening"}
    assert set(Path(path).name for path in estimate["per_dataset"]) == {
        "merged_dev_with_memories.json",
        "merged_eval_in_without_memories.json",
    }
    assert estimate["total_cost"] > 0
    assert estimate["grader_cost"]["total_cost"] > 0
