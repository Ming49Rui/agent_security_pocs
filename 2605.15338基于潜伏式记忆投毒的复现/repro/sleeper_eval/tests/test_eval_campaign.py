from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleeper_eval.eval_campaign.config import (
    CampaignEvalConfig,
    CampaignScoringConfig,
    load_campaign_config,
)
import sleeper_eval.eval_campaign.preflight as preflight_module
import sleeper_eval.eval_campaign.mem0_qdrant as mem0_qdrant_module
from sleeper_eval.eval_campaign.plan import render_plan
from sleeper_eval.eval_campaign.preflight import preflight_campaign
import sleeper_eval.eval_campaign.run as run_module


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")


def _set_subject_only_env(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")


def _allow_mem0_sdk_prereqs(monkeypatch) -> None:
    monkeypatch.setattr(preflight_module, "_mem0_sdk_prereq_errors", lambda config: [])
    monkeypatch.setattr(preflight_module, "_docker_daemon_error", lambda: None)


def _write_dataset(path: Path, *, count: int = 1) -> None:
    path.write_text(json.dumps([{"id": idx} for idx in range(count)]), encoding="utf-8")


def _write_config(
    tmp_path: Path,
    *,
    suffix_file: Path,
    scoring: dict[str, object] | None = None,
    defenses: list[dict[str, object]] | None = None,
    datasets: list[dict[str, object]] | None = None,
    attacks: list[dict[str, object]] | None = None,
    models: list[dict[str, object]] | None = None,
    output_dir: Path | None = None,
) -> Path:
    benign_dataset = tmp_path / "benign_save_dataset.json"
    attack_dataset = tmp_path / "attack_dataset.json"
    _write_dataset(benign_dataset, count=2)
    _write_dataset(attack_dataset, count=3)
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "campaign",
                "output_dir": str(output_dir or (tmp_path / "run")),
                "datasets": datasets
                or [
                    {
                        "label": "eval-in-benign",
                        "dataset_file": str(benign_dataset),
                        "evaluation_mode": "benign_save",
                        "attack": "none",
                    },
                    {
                        "label": "eval-in-attack",
                        "dataset_file": str(attack_dataset),
                        "evaluation_mode": "attack",
                        "attack": "universal_v2_optimized_with_markers",
                    },
                ],
                "attacks": attacks or [],
                "memory_backend": "tool",
                "mention_memory_system": True,
                "grader_model": "openai/gpt-5.4-mini",
                "scoring": scoring
                or {
                    "during_eval": "none",
                    "post_eval_scorers": ["tool_call_scorer"],
                    "post_eval_action": "overwrite",
                },
                "models": models
                or [
                    {
                        "label": "kimi",
                        "model": "openrouter/moonshotai/kimi-k2.5",
                        "provider": "generic",
                    },
                    {
                        "label": "gpt",
                        "model": "openrouter/openai/gpt-5.4",
                        "provider": "gpt",
                    },
                ],
                "defenses": defenses
                or [
                    {"label": "baseline", "defense": "system_prompt_hardening"},
                    {"label": "optimized", "defense_suffix_file": str(suffix_file)},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


class FakeTask:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.metadata = metadata
        self.tags: list[str] = []
        self.config = SimpleNamespace()


def _fake_sleeper_eval(**kwargs):
    return FakeTask(metadata={"evaluation_mode": kwargs["evaluation_mode"]})


def test_preflight_and_build_campaign_tasks_expand_full_matrix(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("Use direct-user provenance checks.", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)
    monkeypatch.setattr(run_module, "sleeper_eval", _fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    tasks = run_module.build_campaign_tasks(preflight)

    assert len(preflight.datasets) == 2
    assert preflight.datasets[0].sample_count == 2
    assert preflight.datasets[1].sample_count == 3
    assert len(tasks) == 8
    labels = {
        (
            task.metadata["campaign_dataset_label"],
            task.metadata["campaign_model_label"],
            task.metadata["campaign_defense_label"],
        )
        for task in tasks
    }
    assert labels == {
        ("eval-in-benign", "kimi", "baseline"),
        ("eval-in-benign", "kimi", "optimized"),
        ("eval-in-benign", "gpt", "baseline"),
        ("eval-in-benign", "gpt", "optimized"),
        ("eval-in-attack", "kimi", "baseline"),
        ("eval-in-attack", "kimi", "optimized"),
        ("eval-in-attack", "gpt", "baseline"),
        ("eval-in-attack", "gpt", "optimized"),
    }
    attack_labels = {task.metadata["campaign_attack_label"] for task in tasks}
    assert attack_labels == {"none", "universal_v2_optimized_with_markers"}


def test_mem0_campaign_settings_are_passed_to_tasks(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "anthropic",
            "mem0_model": "claude-3-7-sonnet-latest",
            "mem0_include_document_content": False,
            "mem0_thinking": "disabled",
            "mem0_reasoning_effort": "max",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    captured_kwargs: list[dict[str, object]] = []

    def fake_sleeper_eval(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTask(metadata={"evaluation_mode": str(kwargs["evaluation_mode"])})

    monkeypatch.setattr(run_module, "sleeper_eval", fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    tasks = run_module.build_campaign_tasks(preflight)

    assert tasks
    assert captured_kwargs
    assert all(kwargs["memory_backend"] == "mem0" for kwargs in captured_kwargs)
    assert all(kwargs["mem0_runtime"] == "sdk" for kwargs in captured_kwargs)
    assert all(kwargs["mem0_provider"] == "anthropic" for kwargs in captured_kwargs)
    assert all(kwargs["mem0_model"] == "claude-3-7-sonnet-latest" for kwargs in captured_kwargs)
    assert all(kwargs["mem0_include_document_content"] is False for kwargs in captured_kwargs)
    assert all(kwargs["mem0_thinking"] == "disabled" for kwargs in captured_kwargs)
    assert all(kwargs["mem0_reasoning_effort"] == "max" for kwargs in captured_kwargs)


def test_preflight_renders_plan_with_counts_and_env_status(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)

    plan = render_plan(preflight_campaign(config, config_path=config_path))

    assert "samples=2" in plan
    assert "samples=3" in plan
    assert "attacks=0" in plan
    assert "required_env_vars:" in plan
    assert "OPENROUTER_API_KEY: set" in plan
    assert "mem0_runtime=local" in plan
    assert "mem0_provider=deepseek" in plan
    assert "mem0_model=deepseek-v4-pro" in plan
    assert "mem0_thinking=enabled" in plan
    assert "mem0_reasoning_effort=high" in plan
    assert "retry_attempts=10" in plan
    assert "limit=None" in plan
    assert "max_samples=None" in plan
    assert "grader_model=openai/gpt-5.4-mini" in plan
    assert "expanded_tasks=8" in plan


def test_preflight_and_task_build_preserve_model_reasoning_settings(
    tmp_path, monkeypatch
) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        models=[
            {
                "label": "gpt",
                "model": "openai/gpt-5.4",
                "provider": "gpt",
                "reasoning_effort": "medium",
            },
            {
                "label": "claude",
                "model": "anthropic/claude-sonnet-4-6",
                "provider": "claude",
                "reasoning_effort": "high",
                "reasoning_tokens": 2048,
            },
        ],
    )
    config = load_campaign_config(config_path)

    captured_kwargs: list[dict[str, object]] = []

    def fake_sleeper_eval(**kwargs):
        captured_kwargs.append(kwargs)
        task = FakeTask(metadata={"evaluation_mode": str(kwargs["evaluation_mode"])})
        task.config = SimpleNamespace(
            reasoning_effort=kwargs.get("reasoning_effort"),
            reasoning_tokens=kwargs.get("reasoning_tokens"),
        )
        return task

    monkeypatch.setattr(run_module, "sleeper_eval", fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    plan = render_plan(preflight)
    tasks = run_module.build_campaign_tasks(preflight)

    assert any(model.reasoning_effort == "medium" for model in preflight.models)
    assert any(model.reasoning_tokens == 2048 for model in preflight.models)
    assert "reasoning_effort=medium" in plan
    assert "reasoning_tokens=2048" in plan
    assert tasks
    assert any(kwargs["reasoning_effort"] == "medium" for kwargs in captured_kwargs)
    assert any(kwargs["reasoning_tokens"] == 2048 for kwargs in captured_kwargs)


def test_preflight_and_task_build_preserve_model_batch_settings(
    tmp_path, monkeypatch
) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        models=[
            {
                "label": "gemini-bool",
                "model": "google/gemini-3.1-pro-preview",
                "provider": "gemini",
                "batch": True,
            },
            {
                "label": "gemini-size",
                "model": "google/gemini-3.1-pro-preview",
                "provider": "gemini",
                "batch": 1000,
            },
            {
                "label": "gemini-structured",
                "model": "google/gemini-3.1-pro-preview",
                "provider": "gemini",
                "batch": {
                    "size": 1000,
                    "max_size": 1000,
                    "send_delay": 60,
                    "tick": 60,
                    "max_batches": 2,
                },
            },
        ],
    )
    config = load_campaign_config(config_path)

    captured_kwargs: list[dict[str, object]] = []

    def fake_sleeper_eval(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTask(metadata={"evaluation_mode": str(kwargs["evaluation_mode"])})

    monkeypatch.setattr(run_module, "sleeper_eval", fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    plan = render_plan(preflight)
    tasks = run_module.build_campaign_tasks(preflight)

    batch_by_label = {model.label: model.batch for model in preflight.models}
    assert batch_by_label["gemini-bool"] == {}
    assert batch_by_label["gemini-size"] == {"size": 1000}
    assert batch_by_label["gemini-structured"] == {
        "size": 1000,
        "max_size": 1000,
        "send_delay": 60.0,
        "tick": 60.0,
        "max_batches": 2,
    }
    assert "batch=enabled" in plan
    assert 'batch={"size":1000}' in plan
    assert '"max_batches":2' in plan
    assert tasks
    assert any(kwargs["batch"] == {} for kwargs in captured_kwargs)
    assert any(kwargs["batch"] == {"size": 1000} for kwargs in captured_kwargs)
    assert any(
        kwargs["batch"]
        == {
            "size": 1000,
            "max_size": 1000,
            "send_delay": 60.0,
            "tick": 60.0,
            "max_batches": 2,
        }
        for kwargs in captured_kwargs
    )


def test_preflight_and_task_build_preserve_sample_shuffle_settings(
    tmp_path, monkeypatch
) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        scoring={
            "during_eval": "none",
            "post_eval_scorers": ["tool_call_scorer"],
            "post_eval_action": "overwrite",
        },
    )
    config = load_campaign_config(config_path).model_copy(
        update={
            "eval": CampaignEvalConfig(
                limit=15,
                sample_shuffle=True,
                sample_shuffle_seed=123,
                max_tasks=1,
            )
        }
    )

    captured_kwargs: list[dict[str, object]] = []

    def fake_sleeper_eval(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTask(metadata={"evaluation_mode": str(kwargs["evaluation_mode"])})

    monkeypatch.setattr(run_module, "sleeper_eval", fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    plan = render_plan(preflight)
    tasks = run_module.build_campaign_tasks(preflight)

    assert "sample_shuffle=True" in plan
    assert "sample_shuffle_seed=123" in plan
    assert tasks
    assert all(kwargs["sample_shuffle"] is True for kwargs in captured_kwargs)
    assert all(kwargs["sample_shuffle_seed"] == 123 for kwargs in captured_kwargs)


def test_tool_only_scoring_does_not_require_grader_env(tmp_path, monkeypatch) -> None:
    _set_subject_only_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        models=[
            {
                "label": "gemini",
                "model": "google/gemini-3.1-pro-preview",
                "provider": "gemini",
                "batch": True,
            }
        ],
    )

    preflight = preflight_campaign(load_campaign_config(config_path), config_path=config_path)

    assert all(status.requirement != "OPENAI_API_KEY" for status in preflight.env_statuses)


def test_preflight_resolves_builtin_named_defense_suffix(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("placeholder", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        defenses=[{"label": "gepa", "defense": "gepa_prompt_hardening_suffix"}],
    )

    preflight = preflight_campaign(load_campaign_config(config_path), config_path=config_path)

    assert len(preflight.defenses) == 1
    assert preflight.defenses[0].suffix_length > 0


def test_run_campaign_calls_eval_set_and_writes_artifacts(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("Use direct-user provenance checks.", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)
    preflight = preflight_campaign(config, config_path=config_path)
    captured: dict[str, object] = {}

    def fake_eval_set(*, tasks, log_dir, **kwargs):
        captured["task_count"] = len(tasks)
        captured["log_dir"] = log_dir
        captured["kwargs"] = kwargs
        return True, []

    monkeypatch.setattr(run_module, "eval_set", fake_eval_set)
    monkeypatch.setattr(run_module, "sleeper_eval", _fake_sleeper_eval)
    monkeypatch.setattr(run_module, "apply_post_eval_scoring", lambda **_kwargs: ["log-1.eval"])
    monkeypatch.setattr(
        run_module,
        "load_analysis_frame",
        lambda _path: pd.DataFrame(
            [
                {
                    "id": "sample-1",
                    "eval_id": "eval-1",
                    "model": "openrouter/openai/gpt-5.4",
                    "task_arg_dataset_file": str(tmp_path / "benign_save_dataset.json"),
                    "task_arg_defense": "system_prompt_hardening",
                    "task_arg_attack": "none",
                    "task_arg_evaluation_mode": "benign_save",
                    "metadata_campaign_dataset_label": "eval-in-benign",
                    "metadata_campaign_model_label": "gpt",
                    "metadata_campaign_defense_label": "baseline",
                    "score_memory_write_numeric": 1.0,
                    "score_semantic_match_numeric": 1.0,
                }
            ]
        ),
    )

    success, _logs, written = run_module.run_campaign(
        config,
        config_path=config_path,
        preflight=preflight,
    )

    assert success is True
    assert captured["task_count"] == 8
    assert captured["kwargs"]["score"] is False
    assert captured["kwargs"]["limit"] is None
    assert os.environ["INSPECT_LOG_DIR"] == str((Path(config.output_dir) / "logs").resolve())
    assert os.environ["INSPECT_TRACE_FILE"] == str(
        (Path(config.output_dir) / "logs" / "inspect-trace.log").resolve()
    )
    assert os.environ["HOME"] == str((Path(config.output_dir) / ".inspect_home").resolve())
    assert Path(config.output_dir, "config.yaml").exists()
    assert Path(config.output_dir, "run_manifest.yaml").exists()
    assert Path(written["summary_json"]).exists()
    assert Path(written["summary_csv"]).exists()
    assert Path(written["summary_md"]).exists()


def test_log_lineage_by_eval_id_records_task_and_timestamp(monkeypatch, tmp_path) -> None:
    logs = [
        SimpleNamespace(name=str(tmp_path / "old.eval")),
        SimpleNamespace(name=str(tmp_path / "new.eval")),
        SimpleNamespace(name=str(tmp_path / "other.eval")),
    ]

    def fake_read(name: str, **_kwargs):
        if name.endswith("old.eval"):
            return SimpleNamespace(
                eval=SimpleNamespace(task_id="task-1", eval_id="eval-old", created="2026-04-18T10:00:00+00:00"),
                stats=SimpleNamespace(completed_at="2026-04-18T10:05:00+00:00"),
            )
        if name.endswith("new.eval"):
            return SimpleNamespace(
                eval=SimpleNamespace(task_id="task-1", eval_id="eval-new", created="2026-04-18T10:10:00+00:00"),
                stats=SimpleNamespace(completed_at="2026-04-18T10:15:00+00:00"),
            )
        return SimpleNamespace(
            eval=SimpleNamespace(task_id="task-2", eval_id="eval-other", created="2026-04-18T10:20:00+00:00"),
            stats=SimpleNamespace(completed_at="2026-04-18T10:25:00+00:00"),
        )

    monkeypatch.setattr(run_module, "list_eval_logs", lambda _log_dir: logs)
    monkeypatch.setattr(run_module, "read_eval_log", fake_read)

    lineage = run_module.log_lineage_by_eval_id(tmp_path)

    assert lineage["eval-old"]["task_id"] == "task-1"
    assert lineage["eval-new"]["timestamp"] == "2026-04-18T10:15:00+00:00"
    assert lineage["eval-other"]["task_id"] == "task-2"


def test_load_campaign_analysis_frame_prefers_latest_sample_row_for_duplicate_sample(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        run_module,
        "load_analysis_frame",
        lambda _path: pd.DataFrame(
            [
                {
                    "eval_id": "eval-old",
                    "dataset_label": "eval-in-attack",
                    "task_arg_attack": "universal_v2_optimized_with_markers",
                    "model_label": "kimi",
                    "defense_label": "baseline",
                    "id": "sample-1",
                    "value": 1,
                },
                {
                    "eval_id": "eval-new",
                    "dataset_label": "eval-in-attack",
                    "task_arg_attack": "universal_v2_optimized_with_markers",
                    "model_label": "kimi",
                    "defense_label": "baseline",
                    "id": "sample-1",
                    "value": 2,
                },
                {
                    "eval_id": "eval-old",
                    "dataset_label": "eval-in-attack",
                    "task_arg_attack": "universal_v2_optimized_with_markers",
                    "model_label": "kimi",
                    "defense_label": "baseline",
                    "id": "sample-2",
                    "value": 3,
                },
            ]
        ),
    )
    monkeypatch.setattr(run_module, "with_display_columns", lambda df: df)
    monkeypatch.setattr(
        run_module,
        "log_lineage_by_eval_id",
        lambda _log_dir: {
            "eval-old": {"task_id": "task-1", "timestamp": "2026-04-18T10:05:00+00:00"},
            "eval-new": {"task_id": "task-1", "timestamp": "2026-04-18T10:15:00+00:00"},
        },
    )

    filtered = run_module.load_campaign_analysis_frame(tmp_path)

    assert len(filtered) == 2
    assert set(filtered["id"]) == {"sample-1", "sample-2"}
    assert filtered.loc[filtered["id"] == "sample-1", "value"].iloc[0] == 2


def test_discover_retry_logs_finds_latest_retry_worthy_matching_cell(monkeypatch, tmp_path) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)
    preflight = preflight_campaign(config, config_path=config_path)
    Path(preflight.output_dir, "logs").mkdir(parents=True, exist_ok=True)

    logs = [
        SimpleNamespace(name=str(tmp_path / "older.eval")),
        SimpleNamespace(name=str(tmp_path / "latest.eval")),
        SimpleNamespace(name=str(tmp_path / "good.eval")),
    ]

    def make_log(*, completed, total, sample_error_count, completed_at, task_id, dataset, attack, model, defense, location):
        samples = [SimpleNamespace(error={"message": "x"}) for _ in range(sample_error_count)]
        return SimpleNamespace(
            status="success",
            invalidated=False,
            samples=samples,
            location=location,
            eval=SimpleNamespace(task_id=task_id, eval_id=task_id, created="2026-04-18T10:00:00+00:00"),
            stats=SimpleNamespace(completed_at=completed_at),
            results=SimpleNamespace(total_samples=total, completed_samples=completed),
            metadata={
                "campaign_dataset_label": dataset,
                "campaign_attack_label": attack,
                "campaign_model_label": model,
                "campaign_defense_label": defense,
            },
        )

    older = make_log(
        completed=100,
        total=100,
        sample_error_count=0,
        completed_at="2026-04-18T10:05:00+00:00",
        task_id="task-1",
        dataset="eval-in-attack",
        attack="universal_v2_optimized_with_markers",
        model="kimi",
        defense="baseline",
        location=str(tmp_path / "older.eval"),
    )
    latest = make_log(
        completed=99,
        total=100,
        sample_error_count=1,
        completed_at="2026-04-18T10:15:00+00:00",
        task_id="task-1",
        dataset="eval-in-attack",
        attack="universal_v2_optimized_with_markers",
        model="kimi",
        defense="baseline",
        location=str(tmp_path / "latest.eval"),
    )
    good = make_log(
        completed=60,
        total=60,
        sample_error_count=0,
        completed_at="2026-04-18T10:20:00+00:00",
        task_id="task-2",
        dataset="eval-in-benign",
        attack="none",
        model="gpt",
        defense="optimized",
        location=str(tmp_path / "good.eval"),
    )

    monkeypatch.setattr(run_module, "list_eval_logs", lambda _log_dir: logs)
    monkeypatch.setattr(
        run_module,
        "read_eval_log",
        lambda name: older if name.endswith("older.eval") else latest if name.endswith("latest.eval") else good,
    )

    retry_logs = run_module.discover_retry_logs(preflight)

    assert len(retry_logs) == 1
    assert retry_logs[0]["log_path"] == str(tmp_path / "latest.eval")
    assert retry_logs[0]["sample_error_count"] == 1
    assert retry_logs[0]["completed_samples"] == 99
    assert retry_logs[0]["total_samples"] == 100


def test_retry_campaign_logs_calls_eval_retry_and_scores_only_new_retry_logs(
    tmp_path, monkeypatch
) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("Use direct-user provenance checks.", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)
    preflight = preflight_campaign(config, config_path=config_path)
    retried_log_path = tmp_path / "run" / "logs" / "retry.eval"
    captured: dict[str, object] = {}

    def fake_eval_retry(
        retry_logs,
        *,
        log_dir,
        max_connections,
        attempt_timeout,
        max_retries,
        score,
    ):
        captured["retry_logs"] = retry_logs
        captured["log_dir"] = log_dir
        captured["max_connections"] = max_connections
        captured["attempt_timeout"] = attempt_timeout
        captured["max_retries"] = max_retries
        captured["score"] = score
        return [SimpleNamespace(status="success", location=str(retried_log_path))]

    monkeypatch.setattr(run_module, "eval_retry", fake_eval_retry)
    monkeypatch.setattr(
        run_module,
        "apply_post_eval_scoring_to_logs",
        lambda **kwargs: captured.setdefault("scored_paths", kwargs["log_paths"]) or [str(retried_log_path)],
    )
    monkeypatch.setattr(run_module, "load_campaign_analysis_frame", lambda _path: pd.DataFrame())
    def fake_write_analysis_outputs(_df, output_dir):
        path = output_dir / "summary.json"
        path.write_text("{}", encoding="utf-8")
        return {"summary_json": path}

    monkeypatch.setattr(run_module, "write_analysis_outputs", fake_write_analysis_outputs)

    success, _logs, written = run_module.retry_campaign_logs(
        config,
        config_path=config_path,
        retry_logs=["broken.eval"],
        preflight=preflight,
        max_connections=1,
        attempt_timeout=300,
        max_retries=7,
    )

    assert success is True
    assert captured["retry_logs"] == ["broken.eval"]
    assert captured["max_connections"] == 1
    assert captured["attempt_timeout"] == 300
    assert captured["max_retries"] == 7
    assert captured["score"] is False
    assert captured["scored_paths"] == [retried_log_path]
    assert Path(config.output_dir, "run_manifest.yaml").exists()
    assert Path(written["summary_json"]).exists()


def test_global_attacks_expand_attack_datasets_but_not_benign(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        datasets=[
            {
                "label": "benign-only",
                "dataset_file": str(tmp_path / "benign.json"),
                "evaluation_mode": "benign_save",
                "attack": "none",
            },
            {
                "label": "attack-only",
                "dataset_file": str(tmp_path / "attack.json"),
                "evaluation_mode": "attack",
            },
        ],
        attacks=[
            {"label": "literature", "attack": "attack_from_literature"},
            {"label": "markers", "attack": "universal_v2_optimized_with_markers"},
        ],
    )
    _write_dataset(tmp_path / "benign.json", count=2)
    _write_dataset(tmp_path / "attack.json", count=3)
    config = load_campaign_config(config_path)
    monkeypatch.setattr(run_module, "sleeper_eval", _fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    tasks = run_module.build_campaign_tasks(preflight)

    assert len(preflight.attacks) == 2
    assert len(tasks) == 12
    labels = {
        (
            task.metadata["campaign_dataset_label"],
            task.metadata["campaign_attack_label"],
        )
        for task in tasks
    }
    assert labels == {
        ("benign-only", "none"),
        ("attack-only", "literature"),
        ("attack-only", "markers"),
    }


def test_attack_defense_labels_can_restrict_no_attack_to_baseline_only(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        datasets=[
            {
                "label": "attack-only",
                "dataset_file": str(tmp_path / "attack.json"),
                "evaluation_mode": "attack",
            },
        ],
        defenses=[
            {"label": "no-defense", "defense": ""},
            {"label": "baseline", "defense": "system_prompt_hardening"},
            {"label": "optimized", "defense_suffix_file": str(suffix_file)},
        ],
        attacks=[
            {"label": "no-attack", "attack": "none", "defense_labels": ["no-defense"]},
            {"label": "markers", "attack": "universal_v2_optimized_with_markers"},
        ],
    )
    _write_dataset(tmp_path / "attack.json", count=3)
    config = load_campaign_config(config_path)
    monkeypatch.setattr(run_module, "sleeper_eval", _fake_sleeper_eval)

    preflight = preflight_campaign(config, config_path=config_path)
    tasks = run_module.build_campaign_tasks(preflight)

    assert len(tasks) == 8
    labels = {
        (
            task.metadata["campaign_attack_label"],
            task.metadata["campaign_model_label"],
            task.metadata["campaign_defense_label"],
        )
        for task in tasks
    }
    assert labels == {
        ("no-attack", "kimi", "no-defense"),
        ("no-attack", "gpt", "no-defense"),
        ("markers", "kimi", "no-defense"),
        ("markers", "kimi", "baseline"),
        ("markers", "kimi", "optimized"),
        ("markers", "gpt", "no-defense"),
        ("markers", "gpt", "baseline"),
        ("markers", "gpt", "optimized"),
    }

    plan = render_plan(preflight)
    assert "no-attack: attack=none defense_labels=['no-defense']" in plan


def test_dataset_filters_are_counted_and_passed_to_tasks(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    dataset_file = tmp_path / "filtered.json"
    dataset_file.write_text(
        json.dumps(
            [
                {
                    "document": {"domain": "legal"},
                    "goal": {"subcategory_id": "1a", "category_id": "1", "domain_seed": "seed-1"},
                },
                {
                    "document": {"domain": "news"},
                    "goal": {"subcategory_id": "1a", "category_id": "1", "domain_seed": "seed-1"},
                },
            ]
        ),
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        datasets=[
            {
                "label": "filtered-attack",
                "dataset_file": str(dataset_file),
                "evaluation_mode": "attack",
                "attack": "attack_from_literature",
                "doc_domain": "legal",
                "subcategory": "1a",
            }
        ],
    )
    config = load_campaign_config(config_path)
    captured_kwargs: list[dict[str, object]] = []

    def fake_task(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTask(metadata={"evaluation_mode": kwargs["evaluation_mode"]})

    monkeypatch.setattr(run_module, "sleeper_eval", fake_task)

    preflight = preflight_campaign(config, config_path=config_path)
    run_module.build_campaign_tasks(preflight)

    assert preflight.datasets[0].sample_count == 1
    assert all(kwargs["doc_domain"] == "legal" for kwargs in captured_kwargs)
    assert all(kwargs["subcategory"] == "1a" for kwargs in captured_kwargs)


def test_run_campaign_passes_limit_to_eval_set(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("Use direct-user provenance checks.", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "eval": load_campaign_config(config_path).eval.model_copy(update={"limit": 3})
        }
    )
    preflight = preflight_campaign(config, config_path=config_path)
    captured: dict[str, object] = {}

    def fake_eval_set(*, tasks, log_dir, **kwargs):
        captured["kwargs"] = kwargs
        return True, []

    monkeypatch.setattr(run_module, "eval_set", fake_eval_set)
    monkeypatch.setattr(run_module, "sleeper_eval", _fake_sleeper_eval)
    monkeypatch.setattr(run_module, "apply_post_eval_scoring", lambda **_kwargs: [])
    monkeypatch.setattr(run_module, "load_analysis_frame", lambda _path: pd.DataFrame())
    monkeypatch.setattr(
        run_module,
        "write_analysis_outputs",
        lambda _df, output_dir: {"summary_json": output_dir / "summary.json"},
    )

    run_module.run_campaign(config, config_path=config_path, preflight=preflight)

    assert captured["kwargs"]["limit"] == 3


def test_load_campaign_config_supports_legacy_single_dataset_shape(tmp_path) -> None:
    dataset_file = tmp_path / "attack_dataset.json"
    _write_dataset(dataset_file, count=1)
    config_path = tmp_path / "campaign-legacy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "legacy-campaign",
                "output_dir": str(tmp_path / "run"),
                "dataset_file": str(dataset_file),
                "evaluation_mode": "attack",
                "attack": "universal_v1",
                "models": [
                    {
                        "label": "kimi",
                        "model": "openrouter/moonshotai/kimi-k2.5",
                        "provider": "generic",
                    }
                ],
                "defenses": [{"label": "baseline", "defense": ""}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_campaign_config(config_path)

    assert len(config.datasets) == 1
    assert config.datasets[0].dataset_file == str(dataset_file)
    assert config.datasets[0].evaluation_mode == "attack"
    assert config.datasets[0].attack == "universal_v1"


def test_duplicate_dataset_labels_fail(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    dataset_file = tmp_path / "dataset.json"
    _write_dataset(dataset_file)
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "output_dir": str(tmp_path / "run"),
                "datasets": [
                    {"label": "dup", "dataset_file": str(dataset_file)},
                    {"label": "dup", "dataset_file": str(dataset_file)},
                ],
                "models": [{"label": "m", "model": "openrouter/moonshotai/kimi-k2.5", "provider": "generic"}],
                "defenses": [{"label": "d", "defense": ""}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate dataset labels"):
        load_campaign_config(config_path)


def test_missing_suffix_file_fails_preflight(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    missing_suffix = tmp_path / "missing.txt"
    config_path = _write_config(
        tmp_path,
        suffix_file=missing_suffix,
        defenses=[{"label": "optimized", "defense_suffix_file": str(missing_suffix)}],
    )
    config = load_campaign_config(config_path)

    with pytest.raises(FileNotFoundError):
        preflight_campaign(config, config_path=config_path)


def test_unknown_defense_fails_preflight(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        defenses=[{"label": "bad-defense", "defense": "not_a_real_defense"}],
    )
    config = load_campaign_config(config_path)

    with pytest.raises(ValueError, match="Unknown defenses"):
        preflight_campaign(config, config_path=config_path)


def test_missing_env_var_fails_preflight(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)

    with pytest.raises(EnvironmentError, match="Missing required environment variables"):
        preflight_campaign(config, config_path=config_path)


def test_mem0_sdk_openai_env_can_use_openrouter_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "openai",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    preflight = preflight_campaign(config, config_path=config_path)

    assert any(status.requirement == "OPENAI_API_KEY|OPENROUTER_API_KEY" for status in preflight.env_statuses)


def test_mem0_prompt_only_openrouter_env_required(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "prompt_only",
            "mem0_provider": "openrouter",
            "mem0_model": "x-ai/grok-4-fast",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    preflight = preflight_campaign(config, config_path=config_path)

    assert any(status.requirement == "OPENROUTER_API_KEY" for status in preflight.env_statuses)


def test_mem0_sdk_anthropic_env_required(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "anthropic",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        preflight_campaign(config, config_path=config_path)


def test_mem0_sdk_deepseek_env_required(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "deepseek",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    with pytest.raises(EnvironmentError, match="DEEPSEEK_API_KEY"):
        preflight_campaign(config, config_path=config_path)


def test_mem0_sdk_gemini_env_required(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "gemini",
            "mem0_model": "gemini-2.5-flash-lite",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
        preflight_campaign(config, config_path=config_path)


def test_mem0_sdk_gemini_env_status_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "gemini",
            "mem0_model": "gemini-2.5-flash-lite",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    preflight = preflight_campaign(config, config_path=config_path)

    assert any(status.requirement == "GOOGLE_API_KEY" for status in preflight.env_statuses)


def test_mem0_sdk_local_qdrant_requires_serial_eval_settings(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "deepseek",
            "eval": CampaignEvalConfig(
                max_tasks=2,
                max_connections=1,
            ),
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    with pytest.raises(ValueError, match="eval.max_tasks <= 1"):
        preflight_campaign(config, config_path=config_path)

    config = config.model_copy(
        update={
            "eval": CampaignEvalConfig(
                max_tasks=1,
                max_connections=2,
            ),
        }
    )

    with pytest.raises(ValueError, match="eval.max_connections <= 1"):
        preflight_campaign(config, config_path=config_path)


def test_mem0_sdk_managed_qdrant_allows_concurrency(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "deepseek",
            "mem0_qdrant_mode": "managed",
            "eval": CampaignEvalConfig(
                max_tasks=4,
                max_connections=2,
            ),
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    preflight = preflight_campaign(config, config_path=config_path)

    assert preflight.resolved_config["mem0_qdrant_mode"] == "managed"


def test_mem0_sdk_server_qdrant_requires_url(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "deepseek",
            "mem0_qdrant_mode": "server",
            "mem0_qdrant_url": "",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    with pytest.raises(EnvironmentError, match="mem0_qdrant_url"):
        preflight_campaign(config, config_path=config_path)


def test_preflight_renders_mem0_qdrant_details(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    _allow_mem0_sdk_prereqs(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "deepseek",
            "mem0_qdrant_mode": "managed",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    plan = render_plan(preflight_campaign(config, config_path=config_path))

    assert "mem0_qdrant_mode=managed" in plan
    assert "mem0_qdrant_collection_name=" in plan


def test_mem0_sdk_managed_qdrant_requires_docker_daemon(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setattr(preflight_module, "_mem0_sdk_prereq_errors", lambda config: ["Docker daemon is unavailable, but mem0_qdrant_mode='managed' requires it: boom"])
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_provider": "deepseek",
            "mem0_qdrant_mode": "managed",
            "scoring": CampaignScoringConfig(
                during_eval="none",
                post_eval_scorers=["mem0_write_scorer"],
                post_eval_action="overwrite",
            ),
        }
    )

    with pytest.raises(EnvironmentError, match="Docker daemon is unavailable"):
        preflight_campaign(config, config_path=config_path)


def test_managed_mem0_qdrant_runtime_sets_and_restores_env(tmp_path, monkeypatch) -> None:
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path).model_copy(
        update={
            "memory_backend": "mem0",
            "mem0_runtime": "sdk",
            "mem0_qdrant_mode": "managed",
        }
    )
    runtime = mem0_qdrant_module.Mem0QdrantRuntime(
        mode="managed",
        url="http://127.0.0.1:6333",
        container_name="qdrant-test",
        storage_dir=str(tmp_path / "storage"),
    )
    removed: list[list[str]] = []
    monkeypatch.setattr(mem0_qdrant_module, "_managed_qdrant_runtime", lambda cfg, output_dir: runtime)
    monkeypatch.setattr(
        mem0_qdrant_module,
        "_docker_run",
        lambda args: removed.append(args) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.delenv("SLEEPER_EVAL_MEM0_QDRANT_URL", raising=False)

    with mem0_qdrant_module.mem0_qdrant_runtime(config, output_dir=tmp_path) as active:
        assert active == runtime
        assert os.environ["SLEEPER_EVAL_MEM0_QDRANT_URL"] == "http://127.0.0.1:6333"

    assert "SLEEPER_EVAL_MEM0_QDRANT_URL" not in os.environ
    assert removed == [["rm", "-f", "qdrant-test"]]


def test_incompatible_scorer_backend_fails_preflight(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        scoring={
            "during_eval": "none",
            "post_eval_scorers": ["tool_call_scorer"],
            "post_eval_action": "overwrite",
        },
    )
    config = load_campaign_config(config_path).model_copy(update={"memory_backend": "mem0"})

    with pytest.raises(ValueError, match="incompatible"):
        preflight_campaign(config, config_path=config_path)


def test_incompatible_scorer_mode_fails_preflight(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        suffix_file=suffix_file,
        datasets=[
            {
                "label": "benign-only",
                "dataset_file": str(tmp_path / "benign.json"),
                "evaluation_mode": "benign_save",
                "attack": "none",
            }
        ],
        scoring={
            "during_eval": "none",
            "post_eval_scorers": ["goal_match_scorer"],
            "post_eval_action": "append",
        },
    )
    _write_dataset(tmp_path / "benign.json", count=1)
    config = load_campaign_config(config_path)

    with pytest.raises(ValueError, match="incompatible"):
        preflight_campaign(config, config_path=config_path)


def test_output_dir_config_hash_mismatch_fails_preflight(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump({"config_hash": "different-hash"}, sort_keys=False),
        encoding="utf-8",
    )
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file, output_dir=output_dir)
    config = load_campaign_config(config_path)

    with pytest.raises(ValueError, match="different campaign"):
        preflight_campaign(config, config_path=config_path)


def test_matching_output_dir_hash_is_resume(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    output_dir = tmp_path / "run"
    config_path = _write_config(tmp_path, suffix_file=suffix_file, output_dir=output_dir)
    config = load_campaign_config(config_path)
    first = preflight_campaign(config, config_path=config_path)
    output_dir.mkdir()
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump({"config_hash": first.config_hash}, sort_keys=False),
        encoding="utf-8",
    )

    resumed = preflight_campaign(config, config_path=config_path)

    assert resumed.output_mode == "resume"


def test_legacy_config_yaml_shape_is_resume(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    output_dir = tmp_path / "run"
    config_path = _write_config(tmp_path, suffix_file=suffix_file, output_dir=output_dir)
    config = load_campaign_config(config_path)
    first = preflight_campaign(config, config_path=config_path)
    output_dir.mkdir()

    legacy_config = json.loads(json.dumps(first.resolved_config))
    legacy_config["eval"].pop("sample_shuffle", None)
    legacy_config["eval"].pop("sample_shuffle_seed", None)
    for attack in legacy_config["attacks"]:
        attack.pop("defense_labels", None)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(legacy_config, sort_keys=False),
        encoding="utf-8",
    )

    resumed = preflight_campaign(config, config_path=config_path)

    assert resumed.output_mode == "resume"


def test_manifest_hash_mismatch_but_matching_config_yaml_is_resume(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    output_dir = tmp_path / "run"
    config_path = _write_config(tmp_path, suffix_file=suffix_file, output_dir=output_dir)
    config = load_campaign_config(config_path)
    first = preflight_campaign(config, config_path=config_path)
    output_dir.mkdir()
    (output_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump({"config_hash": "stale-hash"}, sort_keys=False),
        encoding="utf-8",
    )

    legacy_config = json.loads(json.dumps(first.resolved_config))
    legacy_config["eval"].pop("sample_shuffle", None)
    legacy_config["eval"].pop("sample_shuffle_seed", None)
    for attack in legacy_config["attacks"]:
        attack.pop("defense_labels", None)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(legacy_config, sort_keys=False),
        encoding="utf-8",
    )

    resumed = preflight_campaign(config, config_path=config_path)

    assert resumed.output_mode == "resume"


def test_main_dry_run_prints_plan_and_skips_eval_set(tmp_path, monkeypatch, capsys) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    called = {"eval_set": False}

    monkeypatch.setattr(
        run_module,
        "eval_set",
        lambda **_kwargs: called.__setitem__("eval_set", True),
    )

    exit_code = run_module.main([str(config_path), "--dry-run"])

    assert exit_code == 0
    assert called["eval_set"] is False
    assert "== Eval Campaign Plan ==" in capsys.readouterr().out


def test_main_prompts_by_default_in_tty(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    exit_code = run_module.main([str(config_path)])

    assert exit_code == 1


def test_main_yes_bypasses_prompt(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(run_module, "run_campaign", lambda *_args, **_kwargs: (True, [], {}))

    exit_code = run_module.main([str(config_path), "--yes"])

    assert exit_code == 0


def test_analyze_campaign_rewrites_outputs_without_running_eval(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    config = load_campaign_config(config_path)
    preflight = preflight_campaign(config, config_path=config_path)
    output_dir = Path(preflight.output_dir)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_module, "load_campaign_analysis_frame", lambda _path: pd.DataFrame([{"x": 1}]))

    def fake_write_campaign_analysis(target_output_dir: Path):
        path = target_output_dir / "analysis" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {"summary_json": path}

    monkeypatch.setattr(run_module, "write_campaign_analysis", fake_write_campaign_analysis)

    written = run_module.analyze_campaign(config, config_path=config_path, preflight=preflight)

    assert Path(config.output_dir, "config.yaml").exists()
    assert Path(config.output_dir, "run_manifest.yaml").exists()
    assert Path(written["summary_json"]).exists()


def test_main_analyze_only_uses_analysis_path(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    captured: dict[str, object] = {}

    def fake_analyze_campaign(config, *, config_path, preflight):
        captured["config_path"] = config_path
        captured["output_dir"] = config.output_dir
        return {}

    monkeypatch.setattr(run_module, "analyze_campaign", fake_analyze_campaign)

    exit_code = run_module.main([str(config_path), "--yes", "--analyze-only"])

    assert exit_code == 0
    assert captured["config_path"] == Path(config_path).resolve()


def test_main_analyze_only_conflicts_with_retry_flags(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)

    with pytest.raises(SystemExit, match="cannot be combined"):
        run_module.main([str(config_path), "--analyze-only", "--retry"])


def test_main_retry_log_uses_retry_campaign_path(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    captured: dict[str, object] = {}

    def fake_retry_campaign(config, *, config_path, retry_logs, preflight, max_connections, attempt_timeout, max_retries):
        captured["retry_logs"] = retry_logs
        captured["max_connections"] = max_connections
        captured["attempt_timeout"] = attempt_timeout
        captured["max_retries"] = max_retries
        return True, [], {}

    monkeypatch.setattr(run_module, "retry_campaign_logs", fake_retry_campaign)

    exit_code = run_module.main(
        [
            str(config_path),
            "--yes",
            "--retry-log",
            "broken.eval",
            "--retry-max-connections",
            "1",
            "--retry-attempt-timeout",
            "300",
            "--retry-max-retries",
            "7",
        ]
    )

    assert exit_code == 0
    assert captured["retry_logs"] == [str(Path("broken.eval").resolve())]
    assert captured["max_connections"] == 1
    assert captured["attempt_timeout"] == 300
    assert captured["max_retries"] == 7


def test_main_retry_auto_discovers_logs(tmp_path, monkeypatch) -> None:
    _set_required_env(monkeypatch)
    suffix_file = tmp_path / "suffix.txt"
    suffix_file.write_text("suffix", encoding="utf-8")
    config_path = _write_config(tmp_path, suffix_file=suffix_file)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        run_module,
        "discover_retry_logs",
        lambda _preflight: [
            {
                "log_path": "/tmp/broken.eval",
                "dataset_label": "eval-in-attack",
                "attack_label": "universal_v2_optimized_with_markers",
                "model_label": "kimi",
                "defense_label": "baseline",
                "status": "success",
                "completed_samples": 99,
                "total_samples": 100,
                "sample_error_count": 1,
                "invalidated_sample_count": 0,
            }
        ],
    )
    captured: dict[str, object] = {}

    def fake_retry_campaign(config, *, config_path, retry_logs, preflight, max_connections, attempt_timeout, max_retries):
        captured["retry_logs"] = retry_logs
        return True, [], {}

    monkeypatch.setattr(run_module, "retry_campaign_logs", fake_retry_campaign)

    exit_code = run_module.main([str(config_path), "--yes", "--retry"])

    assert exit_code == 0
    assert captured["retry_logs"] == ["/tmp/broken.eval"]


def test_script_wrapper_delegates_to_package_main(monkeypatch) -> None:
    import scripts.run_eval_campaign as wrapper

    assert wrapper.main is run_module.main


def test_apply_post_eval_scoring_skips_incompatible_logs(monkeypatch, tmp_path) -> None:
    fake_logs = [
        SimpleNamespace(name=str(tmp_path / "attack.eval")),
        SimpleNamespace(name=str(tmp_path / "benign.eval")),
    ]
    attack_log = SimpleNamespace(
        eval=SimpleNamespace(task_args={"memory_backend": "tool", "evaluation_mode": "attack"})
    )
    benign_log = SimpleNamespace(
        eval=SimpleNamespace(task_args={"memory_backend": "tool", "evaluation_mode": "benign_save"})
    )
    scored_with: list[list[str]] = []

    monkeypatch.setattr(run_module, "list_eval_logs", lambda _log_dir: fake_logs)
    monkeypatch.setattr(
        run_module,
        "read_eval_log",
        lambda name: attack_log if name.endswith("attack.eval") else benign_log,
    )
    monkeypatch.setattr(
        run_module,
        "resolve_scorers",
        lambda names: names,
    )
    monkeypatch.setattr(
        run_module,
        "inspect_score",
        lambda log, scorers, action, copy: scored_with.append(list(scorers)) or log,
    )
    monkeypatch.setattr(run_module, "write_eval_log", lambda _log, location: location)

    scored_logs = run_module.apply_post_eval_scoring(
        log_dir=tmp_path,
        scorer_names=["goal_match_scorer", "benign_save_goal_match_scorer"],
        action="append",
    )

    assert len(scored_logs) == 2
    assert scored_with == [
        ["goal_match_scorer"],
        ["benign_save_goal_match_scorer"],
    ]
