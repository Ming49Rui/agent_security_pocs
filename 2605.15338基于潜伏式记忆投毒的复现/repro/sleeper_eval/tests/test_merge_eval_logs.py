from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.support.merge_eval_logs as merge_module


def _write_eval(path: Path, text: str = "stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolve_source_log_dir_accepts_run_dir_and_logs_dir(tmp_path) -> None:
    run_dir = tmp_path / "run-a"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)

    assert merge_module.resolve_source_log_dir(run_dir) == logs_dir.resolve()
    assert merge_module.resolve_source_log_dir(logs_dir) == logs_dir.resolve()


def test_merge_logs_copies_logs_and_writes_manifest(tmp_path, monkeypatch) -> None:
    source_a = tmp_path / "run-a" / "logs"
    source_b = tmp_path / "run-b" / "logs"
    file_a = source_a / "a.eval"
    file_b = source_b / "b.eval"
    _write_eval(file_a, "a")
    _write_eval(file_b, "b")

    eval_ids = {str(file_a.resolve()): "eval-a", str(file_b.resolve()): "eval-b"}

    def fake_read_eval_log(path: str, **_kwargs):
        return SimpleNamespace(eval=SimpleNamespace(eval_id=eval_ids[str(Path(path).resolve())]))

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)

    manifest = merge_module.merge_logs(
        sources=[str(source_a), str(source_b.parent)],
        output_dir=tmp_path / "combined",
    )

    target_logs = tmp_path / "combined" / "logs"
    assert (target_logs / "a.eval").exists()
    assert (target_logs / "b.eval").exists()
    assert manifest["merged_count"] == 2
    assert (tmp_path / "combined" / "merge_manifest.yaml").exists()


def test_merge_logs_skips_duplicate_eval_ids(tmp_path, monkeypatch) -> None:
    existing = tmp_path / "combined" / "logs" / "existing.eval"
    incoming = tmp_path / "run-a" / "logs" / "incoming.eval"
    _write_eval(existing, "existing")
    _write_eval(incoming, "incoming")

    eval_ids = {
        str(existing.resolve()): "same-eval-id",
        str(incoming.resolve()): "same-eval-id",
    }

    def fake_read_eval_log(path: str, **_kwargs):
        return SimpleNamespace(eval=SimpleNamespace(eval_id=eval_ids[str(Path(path).resolve())]))

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)

    manifest = merge_module.merge_logs(
        sources=[str(incoming.parent)],
        output_dir=tmp_path / "combined",
        skip_duplicate_eval_id=True,
    )

    assert manifest["merged_count"] == 0
    assert manifest["skipped_duplicate_eval_id_count"] == 1
    assert (tmp_path / "combined" / "logs" / "incoming.eval").exists() is False


def test_merge_logs_can_run_analysis(tmp_path, monkeypatch) -> None:
    source_a = tmp_path / "run-a" / "logs"
    file_a = source_a / "a.eval"
    _write_eval(file_a, "a")

    eval_ids = {str(file_a.resolve()): "eval-a"}

    def fake_read_eval_log(path: str, **_kwargs):
        resolved = str(Path(path).resolve())
        return SimpleNamespace(
            eval=SimpleNamespace(eval_id=eval_ids.get(resolved, "eval-a")),
            metadata={},
        )

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)
    monkeypatch.setattr(merge_module, "load_analysis_frame", lambda _path: pd.DataFrame([{"x": 1}]))
    monkeypatch.setattr(merge_module, "with_display_columns", lambda df: df)
    monkeypatch.setattr(
        merge_module,
        "write_analysis_outputs",
        lambda _df, out: {"summary_md": out / "summary.md"},
    )

    manifest = merge_module.merge_logs(
        sources=[str(source_a)],
        output_dir=tmp_path / "combined",
        analyze=True,
    )

    assert manifest["analysis_outputs"]["summary_md"].endswith("summary.md")


def test_merge_logs_fails_on_overlapping_campaign_task_cells(tmp_path, monkeypatch) -> None:
    source_a = tmp_path / "run-a" / "logs"
    source_b = tmp_path / "run-b" / "logs"
    file_a = source_a / "a.eval"
    file_b = source_b / "b.eval"
    _write_eval(file_a, "a")
    _write_eval(file_b, "b")

    def fake_read_eval_log(path: str, **_kwargs):
        path_name = Path(path).name
        return SimpleNamespace(
            eval=SimpleNamespace(eval_id=path_name, metadata={}),
            metadata={
                "campaign_dataset_label": "heldout",
                "campaign_attack_label": "markers",
                "campaign_model_label": "gpt",
                "campaign_defense_label": "baseline",
            },
        )

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)

    try:
        merge_module.merge_logs(
            sources=[str(source_a), str(source_b)],
            output_dir=tmp_path / "combined",
        )
    except ValueError as exc:
        assert "Overlapping task cell detected" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("Expected merge compatibility failure")


def test_merge_logs_allows_retried_task_cell_within_single_source(tmp_path, monkeypatch) -> None:
    run_a = tmp_path / "run-a"
    source_a = run_a / "logs"
    file_a = source_a / "a.eval"
    file_b = source_a / "b.eval"
    _write_eval(file_a, "a")
    _write_eval(file_b, "b")
    (run_a / "config.yaml").write_text(
        """
name: campaign
memory_backend: tool
mention_memory_system: true
grader_model: openai/gpt-5.4-mini
scoring:
  post_eval_scorers: [tool_call_scorer]
datasets: []
attacks: []
models: []
defenses: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fake_read_eval_log(path: str, **_kwargs):
        path_name = Path(path).name
        return SimpleNamespace(
            eval=SimpleNamespace(eval_id=path_name, metadata={}),
            metadata={
                "campaign_dataset_label": "heldout",
                "campaign_attack_label": "markers",
                "campaign_model_label": "gpt",
                "campaign_defense_label": "baseline",
            },
        )

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)

    manifest = merge_module.merge_logs(
        sources=[str(run_a)],
        output_dir=tmp_path / "combined",
    )

    assert manifest["merged_count"] == 2


def test_merge_logs_fails_on_conflicting_defense_definitions(tmp_path, monkeypatch) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    source_a = run_a / "logs"
    source_b = run_b / "logs"
    file_a = source_a / "a.eval"
    file_b = source_b / "b.eval"
    _write_eval(file_a, "a")
    _write_eval(file_b, "b")

    config_base = {
        "name": "campaign",
        "memory_backend": "tool",
        "mention_memory_system": True,
        "grader_model": "openai/gpt-5.4-mini",
        "scoring": {"post_eval_scorers": ["tool_call_scorer"]},
        "datasets": [],
        "attacks": [],
        "models": [],
    }
    (run_a / "config.yaml").write_text(
        """
name: campaign
memory_backend: tool
mention_memory_system: true
grader_model: openai/gpt-5.4-mini
scoring:
  post_eval_scorers: [tool_call_scorer]
datasets: []
attacks: []
models: []
defenses:
  - label: baseline
    defense: ""
    suffix_sha256: aaa
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (run_b / "config.yaml").write_text(
        """
name: campaign
memory_backend: tool
mention_memory_system: true
grader_model: openai/gpt-5.4-mini
scoring:
  post_eval_scorers: [tool_call_scorer]
datasets: []
attacks: []
models: []
defenses:
  - label: baseline
    defense: ""
    suffix_sha256: bbb
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fake_read_eval_log(path: str, **_kwargs):
        path_name = Path(path).name
        return SimpleNamespace(
            eval=SimpleNamespace(eval_id=path_name, metadata={}),
            metadata={
                "campaign_dataset_label": "heldout",
                "campaign_attack_label": "markers",
                "campaign_model_label": "gpt" if path_name == "a.eval" else "kimi",
                "campaign_defense_label": "baseline",
            },
        )

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)

    try:
        merge_module.merge_logs(
            sources=[str(run_a), str(run_b)],
            output_dir=tmp_path / "combined",
        )
    except ValueError as exc:
        assert "defenses.baseline" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("Expected merge compatibility failure")


def test_merge_logs_uses_campaign_analysis_when_logs_have_campaign_metadata(tmp_path, monkeypatch) -> None:
    run_a = tmp_path / "run-a"
    source_a = run_a / "logs"
    file_a = source_a / "a.eval"
    _write_eval(file_a, "a")
    (run_a / "config.yaml").write_text(
        """
name: campaign-a
memory_backend: tool
mention_memory_system: true
grader_model: openai/gpt-5.4-mini
scoring:
  post_eval_scorers: [tool_call_scorer]
datasets: []
attacks: []
models: []
defenses: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    eval_ids = {str(file_a.resolve()): "eval-a"}

    def fake_read_eval_log(path: str, **_kwargs):
        return SimpleNamespace(
            eval=SimpleNamespace(eval_id=eval_ids.get(str(Path(path).resolve()), "eval-out")),
            metadata={
                "campaign_dataset_label": "heldout",
                "campaign_attack_label": "markers",
                "campaign_model_label": "gpt",
                "campaign_defense_label": "baseline",
            },
        )

    monkeypatch.setattr(merge_module, "read_eval_log", fake_read_eval_log)
    monkeypatch.setattr(
        merge_module,
        "load_campaign_analysis_frame",
        lambda _path: pd.DataFrame([{"x": 1}]),
    )
    monkeypatch.setattr(merge_module, "load_analysis_frame", lambda _path: pd.DataFrame([{"x": 2}]))
    monkeypatch.setattr(merge_module, "with_display_columns", lambda df: df)
    monkeypatch.setattr(
        merge_module,
        "write_analysis_outputs",
        lambda _df, out: {"summary_md": out / "summary.md"},
    )

    manifest = merge_module.merge_logs(
        sources=[str(run_a)],
        output_dir=tmp_path / "combined",
        analyze=True,
    )

    assert manifest["analysis_mode"] == "campaign"
