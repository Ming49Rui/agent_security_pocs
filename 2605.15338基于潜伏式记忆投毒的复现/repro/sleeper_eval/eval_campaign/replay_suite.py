"""Helpers for launching a suite of mem0 manager replay configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import load_campaign_config


@dataclass(frozen=True)
class ReplaySuiteTarget:
    manager_key: str
    config_path: Path
    output_dir: Path


def _manager_key_from_config_name(
    *,
    stem: str,
    run_prefix: str,
    subject_family: str,
) -> str | None:
    prefix = f"{run_prefix}_{subject_family}_"
    if not stem.startswith(prefix):
        return None
    manager_key = stem[len(prefix) :].strip()
    if not manager_key or manager_key == "transcript":
        return None
    return manager_key


def _manager_key_from_family_replay_stem(stem: str) -> str | None:
    prefix = "replay_"
    if not stem.startswith(prefix):
        return None
    manager_key = stem[len(prefix) :].strip()
    if not manager_key:
        return None
    return manager_key


def discover_replay_suite_targets(
    *,
    config_dir: Path,
    run_prefix: str,
    subject_family: str,
    manager_filters: Iterable[str] = (),
) -> list[ReplaySuiteTarget]:
    normalized_filters = {item.strip() for item in manager_filters if item.strip()}
    targets: list[ReplaySuiteTarget] = []
    family_dir = config_dir / subject_family
    if family_dir.is_dir():
        config_paths = sorted(family_dir.glob("replay_*.yaml"))
        for config_path in config_paths:
            manager_key = _manager_key_from_family_replay_stem(config_path.stem)
            if manager_key is None:
                continue
            if normalized_filters and manager_key not in normalized_filters:
                continue
            config = load_campaign_config(config_path)
            targets.append(
                ReplaySuiteTarget(
                    manager_key=manager_key,
                    config_path=config_path.resolve(),
                    output_dir=Path(config.output_dir).resolve(),
                )
            )
        return targets

    for config_path in sorted(config_dir.glob(f"{run_prefix}_{subject_family}_*.yaml")):
        manager_key = _manager_key_from_config_name(
            stem=config_path.stem,
            run_prefix=run_prefix,
            subject_family=subject_family,
        )
        if manager_key is None:
            continue
        if normalized_filters and manager_key not in normalized_filters:
            continue
        config = load_campaign_config(config_path)
        targets.append(
            ReplaySuiteTarget(
                manager_key=manager_key,
                config_path=config_path.resolve(),
                output_dir=Path(config.output_dir).resolve(),
            )
        )
    return targets


def replay_output_is_complete(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    manifest = output_dir / "run_manifest.yaml"
    log_dir = output_dir / "logs"
    analysis_summary = output_dir / "analysis" / "summary.json"
    if not manifest.is_file() or not analysis_summary.is_file() or not log_dir.is_dir():
        return False
    return any(log_dir.glob("*.eval"))


def build_replay_command(
    *,
    python_executable: str,
    script_path: Path,
    config_path: Path,
    sources: Iterable[str | Path],
    output_dir: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(script_path),
        str(config_path),
    ]
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir.resolve())])
    for source in sources:
        command.extend(["--replay-mem0-manager-from", str(Path(source).resolve())])
    command.append("--yes")
    return command
