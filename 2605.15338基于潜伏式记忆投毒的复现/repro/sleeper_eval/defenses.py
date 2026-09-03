"""Defense configuration helpers for sleeper memory evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

GEPA_PROMPT_HARDENING_SUFFIX = "gepa_prompt_hardening_suffix"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILTIN_DEFENSE_SUFFIX_FILES = {
    GEPA_PROMPT_HARDENING_SUFFIX: _REPO_ROOT
    / "experiments/gepa_defense_optimization/gepa_prompt_hardening_suffix.txt",
}

VALID_DEFENSES = {
    "system_prompt_hardening",
    "untrusted_content_markers",
    GEPA_PROMPT_HARDENING_SUFFIX,
}


@dataclass(frozen=True)
class DefenseConfig:
    system_prompt_hardening: bool = False
    untrusted_content_markers: bool = False
    gepa_prompt_hardening_suffix: bool = False

    def to_metadata(self) -> dict[str, bool]:
        return asdict(self)


def resolve_builtin_defense_suffix(defense_config: DefenseConfig) -> str:
    if defense_config.gepa_prompt_hardening_suffix:
        suffix_path = _BUILTIN_DEFENSE_SUFFIX_FILES[GEPA_PROMPT_HARDENING_SUFFIX]
        return suffix_path.read_text(encoding="utf-8").strip()
    return ""


def defense_config_from_names(names_csv: str | Sequence[str] | None) -> DefenseConfig:
    """Build config from defense names provided as a string or list."""

    if names_csv is None:
        return DefenseConfig()

    raw_names: list[str]
    if isinstance(names_csv, str):
        raw_names = names_csv.split(",")
    else:
        raw_names = []
        for item in names_csv:
            raw_names.extend(part for part in item.split(","))

    names = {name.strip() for name in raw_names if name.strip()}
    if not names:
        return DefenseConfig()

    unknown = sorted(names - VALID_DEFENSES)
    if unknown:
        raise ValueError(
            f"Unknown defenses: {unknown}. Available: {sorted(VALID_DEFENSES)}"
        )

    return DefenseConfig(
        system_prompt_hardening="system_prompt_hardening" in names,
        untrusted_content_markers="untrusted_content_markers" in names,
        gepa_prompt_hardening_suffix=GEPA_PROMPT_HARDENING_SUFFIX in names,
    )
