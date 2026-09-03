from __future__ import annotations

from pathlib import Path

import pytest

import sleeper_eval.defenses as defenses_module
from sleeper_eval.defenses import (
    DefenseConfig,
    GEPA_PROMPT_HARDENING_SUFFIX,
    defense_config_from_names,
    resolve_builtin_defense_suffix,
)


def test_defense_config_from_names_returns_empty_config() -> None:
    assert defense_config_from_names("") == DefenseConfig()


def test_defense_config_from_names_parses_multiple_defenses() -> None:
    config = defense_config_from_names(
        "system_prompt_hardening,untrusted_content_markers"
    )

    assert config.system_prompt_hardening is True
    assert config.untrusted_content_markers is True


def test_defense_config_from_names_accepts_list_values() -> None:
    config = defense_config_from_names(
        ["system_prompt_hardening", "untrusted_content_markers"]
    )

    assert config.system_prompt_hardening is True
    assert config.untrusted_content_markers is True


def test_defense_config_from_names_parses_builtin_gepa_suffix_defense() -> None:
    config = defense_config_from_names(GEPA_PROMPT_HARDENING_SUFFIX)

    assert config.gepa_prompt_hardening_suffix is True


def test_resolve_builtin_defense_suffix_reads_expected_file(tmp_path, monkeypatch) -> None:
    suffix_file = tmp_path / "best_suffix.txt"
    suffix_file.write_text("suffix text", encoding="utf-8")
    monkeypatch.setattr(
        defenses_module,
        "_BUILTIN_DEFENSE_SUFFIX_FILES",
        {GEPA_PROMPT_HARDENING_SUFFIX: Path(suffix_file)},
    )

    suffix = resolve_builtin_defense_suffix(
        DefenseConfig(gepa_prompt_hardening_suffix=True)
    )

    assert suffix == "suffix text"


def test_defense_config_from_names_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        defense_config_from_names("system_prompt_hardening,unknown_defense")
