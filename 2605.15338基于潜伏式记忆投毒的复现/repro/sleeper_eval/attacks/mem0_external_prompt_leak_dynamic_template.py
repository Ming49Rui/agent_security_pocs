"""Dynamic mem0 attack variant driven by a runtime-provided template."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ATTACK_ID = "mem0_external_prompt_leak_dynamic_template"
ATTACK_META = {
    "description": (
        "Dynamic mem0 attack variant that appends a runtime-provided universal "
        "template containing a {goal_text} placeholder."
    ),
    "position": "end",
    "type": "dynamic_external_mem0",
    "variant": "dynamic_template",
    "origin": "actor_critic_runtime",
}

_TEMPLATE_TEXT_ENV = "SLEEPER_EVAL_MEM0_AC_ATTACK_TEMPLATE"
_TEMPLATE_FILE_ENV = "SLEEPER_EVAL_MEM0_AC_ATTACK_TEMPLATE_FILE"


def _load_template() -> str:
    file_path = os.getenv(_TEMPLATE_FILE_ENV, "").strip()
    if file_path:
        template = Path(file_path).read_text(encoding="utf-8")
    else:
        template = os.getenv(_TEMPLATE_TEXT_ENV, "")
    template = template.strip("\n")
    if not template:
        raise ValueError(
            "Dynamic mem0 attack template is empty. Set "
            f"{_TEMPLATE_TEXT_ENV} or {_TEMPLATE_FILE_ENV}."
        )
    if "{goal_text}" not in template:
        raise ValueError(
            "Dynamic mem0 attack template must contain the literal '{goal_text}' placeholder."
        )
    return template


def apply(doc_text: str, goal_text: str, sample_metadata: dict[str, Any]) -> str:
    """Append the runtime template to the end of the document."""
    del sample_metadata
    injection = _load_template().format(goal_text=goal_text)
    return f"{doc_text}{injection}" if doc_text else injection
