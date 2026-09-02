"""No-attack baseline."""

from __future__ import annotations

from typing import Any

ATTACK_ID = "none"
ATTACK_META = {
    "description": "No attack; leave the document unchanged.",
    "type": "baseline",
}


def apply(doc_text: str, goal_text: str, sample_metadata: dict[str, Any]) -> str:
    """Return the original document unchanged."""

    del goal_text, sample_metadata
    return doc_text
