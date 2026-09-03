"""GEPA callbacks for local logging of defense optimization runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ExperimentLoggerCallback:
    root_dir: Path

    @property
    def events_path(self) -> Path:
        return self.root_dir / "artifacts" / "gepa_events.jsonl"

    def _write(self, kind: str, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event_type": kind, **event}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        candidate = event["candidate"]
        event = dict(event)
        event["candidate_suffix_length"] = len(candidate.get("defense_suffix", ""))
        self._write("candidate_selected", event)

    def on_reflective_dataset_built(self, event: dict[str, Any]) -> None:
        self._write("reflective_dataset_built", event)

    def on_proposal_end(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["new_instruction_lengths"] = {
            key: len(value) for key, value in event["new_instructions"].items()
        }
        self._write("proposal_end", event)

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        self._write("candidate_accepted", event)

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        self._write("candidate_rejected", event)

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["candidate_suffix_length"] = len(event["candidate"].get("defense_suffix", ""))
        self._write("valset_evaluated", event)


@dataclass
class ConsoleProgressCallback:
    """Compact console logging for local runs."""

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        suffix = event["candidate"].get("defense_suffix", "")
        print(
            f"[gepa] iter={event['iteration']} selected={event['candidate_idx']} "
            f"score={event['score']:.3f} len={len(suffix)}"
        )

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        print(
            f"[gepa] iter={event['iteration']} accepted candidate={event['new_candidate_idx']} "
            f"score={event['new_score']:.3f}"
        )

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        print(
            f"[gepa] iter={event['iteration']} rejected old={event['old_score']:.3f} "
            f"new={event['new_score']:.3f} reason={event['reason']}"
        )

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        print(
            f"[gepa] iter={event['iteration']} val candidate={event['candidate_idx']} "
            f"avg={event['average_score']:.3f} examples={event['num_examples_evaluated']}"
        )
