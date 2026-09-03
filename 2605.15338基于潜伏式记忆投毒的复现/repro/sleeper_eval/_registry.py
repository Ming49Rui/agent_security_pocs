"""Inspect task registration entry point."""

from .followup_eval.task import followup_eval
from .task import sleeper_eval

__all__ = ["sleeper_eval", "followup_eval"]
