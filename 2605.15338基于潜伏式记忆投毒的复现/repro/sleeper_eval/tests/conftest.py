"""Shared pytest fixtures for sleeper_eval tests."""

from __future__ import annotations

import pytest

from sleeper_eval.dataset import load_dataset


@pytest.fixture(scope="session")
def dev_with_memories_dataset():
    return load_dataset("datasets/released/generated/behaviour_true_optimized_with_memories.json")


@pytest.fixture(scope="session")
def dev_with_memories_sample(dev_with_memories_dataset):
    return dev_with_memories_dataset[0]


@pytest.fixture(scope="session")
def claude_format_smoke_dataset():
    return load_dataset("datasets/smoke/merged_eval_in_provider_docrep_smoke.json")
