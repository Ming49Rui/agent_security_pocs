"""Dataset loading for follow-up behavioral influence evaluations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from inspect_ai.dataset import Dataset, Sample, json_dataset

DATASET_ROOT = Path(__file__).resolve().parents[2] / "eval_dataset" / "datasets"
REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_TURN_ROLES = {"system", "user", "assistant"}
VALID_SPLITS = {"smoke", "dev", "eval_in", "eval_ood"}


class ConversationTurn(BaseModel):
    """One turn from a follow-up conversation transcript."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class FollowupSampleMeta(BaseModel):
    """Typed metadata for follow-up behavioral influence samples."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    initial_turns: list[ConversationTurn]
    user_queries: list[str]
    memories: list[str]
    injected_memory: str
    memory_count: int
    injected_memory_index: int
    goal_text: str | None = None
    doc_id: str | None = None
    doc_domain: str | None = None
    source_query: str | None = None
    query_style: str | None = None
    multi_turn_split: str | None = None
    source_format: str
    category: str | None = None
    source_goal_id: str | None = None
    variant: str
    split: str
    source_file: str
    sample_id: str


class FollowupDatasetInfo(BaseModel):
    """Static metadata inferred from the dataset path."""

    model_config = ConfigDict(frozen=True)

    variant: str
    split: str
    source_file: str


FilterFn = Callable[[Sample], bool]


def resolve_dataset_path(file_path: str | Path) -> Path:
    """Resolve a follow-up dataset path."""

    path = Path(file_path)
    if path.is_absolute():
        resolved = path
    else:
        resolved = DATASET_ROOT / path
        if not resolved.exists():
            alt = REPO_ROOT / path
            if alt.exists():
                resolved = alt

    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}. Pass a short relative path like "
            "'followup/smoke/file.json', a repo-relative path, or an absolute path."
        )

    return resolved


def infer_dataset_info(file_path: str | Path) -> FollowupDatasetInfo:
    """Infer split and variant for a follow-up dataset."""

    path = resolve_dataset_path(file_path)
    split = next((part for part in path.parts if part in VALID_SPLITS), None)
    if split is None:
        stem = path.stem
        if stem.startswith("merged_dev_"):
            split = "dev"
        elif stem.startswith("merged_eval_in_"):
            split = "eval_in"
        elif stem.startswith("merged_ood_"):
            split = "eval_ood"
    if split is None:
        raise ValueError(
            "Unsupported follow-up dataset path. Expected one path segment in "
            f"{sorted(VALID_SPLITS)}: {path}"
        )

    return FollowupDatasetInfo(
        variant=path.stem,
        split=split,
        source_file=path.name,
    )


def _validate_record(record: dict[str, Any]) -> None:
    if "memory_state" in record and "conversation" in record:
        conversation = record.get("conversation") or {}
        turns = conversation.get("turns") or []
        if not turns:
            raise ValueError("Follow-up sample must include at least one conversation turn.")

        last_role = turns[-1].get("role")
        if last_role != "user":
            raise ValueError("Follow-up samples must end on a user turn so the model can respond.")

        for turn in turns:
            role = turn.get("role")
            if role not in VALID_TURN_ROLES:
                raise ValueError(f"Unsupported conversation role: {role!r}")

        memory_state = record.get("memory_state") or {}
        memories = list(memory_state.get("memories") or [])
        injected_memory = memory_state.get("injected_memory")
        if not isinstance(injected_memory, str) or not injected_memory.strip():
            raise ValueError("Follow-up sample must include a non-empty injected_memory.")
        if injected_memory not in memories:
            raise ValueError("Injected memory must appear in the full memory list.")
        return

    if "multi_turn_queries" in record and "goal" in record and "preexisting_memories" in record:
        user_queries = list(record.get("multi_turn_queries") or [])
        if not user_queries:
            raise ValueError("Multiturn follow-up sample must include multi_turn_queries.")
        if not all(isinstance(query, str) and query.strip() for query in user_queries):
            raise ValueError("All multiturn queries must be non-empty strings.")
        goal_text = (record.get("goal") or {}).get("goal_text")
        if not isinstance(goal_text, str) or not goal_text.strip():
            raise ValueError("Multiturn follow-up sample must include a non-empty goal_text.")
        return

    raise ValueError("Unsupported follow-up record format.")


def make_record_to_sample(file_path: str | Path) -> Callable[[dict[str, Any]], Sample]:
    """Build a record-to-sample mapper bound to a follow-up dataset file."""

    info = infer_dataset_info(file_path)

    def record_to_sample(record: dict[str, Any]) -> Sample:
        _validate_record(record)

        if "memory_state" in record and "conversation" in record:
            conversation = record["conversation"]
            memory_state = record["memory_state"]
            metadata = record.get("metadata") or {}
            memories = list(memory_state.get("memories") or [])
            injected_memory = str(memory_state["injected_memory"])
            turns = [
                ConversationTurn(role=turn["role"], content=turn["content"])
                for turn in conversation["turns"]
            ]
            initial_turns = turns[:-1]
            user_queries = [turns[-1].content]
            sample_id = str(metadata.get("sample_id") or conversation["conversation_id"])
            category = metadata.get("category")
            source_goal_id = metadata.get("source_goal_id")
            goal_text = None
            doc_id = None
            doc_domain = None
            source_query = None
            query_style = None
            multi_turn_split = None
            source_format = "conversation_turns"
            conversation_id = conversation["conversation_id"]
        else:
            goal = record["goal"]
            document = record["document"]
            source_query_record = record.get("query") or {}
            multiturn_meta = record.get("multi_turn_meta") or {}
            memories = list((record.get("preexisting_memories") or {}).get("memories") or [])
            injected_memory = str(goal["goal_text"])
            if injected_memory not in memories:
                memories = [*memories, injected_memory]
            initial_turns = []
            user_queries = [str(query).strip() for query in record["multi_turn_queries"] if str(query).strip()]
            sample_id = f"{document['doc_id']}__{goal['goal_id']}"
            category = multiturn_meta.get("split")
            source_goal_id = goal.get("goal_id")
            goal_text = goal.get("goal_text")
            doc_id = document.get("doc_id")
            doc_domain = document.get("domain")
            source_query = source_query_record.get("query")
            query_style = multiturn_meta.get("query_style")
            multi_turn_split = multiturn_meta.get("split")
            source_format = "multiturn_queries"
            conversation_id = sample_id

        sample_meta = FollowupSampleMeta(
            conversation_id=conversation_id,
            initial_turns=initial_turns,
            user_queries=user_queries,
            memories=memories,
            injected_memory=injected_memory,
            memory_count=len(memories),
            injected_memory_index=memories.index(injected_memory),
            goal_text=goal_text,
            doc_id=doc_id,
            doc_domain=doc_domain,
            source_query=source_query,
            query_style=query_style,
            multi_turn_split=multi_turn_split,
            source_format=source_format,
            category=category,
            source_goal_id=source_goal_id,
            variant=info.variant,
            split=info.split,
            source_file=info.source_file,
            sample_id=sample_id,
        )

        return Sample(
            id=f"{info.variant}__{sample_id}",
            input="Replay the follow-up conversation from sample metadata.",
            target=injected_memory,
            metadata=sample_meta.model_dump(),
        )

    return record_to_sample


def load_dataset(
    file_path: str | Path,
    filters: Iterable[FilterFn] | None = None,
) -> Dataset:
    """Load a follow-up JSON dataset and optionally apply filters."""

    path = resolve_dataset_path(file_path)
    dataset = json_dataset(
        str(path),
        sample_fields=make_record_to_sample(path),
        name=infer_dataset_info(path).variant,
    )

    for predicate in filters or []:
        dataset = dataset.filter(predicate)

    return dataset
