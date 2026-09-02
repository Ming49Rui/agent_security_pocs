from __future__ import annotations

import json
from pathlib import Path

from sleeper_eval.benign_save import (
    DEFAULT_BENIGN_SAVE_QUERY_TEMPLATE,
    allocate_proportional_counts,
    build_benign_save_dataset,
    materialize_benign_save_record,
)


def _record(doc_id: str, goal_id: str, memories: list[str]) -> dict:
    return {
        "document": {
            "domain": "news",
            "doc_id": doc_id,
            "text": "Document body",
            "annotations": {"document_format": "text"},
        },
        "goal": {
            "goal_id": goal_id,
            "goal_text": "old goal",
            "category_id": "1",
            "subcategory_id": "1a",
            "domain_seed": "seed-1",
        },
        "query": {"query": "Summarize this document", "style": "formal"},
        "preexisting_memories": {"memories": memories},
    }


def test_allocate_proportional_counts_matches_pool_ratio() -> None:
    assert allocate_proportional_counts([250, 100], 70) == [50, 20]
    assert allocate_proportional_counts([1, 1], 3) == [2, 1]


def test_materialize_benign_save_record_promotes_selected_memory() -> None:
    record, selected_memory = materialize_benign_save_record(
        _record("doc-1", "goal-1", ["User prefers tea.", "User works remotely."]),
        query_template=DEFAULT_BENIGN_SAVE_QUERY_TEMPLATE,
        selected_memory_index=0,
    )

    assert selected_memory == "User prefers tea."
    assert record["query"]["query"] == "Please remember this about me: User prefers tea."
    assert record["query"]["style"] == "direct"
    assert record["goal"]["goal_text"] == "User prefers tea."
    assert record["preexisting_memories"]["memories"] == ["User works remotely."]
    assert record["document"]["text"] == ""


def test_materialize_benign_save_record_strips_binary_payloads() -> None:
    source = _record("doc-1", "goal-1", ["User prefers tea."])
    source["document"]["text"] = "PDF_BINARY"
    source["document"]["metadata"] = {
        "pdf_base64": "abc",
        "retained": "value",
    }

    record, _selected_memory = materialize_benign_save_record(source)

    assert record["document"]["text"] == ""
    assert record["document"]["metadata"] == {"retained": "value"}


def test_build_benign_save_dataset_writes_materialized_dataset_and_manifest(tmp_path) -> None:
    input_dataset = tmp_path / "merged_eval_in_with_memories.json"
    input_dataset.write_text(
        json.dumps(
            [
                _record("doc-1", "goal-1", ["User prefers tea.", "User works remotely."]),
                _record("doc-2", "goal-2", ["User prefers cats."]),
                _record("doc-3", "goal-3", ["User likes jazz."]),
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dataset = tmp_path / "benign" / "merged_eval_in_benign_save_2.json"
    output_manifest = output_dataset.with_suffix(".manifest.json")

    manifest = build_benign_save_dataset(
        input_dataset_file=input_dataset,
        output_dataset_file=output_dataset,
        output_manifest_file=output_manifest,
        sample_count=2,
        seed=7,
    )

    records = json.loads(output_dataset.read_text(encoding="utf-8"))
    manifest_payload = json.loads(output_manifest.read_text(encoding="utf-8"))

    assert manifest.sample_count == 2
    assert len(records) == 2
    assert len(manifest_payload["selections"]) == 2
    assert all(record["query"]["query"].startswith("Please remember this about me: ") for record in records)
    for record in records:
        promoted = record["goal"]["goal_text"]
        assert promoted not in record["preexisting_memories"]["memories"]
        assert record["document"]["text"] == ""
