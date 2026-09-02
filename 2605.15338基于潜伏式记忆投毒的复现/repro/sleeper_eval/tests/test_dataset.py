from __future__ import annotations

from pathlib import Path
import json

import pytest

from sleeper_eval.dataset import (
    SampleMeta,
    infer_dataset_info,
    infer_record_memory_condition,
    is_supported_provider_doc_format,
    load_dataset,
    resolve_dataset_path,
)

DATASET_FILES = [
    "datasets/released/generated/behaviour_true_optimized_with_memories.json",
    "datasets/released/generated/behaviour_true_optimized_without_memories.json",
    "datasets/released/generated/agent_true_optimized_with_memories.json",
    "datasets/released/generated/agent_true_optimized_without_memories.json",
]


@pytest.mark.parametrize("dataset_file", DATASET_FILES)
def test_all_dataset_files_load(dataset_file: str) -> None:
    dataset = load_dataset(dataset_file)
    assert len(dataset) > 0

    ids = [sample.id for sample in dataset]
    assert len(ids) == len(set(ids))

    meta = SampleMeta.model_validate(dataset[0].metadata)
    info = infer_dataset_info(dataset_file)
    assert meta.variant == info.variant
    assert meta.split == info.split
    assert meta.memory_condition == info.memory_condition
    assert meta.source_file == Path(dataset_file).name


def test_with_memories_variant_sets_memory_flags() -> None:
    dataset = load_dataset("datasets/released/generated/behaviour_true_optimized_with_memories.json")
    sample = dataset[0]
    meta = SampleMeta.model_validate(sample.metadata)

    assert meta.has_memories is True
    assert meta.memories
    assert meta.has_contradiction is False


def test_without_memories_variant_sets_empty_memories() -> None:
    dataset = load_dataset("datasets/released/generated/behaviour_true_optimized_without_memories.json")
    sample = dataset[0]
    meta = SampleMeta.model_validate(sample.metadata)

    assert meta.has_memories is False
    assert meta.memories == []
    assert meta.has_contradiction is False



def test_dataset_filtering_by_subcategory(dev_with_memories_dataset) -> None:
    sample = dev_with_memories_dataset[0]
    selected = sample.metadata["subcategory_id"]

    filtered = load_dataset(
        "datasets/released/generated/behaviour_true_optimized_with_memories.json",
        filters=[lambda current: current.metadata["subcategory_id"] == selected],
    )

    assert len(filtered) > 0
    assert all(item.metadata["subcategory_id"] == selected for item in filtered)


def test_resolve_dataset_path_handles_relative_and_absolute() -> None:
    relative = resolve_dataset_path(
        "datasets/released/generated/behaviour_true_optimized_with_memories.json"
    )
    absolute = resolve_dataset_path(relative)

    assert relative.is_absolute()
    assert absolute == relative


def test_resolve_dataset_path_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_dataset_path("datasets/released/generated/does_not_exist.json")


def test_infer_record_memory_condition_uses_sample_content() -> None:
    assert infer_record_memory_condition(["known fact"], None) == "with_memories"
    assert infer_record_memory_condition([], None) == "without_memories"
    assert infer_record_memory_condition([], "contradiction") == "contradiction"


def test_infer_dataset_info_defaults_unknown_memory_filenames_to_mixed(tmp_path: Path) -> None:
    dataset_path = tmp_path / "merged_dev_custom_smoke.json"
    dataset_path.write_text("[]", encoding="utf-8")

    info = infer_dataset_info(dataset_path)

    assert info.split == "dev"
    assert info.memory_condition == "mixed"


def test_infer_dataset_info_supports_ood_filenames(tmp_path: Path) -> None:
    dataset_path = tmp_path / "merged_ood_custom.json"
    dataset_path.write_text("[]", encoding="utf-8")

    info = infer_dataset_info(dataset_path)

    assert info.split == "eval_ood"
    assert info.memory_condition == "mixed"


def test_infer_dataset_info_supports_released_generated_eval_in_filenames() -> None:
    info = infer_dataset_info(
        "datasets/released/generated/behaviour_true_optimized_with_memories.json"
    )

    assert info.split == "eval_in"
    assert info.memory_condition == "with_memories"


def test_infer_dataset_info_supports_released_generated_non_english_filenames() -> None:
    info = infer_dataset_info(
        "datasets/released/generated/non_english_true_optimized_without_memories.json"
    )

    assert info.split == "eval_ood"
    assert info.memory_condition == "without_memories"


def test_synthetic_ood_dataset_populates_generic_document_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "merged_ood_with_memories.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "document": {
                        "domain": "synthetic_domain",
                        "doc_id": "doc-1",
                        "text": "Document body",
                        "metadata": {
                            "filename": "notes.txt",
                            "mime": "text/plain",
                            "title": "Synthetic Title",
                            "local_path": "synthetic/notes.txt",
                        },
                        "annotations": {
                            "document_format": "markdown",
                            "language_code": "en",
                            "language": "English",
                        },
                        "source_name": "synthetic_source",
                        "source_path": "synthetic/path",
                    },
                    "goal": {
                        "goal_text": "Remember a fact",
                        "subcategory_id": "1a",
                        "category_id": "1",
                        "category_name": "Category",
                        "subcategory_name": "Subcategory",
                        "domain_seed": "Synthetic",
                        "goal_id": "goal-1",
                    },
                    "query": {
                        "query": "Summarize the document",
                        "style": "formal",
                    },
                    "preexisting_memories": {
                        "memories": ["Existing memory"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_path)
    meta = SampleMeta.model_validate(dataset[0].metadata)

    assert meta.split == "eval_ood"
    assert meta.memory_condition == "with_memories"
    assert meta.doc_format == "markdown"
    assert meta.doc_language_code == "en"
    assert meta.doc_language == "English"
    assert meta.doc_source_name == "synthetic_source"
    assert meta.doc_source_path == "synthetic/path"
    assert meta.doc_metadata_filename == "notes.txt"
    assert meta.doc_metadata_mime == "text/plain"
    assert meta.doc_metadata_title == "Synthetic Title"
    assert meta.doc_metadata_local_path == "synthetic/notes.txt"


def test_pdf_placeholder_samples_are_normalized() -> None:
    dataset = load_dataset("datasets/smoke/merged_eval_in_provider_docrep_smoke.json")
    sample = next(item for item in dataset if item.metadata["doc_domain"] == "science_pdfs")
    meta = SampleMeta.model_validate(sample.metadata)

    assert meta.doc_text_source == "pdf_base64"
    assert meta.doc_text != "PDF_BINARY"
    assert len(meta.doc_text) > 100
    assert meta.doc_pdf_pages is not None
    assert len(meta.doc_pdf_pages) >= 1
    assert any(page for page in meta.doc_pdf_pages)


def test_html_samples_keep_raw_html() -> None:
    dataset = load_dataset("datasets/smoke/merged_eval_in_provider_docrep_smoke.json")
    sample = next(item for item in dataset if item.metadata["doc_domain"] == "web_html_wikipedia")
    meta = SampleMeta.model_validate(sample.metadata)

    assert meta.doc_text_source == "document.text"
    assert "<html" in meta.doc_text.lower()
    assert meta.doc_source_path is not None


def test_supported_provider_doc_format_helper() -> None:
    assert is_supported_provider_doc_format("text") is True
    assert is_supported_provider_doc_format("pdf") is True
    assert is_supported_provider_doc_format("csv") is False
    assert is_supported_provider_doc_format("excel") is False
    assert is_supported_provider_doc_format("xml") is False


def test_provider_docrep_smoke_dataset_contains_only_supported_formats() -> None:
    dataset = load_dataset("datasets/smoke/merged_eval_in_provider_docrep_smoke.json")

    assert len(dataset) == 12
    formats = {sample.metadata["doc_format"] for sample in dataset}
    assert formats == {"text", "html", "code", "email", "tweet", "pdf"}
