"""Dataset loading helpers for sleeper memory evaluations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from inspect_ai.dataset import Dataset, Sample, json_dataset
from sleeper_eval.document_text import extract_pdf_pages_from_base64, normalize_document_text

DATASET_ROOT = Path(__file__).resolve().parent.parent / "eval_dataset" / "datasets"
REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_PROVIDER_DOC_FORMATS = frozenset(
    {"text", "html", "code", "email", "tweet", "pdf"}
)
VALID_SPLITS = ("smoke", "dev", "eval_in", "eval_ood")


class SampleMeta(BaseModel):
    """Typed metadata for sleeper eval samples."""

    model_config = ConfigDict(frozen=True)

    doc_text: str
    doc_text_source: str
    doc_domain: str
    doc_id: str
    doc_format: str | None = None
    doc_language_code: str | None = None
    doc_language: str | None = None
    doc_source_name: str | None = None
    doc_source_path: str | None = None
    doc_metadata_filename: str | None = None
    doc_metadata_mime: str | None = None
    doc_metadata_title: str | None = None
    doc_metadata_local_path: str | None = None
    doc_pdf_pages: list[str] | None = None
    goal_text: str
    subcategory_id: str
    category_id: str
    category_name: str | None = None
    subcategory_name: str | None = None
    domain_seed: str
    goal_id: str
    query_text: str
    query_style: str | None = None
    memories: list[str]
    has_memories: bool
    has_contradiction: bool
    contradiction_memory: str | None = None
    variant: str
    split: str
    memory_condition: str
    source_file: str


class DatasetInfo(BaseModel):
    """Static metadata derived from the dataset filename."""

    model_config = ConfigDict(frozen=True)

    variant: str
    split: str
    memory_condition: str
    source_file: str


FilterFn = Callable[[Sample], bool]


def is_supported_provider_doc_format(doc_format: str | None) -> bool:
    """Return whether a document format is in the current provider-accurate eval set."""

    if doc_format is None:
        return False
    return doc_format.casefold() in SUPPORTED_PROVIDER_DOC_FORMATS


def resolve_dataset_path(file_path: str | Path) -> Path:
    """Resolve a dataset path relative to the repo dataset directory."""

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
            "'in_dist/file.json', a repo-relative path, or an absolute path."
        )

    return resolved


def infer_dataset_info(file_path: str | Path) -> DatasetInfo:
    """Infer split and filename-level dataset metadata.

    Historically this loader assumed a narrow set of legacy filenames. Public-release
    datasets and generated paper slices do not always preserve those exact stems, so
    we fall back to path/stem heuristics instead of rejecting an otherwise valid JSON
    dataset solely because of its filename.
    """

    path = resolve_dataset_path(file_path)
    stem = path.stem

    split = next((part for part in path.parts if part in VALID_SPLITS), None)
    if split is None:
        if stem.startswith("merged_dev_"):
            split = "dev"
        elif stem.startswith("merged_eval_in_"):
            split = "eval_in"
        elif stem.startswith("merged_ood_") or "_ood_" in stem or stem.startswith("non_english_"):
            split = "eval_ood"
        elif "smoke" in stem:
            split = "smoke"
        else:
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = None
            if rel is not None and len(rel.parts) >= 2 and rel.parts[0] == "datasets" and rel.parts[1] == "released":
                split = "eval_ood" if stem.startswith("non_english_") else "eval_in"
            else:
                split = "custom"

    if "goal_contradiction" in stem:
        memory_condition = "contradiction"
    elif stem.endswith("with_memories"):
        memory_condition = "with_memories"
    elif stem.endswith("without_memories"):
        memory_condition = "without_memories"
    else:
        memory_condition = "mixed"

    return DatasetInfo(
        variant=stem,
        split=split,
        memory_condition=memory_condition,
        source_file=path.name,
    )


def infer_record_memory_condition(
    memories: list[str],
    contradiction_memory: str | None,
) -> str:
    """Infer memory condition from the sample itself, not just the filename."""
    if contradiction_memory is not None:
        return "contradiction"
    if memories:
        return "with_memories"
    return "without_memories"


def make_record_to_sample(file_path: str | Path) -> Callable[[dict[str, Any]], Sample]:
    """Build a record-to-sample mapper bound to a dataset file."""

    info = infer_dataset_info(file_path)

    def record_to_sample(record: dict[str, Any]) -> Sample:
        document = record["document"]
        goal = record["goal"]
        query = record["query"]
        annotations = document.get("annotations", {}) or {}
        doc_metadata = document.get("metadata", {}) or {}
        preexisting_memories = record.get("preexisting_memories") or {}
        doc_text, doc_text_source = normalize_document_text(document)
        doc_pdf_pages: list[str] | None = None
        if document.get("text") == "PDF_BINARY" and "pdf_base64" in doc_metadata:
            doc_pdf_pages = extract_pdf_pages_from_base64(doc_metadata["pdf_base64"])

        memories = list(preexisting_memories.get("memories") or [])
        contradiction_memory = preexisting_memories.get("goal_contradiction_memory")
        memory_condition = infer_record_memory_condition(
            memories=memories,
            contradiction_memory=contradiction_memory,
        )

        metadata = SampleMeta(
            doc_text=doc_text,
            doc_text_source=doc_text_source,
            doc_domain=document["domain"],
            doc_id=document["doc_id"],
            doc_format=annotations.get("document_format"),
            doc_language_code=annotations.get("language_code"),
            doc_language=annotations.get("language"),
            doc_source_name=document.get("source_name"),
            doc_source_path=document.get("source_path"),
            doc_metadata_filename=doc_metadata.get("filename"),
            doc_metadata_mime=doc_metadata.get("mime") or doc_metadata.get("content_type"),
            doc_metadata_title=doc_metadata.get("title"),
            doc_metadata_local_path=(
                doc_metadata.get("local_pdf_path")
                or doc_metadata.get("path")
                or doc_metadata.get("local_path")
            ),
            doc_pdf_pages=doc_pdf_pages,
            goal_text=goal["goal_text"],
            subcategory_id=goal["subcategory_id"],
            category_id=goal["category_id"],
            category_name=goal.get("category_name"),
            subcategory_name=goal.get("subcategory_name"),
            domain_seed=goal["domain_seed"],
            goal_id=goal["goal_id"],
            query_text=query["query"],
            query_style=query.get("style"),
            memories=memories,
            has_memories=bool(memories),
            has_contradiction=contradiction_memory is not None,
            contradiction_memory=contradiction_memory,
            variant=info.variant,
            split=info.split,
            memory_condition=memory_condition,
            source_file=info.source_file,
        )

        return Sample(
            id=f"{info.variant}__{document['doc_id']}__{goal['goal_id']}",
            input="Compose the full prompt from sample metadata.",
            target=goal["goal_text"],
            metadata=metadata.model_dump(),
        )

    return record_to_sample


def load_dataset(
    file_path: str | Path,
    filters: Iterable[FilterFn] | None = None,
) -> Dataset:
    """Load a JSON dataset and optionally apply filters."""

    path = resolve_dataset_path(file_path)
    dataset = json_dataset(
        str(path),
        sample_fields=make_record_to_sample(path),
        name=infer_dataset_info(path).variant,
    )

    for predicate in filters or []:
        dataset = dataset.filter(predicate)

    return dataset
