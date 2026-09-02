"""Config models and path resolution for eval campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
import yaml

from sleeper_eval.dataset import resolve_dataset_path


def _ensure_unique(labels: list[str], *, kind: str) -> None:
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {kind} labels are not allowed: {duplicates}")


class CampaignModelConfig(BaseModel):
    class CampaignBatchConfig(BaseModel):
        size: int | None = Field(default=None, ge=1)
        max_size: int | None = Field(default=None, ge=1)
        send_delay: float | None = Field(default=None, ge=0)
        tick: float | None = Field(default=None, ge=0)
        max_batches: int | None = Field(default=None, ge=1)
        max_consecutive_check_failures: int | None = Field(default=None, ge=1)

    label: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    prompt_model: str | None = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    reasoning_tokens: int | None = Field(default=None, ge=0)
    batch: bool | int | CampaignBatchConfig | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_batch(self) -> "CampaignModelConfig":
        if isinstance(self.batch, int) and self.batch <= 0:
            raise ValueError("models[].batch integer shorthand must be positive.")
        return self

    @property
    def resolved_prompt_model(self) -> str:
        return self.prompt_model or self.model

    @property
    def resolved_batch(self) -> dict[str, Any] | None:
        if self.batch is None or self.batch is False:
            return None
        if self.batch is True:
            return {}
        if isinstance(self.batch, int):
            return {"size": self.batch}
        return self.batch.model_dump(mode="json", exclude_none=True)

    @property
    def resolved_extra_headers(self) -> dict[str, str] | None:
        if not self.extra_headers:
            return None
        return dict(self.extra_headers)

    @property
    def resolved_extra_body(self) -> dict[str, Any] | None:
        if not self.extra_body:
            return None
        return dict(self.extra_body)


class CampaignDefenseConfig(BaseModel):
    label: str = Field(min_length=1)
    defense: str = ""
    defense_suffix_file: str = ""
    defense_suffix_text: str = ""

    @model_validator(mode="after")
    def validate_suffix_sources(self) -> "CampaignDefenseConfig":
        if self.defense_suffix_file and self.defense_suffix_text:
            raise ValueError("Specify at most one of defense_suffix_file or defense_suffix_text.")
        return self


class CampaignRetryConfig(BaseModel):
    retry_attempts: int = 10
    retry_immediate: bool = True
    retry_wait: int = 30
    retry_connections: float = 1.0
    retry_on_error: int = 2


class CampaignEvalConfig(BaseModel):
    limit: int | tuple[int, int] | None = None
    sample_shuffle: bool = False
    sample_shuffle_seed: int | None = None
    max_tasks: int | None = None
    max_samples: int | None = None
    max_connections: int | None = None
    fail_on_error: bool | float | None = None
    bundle: bool = True
    bundle_overwrite: bool = True
    max_retries: int | None = None
    attempt_timeout: int | None = None


class CampaignAttackConfig(BaseModel):
    label: str = Field(min_length=1)
    attack: str = Field(min_length=1)
    defense_labels: list[str] = Field(default_factory=list)


class CampaignDatasetConfig(BaseModel):
    label: str = Field(min_length=1)
    dataset_file: str = Field(min_length=1)
    evaluation_mode: str = "attack"
    attack: str = ""
    subcategory: str = ""
    category: str = ""
    doc_domain: str = ""
    domain_seed: str = ""

    @model_validator(mode="after")
    def validate_mode(self) -> "CampaignDatasetConfig":
        if self.evaluation_mode not in {"attack", "benign_save"}:
            raise ValueError(
                f"Unsupported evaluation_mode '{self.evaluation_mode}'. "
                "Expected 'attack' or 'benign_save'."
            )
        return self


class CampaignScoringConfig(BaseModel):
    during_eval: str = "full"
    post_eval_scorers: list[str] = Field(default_factory=list)
    post_eval_action: str = "overwrite"

    @model_validator(mode="after")
    def validate_scoring(self) -> "CampaignScoringConfig":
        if self.during_eval not in {"full", "none"}:
            raise ValueError("scoring.during_eval must be 'full' or 'none'.")
        if self.post_eval_action not in {"overwrite", "append"}:
            raise ValueError("scoring.post_eval_action must be 'overwrite' or 'append'.")
        return self


class CampaignConfig(BaseModel):
    name: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    dataset_file: str = ""
    evaluation_mode: str = "attack"
    attack: str = "none"
    datasets: list[CampaignDatasetConfig] = Field(default_factory=list)
    attacks: list[CampaignAttackConfig] = Field(default_factory=list)
    memory_backend: str = "tool"
    mem0_runtime: Literal["local", "sdk", "prompt_only", "transcript_only"] = "local"
    mem0_provider: Literal["openai", "openrouter", "anthropic", "deepseek", "gemini"] = "deepseek"
    mem0_model: str = "deepseek-v4-pro"
    mem0_prompt_variant: Literal["mem0_default", "lenient_assistant"] = "mem0_default"
    mem0_include_document_content: bool = True
    mem0_thinking: Literal["enabled", "disabled"] = "enabled"
    mem0_reasoning_effort: Literal["high", "max"] = "high"
    mem0_qdrant_mode: Literal["local", "server", "managed"] = "local"
    mem0_qdrant_url: str = ""
    mem0_qdrant_api_key_env: str = "QDRANT_API_KEY"
    mem0_qdrant_collection_name: str = ""
    mem0_qdrant_managed_image: str = "qdrant/qdrant:latest"
    mem0_qdrant_managed_storage_dir: str = ""
    replay_max_concurrency: int = Field(default=1, ge=1)
    mention_memory_system: bool = True
    grader_model: str = "openai/gpt-5.4-mini"
    retry: CampaignRetryConfig = Field(default_factory=CampaignRetryConfig)
    eval: CampaignEvalConfig = Field(default_factory=CampaignEvalConfig)
    scoring: CampaignScoringConfig = Field(default_factory=CampaignScoringConfig)
    models: list[CampaignModelConfig]
    defenses: list[CampaignDefenseConfig]

    @model_validator(mode="after")
    def validate_campaign(self) -> "CampaignConfig":
        if self.datasets:
            if self.dataset_file:
                raise ValueError("Specify either dataset_file or datasets, not both.")
        else:
            if not self.dataset_file:
                raise ValueError("Specify dataset_file or provide datasets.")
            self.datasets = [
                CampaignDatasetConfig(
                    label=Path(self.dataset_file).stem,
                    dataset_file=self.dataset_file,
                    evaluation_mode=self.evaluation_mode,
                    attack=self.attack,
                )
            ]

        _ensure_unique([dataset.label for dataset in self.datasets], kind="dataset")
        _ensure_unique([attack.label for attack in self.attacks], kind="attack")
        _ensure_unique([model.label for model in self.models], kind="model")
        _ensure_unique([defense.label for defense in self.defenses], kind="defense")
        return self


def load_campaign_config(config_file: str | Path) -> CampaignConfig:
    payload = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Campaign config must deserialize to a mapping.")
    return CampaignConfig.model_validate(payload)


def resolve_dataset_file(dataset_file: str, *, config_dir: Path) -> Path:
    dataset_path = Path(dataset_file)
    if dataset_path.is_absolute():
        return resolve_dataset_path(dataset_path)

    config_relative = (config_dir / dataset_path).resolve()
    if config_relative.exists():
        return config_relative

    return resolve_dataset_path(dataset_file)


def resolve_defense_suffix(defense: CampaignDefenseConfig, *, config_dir: Path) -> str:
    if defense.defense_suffix_text:
        return defense.defense_suffix_text.strip()
    if defense.defense_suffix_file:
        suffix_path = Path(defense.defense_suffix_file)
        if not suffix_path.is_absolute():
            suffix_path = (config_dir / suffix_path).resolve()
        return suffix_path.read_text(encoding="utf-8").strip()
    return ""
