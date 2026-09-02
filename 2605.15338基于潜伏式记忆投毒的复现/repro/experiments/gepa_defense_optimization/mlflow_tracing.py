"""Optional local MLflow tracing for raw-GEPA experiment runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator


class _NoOpSpan:
    trace_id = ""

    def set_inputs(self, _: Any) -> None:
        return

    def set_outputs(self, _: Any) -> None:
        return

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value
        return

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        del attributes
        return


@dataclass
class MlflowTracing:
    enabled: bool
    tracking_uri: str = ""
    experiment_name: str = "gepa-defense-optim-v2"
    source_repo: str = ""

    _mlflow: Any | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    _root_span: Any | None = None
    _root_dir: Path | None = None
    _reflection_call_index: int = 0
    _reflective_dataset_index: int = 0

    @classmethod
    def disabled(cls) -> "MlflowTracing":
        return cls(enabled=False)

    def configure_defaults(self, root_dir: Path) -> None:
        if not self.enabled:
            return
        if not self.tracking_uri:
            experiment_dir = Path(__file__).resolve().parent
            db_path = (experiment_dir / "mlflow" / "mlflow.db").resolve()
            self.tracking_uri = f"sqlite:///{db_path}"

    def _load_mlflow(self) -> Any:
        if not self.enabled:
            return None
        if self._mlflow is not None:
            return self._mlflow
        if self.source_repo:
            source_repo = str(Path(self.source_repo).resolve())
            if source_repo not in sys.path:
                sys.path.insert(0, source_repo)
        try:
            self._mlflow = importlib.import_module("mlflow")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MLflow tracing was requested but `mlflow` is not importable in this environment. "
                "Install it first, for example with `uv add mlflow`."
            ) from exc
        return self._mlflow

    @contextmanager
    def optimization_session(
        self,
        *,
        root_dir: Path,
        preset_name: str,
        seed: int,
        reflection_model: str,
        max_metric_calls: int,
        train_examples: int,
        val_examples: int,
    ) -> Iterator["MlflowTracing"]:
        if not self.enabled:
            yield self
            return

        mlflow = self._load_mlflow()
        self.configure_defaults(root_dir)
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        run_name = f"{preset_name}-{root_dir.name}"
        with mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id
            self._root_dir = root_dir
            self._reflection_call_index = 0
            self._reflective_dataset_index = 0
            mlflow.log_params(
                {
                    "preset": preset_name,
                    "seed": seed,
                    "reflection_model": reflection_model,
                    "max_metric_calls": max_metric_calls,
                    "train_examples": train_examples,
                    "val_examples": val_examples,
                }
            )
            with mlflow.start_span(name="gepa_optimize", span_type="WORKFLOW") as span:
                self._root_span = span
                span.set_inputs(
                    {
                        "preset": preset_name,
                        "seed": seed,
                        "reflection_model": reflection_model,
                        "max_metric_calls": max_metric_calls,
                        "train_examples": train_examples,
                        "val_examples": val_examples,
                    }
                )
                mlflow.update_current_trace(
                    tags={"component": "gepa-defense-optim-v2"},
                    metadata={
                        "run_dir": str(root_dir),
                        "preset": preset_name,
                    },
                    request_preview=f"preset={preset_name} train={train_examples} val={val_examples}",
                )
                yield self
                self._root_span = None
                self._root_dir = None

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        span_type: str = "CHAIN",
        inputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled:
            yield _NoOpSpan()
            return
        mlflow = self._load_mlflow()
        with mlflow.start_span(name=name, span_type=span_type) as span:
            if inputs is not None:
                span.set_inputs(inputs)
            if attributes:
                span.set_attributes(attributes)
            yield span

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        mlflow = self._load_mlflow()
        mlflow.log_metrics(metrics)

    def log_tags(self, tags: dict[str, str]) -> None:
        if not self.enabled:
            return
        mlflow = self._load_mlflow()
        mlflow.set_tags(tags)

    def export_traces(self, root_dir: Path) -> dict[str, Any] | None:
        if not self.enabled or not self.experiment_id:
            return None
        mlflow = self._load_mlflow()
        flush_fn = getattr(mlflow, "flush_trace_async_logging", None)
        if callable(flush_fn):
            flush_fn()
        traces = mlflow.search_traces(
            locations=[self.experiment_id],
            run_id=self.run_id,
            return_type="list",
        )
        payload = {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "trace_count": len(traces),
            "traces": [trace.to_dict() for trace in traces],
        }
        artifacts_dir = root_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "mlflow_trace_export.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary = {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "trace_count": len(traces),
            "trace_ids": [trace.info.trace_id for trace in traces],
        }
        (artifacts_dir / "mlflow_trace_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def set_root_outputs(self, outputs: dict[str, Any]) -> None:
        if not self.enabled or self._root_span is None:
            return
        self._root_span.set_outputs(outputs)

    def log_reflection_io(self, *, model_slug: str, prompt_full: str, response_full: str) -> dict[str, str] | None:
        if not self.enabled or self._root_dir is None:
            return None
        self._reflection_call_index += 1
        safe_model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_slug).strip("-") or "model"
        reflection_dir = self._root_dir / "artifacts" / "reflection_lm_calls"
        reflection_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self._reflection_call_index:03d}-{safe_model_slug}"
        prompt_path = reflection_dir / f"{prefix}-prompt.txt"
        response_path = reflection_dir / f"{prefix}-response.txt"
        prompt_path.write_text(prompt_full, encoding="utf-8")
        response_path.write_text(response_full, encoding="utf-8")
        return {
            "prompt_file": str(prompt_path),
            "response_file": str(response_path),
        }

    def log_reflective_dataset(
        self,
        *,
        component_name: str,
        candidate: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        if not self.enabled or self._root_dir is None:
            return None
        self._reflective_dataset_index += 1
        reflective_dir = self._root_dir / "artifacts" / "reflective_datasets"
        reflective_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self._reflective_dataset_index:03d}-{component_name}"
        dataset_path = reflective_dir / f"{prefix}.json"
        payload = {
            "component_name": component_name,
            "candidate": candidate,
            "item_count": len(items),
            "items": items,
        }
        dataset_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"reflective_dataset_file": str(dataset_path)}
