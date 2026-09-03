"""Run raw-GEPA defense suffix optimization over the real sleeper-eval stack."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from gepa.api import optimize

from .adapter import SleeperDefenseAdapter
from .callbacks import ConsoleProgressCallback, ExperimentLoggerCallback
from .config import (
    DEFAULT_OPTIMIZATION_ATTACK,
    FULL_PRESET,
    FULL_V2_TRANSFER_PRESET,
    OPTIMIZATION_TARGETS,
    SEED_SUFFIX,
    SMOKE_PRESET,
    TINY_PRESET,
    V2_TRANSFER_OPTIMIZATION_ATTACK,
    default_reflection_model_for_preset,
)
from .dataset_builder import (
    DEV_WITH_MEMORIES,
    DEV_WITHOUT_MEMORIES,
    build_optimization_bundle,
    optimization_bundle_from_dict,
    optimization_bundle_to_dict,
    write_manifest,
)
from .mlflow_tracing import MlflowTracing
from .reflection import REFLECTION_PROMPT_TEMPLATE, build_reflection_lm
from .runner import InspectEvalRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("tiny", "smoke", "full", "full_v2_transfer"), default="smoke")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--optimization-bundle", default="")
    parser.add_argument("--build-bundle", action="store_true")
    parser.add_argument(
        "--seed-suffix-file",
        default="",
        help="Optional path to a text file whose contents should replace the default seed suffix.",
    )
    parser.add_argument("--reflection-model", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-metric-calls", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    parser.add_argument("--mlflow-trace", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="")
    parser.add_argument("--mlflow-experiment-name", default="")
    parser.add_argument("--mlflow-source-repo", default="")
    return parser.parse_args(argv)


def _preset_from_name(name: str):
    if name == "tiny":
        return TINY_PRESET
    if name == "smoke":
        return SMOKE_PRESET
    if name == "full_v2_transfer":
        return FULL_V2_TRANSFER_PRESET
    return FULL_PRESET


def _optimization_attack_for_preset(preset_name: str) -> str:
    if preset_name == "full_v2_transfer":
        return V2_TRANSFER_OPTIMIZATION_ATTACK
    return DEFAULT_OPTIMIZATION_ATTACK


def _seed_suffix_from_args(args: argparse.Namespace) -> str:
    if not args.seed_suffix_file:
        return SEED_SUFFIX
    return Path(args.seed_suffix_file).read_text(encoding="utf-8").strip()


def _seed_suffix_source(args: argparse.Namespace) -> str:
    return args.seed_suffix_file or "(default SEED_SUFFIX)"


def _default_output_dir(preset_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(__file__).resolve().parent / "runs" / f"{preset_name}-{ts}"


def _default_mlflow_experiment_name(preset_name: str) -> str:
    return f"gepa-defense-optim-v2-{preset_name}"


def _bundle_path_from_args(args: argparse.Namespace, root_dir: Path) -> Path:
    if args.optimization_bundle:
        return Path(args.optimization_bundle)
    return root_dir / "artifacts" / "optimization_bundle.json"


def _write_run_config(
    root_dir: Path,
    *,
    preset,
    seed_suffix: str,
    seed_suffix_file: str,
    optimization_attack: str,
    seed: int,
    reflection_model: str,
    max_metric_calls: int,
) -> None:
    write_manifest(
        root_dir / "artifacts" / "config.json",
        {
            "preset": asdict(preset),
            "seed": seed,
            "seed_suffix_length": len(seed_suffix),
            "seed_suffix_file": seed_suffix_file,
            "optimization_attack": optimization_attack,
            "reflection_model": reflection_model,
            "targets": [asdict(target) for target in OPTIMIZATION_TARGETS],
            "reflection_prompt_template": REFLECTION_PROMPT_TEMPLATE,
            "max_metric_calls": max_metric_calls,
            "mlflow_trace": False,
        },
    )


def _required_env_var_for_model_slug(model_slug: str) -> str | None:
    parts = [part.strip() for part in model_slug.split("/") if part.strip()]
    if not parts:
        return None
    provider = parts[0]
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "google":
        return "GOOGLE_API_KEY"
    return None


def _required_env_status(*, reflection_model: str) -> dict[str, dict[str, object]]:
    required: dict[str, list[str]] = {}
    for target in OPTIMIZATION_TARGETS:
        env_var = _required_env_var_for_model_slug(target.model_slug)
        if env_var is not None:
            required.setdefault(env_var, []).append(target.model_slug)
    reflection_env_var = _required_env_var_for_model_slug(reflection_model)
    if reflection_env_var is not None:
        required.setdefault(reflection_env_var, []).append(reflection_model)

    return {
        env_var: {
            "model_slugs": model_slugs,
            "is_set": bool(os.environ.get(env_var)),
        }
        for env_var, model_slugs in sorted(required.items())
    }


def _validate_runtime_env(*, reflection_model: str) -> dict[str, list[str]]:
    env_status = _required_env_status(reflection_model=reflection_model)

    print("required_env_vars:")
    for env_var, payload in env_status.items():
        model_slugs = payload["model_slugs"]
        status = "set" if payload["is_set"] else "missing"
        joined_models = ", ".join(model_slugs)
        print(f"  - {env_var}: {status} ({joined_models})")

    missing = {
        env_var: payload["model_slugs"]
        for env_var, payload in env_status.items()
        if not payload["is_set"]
    }
    if missing:
        details = "; ".join(f"{env_var} for {', '.join(model_slugs)}" for env_var, model_slugs in sorted(missing.items()))
        raise EnvironmentError(
            "Missing required API keys for this optimization run: "
            f"{details}. Either set those env vars or change the target/reflection models."
        )
    return {env_var: list(payload["model_slugs"]) for env_var, payload in env_status.items()}


def _render_plan(
    *,
    mode: str,
    preset_name: str,
    root_dir: Path,
    bundle_path: Path,
    reflection_model: str,
    seed_suffix_source: str,
    seed_suffix_length: int,
    optimization_attack: str,
    max_metric_calls: int,
    env_status: dict[str, dict[str, object]],
    train_source_total: int,
    val_source_total: int,
    train_examples: int,
    val_examples: int,
    train_dataset_file: str,
    val_dataset_file: str,
    bundle_optimization_attack: str | None = None,
) -> str:
    lines = [
        "== Raw GEPA Optimization Plan ==",
        f"mode={mode}",
        f"preset={preset_name}",
        f"run_dir={root_dir}",
        f"bundle={bundle_path}",
        "",
        "seed:",
        f"  - source={seed_suffix_source}",
        f"  - suffix_length={seed_suffix_length}",
        "",
        "optimization:",
        f"  - attack={optimization_attack}",
        f"  - reflection_model={reflection_model}",
        f"  - max_metric_calls={max_metric_calls}",
        "",
        "targets:",
        *(f"  - provider={t.provider} model={t.model_slug} prompt_model={t.prompt_model}" for t in OPTIMIZATION_TARGETS),
        "",
    ]
    if mode == "build_bundle":
        lines.extend(
            [
                "source_datasets:",
                f"  - with_memories={DEV_WITH_MEMORIES}",
                f"  - without_memories={DEV_WITHOUT_MEMORIES}",
                "",
            ]
        )
    if bundle_optimization_attack is not None:
        lines.extend(["bundle:", f"  - optimization_attack={bundle_optimization_attack}", ""])
    lines.extend(
        [
            "splits:",
            f"  - train_source_total={train_source_total}",
            f"  - val_source_total={val_source_total}",
            f"  - train_examples={train_examples}",
            f"  - val_examples={val_examples}",
            f"  - train_dataset={train_dataset_file}",
            f"  - val_dataset={val_dataset_file}",
            "",
            "required_env_vars:",
            *(
                f"  - {env_var}: {'set' if payload['is_set'] else 'missing'} "
                f"({', '.join(payload['model_slugs'])})"
                for env_var, payload in env_status.items()
            ),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preset = _preset_from_name(args.preset)
    seed_suffix = _seed_suffix_from_args(args)
    seed_suffix_source = _seed_suffix_source(args)
    optimization_attack = _optimization_attack_for_preset(preset.name)
    reflection_model = args.reflection_model or default_reflection_model_for_preset(preset.name)
    max_metric_calls = args.max_metric_calls or preset.max_metric_calls
    if not args.build_bundle and not args.output_dir and not args.optimization_bundle:
        raise ValueError(
            "Optimization now expects a prebuilt bundle. Pass --build-bundle first, "
            "or provide --output-dir pointing at an existing run dir, "
            "or pass --optimization-bundle explicitly."
    )
    root_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.preset)
    bundle_path = _bundle_path_from_args(args, root_dir)
    env_status = _required_env_status(reflection_model=reflection_model)

    if args.build_bundle:
        train_dataset_file = str(root_dir / "artifacts" / "datasets" / f"merged_dev_gepa_v2_{preset.name}_train.json")
        val_dataset_file = str(root_dir / "artifacts" / "datasets" / f"merged_dev_gepa_v2_{preset.name}_val.json")
        print(
            _render_plan(
                mode="build_bundle",
                preset_name=preset.name,
                root_dir=root_dir,
                bundle_path=bundle_path,
                reflection_model=reflection_model,
                seed_suffix_source=seed_suffix_source,
                seed_suffix_length=len(seed_suffix),
                optimization_attack=optimization_attack,
                max_metric_calls=max_metric_calls,
                env_status=env_status,
                train_source_total=preset.train.total,
                val_source_total=preset.val.total,
                train_examples=preset.train.total * len(OPTIMIZATION_TARGETS),
                val_examples=preset.val.total * len(OPTIMIZATION_TARGETS),
                train_dataset_file=train_dataset_file,
                val_dataset_file=val_dataset_file,
            )
        )
        if args.dry_run:
            return
        (root_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = build_optimization_bundle(
            root_dir,
            preset,
            seed=args.seed,
            optimization_attack=optimization_attack,
        )
        write_manifest(bundle_path, optimization_bundle_to_dict(bundle))
        _write_run_config(
            root_dir,
            preset=preset,
            seed_suffix=seed_suffix,
            seed_suffix_file=args.seed_suffix_file,
            optimization_attack=optimization_attack,
            seed=args.seed,
            reflection_model=reflection_model,
            max_metric_calls=max_metric_calls,
        )
        print("== Raw GEPA Defense Optimization Bundle ==")
        print(f"preset={preset.name}")
        print(f"run_dir={root_dir}")
        print(f"bundle={bundle_path}")
        print(f"optimization_attack={optimization_attack}")
        print(f"train_examples={len(bundle.train.examples)} val_examples={len(bundle.val.examples)}")
        print("Bundle build complete. Re-run without --build-bundle to start optimization.")
        return

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Optimization bundle not found at {bundle_path}. "
            "Build it first with --build-bundle or pass --optimization-bundle."
        )

    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle = optimization_bundle_from_dict(bundle_payload)
    if bundle.optimization_attack != optimization_attack:
        raise ValueError(
            f"Optimization bundle attack mismatch: bundle was built for "
            f"{bundle.optimization_attack!r} but preset {preset.name!r} expects "
            f"{optimization_attack!r}. Rebuild the bundle with --build-bundle."
        )
    if not args.output_dir:
        root_dir = bundle_path.parent.parent
    print(
        _render_plan(
            mode="optimize",
            preset_name=preset.name,
            root_dir=root_dir,
            bundle_path=bundle_path,
            reflection_model=reflection_model,
            seed_suffix_source=seed_suffix_source,
            seed_suffix_length=len(seed_suffix),
            optimization_attack=optimization_attack,
            max_metric_calls=max_metric_calls,
            env_status=env_status,
            train_source_total=len(bundle.train.source_sample_keys),
            val_source_total=len(bundle.val.source_sample_keys),
            train_examples=len(bundle.train.examples),
            val_examples=len(bundle.val.examples),
            train_dataset_file=bundle.train.dataset_file,
            val_dataset_file=bundle.val.dataset_file,
            bundle_optimization_attack=bundle.optimization_attack,
        )
    )
    if args.dry_run:
        return
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Refusing to start optimization without --yes in non-interactive mode. "
                "Use --dry-run to inspect the plan first."
            )
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return
    (root_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    _write_run_config(
        root_dir,
        preset=preset,
        seed_suffix=seed_suffix,
        seed_suffix_file=args.seed_suffix_file,
        optimization_attack=optimization_attack,
        seed=args.seed,
        reflection_model=reflection_model,
        max_metric_calls=max_metric_calls,
    )

    tracer = MlflowTracing(
        enabled=args.mlflow_trace,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name or _default_mlflow_experiment_name(preset.name),
        source_repo=args.mlflow_source_repo,
    )
    tracer.configure_defaults(root_dir)
    write_manifest(
        root_dir / "artifacts" / "config.json",
        {
            **json.loads((root_dir / "artifacts" / "config.json").read_text(encoding="utf-8")),
            "mlflow_trace": args.mlflow_trace,
            "mlflow_tracking_uri": tracer.tracking_uri,
            "mlflow_experiment_name": tracer.experiment_name,
            "mlflow_source_repo": args.mlflow_source_repo,
        },
    )

    runner = InspectEvalRunner(root_dir=root_dir, tracer=tracer)
    adapter = SleeperDefenseAdapter(runner=runner, tracer=tracer)
    reflection_lm = build_reflection_lm(reflection_model, tracer=tracer)

    print("== Raw GEPA Defense Optimization ==")
    print(f"preset={preset.name}")
    print(f"run_dir={root_dir}")
    print(f"reflection_model={reflection_model}")
    print(f"optimization_attack={optimization_attack}")
    print(f"seed_suffix_length={len(seed_suffix)}")
    print(f"train_examples={len(bundle.train.examples)} val_examples={len(bundle.val.examples)}")
    print(f"max_metric_calls={max_metric_calls}")
    _validate_runtime_env(reflection_model=reflection_model)
    if args.mlflow_trace:
        print(f"mlflow_tracking_uri={tracer.tracking_uri}")
        print(f"mlflow_experiment_name={tracer.experiment_name}")

    with tracer.optimization_session(
        root_dir=root_dir,
        preset_name=preset.name,
        seed=args.seed,
        reflection_model=reflection_model,
        max_metric_calls=max_metric_calls,
        train_examples=len(bundle.train.examples),
        val_examples=len(bundle.val.examples),
    ):
        result = optimize(
            seed_candidate={"defense_suffix": seed_suffix},
            trainset=bundle.train.examples,
            valset=bundle.val.examples,
            adapter=adapter,
            reflection_lm=reflection_lm,
            reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=preset.reflection_minibatch_size,
            use_merge=preset.use_merge,
            max_merge_invocations=preset.max_merge_invocations,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            cache_evaluation=True,
            run_dir=str(root_dir / "gepa"),
            callbacks=[
                ConsoleProgressCallback(),
                ExperimentLoggerCallback(root_dir=root_dir),
            ],
            seed=args.seed,
        )

    best_candidate = result.best_candidate
    assert isinstance(best_candidate, dict)
    best_suffix = best_candidate["defense_suffix"]
    (root_dir / "artifacts" / "best_suffix.txt").write_text(best_suffix, encoding="utf-8")
    (root_dir / "artifacts" / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = (
        "# Raw GEPA Defense Optimization\n\n"
        f"- Preset: `{preset.name}`\n"
        f"- Best candidate index: `{result.best_idx}`\n"
        f"- Best aggregate val score: `{result.val_aggregate_scores[result.best_idx]:.3f}`\n"
        f"- Best suffix length: `{len(best_suffix)}` chars\n"
        f"- Candidates explored: `{len(result.candidates)}`\n"
        f"- Total metric calls: `{result.total_metric_calls}`\n"
    )
    (root_dir / "artifacts" / "summary.md").write_text(summary, encoding="utf-8")
    tracer.set_root_outputs(
        {
            "best_candidate_idx": result.best_idx,
            "best_val_score": result.val_aggregate_scores[result.best_idx],
            "best_suffix_length": len(best_suffix),
            "candidate_count": len(result.candidates),
            "total_metric_calls": result.total_metric_calls,
        }
    )
    mlflow_summary = tracer.export_traces(root_dir)
    if mlflow_summary:
        print(
            f"mlflow_traces={mlflow_summary['trace_count']} "
            f"run_id={mlflow_summary['run_id']} experiment_id={mlflow_summary['experiment_id']}"
        )

    print(f"best_candidate_idx={result.best_idx}")
    print(f"best_val_score={result.val_aggregate_scores[result.best_idx]:.3f}")
    print(f"best_suffix_length={len(best_suffix)}")
    print(f"artifacts={root_dir / 'artifacts'}")


if __name__ == "__main__":
    main()
