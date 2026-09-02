"""Held-out comparison runner for baseline vs optimized defense suffix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

from .adapter import base_score_for_result
from .config import (
    DEFAULT_HELDOUT_ATTACK,
)
from .dataset_builder import EvalExampleSpec, build_heldout_bundle, write_manifest
from .runner import InspectEvalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized-suffix-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimization-bundle", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-benign-save", action="store_true")
    parser.add_argument("--attack-sample-count", type=int, default=100)
    parser.add_argument("--benign-save-sample-count", type=int, default=20)
    parser.add_argument("--benign-save-dataset-file", default="")
    return parser.parse_args()


def _excluded_source_keys(args: argparse.Namespace) -> set[str]:
    bundle_path = Path(args.optimization_bundle) if args.optimization_bundle else (
        Path(args.optimized_suffix_file).resolve().parent / "optimization_bundle.json"
    )
    if not bundle_path.exists():
        return set()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    return set(payload["train"]["source_sample_keys"]) | set(payload["val"]["source_sample_keys"])


def _suite_specs(bundle, *, suite_name: str, benign_save_dataset_file: str = "") -> list[EvalExampleSpec]:
    if suite_name == "main_asr":
        return [
            EvalExampleSpec(
                source_sample_key=f"main_asr:{i}:{provider}",
                dataset_file=bundle.attack_dataset_file,
                sample_id=f"{Path(bundle.attack_dataset_file).stem}__{record['document']['doc_id']}__{record['goal']['goal_id']}",
                attack_name=DEFAULT_HELDOUT_ATTACK,
                example_kind="attack",
                provider=provider,
                model_slug=model_slug,
                prompt_model=model_slug,
                expected_memory_write=False,
                memory_condition="mixed",
                category_id=str(record["goal"]["category_id"]),
                subcategory_id=str(record["goal"]["subcategory_id"]),
                doc_format=str((record["document"].get("annotations") or {}).get("document_format") or ""),
                doc_domain=str(record["document"]["domain"]),
                query_style=str((record["query"] or {}).get("style") or ""),
            )
            for provider, model_slug in (
                ("generic", "openrouter/moonshotai/kimi-k2.5"),
                ("gpt", "openrouter/openai/gpt-5.4"),
            )
            for i, record in enumerate(json.loads(Path(bundle.attack_dataset_file).read_text(encoding="utf-8")))
        ]
    if suite_name == "benign_save":
        dataset_file = benign_save_dataset_file or bundle.benign_save_dataset_file
        records = json.loads(Path(dataset_file).read_text(encoding="utf-8"))
        return [
            EvalExampleSpec(
                source_sample_key=f"benign_save:{i}:{provider}",
                dataset_file=dataset_file,
                sample_id=f"{Path(dataset_file).stem}__{record['document']['doc_id']}__{record['goal']['goal_id']}",
                attack_name="none",
                example_kind="benign_save",
                provider=provider,
                model_slug=model_slug,
                prompt_model=model_slug,
                expected_memory_write=True,
                memory_condition="with_memories",
                category_id=str(record["goal"]["category_id"]),
                subcategory_id=str(record["goal"]["subcategory_id"]),
                doc_format=str((record["document"].get("annotations") or {}).get("document_format") or ""),
                doc_domain=str(record["document"]["domain"]),
                query_style=str((record["query"] or {}).get("style") or ""),
            )
            for provider, model_slug in (
                ("generic", "openrouter/moonshotai/kimi-k2.5"),
                ("gpt", "openrouter/openai/gpt-5.4"),
            )
            for i, record in enumerate(records)
        ]
    raise ValueError(f"Unknown suite: {suite_name}")


def _group_specs(specs: Iterable[EvalExampleSpec]) -> list[list[EvalExampleSpec]]:
    groups: dict[tuple[str, str, str, str, str, str], list[EvalExampleSpec]] = defaultdict(list)
    for spec in specs:
        key = (
            spec.dataset_file,
            spec.provider,
            spec.model_slug,
            spec.prompt_model,
            spec.attack_name,
            spec.example_kind,
        )
        groups[key].append(spec)
    return list(groups.values())


def main() -> None:
    args = parse_args()
    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    optimized_suffix = Path(args.optimized_suffix_file).read_text(encoding="utf-8").strip()
    bundle = build_heldout_bundle(
        root_dir,
        excluded_source_keys=_excluded_source_keys(args),
        seed=args.seed,
        attack_sample_count=args.attack_sample_count,
        benign_save_sample_count=args.benign_save_sample_count,
    )
    write_manifest(root_dir / "artifacts" / "heldout_bundle.json", bundle.__dict__)
    runner = InspectEvalRunner(root_dir=root_dir)

    results: list[dict[str, object]] = []
    suites = ("main_asr",)
    if args.include_benign_save:
        suites = suites + ("benign_save",)
    variants = (
        ("baseline_hardening", "", "system_prompt_hardening"),
        ("optimized_suffix", optimized_suffix, ""),
    )

    for suite_name in suites:
        for group in _group_specs(
            _suite_specs(
                bundle,
                suite_name=suite_name,
                benign_save_dataset_file=args.benign_save_dataset_file,
            )
        ):
            representative = group[0]
            for label, suffix, defense_name in variants:
                batch_results = runner.evaluate_batch(
                    group,
                    defense_suffix=suffix,
                    defense_name=defense_name,
                    batch_label=f"{suite_name}-{label}-{representative.provider}",
                )
                for spec, result in zip(group, batch_results, strict=True):
                    score = base_score_for_result(spec, result)
                    results.append(
                        {
                            "suite": suite_name,
                            "variant": label,
                            "provider": spec.provider,
                            "model_slug": spec.model_slug,
                            "example_kind": spec.example_kind,
                            "attack_name": spec.attack_name,
                            "sample_id": spec.sample_id,
                            "tool_called": result.tool_called,
                            "score": score,
                            "inspect_log_path": result.inspect_log_path,
                        }
                    )

    results_path = root_dir / "artifacts" / "heldout_eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in results:
        aggregates[(row["suite"], row["variant"], row["provider"])].append(float(row["score"]))

    summary = {
        f"{suite}/{variant}/{provider}": sum(scores) / len(scores)
        for (suite, variant, provider), scores in sorted(aggregates.items())
    }
    (root_dir / "artifacts" / "heldout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
