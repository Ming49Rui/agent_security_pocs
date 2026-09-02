"""Create a reusable direct-only benign-save dataset from eval-in with-memories samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from sleeper_eval.benign_save import (
    DEFAULT_BENIGN_SAVE_QUERY_TEMPLATE,
    build_benign_save_dataset,
    build_benign_save_dataset_proportional_strata,
)

DEFAULT_INPUT = "in_dist/merged_eval_in_with_memories.json"
DEFAULT_OUTPUT = "datasets/released/benign_save/merged_eval_in_benign_save_60.json"

PAPER_TRUE_OPT_BEHAVIOUR = (
    "datasets/released/behaviour_eval_in_with_true_optimized_goals.json"
)
PAPER_TRUE_OPT_AGENT = (
    "datasets/released/agent_eval_in_with_true_optimized_goals.json"
)
DEFAULT_PAPER_TRUE_OPT_OUTPUT = (
    "datasets/released/benign_save/merged_eval_in_benign_save_70_true_opt.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query-template",
        default=DEFAULT_BENIGN_SAVE_QUERY_TEMPLATE,
        help="Template for the rewritten direct user save request. Must include '{memory}'.",
    )
    parser.add_argument(
        "--behaviour-dataset",
        default="",
        help=(
            "If set together with --agent-dataset, build a proportional benign-save dataset from "
            "behaviour-related and agent-related true-optimized with-memories sources "
            "(pool-size proportional allocation)."
        ),
    )
    parser.add_argument(
        "--agent-dataset",
        default="",
        help="Paired with --behaviour-dataset for proportional multi-source benign-save (see --behaviour-dataset).",
    )
    parser.add_argument(
        "--paper-true-opt-proportional",
        action="store_true",
        help=(
            f"Shorthand: --behaviour-dataset {PAPER_TRUE_OPT_BEHAVIOUR!r}, "
            f"--agent-dataset {PAPER_TRUE_OPT_AGENT!r}, --sample-count 70, "
            f"--output {DEFAULT_PAPER_TRUE_OPT_OUTPUT!r} (override with explicit flags)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    manifest_path = output_path.with_suffix(".manifest.json")
    if args.paper_true_opt_proportional:
        behaviour = args.behaviour_dataset or PAPER_TRUE_OPT_BEHAVIOUR
        agent = args.agent_dataset or PAPER_TRUE_OPT_AGENT
        # Default CLI sample-count is 60; for this paper slice we want 70 unless overridden.
        sample_count = 70 if args.sample_count == 60 else args.sample_count
        if args.output == DEFAULT_OUTPUT:
            output_path = Path(DEFAULT_PAPER_TRUE_OPT_OUTPUT)
            manifest_path = output_path.with_suffix(".manifest.json")
        else:
            output_path = Path(args.output)
            manifest_path = output_path.with_suffix(".manifest.json")
        manifest = build_benign_save_dataset_proportional_strata(
            strata=[
                ("behaviour_related", behaviour),
                ("agent_related", agent),
            ],
            output_dataset_file=output_path,
            output_manifest_file=manifest_path,
            sample_count=sample_count,
            seed=args.seed,
            query_template=args.query_template,
        )
    elif args.behaviour_dataset and args.agent_dataset:
        manifest = build_benign_save_dataset_proportional_strata(
            strata=[
                ("behaviour_related", args.behaviour_dataset),
                ("agent_related", args.agent_dataset),
            ],
            output_dataset_file=output_path,
            output_manifest_file=manifest_path,
            sample_count=args.sample_count,
            seed=args.seed,
            query_template=args.query_template,
        )
    else:
        manifest = build_benign_save_dataset(
            input_dataset_file=args.input,
            output_dataset_file=output_path,
            output_manifest_file=manifest_path,
            sample_count=args.sample_count,
            seed=args.seed,
            query_template=args.query_template,
        )
    lines = [
        "== Benign Save Dataset ==",
        f"output={output_path}",
        f"manifest={manifest_path}",
        f"sample_count={manifest.sample_count}",
        f"seed={manifest.seed}",
    ]
    if manifest.stratum_sample_counts:
        lines.append(f"stratum_sample_counts={manifest.stratum_sample_counts}")
        lines.append(f"proportional_strata={manifest.proportional_strata}")
    else:
        lines.append(f"input={args.input}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
