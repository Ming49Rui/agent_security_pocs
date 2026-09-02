# Claude Prompt Validation

This experiment compares the **full** Claude provider prompt,
[`claude.md`](../../sleeper_eval/prompts/provider/claude.md),
against the **truncated** variant,
[`claude_truncated.md`](../../sleeper_eval/prompts/provider/claude_truncated.md),
on the same fixed in-distribution sample set.

Current dataset policy:
- supported formats only: `text`, `html`, `code`, `email`, `tweet`, `pdf`
- excluded for now: `csv`, `xlsx`, `xml`
- fixed comparison cohort: `50` records
- memory split: `25` with memories / `25` without memories
- selection seed: `20260329`

Committed input:
- provider-docrep smoke dataset: [`datasets/smoke/merged_eval_in_provider_docrep_smoke.json`](../../datasets/smoke/merged_eval_in_provider_docrep_smoke.json)

Generated local artifacts:
- `experiments/claude_prompt_validation/data/selection_seed_20260329.json`
- `experiments/claude_prompt_validation/data/merged_eval_in_claude_prompt_validation_seed_20260329.json`

The generated artifacts are reproducible local outputs and are not part of the committed public tree.

## Scripts

The workflow is intentionally reduced to three Python entrypoints:

- [`create_samples.py`](scripts/create_samples.py)
  Deterministically regenerates the fixed validation manifest and merged dataset.
- [`run_validation.py`](scripts/run_validation.py)
  Optionally regenerates datasets, runs full/truncated/both conditions, and scores `tool_call_scorer`.
- [`review_validation.py`](scripts/review_validation.py)
  Verifies sample-set consistency and provenance, prints condition summaries, and prints tool-call saves.

## Typical Usage

Load env first:

```bash
set -a
source .env
set +a
```

Run both conditions on the fixed `50`-sample validation dataset with a custom attack:

```bash
uv run python experiments/claude_prompt_validation/scripts/run_validation.py \
  --prepare-samples \
  --condition both \
  --dataset-kind full \
  --attack universal_v2_optimized_with_markers
```

This now runs the normal Inspect way:
- one `inspect eval` for the full condition
- one `inspect eval` for the truncated condition
- one `.eval` log file per condition
- tool-only scoring afterward

Review the latest full vs truncated runs:

```bash
uv run python experiments/claude_prompt_validation/scripts/review_validation.py
```

## Useful Options

Run only one condition:

```bash
uv run python experiments/claude_prompt_validation/scripts/run_validation.py \
  --condition truncated \
  --attack universal_v2_optimized_with_markers
```

Run only the format-smoke subset:

```bash
uv run python experiments/claude_prompt_validation/scripts/run_validation.py \
  --condition truncated \
  --dataset-kind format_smoke \
  --attack universal_v2_optimized_with_markers
```

Limit to the first few sample ids for debugging:

```bash
uv run python experiments/claude_prompt_validation/scripts/run_validation.py \
  --condition full \
  --attack universal_v2_optimized_with_markers \
  --sample-limit 3
```

Print commands without launching model calls:

```bash
uv run python experiments/claude_prompt_validation/scripts/run_validation.py \
  --condition both \
  --attack universal_v2_optimized_with_markers \
  --dry-run
```

Review explicit log dirs instead of the latest ones:

```bash
uv run python experiments/claude_prompt_validation/scripts/review_validation.py \
  --full-log-dir logs/claude_docrep_actor_critic_markers_aware_validation/claude_sonnet_4.6_full-<timestamp> \
  --truncated-log-dir logs/claude_docrep_actor_critic_markers_aware_validation/claude_sonnet_4.6_truncated-<timestamp>
```

## Defaults

`run_validation.py` defaults to:
- model: `anthropic/claude-sonnet-4-6`
- dataset kind: `full`
- condition: `both`
- attack: `universal_v1`
- execution mode: `batch`
- full sleep: `60s`
- truncated sleep: `30s`
- `inspect eval` with `--no-score`, followed by `inspect score` using `tool_call_scorer`

If you ever want the old one-sample-per-eval behavior, use:

```bash
uv run python experiments/claude_prompt_validation/scripts/run_validation.py \
  --condition both \
  --execution-mode serial
```

## Output Layout

Runs write to:
- `logs/claude_docrep_actor_critic_markers_aware_validation/`

With timestamped directories such as:
- `claude_sonnet_4.6_full-YYYYMMDD-HHMMSS`
- `claude_sonnet_4.6_truncated-YYYYMMDD-HHMMSS`
- `claude_sonnet_4.6_truncated_format_smoke-YYYYMMDD-HHMMSS`

## What We Care About

- primary: tool-call rate on the same fixed sample ids
- secondary: whether saved memory text tracks the adversarial goal text
