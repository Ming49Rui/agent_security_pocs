# Eval Campaign Runner

This package implements the config-driven runner behind
[`scripts/run_eval_campaign.py`](../../scripts/run_eval_campaign.py) for matrix-style
`sleeper_eval` experiments. For the public config surface and per-experiment runbook,
start at [`scripts/configs/paper/README.md`](../../scripts/configs/paper/README.md);
this document explains the runner's mechanics (planning, retry, analysis).

Use the same entrypoint for:
- planning a campaign
- running it
- retrying failed / cancelled samples in existing logs
- rebuilding analysis from logs

## Entry Point

```bash
uv run python scripts/run_eval_campaign.py <config.yaml>
```

## Core Workflow

### 1. Dry-run a campaign

```bash
uv run python scripts/run_eval_campaign.py <config.yaml> --dry-run
```

This resolves:
- datasets and sample counts
- fixed dataset attacks vs global attacks
- models / providers / prompt templates
- defenses and suffix hashes
- expanded task cells
- required environment variables

### 2. Run a campaign

```bash
uv run python scripts/run_eval_campaign.py <config.yaml>
```

Or skip the prompt:

```bash
uv run python scripts/run_eval_campaign.py <config.yaml> --yes
```

Outputs are written under the campaign `output_dir`:
- `logs/`
- `analysis/`
- `config.yaml`
- `run_manifest.yaml`
- optionally `bundle/`

### 3. Retry errored samples from an existing campaign

Auto-discover retry-worthy logs for the config:

```bash
uv run python scripts/run_eval_campaign.py <config.yaml> --retry
```

Common overrides for pathological samples:

```bash
uv run python scripts/run_eval_campaign.py <config.yaml> \
  --retry \
  --retry-max-connections 1 \
  --retry-attempt-timeout 300
```

You can also explicitly point at one or more logs:

```bash
uv run python scripts/run_eval_campaign.py <config.yaml> \
  --retry-log path/to/log.eval
```

Retry behavior is delegated to Inspect `eval-retry`, which:
- preserves already-completed samples
- reruns only errored / invalidated / cancelled samples

### 4. Rebuild analysis only

```bash
uv run python scripts/run_eval_campaign.py <config.yaml> --analyze-only
```

This:
- does not run generation
- does not run retry
- reloads the campaign logs
- rewrites `analysis/`

Use this after:
- analysis code changes
- scorer/summary formatting changes
- retry reconciliation changes

`--analyze-only` cannot be combined with `--retry` or `--retry-log`.

## Retry-Aware Analysis Semantics

Campaign analysis is not a naive "all eval_ids in the log dir" aggregation.

When Inspect `eval-retry` is used:
- old completed samples may remain associated with the original `eval_id`
- retried samples may appear under a new `eval_id`

So campaign analysis reconciles at the sample level:
- for each `(dataset, attack, model, defense, sample id)` key
- keep the newest row

This prevents:
- double-counting original + retried rows
- dropping preserved old rows when only one sample was retried

For campaign runs that include retries, prefer `--analyze-only` over calling `scripts/analyze.py` directly.

## Config Notes

### Datasets

Each dataset entry may specify:
- `dataset_file`
- `evaluation_mode`: `attack` or `benign_save`
- optional fixed `attack`
- optional filters:
  - `subcategory`
  - `category`
  - `doc_domain`
  - `domain_seed`

Benign-save datasets default to `attack=none`.

### Attacks

There are two patterns:

1. Dataset-level fixed attack:
- set `attack:` on the dataset
- no top-level `attacks:` expansion is needed

2. Global attack expansion:
- use top-level `attacks:`
- attack datasets expand across them
- benign-save datasets do not

Per-attack defense subsets are supported:

```yaml
attacks:
  - label: no-attack
    attack: none
    defense_labels:
      - no-defense
```

### Defenses

Defense entries can use:
- `defense`
- `defense_suffix_file`
- `defense_suffix_text`

Suffix-based defenses are passed through `defense_suffix_override`.

## Recommended Pattern

For normal campaign use:
1. `--dry-run`
2. run the campaign
3. if anything errors, use `--retry`
4. if needed later, use `--analyze-only`
