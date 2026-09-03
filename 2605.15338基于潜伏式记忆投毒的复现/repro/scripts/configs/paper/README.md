# Paper Configs

This directory holds the canonical, paper-facing campaign configs for the public
release. A **campaign config** is a YAML file that fully specifies one experiment —
its datasets, models, attacks, defenses, memory backend, and scoring — so the run is
reproducible from the file alone.

Every config runs through one entrypoint and the same two-step contract: preview with
`--dry-run`, then execute with `--yes`.

```bash
uv run python scripts/run_eval_campaign.py <config> --dry-run   # validate + cost estimate, no model calls
uv run python scripts/run_eval_campaign.py <config> --yes       # execute
```

For the threat model, metrics (IR / RR / AUR), and the regime distinction, see the
[top-level README](../../../README.md). For how a campaign executes internally, see
[`sleeper_eval/eval_campaign/README.md`](../../../sleeper_eval/eval_campaign/README.md).

## Config map

| Config (relative to repo root) | Experiment | Regime | Smoke counterpart |
|--------------------------------|------------|--------|-------------------|
| `scripts/configs/paper/tool_based_main.yaml` | Main injection evaluation | Tool-based | `smoke/tool_based_main.smoke.yaml`, `smoke/tool_based_main.limit1.smoke.yaml` |
| `scripts/configs/paper/tool_based_multilingual.yaml` | Multilingual / non-English OOD | Tool-based | `smoke/tool_based_multilingual.smoke.yaml` |
| `scripts/configs/paper/external_manager_main.yaml` | Main injection evaluation | External-manager | `smoke/external_manager_main.smoke.yaml` |
| `scripts/configs/paper/benign_save_ablation.yaml` | Benign-save preservation ablation | Tool-based | `smoke/benign_save_ablation.smoke.yaml` |
| `scripts/configs/paper/provider_setup_swap_ablation.yaml` | Provider-setup swap ablation | Tool-based | `smoke/provider_setup_swap_ablation.smoke.yaml` |
| `scripts/configs/paper/external_manager_model_ablation/all/*.yaml` | Manager-model ablation | External-manager | run with small limits (see below) |

Smoke configs mirror their canonical experiment with small per-cell limits.
`tool_based_main.limit1.smoke.yaml` is a *full-matrix* smoke: it touches every
model × attack × defense cell with one item each, to catch matrix wiring issues
before a full run.

All generated logs, bundles, and analyses default to `.artifacts/` (typically
`.artifacts/paper/<experiment-family>/`); mem0 replay pipelines use working trees
under `logs/`.

## Runbook

### Dataset prep (once)

```bash
uv run python datasets/builders/materialize_released_splits.py
uv run python scripts/create_benign_save_dataset.py --paper-true-opt-proportional --seed 42
```

### Standard campaigns

These all follow the identical `--dry-run` → smoke → `--yes` pattern; only the config
path changes.

```bash
# Tool-based main
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_main.yaml --dry-run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/smoke/tool_based_main.limit1.smoke.yaml --yes
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_main.yaml --yes

# Tool-based multilingual
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_multilingual.yaml --dry-run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_multilingual.yaml --yes

# External-manager main
uv run python scripts/run_eval_campaign.py scripts/configs/paper/external_manager_main.yaml --dry-run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/external_manager_main.yaml --yes

# Benign-save ablation
uv run python scripts/run_eval_campaign.py scripts/configs/paper/benign_save_ablation.yaml --dry-run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/benign_save_ablation.yaml --yes

# Provider-setup swap ablation
uv run python scripts/run_eval_campaign.py scripts/configs/paper/provider_setup_swap_ablation.yaml --dry-run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/provider_setup_swap_ablation.yaml --yes
```

### External-manager model ablation

This ablation generates **subject transcripts** (the target model's conversation),
then **replays** them through one or more memory managers. Pick the path by intent.

**Single manager, cleanest end-to-end** (transcripts → replay → semantic scoring →
analysis in one command):

```bash
uv run python scripts/run_mem0_replay_pipeline.py \
  scripts/configs/paper/external_manager_model_ablation/all/replay_gemini31flashlite.yaml \
  --output-dir logs/extmgr_replay_gemini31flashlite \
  --score-semantic --yes
```

Produces `logs/extmgr_replay_gemini31flashlite/transcript/` and `.../replay/`. The
replay tree holds the final `.eval` logs plus memory-write and semantic-match rates.

**Whole suite from one shared transcript set:**

```bash
uv run python scripts/run_mem0_replay_suite_pipeline.py \
  scripts/configs/paper/external_manager_model_ablation/all/subject_transcript.yaml \
  --output-dir logs/extmgr_suite \
  --score-semantic --max-parallel-managers 3 --yes
```

Produces `logs/extmgr_suite/transcript/` and `logs/extmgr_suite/replays/<manager-key>/`,
one replay tree per `replay_*.yaml`. The suite includes both
`replay_gemini31flashlite.yaml` (native Gemini / `GOOGLE_API_KEY`) and
`replay_gemini31flashlite_openrouter.yaml` (OpenRouter / `OPENROUTER_API_KEY`).

**Stepwise** (generate transcripts once, then replay separately — best for
inspection or repeated fan-out):

```bash
uv run python scripts/run_eval_campaign.py \
  scripts/configs/paper/external_manager_model_ablation/all/subject_transcript.yaml \
  --output-dir logs/extmgr_subjects/transcript --yes

# Replay one manager against the existing transcripts:
uv run python scripts/replay_mem0_manager.py \
  scripts/configs/paper/external_manager_model_ablation/all/replay_gemini31flashlite.yaml \
  --from logs/extmgr_subjects/transcript/logs \
  --output-dir logs/extmgr_subjects/replay_gemini31flashlite \
  --score-semantic --yes

# Or fan out over every manager config from the same transcripts:
uv run python scripts/run_mem0_manager_replay_suite.py \
  --config-dir scripts/configs/paper/external_manager_model_ablation \
  --subject-family all \
  --from logs/extmgr_subjects/transcript/logs \
  --max-parallel-managers 3 --yes
```

See
[`experiments/external_manager_model_ablation/README.md`](../../../experiments/external_manager_model_ablation/README.md)
for a fuller walkthrough of the manager suite.
