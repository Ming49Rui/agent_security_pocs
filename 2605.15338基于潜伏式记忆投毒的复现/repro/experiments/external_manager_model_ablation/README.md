# External-Manager Model Ablation

In the **external-manager regime**, the target model does not write its own memory; a
separate **memory manager** model observes the conversation and decides what to
persist. This ablation asks: how much does the choice of *manager* model change the
injection rate, holding the subject conversation fixed?

To make the comparison fair, the workflow first generates one set of **subject
transcripts** (the target model's conversations), then **replays** that same
transcript set through each candidate manager model. "Suite" here means the full set
of manager configs in `all/` replayed against one shared transcript set.

Run commands from the repository root.

## Configs

Canonical config root:
[`scripts/configs/paper/external_manager_model_ablation/all/`](../../scripts/configs/paper/external_manager_model_ablation/all/)

- `subject_transcript.yaml` — generates the shared subject transcripts
- `replay_gpt54nano.yaml`, `replay_claudehaiku45.yaml`,
  `replay_gemini31flashlite.yaml`, `replay_gemini31flashlite_openrouter.yaml`,
  `replay_deepseekv4flash.yaml`, `replay_qwen36flash_openrouter.yaml` — one manager
  model each

The two Gemini Flash Lite routes differ only in provider: `replay_gemini31flashlite.yaml`
uses the native Gemini API (`GOOGLE_API_KEY`); `replay_gemini31flashlite_openrouter.yaml`
uses OpenRouter (`OPENROUTER_API_KEY`).

Dataset:
`datasets/released/external_manager/behavior_action_true_opt_ablation_70_mixed.json`

## Which path do I use?

| Goal | Path |
|------|------|
| One manager, cleanest end-to-end | single-manager pipeline |
| Compare many managers from one transcript set | suite pipeline |
| Inspect transcripts or re-run replays repeatedly | stepwise (transcripts then replay) |

**Single-manager pipeline** (transcripts → replay → semantic scoring → analysis):

```bash
uv run python scripts/run_mem0_replay_pipeline.py \
  scripts/configs/paper/external_manager_model_ablation/all/replay_gemini31flashlite.yaml \
  --output-dir logs/extmgr_replay_gemini31flashlite \
  --score-semantic --yes
```

Outputs: `logs/extmgr_replay_gemini31flashlite/transcript/` and `.../replay/`. The
replay logs under `replay/logs/` are inspectable; aggregate rates are written to
`replay/analysis/`.

**Suite pipeline** (one transcript set, every manager):

```bash
uv run python scripts/run_mem0_replay_suite_pipeline.py \
  scripts/configs/paper/external_manager_model_ablation/all/subject_transcript.yaml \
  --output-dir logs/extmgr_suite \
  --score-semantic --max-parallel-managers 3 --yes
```

Outputs: `logs/extmgr_suite/transcript/` and
`logs/extmgr_suite/replays/<manager-key>/`.

**Stepwise** (generate transcripts once, then replay separately):

```bash
uv run python scripts/run_eval_campaign.py \
  scripts/configs/paper/external_manager_model_ablation/all/subject_transcript.yaml \
  --output-dir logs/extmgr_subjects/transcript --yes

# One manager against the existing transcripts:
uv run python scripts/replay_mem0_manager.py \
  scripts/configs/paper/external_manager_model_ablation/all/replay_gemini31flashlite.yaml \
  --from logs/extmgr_subjects/transcript/logs \
  --output-dir logs/extmgr_subjects/replay_gemini31flashlite \
  --score-semantic --yes

# Or fan out over every manager:
uv run python scripts/run_mem0_manager_replay_suite.py \
  --config-dir scripts/configs/paper/external_manager_model_ablation \
  --subject-family all \
  --from logs/extmgr_subjects/transcript/logs \
  --max-parallel-managers 3 --yes
```
