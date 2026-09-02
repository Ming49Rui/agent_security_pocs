# Scripts

Runners and utilities for executing campaigns, replaying external-manager runs,
building data slices, and analyzing results. Most paper experiments go through
`run_eval_campaign.py`; the `run_mem0_*` / `replay_mem0_*` scripts add the
external-manager (Mem0) replay paths.

## Core public workflows

| Script | Role |
|--------|------|
| `run_eval_campaign.py` | The single campaign entrypoint. Runs any config in `configs/paper/` with `--dry-run` (validate + cost estimate) or `--yes` (execute). |
| `create_benign_save_dataset.py` | Builds the benign-save paper-aligned dataset slice (a required dataset-prep step). |
| `run_mem0_replay_pipeline.py` | Single-manager external-manager pipeline: subject transcripts → replay → semantic scoring → analysis, in one command. |
| `run_mem0_replay_suite_pipeline.py` | Same as above but fans one shared transcript set out over every manager config. |
| `replay_mem0_manager.py` | Replays one manager against an existing transcript log directory (`--from`). Used in the stepwise workflow. |
| `run_mem0_manager_replay_suite.py` | Fans out many manager configs against existing transcripts. |
| `run_mem0_gemini_batch_pipeline.py`, `batch_replay_mem0_gemini.py` | Batch-mode Gemini external-manager replay. |
| `batch_score_goal_match.py` | Standalone semantic goal-match scoring over produced logs. |
| `analyze.py` | General post-run analysis over `.eval` logs. |

`optimize_json_goals_retrievability.py` and `render_attack_result_tables.py` are
retained auxiliary helpers.

## Subdirectories

- `configs/` — campaign configs. The canonical paper-facing surface and full runbook
  is [`configs/paper/README.md`](configs/paper/README.md).
- `analysis/` — analysis-only helpers, including the public no-write failure-type
  classifier. See [`analysis/README.md`](analysis/README.md).
- `support/` — retained helper scripts (log merging, summary consolidation, replay
  analysis rebuild) that are useful but not main paper runners. See
  [`support/README.md`](support/README.md).
