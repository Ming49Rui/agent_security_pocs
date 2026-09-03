# Support Scripts

This directory contains retained utility scripts that are useful for log inspection,
analysis consolidation, and replay maintenance, but are not the main paper-facing
entrypoints.

Core paper workflows remain at the top level of `scripts/`, for example:

- `run_eval_campaign.py`
- `replay_mem0_manager.py`
- `run_mem0_replay_pipeline.py`
- `run_mem0_replay_suite_pipeline.py`
- `create_benign_save_dataset.py`
- `analyze.py`

Included support utilities:

- `analyze_dataset.py`
- `consolidate_campaign_summaries.py`
- `merge_eval_logs.py`
- `patch_eval_samples_from_log.py`
- `rebuild_mem0_replay_analysis.py`
