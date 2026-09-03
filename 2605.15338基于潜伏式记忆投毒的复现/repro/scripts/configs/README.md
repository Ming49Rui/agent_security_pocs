# Configs

Campaign configs for `scripts/run_eval_campaign.py`. A config is a YAML file that
fully specifies one experiment (datasets, models, attacks, defenses, memory backend,
scoring) so a run is reproducible from the file alone.

## Canonical public surface

All paper-facing configs live under [`paper/`](paper/). That directory's
[`README.md`](paper/README.md) is the authoritative config map and runbook; start
there. In short:

- `paper/*.yaml` — canonical experiment configs (tool-based main, multilingual,
  external-manager main, benign-save, provider-setup swap).
- `paper/smoke/*.yaml` — small-limit smoke counterparts.
- `paper/external_manager_model_ablation/all/*.yaml` — subject-transcript and
  per-manager replay configs for the external-manager model ablation.

## Output convention

Configs write generated logs, bundles, and analyses under `.artifacts/` by default
(typically `.artifacts/paper/<experiment-family>/`). Source datasets and experiment
modules stay under `datasets/` and `experiments/`; generated artifacts are not
committed.
