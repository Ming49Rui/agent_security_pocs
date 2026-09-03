# Benign-Save Ablation

A defense is only useful if it blocks *poisoned* memory writes without also blocking
*legitimate* ones. This ablation measures that side: it checks that the prompting
defenses do not collapse normal, user-requested memory saves. A defense that drives
the injection rate down but also refuses benign saves is not a real win — this
experiment makes that trade-off visible.

The dataset consists of benign user requests that legitimately ask the assistant to
remember something. A "good" defense should leave these writes intact (high benign
save rate) while still reducing adversarial injection in the main experiments.

Run commands from the repository root.

## Inputs

- builder: [`scripts/create_benign_save_dataset.py`](../../scripts/create_benign_save_dataset.py)
- config: [`scripts/configs/paper/benign_save_ablation.yaml`](../../scripts/configs/paper/benign_save_ablation.yaml)
- generated dataset: `datasets/released/benign_save/merged_eval_in_benign_save_70_true_opt.json`

## Build

```bash
uv run python scripts/create_benign_save_dataset.py --paper-true-opt-proportional --seed 42
```

## Run

```bash
# Preflight
uv run python scripts/run_eval_campaign.py scripts/configs/paper/benign_save_ablation.yaml --dry-run

# Generation plus semantic scoring
uv run python scripts/run_eval_campaign.py scripts/configs/paper/benign_save_ablation.yaml --yes
```

Results land under `.artifacts/paper/benign_save_ablation/`. Read this alongside the
tool-based main results: the relevant comparison is the benign save rate here versus
the injection-rate reduction the same defense achieves there.
