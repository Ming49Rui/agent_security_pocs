# GEPA Defense Optimization

One of the prompting defenses is a **prompt-hardening suffix**: extra system-prompt
text instructing the assistant not to form memories from injected external content.
A hand-written suffix is easy for adaptive attacks to evade, so this module instead
*optimizes* the suffix with [GEPA](https://github.com/gepa-ai/gepa) — a
reflective prompt optimizer that reads full execution traces to diagnose why a
candidate failed and proposes a better one — evaluated directly against the real
`sleeper_eval` injection pipeline.

The result is `prompts/defense/gepa_prompt_hardening_suffix.txt`, the GEPA suffix the
paper's tool-based defense comparisons use (selected as a `defense` in the paper
configs). It sits alongside the naive hand-written suffix and the spotlighting
defense in [`prompts/defense/`](../../prompts/defense/).

Run commands from the repository root.

## What is included

- optimization and evaluation source code (`run_optimize.py`, `run_eval.py`,
  `adapter.py`, `reflection.py`, `runner.py`, `dataset_builder.py`, `config.py`)
- the public learned suffix consumed by configs:
  `prompts/defense/gepa_prompt_hardening_suffix.txt`

Historical runs, MLflow state, and scratch notes are intentionally not committed.

## Entry points

```bash
# Optimize the suffix against the sleeper_eval pipeline
uv run python -m experiments.gepa_defense_optimization.run_optimize --help

# Held-out transfer evaluation of an optimized suffix
uv run python -m experiments.gepa_defense_optimization.run_eval --help
```

## Relation to the rest of the repo

The optimized suffix is one of four prompting defenses (naive hardening, GEPA
hardening, extreme spotlighting, hardening + spotlighting) selectable in the paper
configs — see [`scripts/configs/paper/README.md`](../../scripts/configs/paper/README.md).
Whether a defense preserves legitimate memory writes is checked by the
[benign-save ablation](../benign_save_ablation/).
