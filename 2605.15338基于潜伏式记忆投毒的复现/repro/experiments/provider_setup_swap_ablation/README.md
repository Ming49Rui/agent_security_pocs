# Provider-Setup Swap Ablation

**Question:** is the injection rate a property of the *model*, or an artifact of the
*provider harness* it runs under? If swapping the harness around a fixed subject model
moved the injection rate substantially, results would be confounded by setup choices
rather than reflecting genuine model behavior.

This ablation runs the same subject model under a different provider-family harness
and compares injection rates. A provider-family swap changes several setup elements
together — system-prompt family, assistant/model identity framing, memory-tool
naming, memory placement, and document rendering — so it tests robustness to the
whole harness, not one knob. Provider families correspond to the prompt variants in
[`prompts/provider/`](../../prompts/provider/) (`gpt`, `claude`, `gemini`,
`generic`).

Run commands from the repository root.

## Source files

- dataset builder:
  [`experiments/provider_setup_swap_ablation/scripts/create_samples.py`](scripts/create_samples.py)
- result summarizer:
  [`experiments/provider_setup_swap_ablation/scripts/summarize_results.py`](scripts/summarize_results.py)
- config:
  [`scripts/configs/paper/provider_setup_swap_ablation.yaml`](../../scripts/configs/paper/provider_setup_swap_ablation.yaml)
- smoke config:
  [`scripts/configs/paper/smoke/provider_setup_swap_ablation.smoke.yaml`](../../scripts/configs/paper/smoke/provider_setup_swap_ablation.smoke.yaml)

## Build the dataset

Regenerates the 100-sample public ablation dataset (a fixed seeded sample of the
paper true-optimized set):

```bash
uv run python experiments/provider_setup_swap_ablation/scripts/create_samples.py
```

Outputs the dataset plus its `.manifest.json`, `.validation.json`, and
`.validation.md` under this experiment's `data/`.

## Run

```bash
# Preflight
uv run python scripts/run_eval_campaign.py scripts/configs/paper/provider_setup_swap_ablation.yaml --dry-run

# Smoke
uv run python scripts/run_eval_campaign.py scripts/configs/paper/smoke/provider_setup_swap_ablation.smoke.yaml --dry-run

# Full run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/provider_setup_swap_ablation.yaml --yes

# Summarize
uv run python experiments/provider_setup_swap_ablation/scripts/summarize_results.py \
  .artifacts/paper/provider_setup_swap_ablation
```

The summarizer reports per-provider-family injection rates over
`.artifacts/paper/provider_setup_swap_ablation`; the takeaway is whether they stay
close across harnesses.
