# Sleeper Memory Poisoning in LLM Agents

[![arXiv](https://img.shields.io/badge/arXiv-2605.15338-b31b1b.svg)](https://arxiv.org/abs/2605.15338v2)

This repository contains the public evaluation framework and released artifacts for
the paper [*Hidden in Memory: Sleeper Memory Poisoning in LLM Agents*](https://arxiv.org/abs/2605.15338v2).

LLM assistants increasingly carry **persistent memory** so they can personalize and
stay consistent across sessions. This work shows that memory is also a long-term
attack surface. An adversary who controls only ordinary external content — a
document, a web page, an email, a code repository — can plant a fabricated "memory"
that the assistant writes into its own memory store. The poisoned memory stays
**dormant** until a later, unrelated conversation retrieves it, at which point it can
steer the assistant's behavior long after the original malicious content is gone. We
call this a **sleeper memory poisoning** attack, and this repo lets you reproduce the
measurements behind it.

## Key findings

The numbers below summarize the paper; this repo reproduces the measurements. Full
tables, confidence intervals, and end-to-end coupled rates are in the paper.

### Memory injection is highly effective

Injection rate (IR) for the proposed *Actor–Critic* universal attack in the
tool-based regime, by model:

| Subset | GPT-5.4 | GPT-5.5 | Sonnet-4.6 | Gemini-3.1 | Kimi-K2.6 | DeepSeek-v4 |
|--------|--------:|--------:|-----------:|-----------:|----------:|------------:|
| Behavior | 99.4% | 99.8% | 64.2% | 88.6% | 95.0% | 96.2% |
| Agent Action | 97.0% | 91.5% | 6.5% | 67.0% | 81.0% | 88.5% |

- Actor–Critic far outperforms a naive *User Review* baseline: in the tool-based
  Behavior setting User Review is 3.0% on GPT-5.4 and 4.2% on GPT-5.5, while
  Actor–Critic reaches 99.4% and 99.8%, and remains well above User Review on every
  model.
- In the external-manager regime on the Behavior subset, Actor–Critic reaches 80.4%
  on GPT-5.4, 75.0% on GPT-5.5, 78.8% on Kimi-K2.6, and 86.4% on DeepSeek-v4, but is
  much lower on Gemini-3.1 (54.2%) and Sonnet-4.6 (13.6%).
- Claude Sonnet-4.6 is the most robust model, and the harder *Agent Action* goals
  are lower than *Behavior* goals on every model.

### Poisoned memories carry into later sessions — but proximity matters

Retrieval and adversarial usage are strongly dependent on whether the future query is
related to the planted memory:

| Query type | Retrieval rate | Adversarial-usage rate |
|------------|---------------|------------------------|
| Goal-adjacent (related) | 90.0–95.0% (Behavior), 94.0–98.0% (Agent Action) | Behavior 42.0% (GPT-5.4) to 85.0% (DeepSeek-v4); Agent Action 60.0% (Sonnet-4.6) to 89.0% (Gemini-3.1) |
| Goal-distant (unrelated) | 3.0–8.0% (Behavior), 13.0–18.0% (Agent Action) | 0.0–6.0% (Behavior), 6.0–17.0% (Agent Action) |

- Manual checks on production ChatGPT and Claude interfaces reproduced the attack in
  24/25 cases.

### Defenses help but are uneven

- Prompt-based hardening (naive, GEPA-optimized, extreme spotlighting, and
  combinations) can drive injection to near-zero for some models — notably
  Sonnet-4.6 and Gemini-3.1. The paper explicitly verifies preserved legitimate
  memory writes for the naive and GEPA hardening variants.
- It stays brittle for other models and against an adaptive attacker: under GEPA
  hardening an adaptive variant pushes Kimi-K2.6 injection back up from 6.2% to 64.6%.
- Detection is a complementary signal: activation probes exceed 0.95 AUROC with as
  few as ~125 training documents, and an LLM document scanner localizes payloads at
  >0.96.

### Mechanistic analysis

- Injected examples are highly separable in intermediate activations: single-layer
  probes reach 0.93–0.99 AUROC, fused multi-layer probes >0.95.
- Successful attacks draw disproportionate attention to the adversarial payload.
- The signature is partially shared across model families: trivial PCA alignment does
  not transfer (~0.48–0.54 AUROC), but Procrustes alignment recovers substantial
  cross-model transfer (~0.74–0.85).

## Contents

- [Threat model and metrics](#threat-model-and-metrics)
- [What is in this repository](#what-is-in-this-repository)
- [Repository navigation](#repository-navigation)
- [Environment setup](#environment-setup)
- [Prepare datasets](#prepare-datasets)
- [Experiment taxonomy](#experiment-taxonomy)
- [Running experiments](#running-experiments)
- [Prompt artifacts](#prompt-artifacts)
- [Outputs and analysis](#outputs-and-analysis)
- [Current scope](#current-scope)
- [Licensing](#licensing)

## Threat model and metrics

We assume a **black-box, external-content adversary**. The attacker has no access to
model weights, the system prompt, the memory-writing mechanism, the retrieval
mechanism, or the contents of the memory store. The attacker controls only a piece of
external content that the user later asks the assistant to process.

An attack succeeds only if it clears three stages. The framework scores each stage
separately:

| Stage | Question | Metric |
|-------|----------|--------|
| **Injection** | Does processing the malicious content cause the assistant to write a memory aligned with the attacker's goal? | **Injection Rate (IR)** |
| **Retrieval** | In a later, separate conversation, is that poisoned memory retrieved back into context? | **Retrieval Rate (RR)** |
| **Adversarial Usage** | Given the poisoned memory is in context, does the assistant actually behave the way the attacker intended? | **Adversarial Usage Rate (AUR)** |

The framework evaluates two memory-management regimes, because real assistants differ
in how they decide what to remember:

- **Tool-based regime** — the target model writes its own memory through a native
  memory tool (the pattern used by assistants such as ChatGPT, Claude, and Gemini).
- **External-manager regime** — a separate LLM memory manager observes the
  conversation and decides what to persist. This is *simulated*: rather than running
  the [Mem0](https://github.com/mem0ai/mem0) service, the framework drives a manager
  model with Mem0's published memory-extraction prompt (a checked-in copy under
  `sleeper_eval/prompts/mem0_manager/`), so the manager behavior is explicit and
  auditable. The `mem0` naming in configs and scripts refers to this simulated
  manager.

Post-injection probing uses two query types: **goal-adjacent** queries that are
topically related to the planted memory, and **goal-distant** queries that are
unrelated, to test whether a poisoned memory leaks into unrelated conversations.

## What is in this repository

The repository is organized into source areas (committed, hand-authored) and
generated areas (produced locally when you run things).

**Source areas**

- `datasets/` — the canonical data surface: committed released source datasets, a
  smoke dataset, dataset builders, and provenance/taxonomy files.
- `goals_generation/` — source code and procedure for the adversarial behavior-goal
  generation pipeline. The fixed released goal bank itself ships under
  `datasets/released/behavior_goals/`.
- `prompts/` — the stable **prompt-appendix surface**: every prompt artifact the
  paper appendix references (attack, defense, judge, manager, and provider prompts).
- `sleeper_eval/` — the evaluation runtime: the task, attacks, memory backends,
  defenses, scorers, and the campaign orchestrator.
- `scripts/` — campaign runners, mem0 replay/scoring utilities, and analysis helpers.
- `experiments/` — retained source modules for specific ablations: GEPA defense
  optimization, provider-setup swap, benign-save, and the external-manager
  model ablation.

**Generated areas** (not committed; safe to delete and regenerate)

- `.artifacts/` — default destination for eval logs, bundles, and analysis outputs.
- `logs/` — Inspect run logs and replay-pipeline working directories.

## Repository navigation

| Path | What it is | Start here if you want to… |
|------|------------|----------------------------|
| `README.md` | This file | Understand the project and reproduce results |
| `datasets/README.md` | Data surface and builders | Prepare or rebuild datasets |
| `goals_generation/README.md` | Adversarial goal pipeline | Understand or regenerate the goal bank |
| `prompts/README.md` | Prompt-appendix glossary | Find a prompt the paper references |
| `scripts/README.md` | Runner and utility index | Know which script does what |
| `scripts/configs/paper/README.md` | Canonical configs + runbook | Run a specific paper experiment |
| `experiments/*/README.md` | Per-ablation notes | Reproduce a specific ablation |
| `sleeper_eval/eval_campaign/README.md` | Campaign internals | Understand how a campaign executes |
| `sleeper_eval/prompts/document_representation/` | Provider document-rendering study | Audit how attached docs are modeled per provider |

## Environment setup

The project uses [`uv`](https://github.com/astral-sh/uv) and
[`inspect-ai`](https://inspect.aisi.org.uk/). Python 3.11+ is required.

```bash
uv venv
uv pip install -e '.[dev]'
cp .env.example .env
```

Then edit `.env`. `.env.example` lists every variable. You only need credentials for
the providers a given config actually uses, both for the subject models and (in the
external-manager regime) the mem0 manager model:

| Provider / route | Required variables | Used by |
|------------------|--------------------|---------|
| OpenRouter | `OPENROUTER_API_KEY` | many subjects (Gemini-pro, Kimi) and OpenRouter-routed mem0 managers |
| OpenAI (direct) | `OPENAI_API_KEY` | GPT subjects; default graders/judges |
| Anthropic (direct) | `ANTHROPIC_API_KEY` | Claude subjects/managers |
| Google Gemini (native API) | `GOOGLE_API_KEY` | Gemini subjects/managers on the native path |
| DeepSeek (direct) | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | DeepSeek subjects and the DeepSeek mem0 manager |

Inspect runtime variables (`INSPECT_LOG_DIR`, `INSPECT_TRACE_FILE`, `HOME`,
`INSPECT_EVAL_MAX_CONNECTIONS`) are also set in `.env.example`. You do not have to
guess which keys a config needs: `--dry-run` preflight reports exactly which
environment variables are missing for the models that config selects (see
[Running experiments](#running-experiments)). The external-manager mem0 path
additionally needs `uv pip install -e '.[mem0_eval]'`.

## Prepare datasets

The behavior, agent, and multilingual datasets are committed as **mixed-memory source
bundles** under `datasets/released/`. The exact with-memory / without-memory splits
that the paper configs consume are *generated locally* — they are derived from the
committed source, not shipped directly. Run both build steps once before running
experiments:

```bash
# 1. Materialize the paper-facing memory splits into datasets/released/generated/
uv run python datasets/builders/materialize_released_splits.py

# 2. Materialize the benign-save paper-aligned slice
uv run python scripts/create_benign_save_dataset.py --paper-true-opt-proportional --seed 42
```

See [`datasets/README.md`](datasets/README.md) for the full file glossary and for how
to build new document/query corpora.

## Experiment taxonomy

Every paper experiment maps to one canonical config and a matching smoke config. All
configs run through the single campaign entrypoint, `scripts/run_eval_campaign.py`
(the external-manager model ablation adds replay runners — see below).

| Paper experiment | Regime | Canonical config | Smoke config |
|------------------|--------|------------------|--------------|
| Main injection evaluation | Tool-based | `scripts/configs/paper/tool_based_main.yaml` | `smoke/tool_based_main.smoke.yaml`, `smoke/tool_based_main.limit1.smoke.yaml` |
| Multilingual / non-English OOD injection | Tool-based | `scripts/configs/paper/tool_based_multilingual.yaml` | `smoke/tool_based_multilingual.smoke.yaml` |
| Main injection evaluation | External-manager | `scripts/configs/paper/external_manager_main.yaml` | `smoke/external_manager_main.smoke.yaml` |
| Benign-save preservation ablation | Tool-based | `scripts/configs/paper/benign_save_ablation.yaml` | `smoke/benign_save_ablation.smoke.yaml` |
| Provider-setup swap ablation | Tool-based | `scripts/configs/paper/provider_setup_swap_ablation.yaml` | `smoke/provider_setup_swap_ablation.smoke.yaml` |
| Manager-model ablation | External-manager | `scripts/configs/paper/external_manager_model_ablation/all/*.yaml` | (uses limited replay runs; see below) |

Prompting defenses (naive prompt hardening, GEPA prompt hardening, extreme
spotlighting / untrusted-content markers, and hardening + spotlighting) are not a
separate config family — they are selected as `defenses` inside the configs above.
The defense prompt artifacts live in [`prompts/defense/`](prompts/defense/), and the
GEPA suffix optimization source lives in
[`experiments/gepa_defense_optimization/`](experiments/gepa_defense_optimization/).

(All config paths above are relative to the repository root. Smoke paths are under
`scripts/configs/paper/smoke/`.)

## Running experiments

Every campaign config follows the same two-step contract: preview with `--dry-run`,
then execute with `--yes`.

```bash
# 1. Preflight: validate the config and estimate cost without calling any model
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_main.yaml --dry-run

# 2. Quick smoke (small per-cell limits) — confirm wiring end to end
uv run python scripts/run_eval_campaign.py scripts/configs/paper/smoke/tool_based_main.smoke.yaml --yes

# 3. Full run
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_main.yaml --yes
```

For the tool-based main experiment there is also a *full-matrix* smoke config,
`smoke/tool_based_main.limit1.smoke.yaml`, which exercises every model × attack ×
defense cell with one item each — useful for catching matrix wiring problems before a
full run. The multilingual, external-manager-main, benign-save, and provider-setup
configs all follow the identical `--dry-run` → smoke → `--yes` pattern; swap in the
config from the [taxonomy table](#experiment-taxonomy).

### External-manager (mem0) — which path do I use?

The external-manager regime has two parts: generate **subject transcripts** (the
target model's conversation), then **replay** those transcripts through one or more
memory managers. There are three supported ways to do this; pick by intent:

| You want… | Use | Command |
|-----------|-----|---------|
| The full main matrix in one run | the standard campaign runner | `run_eval_campaign.py scripts/configs/paper/external_manager_main.yaml --yes` |
| One manager, cleanest end-to-end (transcripts → replay → semantic scoring → analysis) | the single-manager pipeline | `run_mem0_replay_pipeline.py scripts/configs/paper/external_manager_model_ablation/all/replay_gemini31flashlite.yaml --output-dir logs/extmgr_replay_gemini31flashlite --score-semantic --yes` |
| Many managers compared from one shared transcript set | the suite pipeline | `run_mem0_replay_suite_pipeline.py scripts/configs/paper/external_manager_model_ablation/all/subject_transcript.yaml --output-dir logs/extmgr_suite --score-semantic --max-parallel-managers 3 --yes` |

For step-by-step control (generate transcripts once, then replay a single manager or
fan out repeatedly against the existing transcript directory) see
[`experiments/external_manager_model_ablation/README.md`](experiments/external_manager_model_ablation/README.md)
and [`scripts/configs/paper/README.md`](scripts/configs/paper/README.md).

## Prompt artifacts

There are two prompt trees, and the distinction is intentional:

- **`prompts/`** is the **stable prompt-appendix surface**. Every prompt the paper
  appendix cites lives here, frozen as published, so readers can find exactly what
  the paper refers to. See [`prompts/README.md`](prompts/README.md) for the glossary.
- **`sleeper_eval/prompts/`** is the **runtime prompt area** used by the code. It may
  evolve with the implementation and is not the citation surface. Its
  `document_representation/` subfolder is the exception worth calling out: it holds
  the empirical provider study (Claude/GPT/Gemini, two runs each) that justifies the
  harness's per-provider document-rendering choices — the 16k-character long-document
  threshold, the Claude `antml:` namespace, and the csv/xlsx/xml exclusions.

Keep this split when editing: do not point paper-appendix references at runtime
files, and do not treat runtime prompts as the published artifacts.

## Outputs and analysis

Campaign configs write generated logs, bundles, and analyses under `.artifacts/` by
default (typically `.artifacts/paper/<experiment-family>/`); mem0 replay pipelines
write working trees under `logs/`. Source directories stay source-only.

The retained public no-write failure classifier recomputes deterministic
success / wrong-write / no-write outcomes from `.eval` logs and sub-classifies
`no_write` rows into `refused`, `ignored`, `accepted_but_not_stored`, or `ambiguous`:

```bash
uv run python scripts/analysis/report_injection_failure_types.py --help
```

By default its judge model is `gpt-5.4-nano`. See
[`scripts/analysis/README.md`](scripts/analysis/README.md) and
[`scripts/support/README.md`](scripts/support/README.md) for the other helpers.

## Current scope

Some paper-adjacent components are not included as first-class modules in this
release:

- LLM scanner artifacts
- activation-probe / mechanistic-analysis artifacts
- wording-sensitivity ablation assets outside the retained injection workflows

The injection framework needed for position-ablation runs is retained, but there is
no separate position-ablation module.

## Citation

If you use this work, please cite:

```bibtex
@article{pulipaka2026sleeper,
  title        = {Hidden in Memory: Sleeper Memory Poisoning in LLM Agents},
  author       = {Pulipaka, Sidharth and Hlebik, Stanislau and Raghav, Leonidas and
                  Abdelnabi, Sahar and Raina, Vyas and Sheth, Ivaxi and Fritz, Mario},
  journal      = {arXiv preprint arXiv:2605.15338},
  year         = {2026},
  doi          = {10.48550/arXiv.2605.15338},
  url          = {https://arxiv.org/abs/2605.15338}
}
```

## Licensing

Code, prompts, adversarial goals, and benchmark annotations authored in this project
are released here. Some bundled evaluation data are derived from third-party corpora
that carry their own licenses or terms of use; those upstream terms continue to
govern the underlying source materials and any source-derived portions of the
datasets.
