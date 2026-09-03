# OpenClaw evaluation runbook

Run all commands from the `code_and_data/` directory.

## 1. Install Python and OpenClaw

Supported environment:

- macOS or Linux;
- Python 3.12;
- Node.js 22.14.0 or newer;
- npm and the system `patch` utility.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

mkdir openclaw-2026.4.22
cd openclaw-2026.4.22
npm init -y
npm install --save-exact openclaw@2026.4.22
cd ..
```

Validate and apply the version-pinned runtime patch:

```bash
.venv/bin/python scripts/apply_openclaw_compat_patch.py \
  openclaw-2026.4.22 --check
.venv/bin/python scripts/apply_openclaw_compat_patch.py \
  openclaw-2026.4.22
```

The expected CLI is:

```bash
export OPENCLAW_CLI="$PWD/openclaw-2026.4.22/node_modules/.bin/openclaw"
"$OPENCLAW_CLI" --version
```

## 2. Provide the benchmark input

Obtain the benchmark snapshot separately. The evaluation scripts do not
download or generate benchmark data. Pass the task-suite root for the selected
threat through `--tasks-root`; the root must contain:

```text
/absolute/path/to/one_threat_suite/
├── common/
└── variants/
    ├── 2_skills_chain/
    └── 3_skills_chain/
```

Each evaluated task must already contain its `task.md`, skills, poisoned or
clean data, and any graph-derived `data/execution_guide.json`. The harness reads
these files but never creates or rewrites the source dataset.

## 3. Configure the two APIs

The sequential runner accepts independent Agent and Judge settings. API keys
may be passed as command-line arguments, but environment variables are
recommended so that secrets do not remain in shell history:

```bash
export AGENT_API_KEY="your-agent-api-key"
export JUDGE_API_KEY="your-judge-api-key"
```

The remaining four values are regular runner parameters:

- `--agent-api-base`
- `--agent-model`
- `--judge-api-base`
- `--judge-model`

They may alternatively be supplied through `AGENT_API_BASE`, `AGENT_MODEL`,
`JUDGE_API_BASE`, and `JUDGE_MODEL`.

The default Agent adapter is `openai-completions`. Use
`--agent-api-adapter` for another adapter supported by OpenClaw. The Judge
endpoint must provide OpenAI-compatible Chat Completions.

For each invocation, the runner:

1. creates a random isolated `~/.openclaw-msc-sequential-*` profile;
2. writes the Agent endpoint and model into that profile;
3. runs OpenClaw `config validate`;
4. creates a shared random Gateway token;
5. starts one local Gateway on a free port;
6. evaluates the selected tasks sequentially;
7. stops the Gateway and removes the temporary profile.

## 4. Verify one task without model calls

`--dry-run` checks argument handling, dataset discovery, and workspace
materialization without starting the Gateway or calling either API:

```bash
.venv/bin/python scripts/run_evaluation.py \
  --threat t1 \
  --scenario devops_and_system_admin \
  --persona-id backend_dev \
  --tasks-root /absolute/path/to/tasks_t1 \
  --chain-variant 3_skills_chain \
  --data-type poisoned \
  --agent-api-base https://agent-provider.example/v1 \
  --agent-model agent-model-id \
  --judge-api-base https://judge-provider.example/v1 \
  --judge-model judge-model-id \
  --openclaw-cli "$OPENCLAW_CLI" \
  --result-run-id smoke_t1 \
  --dry-run
```

## 5. Run one real task

```bash
.venv/bin/python scripts/run_evaluation.py \
  --threat t1 \
  --scenario devops_and_system_admin \
  --persona-id backend_dev \
  --tasks-root /absolute/path/to/tasks_t1 \
  --chain-variant 3_skills_chain \
  --data-type poisoned \
  --interaction-turns 2 \
  --timeout-seconds 1120 \
  --max-steps 30 \
  --results-root results/reproduced/t1/custom-model \
  --result-run-id t1_3skill_poison_custom_model \
  --agent-api-base https://agent-provider.example/v1 \
  --agent-model agent-model-id \
  --judge-api-base https://judge-provider.example/v1 \
  --judge-model judge-model-id \
  --openclaw-cli "$OPENCLAW_CLI" \
  --resume
```

## 6. Run all 76 roles sequentially

```bash
.venv/bin/python scripts/run_evaluation.py \
  --threat t1 --all \
  --tasks-root /absolute/path/to/tasks_t1 \
  --chain-variant 3_skills_chain \
  --data-type poisoned \
  --interaction-turns 2 \
  --timeout-seconds 1120 \
  --max-steps 30 \
  --results-root results/reproduced/t1/custom-model \
  --result-run-id t1_3skill_poison_custom_model \
  --agent-api-base https://agent-provider.example/v1 \
  --agent-model agent-model-id \
  --judge-api-base https://judge-provider.example/v1 \
  --judge-model judge-model-id \
  --openclaw-cli "$OPENCLAW_CLI" \
  --resume
```

The underlying harness processes the discovered task list in order and does not
start parallel workers. Change the threat, model configuration, result path,
and stable run ID for another experimental cell. `--resume` skips usable
existing results and automatically retries Agent-timeout records. Add
`--rerun-failed` only when FAILED Agent records should be executed again.

## Scope

This bundle covers only the OpenClaw evaluation stage:

```text
external benchmark snapshot
  -> OpenClaw Agent execution
  -> independent Judge request
  -> raw and aggregate result files
```

Skill-card construction, graph optimization, chain ranking, poisoned-data
generation, and `execution_guide.json` generation are outside this bundle.

## Protocol identity

The Judge contract is embedded in `scripts/eval_openclaw_t1_local.py`; no
external runtime specification is loaded by this artifact.

| File | SHA-256 |
|---|---|
| `scripts/run_evaluation.py` | `d6a39e45808f0b4a8cc9fc06d494a45c6115448156a488fe00c0084d9ad2fd3e` |
| `scripts/eval_openclaw_t1_local.py` | `dc43f7ff3593388e69c492c143155a903ad6f2ff2d728f36881e9e8985e8fba8` |
| `scripts/openclaw_model_compat.py` | `d0109c8b68c5f8337410f59047cfc49499e56620ffc200089f9308a9fad726d6` |
| `scripts/create_openclaw_profile.py` | `bc0d50bfa9be2cebe92caa8ca41d2ad18bcbe75fe69b30b4a2bee980e2dce7c9` |
| `patches/openclaw-2026.4.22-gpt-gemini-compat.patch` | `96a47723e87985d58d6573ebb0bd0dc3d3826ef3d71713864197fcf62d301d4a` |

The evaluator has no runtime Agent or Judge provider default. Select both APIs
explicitly for every reproduction environment.
