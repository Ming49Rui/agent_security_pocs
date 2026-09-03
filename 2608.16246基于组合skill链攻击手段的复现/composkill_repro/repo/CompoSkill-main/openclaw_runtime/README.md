# OpenClaw Sequential Evaluator / OpenClaw 串行评测脚本

## 中文

这是一个通用的 OpenClaw 串行评测包。它会创建临时 Profile 和本地
Gateway，然后按照数据集中的顺序逐条执行任务。Agent API 与 Judge API
分别配置，运行结束后临时 Profile 会自动删除。

本包只负责评测，不下载数据，也不生成技能图、攻击链或
`execution_guide.json`。

### 1. 数据目录

使用 `--tasks-root` 指向当前 threat 的任务根目录：

```text
/absolute/path/to/tasks_t1/
├── common/
└── variants/
    └── 3_skills_chain/
```

`--threat t1/t3/t5/t6/t7` 用于选择对应的评测与 Judge 规则；
`--tasks-root` 可以是任意机器上的实际数据路径。

### 2. 安装

需要 Python 3.12、Node.js 22.14.0 或更新版本、npm 和 `patch`：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

mkdir openclaw-2026.4.22
cd openclaw-2026.4.22
npm init -y
npm install --save-exact openclaw@2026.4.22
cd ..

.venv/bin/python scripts/apply_openclaw_compat_patch.py \
  openclaw-2026.4.22
```

### 3. 全量串行运行

```bash
export AGENT_API_KEY="your-agent-api-key"
export JUDGE_API_KEY="your-judge-api-key"

.venv/bin/python scripts/run_evaluation.py \
  --threat t1 --all \
  --tasks-root /absolute/path/to/tasks_t1 \
  --chain-variant 3_skills_chain \
  --data-type poisoned \
  --agent-api-base https://agent-provider.example/v1 \
  --agent-model agent-model-id \
  --judge-api-base https://judge-provider.example/v1 \
  --judge-model judge-model-id \
  --results-root results/t1 \
  --result-run-id t1_sequential_run \
  --resume
```

运行单个任务时，去掉 `--all`，加入：

```bash
--scenario devops_and_system_admin --persona-id backend_dev
```

`--resume` 跳过已有可用结果；需要重新执行 FAILED 任务时再加入
`--rerun-failed`。API key 也可通过 `--agent-api-key` 和
`--judge-api-key` 传入，但环境变量更安全。

---

## English

This package runs an externally supplied benchmark with OpenClaw, one task at
a time. It creates an isolated temporary profile and local Gateway, configures
the Agent and Judge APIs independently, and removes the profile after the run.

It does not download datasets or generate skill graphs, attack chains, or
`execution_guide.json`.

### 1. Dataset path

Point `--tasks-root` to the task-suite root for the selected threat:

```text
/absolute/path/to/tasks_t1/
├── common/
└── variants/
    └── 3_skills_chain/
```

`--threat t1/t3/t5/t6/t7` selects the threat-specific evaluation and Judge
rules. `--tasks-root` may point to any valid location on another machine.

### 2. Install

Requirements: Python 3.12, Node.js 22.14.0 or newer, npm, and `patch`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

mkdir openclaw-2026.4.22
cd openclaw-2026.4.22
npm init -y
npm install --save-exact openclaw@2026.4.22
cd ..

.venv/bin/python scripts/apply_openclaw_compat_patch.py \
  openclaw-2026.4.22
```

### 3. Run all tasks sequentially

```bash
export AGENT_API_KEY="your-agent-api-key"
export JUDGE_API_KEY="your-judge-api-key"

.venv/bin/python scripts/run_evaluation.py \
  --threat t1 --all \
  --tasks-root /absolute/path/to/tasks_t1 \
  --chain-variant 3_skills_chain \
  --data-type poisoned \
  --agent-api-base https://agent-provider.example/v1 \
  --agent-model agent-model-id \
  --judge-api-base https://judge-provider.example/v1 \
  --judge-model judge-model-id \
  --results-root results/t1 \
  --result-run-id t1_sequential_run \
  --resume
```

For one task, remove `--all` and add:

```bash
--scenario devops_and_system_admin --persona-id backend_dev
```

`--resume` skips usable existing results. Add `--rerun-failed` only when FAILED
tasks should be executed again. API keys may also be passed with
`--agent-api-key` and `--judge-api-key`, but environment variables are safer.

The runner automatically generates its profile, runs OpenClaw `config
validate`, starts the Gateway on a free port, and processes tasks sequentially.
