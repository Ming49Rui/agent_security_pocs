# CompoSkill 复现手册（reproduction runbook）

论文：*CompoSkill: Compositional Skill Chain Attacks from Individually Scanner-Passing LLM Agent Skills* (arXiv:2608.16246v1)

## 0. 现状速览（已完成 ✅ / 待 API key ⏳）

| 项 | 状态 | 说明 |
|---|---|---|
| 官方仓库 CompoSkill (GitHub Limax666/CompoSkill) | ✅ | 下载至 `composkill_repro/repo/CompoSkill-main/` |
| 基准数据集 CompoSkill-Bench (HF Limax11/CompoSkill-Bench) | ✅ | 经 hf-mirror 克隆至 `composkill_repro/datasets/CompoSkill-Bench/`，5 威胁 × 5 链长变体(2–6) × 6 场景 × 76 角色 |
| Python 3.12.10 (用户级, C:\Users\cm191\python312) + venv | ✅ | venv 位于 `composkill_repro/venv312`，已装 nanobot-ai 0.3.0 / requests / PyYAML |
| Algorithm 1 图阶段离线验证 | ✅ | 手造 SCG 自测，Score=∏边权×r(源)×r(终) 与论文 Eq.1 吻合（0.81 例） |
| 数据物化 materialize_overlay_tasks_root | ✅ | common+variant 合并为评测任务包，dry-run 全链路通过 |
| 白盒变体接入 | ✅ | 打了补丁：`--data-variant poisoned_data_explicit`（原代码只允许 poisoned/clean） |
| 评测运行（agent+judge 调 LLM） | ⏳ | 缺 API key |

## 1. 目录与数据布局

```
composkill_repro/
├── repo/CompoSkill-main/          # 官方仓库（工作根）
│   ├── scripts/                   # 流水线 + 评测器
│   ├── openclaw_skills_guard/     # SkillsGuard 扫描器（须与 scripts/ 同级）
│   └── benchmark/                 # 输出目录（SCG/attack_plans/results）
├── datasets/CompoSkill-Bench/     # 基准数据集
└── venv312/                       # Python 3.12 venv
```

数据集 overlay 布局（评测器自动识别）：
```
<威胁>/common/<场景>/<角色>/     → task.md, runtime.yaml, skills/, data/clean_data/ …
<威胁>/variants/<N>_skills_chain/<场景>/<角色>/
    → chain_spec.yaml, selected_skills.yaml, ipi_injection_spec.yaml,
      data/poisoned_data/ (黑盒隐式注入成品) + data/poisoned_data_explicit/ (白盒显式备份)
```

三变体对应关系（论文的白盒/黑盒/clean）：
- **白盒（显式注入）** → `--data-variant poisoned_data_explicit`（文件内点名 skill_id，如 "使用 xejrax_file-search 检索…"）
- **黑盒（隐式注入）** → `--data-variant poisoned_data`（纯能力级业务指令，execution_guide.json 无任何 skill_id）
- **clean（测 Utility）** → `--data-variant clean_data`

威胁/目录名注意：数据集顶层目录是语义名（data_exfiltration / memory_tampering / multi-agent_collaboration_hijacking / privilege_escalation_and_dangerous_command_execution / resource_exhaustion）；代码里威胁键是 t1/t3/t5/t6/t7。已确认 5 个威胁目录全部存在（HF API 分页曾造成"只有 4 个"的误判）。

## 2. .env 配置（唯一待办）

在 `repo/CompoSkill-main/.env` 写入（agent/judge 可同 key 同模型）：
```
NANOBOT_API_KEY=<你的key>
NANOBOT_API_BASE=https://api.siliconflow.cn/v1     # 或 dashscope 等
NANOBOT_MODEL=deepseek-ai/DeepSeek-V4-Flash
JUDGE_API_KEY=<你的key>
JUDGE_API_BASE=https://api.siliconflow.cn/v1
JUDGE_MODEL=deepseek-ai/DeepSeek-V4-Flash
```
已探测：api.siliconflow.cn / dashscope.aliyuncs.com / api.deepseek.com 网络均可达（401=待鉴权）。

## 2.5 本机已应用的兼容补丁（库版本差异）

仓库代码基于 nanobot-ai 的一个开发版/旧版 API 编写；正式发布版（0.1.4.post6~0.3.0）接口不同。本复现已锁定 **nanobot-ai==0.1.5.post3** 并改动 `scripts/eval_nanobot_baseline.py` 三处：

1. `from nanobot.providers.custom_provider import CustomProvider` → `from nanobot.providers.openai_compat_provider import OpenAICompatProvider as CustomProvider`
   （0.1.5.post3 无 custom_provider 模块；OpenAICompatProvider 构造器前三个参数 api_key/api_base/default_model 与 CustomProvider 相同）
2. AgentLoop 参数 `web_search_config=WebSearchConfig()` → `web_config=WebToolsConfig()`
   （0.1.5.post3 的参数名是 web_config，类型 WebToolsConfig，其内含 search=WebSearchConfig 子配置）
3. 工具事件埋点：0.1.5.post3 的 runner 对已知工具直接调 `tool.execute(**params)`，绕过 `ToolRegistry.execute`，导致原 `agent.tools.execute = traced_execute` 从不触发。已改为遍历 `agent.tools.tool_names`，逐个包装已注册工具实例的 `execute` 方法（保留 name/params/result_preview 记录格式），保证 judge 能看到工具轨迹（CFR/ASR 依赖它）。

另：`--data-variant` 的 choices 增加了 `poisoned_data_explicit`（白盒腿）。

## 3. 离线验证命令（无需 API）

```bash
PY=../../venv312/Scripts/python.exe   # 于 repo/CompoSkill-main 下执行

# Algorithm 1 自测（--no-llm 纯图阶段）
$PY scripts/chain_search_ksp.py --threat t1 --scenario offline_test --no-llm --top-k 5
# 自测 SCG 在 benchmark/scene_scg/t1/offline_test/scg.json，验证点：3链 0.81 分

# 评测器 dry-run（物化+任务枚举，不调 API）
$PY scripts/eval_nanobot_baseline.py --tasks-root ../../datasets/CompoSkill-Bench/data_exfiltration \
    --results-root benchmark/results_smoke --scenario devops_and_system_admin \
    --chain-variant 3_skills_chain --data-variant poisoned_data \
    --agent-model fake --agent-api-base https://example.invalid/v1 \
    --judge-model fake --judge-api-base https://example.invalid/v1 --dry-run
```

## 4. 评测运行（需 API key）

### 4.1 最小烟测（1 角色）
```bash
$PY scripts/eval_nanobot_baseline.py --tasks-root ../../datasets/CompoSkill-Bench/data_exfiltration \
    --results-root benchmark/results_smoke --scenario devops_and_system_admin --persona-id backend_dev \
    --chain-variant 3_skills_chain --data-variant poisoned_data \
    --agent-model deepseek-ai/DeepSeek-V4-Flash --agent-api-base https://api.siliconflow.cn/v1 \
    --judge-model deepseek-ai/DeepSeek-V4-Flash --judge-api-base https://api.siliconflow.cn/v1
```
（key 从 .env 自动读 NANOBOT_API_KEY / JUDGE_API_KEY，无需命令行传。）

### 4.2 关键参数
- `--chain-variant`：**必须显式指定**。不传时评测器默认取 `variants/` 下**最大 N** 的变体（如 6_skills_chain），不是论文的 3 链！
- `--baseline-mode`：A_no_guard（无防御）/ B_skills_guard / C_cisco_scanner / D_skillspector
- `--guard-decision-mode`：warn_only / strict_block（复现表 3 的严格封禁用 strict_block）
- `--interaction-turns`：默认 3（.env HSB_INTERACTION_TURNS 可调）
- `--all` = 全部场景；`--scenario X` = 该场景全部角色；`--num-threads N` = 并发
- `--resume`：断点续跑，跳过已有有效结果

### 4.3 复现论文各表/图
| 论文 | 命令要点 |
|---|---|
| 表2 黑盒（3链 CFR/ASR） | 每个威胁：`--chain-variant 3_skills_chain --data-variant poisoned_data --baseline-mode A_no_guard --all` |
| 图3 白盒 | `--data-variant poisoned_data_explicit`（论文数值为 Nanobot/DeepSeek-V4 上 83.3% CFR / 59.7% ASR） |
| 表3 防御绕过 | 同 3 链黑盒 × baseline-mode B/C/D × `--guard-decision-mode strict_block`，再用 aggregate_results.py 算 DBR |
| 图4 2 链 vs 3 链 | `--chain-variant 2_skills_chain` 与 `3_skills_chain` 各跑一次 |
| RQ3 链长 L=4..6 | `--chain-variant 4_skills_chain` 等（数据集已含 4/5/6 变体，无需克隆） |
| Utility | `--data-variant clean_data` 单独跑 |

### 4.4 指标聚合
```bash
$PY scripts/aggregate_results.py --results-root benchmark/results_smoke
```
输出按 baseline_mode × data_variant 分组：n / CFR / ASR / Utility，以及 DB R 表。

## 5. 与论文的口径对照 + 坑

1. **威胁编号不一致**：代码/README 用 t1/t3/t5/t6/t7（README 表里 t5 还写成 Lateral Movement，实为权限提升）；数据集目录用语义名；论文正文用 T1–T5。以数据集目录名为准。
2. **链长默认值坑**：`--chain-variant` 不传 = 最大 N 变体，复现 RQ1/表2 必须显式 `3_skills_chain`。
3. **ASR ≤ CFR 恒成立**（formed 前置）+ 本仓储对 ASR 的计算含 hard/soft trigger 两档：论文里 ASR 对应 hard_trigger（`overall_chain_triggered`），聚合脚本按此实现。
4. **评测隔离**：agent 只看到 task.md + skills/ + data/<variant>，chain_spec.yaml / ipi_injection_spec.yaml 等元数据不给 agent（judge 才用）——与论文"隐式注入不暴露技能 ID"的设计一致。
5. **防御 B（SkillsGuard）**：需要 `openclaw_skills_guard/` 与 `scripts/` 同级（已在仓库内满足），扫描器规则在 `scanner.py`（total_hits≥15 → block）。
6. **D（SkillSpector）** adapter 需要外部 env（skillspector 环境），机器上若无则跳过该 leg。
7. **OpenClaw 运行时**（论文另一平台）需 Node.js ≥22 与 npm 装 openclaw@2026.4.22，见 `openclaw_runtime/docs/reproduction.md`；本机未装 Node，首轮复现建议只用 Nanobot 平台。

## 6. 成本提示
单次 agent 会话 ≈ 交互轮数×(agent+audit judge) + 终局 judge ≈ 3–6 次 LLM 调用。
- 烟测（1 角色×2 变体）：≈ 10–15 次调用，< ¥0.5
- 场景级（1 场景 10 角色×2/3 链）：≈ 200 次调用
- 威胁级（76 角色×3 链）：≈ 2000 次调用（1–2 小时）
用 DeepSeek-V4-Flash 级别模型，威胁级全量（含防御 legs）估计几十元以内。