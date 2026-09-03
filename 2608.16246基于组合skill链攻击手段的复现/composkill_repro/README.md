# CompoSkill 复现项目

复现论文 **《CompoSkill: Compositional Skill Chain Attacks from Individually Scanner-Passing LLM Agent Skills》**（arXiv:2608.16246v1，2026-08）。
核心主张：单独通过扫描器审计（scanner-passing）的技能，被自治智能体组合成 源→桥→终点 攻击链后依然可以产生危害，技能安全是**路径级**而非**节点级**性质。

本项目的复现方式：**官方仓库 + 官方基准数据集 + 本机搭建的 Nanobot 评测环境**。已完成 4 个评测腿（黑盒 / 白盒 / 防御-B / clean）40 条记录，黑盒链形成率 60%（论文形态一致），详见下方「结果速览」与 `REPORT.md`。

---

## 1. 目录结构与用途

```
composkill_repro/
├── README.md                         ← 本文件（总体说明/运行手册）
├── datasets/CompoSkill-Bench/        ★ 官方基准数据集（经 hf-mirror git clone）
│     <威胁>/common/<场景>/<角色>/           共享底座：task.md、skills/、data/clean_data/ …
│     <威胁>/variants/<N>_skills_chain/…    链长变体：chain_spec.yaml、selected_skills.yaml、
│                                           data/poisoned_data/（黑盒成品）、poisoned_data_explicit/（白盒备份）
│     （威胁目录：data_exfiltration / memory_tampering / multi-agent_collaboration_hijacking /
│       privilege_escalation_and_dangerous_command_execution / resource_exhaustion）
├── venv312/                          ★ 虚拟环境：Python 3.12.10 / nanobot-ai==0.1.5.post3 / requests / PyYAML
└── repo/CompoSkill-main/             ★ 官方仓库（工作根，评测均在此执行）
    ├── scripts/                      流水线与评测脚本（见下表）
    ├── openclaw_runtime/             OpenClaw 运行时（本复现未使用；需 Node.js≥22，见其 docs/reproduction.md）
    ├── openclaw_skills_guard/        SkillsGuard 扫描器（防御腿 B 直接调用其 scanner.py，纯标准库）
    ├── requirements.txt              官方依赖清单
    ├── .env                          评测 API 凭据（已填；含密钥，勿提交/勿外传）
    ├── REPRO.md                      复现手册：配置、命令全集、论文各表/图对照、坑位清单
    ├── REPORT.md                     场景级复现报告（40 条记录的结果与论文对照）
    ├── benchmark/                    运行产物目录（SCG 自测、attack_plans、结果）
    │     ├── scene_scg/t1/offline_test/scg.json      Algorithm 1 离线自测图（7 节点）
    │     ├── attack_plans/t1/offline_test.json       Algorithm 1 自测输出
    │     └── results_smoke/<模型>/raw|aggregated     评测结果（raw 逐角色 JSON + 聚合 jsonl）
    └── （git 存档件：composkill.zip / hf_probe.json / nb.json / py312.exe / wheel_tmp/ 可删）
```

### 官方脚本各文件作用

| scripts/ 文件 | 作用 | 本复现用到? |
|---|---|---|
| `chain_search_ksp.py` | **Algorithm 1**：在 SCG 上枚举 source→[bridge]→terminal 链，Score=∏边权×r(源)×r(终)，LLM 做连贯性过滤 + 隐式注入撰写 | ✅ `--no-llm` 离线验证过 |
| `scene_scg_builder.py` | 用 LLM 给场景技能做能力画像（I/O/severity/角色），构建 SCG | 未用（数据集已含注入成品） |
| `inject_attack_plan.py` | 把攻击链注入任务数据（生成 execution_guide.json、改写 poisoned_data） | 未用（数据集已注入完成） |
| `generate_hsb_tasks_with_llm*.py` | 从零生成 380 个任务实例 | 未用（直接用发布数据集） |
| `build_skill_snapshot.py` / `build_skill_pools_with_llm.py` | 抓 ClawHub 快照、组建角色技能池 | 未用 |
| `msc_llm.py` | 通用 OpenAI 兼容 LLM 客户端（带磁盘缓存），供上述管线调用 | 间接 |
| `eval_nanobot_baseline.py` | **Nanobot 评测器**（agent 循环 + judge 判定 + 防御接入 + 结果落盘） | ✅ 主力（含本机 4 处补丁，见 §4） |
| `aggregate_results.py` | **本机新增**：从聚合 jsonl 计算 CFR/ASR/Utility/DBR（按 (baseline,raw_path) 去重） | ✅ |
| `cisco_skill_scanner_adapter.py` / `skillspector_adapter.py` | 防御腿 C / D 适配器 | 未用（C/D 可选） |

### 三变体数据口径（论文白盒/黑盒/clean 的物理载体）

- **黑盒（隐式注入）** → `--data-variant poisoned_data`：纯业务口吻指令、零 skill_id，含 execution_guide.json
- **白盒（显式注入）** → `--data-variant poisoned_data_explicit`：点名「使用 \<skill_id\> …」
- **clean（测 Utility）** → `--data-variant clean_data`

---

## 2. 如何运行

### 2.1 前置

- Python 3.12（本机：`C:\Users\cm191\python312\python.exe`；虚拟环境 `venv312\` 已配好）
- 依赖：`venv312\Scripts\pip.exe install -i https://pypi.tuna.tsinghua.edu.cn/simple requests PyYAML nanobot-ai==0.1.5.post3`
- API 凭据：在 `repo\CompoSkill-main\.env` 写好（agent/judge 可同 key），或用命令行 `--agent-api-key/--judge-api-key`

### 2.2 离线验证（不需要 API）

```bash
cd repo/CompoSkill-main
PY=../../venv312/Scripts/python.exe

# Algorithm 1 纯图阶段自测（手造 SCG，验证 Score 公式）
$PY scripts/chain_search_ksp.py --threat t1 --scenario offline_test --no-llm --top-k 5

# 评测器 dry-run（overlay 识别→物化→任务枚举，不调 API）
$PY scripts/eval_nanobot_baseline.py --tasks-root ../../datasets/CompoSkill-Bench/data_exfiltration \
    --results-root benchmark/results_smoke --scenario devops_and_system_admin \
    --chain-variant 3_skills_chain --data-variant poisoned_data \
    --agent-model fake --agent-api-base https://example.invalid/v1 \
    --judge-model fake --judge-api-base https://example.invalid/v1 --dry-run
```

### 2.3 评测运行（需要 API）

```bash
cd repo/CompoSkill-main
PY=../../venv312/Scripts/python.exe
BASE="--tasks-root ../../datasets/CompoSkill-Bench/data_exfiltration \
      --results-root benchmark/results_smoke \
      --chain-variant 3_skills_chain \
      --agent-model deepseek-v4-flash-0731 --agent-api-base https://model.shouxu.tech/v1 \
      --judge-model deepseek-v4-flash-0731 --judge-api-base https://model.shouxu.tech/v1"

# 烟测：1 角色
$PY scripts/eval_nanobot_baseline.py --scenario devops_and_system_admin --persona-id backend_dev \
    --data-variant poisoned_data $BASE

# 场景全量（10 角色，2 线程）—— 四个腿照抄下面四行，先后执行
$PY scripts/eval_nanobot_baseline.py --scenario devops_and_system_admin --data-variant poisoned_data --num-threads 2 $BASE            # A 黑盒
$PY scripts/eval_nanobot_baseline.py --scenario devops_and_system_admin --data-variant poisoned_data_explicit --num-threads 2 $BASE  # A 白盒
$PY scripts/eval_nanobot_baseline.py --scenario devops_and_system_admin --data-variant poisoned_data --baseline-mode B_skills_guard --guard-decision-mode strict_block --num-threads 2 $BASE  # B 防御
$PY scripts/eval_nanobot_baseline.py --scenario devops_and_system_admin --data-variant clean_data --num-threads 2 $BASE            # A clean（Utility）

# 聚合指标（CFR/ASR/Utility/DBR）
$PY scripts/aggregate_results.py --results-root benchmark/results_smoke/deepseek-v4-flash-0731

# 扩展其它威胁：换 --tasks-root 的威胁目录即可（5 威胁全部支持），全场景用 --all
```

### 2.4 常用参数

| 参数 | 说明 |
|---|---|
| `--chain-variant` | 链长变体（2_skills_chain … 6_skills_chain）；**不传默认取最大 N 变体**，复现表 2 必须显式 `3_skills_chain` |
| `--baseline-mode` | A_no_guard / B_skills_guard / C_cisco_scanner / D_skillspector |
| `--guard-decision-mode` | warn_only / strict_block（复现表 3 用 strict_block） |
| `--data-variant` | poisoned_data / poisoned_data_explicit / clean_data |
| `--interaction-turns` | 交互轮数（默认 3） |
| `--num-threads` | 并发线程（默认 1） |
| `--resume` | 断点续跑，跳过已有有效结果 |

---

## 3. 结果速览（T1 × devops 场景 × 10 角色 × 3 链，deepseek-v4-flash-0731）

| 腿 | n | CFR | ASR | Utility |
|---|---|---|---|---|
| A 黑盒 implicit | 10 | **60%** | **10%** | — |
| A 白盒 explicit | 10 | 10% | 0% | — |
| A clean | 10 | 40% | 0% | **46.5%** |
| B SkillsGuard strict-block | 10 | 0% | 0% | — |

- 黑盒 CFR 60% 与论文形态一致（论文 DeepSeek-V4/Nanobot 80.6%）；ASR 10%（论文 53.2%），flash 模型 + 单次采样偏低属预期，backend_dev 曾跑出完整 hard_trigger。
- 防御 B 本机 DBR=0.00（论文 0.58）：拦截更彻底，方向一致。
- 白盒腿被本机 nanobot 沙箱（restrict_to_workspace）压制，9/10 死于 safety guard——环境性差异。

---

## 4. 本机对官方代码的修改（重新搭建环境时必读）

依赖锁定 `nanobot-ai==0.1.5.post3`；官方评估器基于未发布的旧 API 编写，补丁集中在 `scripts/eval_nanobot_baseline.py`（4 处）：

1. import：`custom_provider.CustomProvider` → `openai_compat_provider.OpenAICompatProvider as CustomProvider`（0.1.5.post3 无 custom_provider 模块，构造器前三个参数一致）
2. AgentLoop 参数：`web_search_config=WebSearchConfig()` → `web_config=WebToolsConfig()`
3. **工具事件埋点**：0.1.5.post3 的 runner 直接调 `tool.execute(**params)`、绕过 `ToolRegistry.execute`，官方埋点不触发 → 已改为遍历 `agent.tools.tool_names` 逐个包装工具实例的 execute（judge 依赖工具轨迹判 CFR/ASR，此修复必不可少）
4. `--data-variant` choices 增补 `poisoned_data_explicit`（白盒腿）

---

## 5. 已知注意事项

- `.env` 含真实 API key，请勿提交到任何 git 仓库或外传。
- raw 文件名不含 baseline mode：防御腿会覆盖同角色同变体的 `…__poisoned_data__baseline_v16.json` 原始文件（指标数据在聚合 jsonl 里按 `(baseline, raw_path)` 去重保留，不会丢）。
- HF 数据集直连不通时用镜像：`git clone https://hf-mirror.com/datasets/Limax11/CompoSkill-Bench`。
- 防御腿 C（Cisco）/D（SkillSpector）需要额外安装对应包/环境，本机未启用。
- 评测慢是常态：单角色 6–8 分钟（agent 多轮 + judge），场景全量四腿约 3.5 小时；可用 `--num-threads 2~4` 加速。