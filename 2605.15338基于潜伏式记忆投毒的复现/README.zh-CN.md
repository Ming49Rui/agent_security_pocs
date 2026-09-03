# LLM Agent 记忆投毒（Sleeper Memory Poisoning）——中文 README

> 本文件是官方仓库 `README.md` 的中文翻译与本地化说明，内容忠实于原作者（SPAR/ELLIS/CISPA）的发布说明，并补充了本机（shouxu.tech 端点 + Windows）复现时的实际使用说明。论文见此：[*Hidden in Memory: Sleeper Memory Poisoning in LLM Agents*](https://arxiv.org/abs/2605.15338v2)（arXiv:2605.15338v2）。

LLM 助手越来越多地携带**持久记忆**（persistent memory），以便跨会话个性化与保持一致。本工作证明：记忆本身也是一个长期攻击面。只控制普通外部内容的攻击者——一份文档、一个网页、一封邮件、一个代码仓库——就能在助手自己的记忆库里"种下"一条伪造的"记忆"。这条被投毒的记忆会**潜伏**到之后某个无关会话中被检索出来，此时它就能引导助手的行为——而最初的恶意内容早已消失。我们称之为**潜伏式记忆投毒（sleeper memory poisoning）**，本仓库提供复现论文全部测量的评估框架。

## 关键发现

仓库复现论文的测量结果。完整表格、置信区间与端到端耦合率见论文。

### 记忆注入高度有效

工具型记忆机制下，论文提出的 **Actor-Critic 通用攻击** 的注入率（IR），按模型：

| 子集 | GPT-5.4 | GPT-5.5 | Sonnet-4.6 | Gemini-3.1 | Kimi-K2.6 | DeepSeek-v4 |
|--------|--------:|--------:|-----------:|-----------:|----------:|------------:|
| Behavior | 99.4% | 99.8% | 64.2% | 88.6% | 95.0% | 96.2% |
| Agent Action | 97.0% | 91.5% | 6.5% | 67.0% | 81.0% | 88.5% |

- Actor-Critic 远胜朴素的 *User Review* 基线：工具型 Behavior 设置下，User Review 在 GPT-5.4 上仅 3.0%、GPT-5.5 上 4.2%，而 Actor-Critic 达到 99.4% / 99.8%，且在每个模型上都明显高于 User Review。
- 外部管理器机制下 Behavior 子集：Actor-Critic 在 GPT-5.4 达 80.4%、GPT-5.5 达 75.0%、Kimi-K2.6 达 78.8%、DeepSeek-v4 达 86.4%，但在 Gemini-3.1（54.2%）与 Sonnet-4.6（13.6%）上低得多。
- Claude Sonnet-4.6 是最稳健的模型；更难的 *Agent Action* 目标在**所有模型**上的注入率都低于 *Behavior* 目标。

### 被投毒的记忆会进入后续会话——但邻近度是关键

毒记忆的检索与恶意使用，强烈依赖未来查询是否与记忆相关：

| 查询类型 | 检索率（RR） | 恶意使用率（AUR） |
|------------|---------------|----------------------|
| 目标邻近（相关） | Behavior 90.0–95.0%；Agent Action 94.0–98.0% | Behavior 42.0%（GPT-5.4）～85.0%（DeepSeek-v4）；Agent Action 60.0%（Sonnet-4.6）～89.0%（Gemini-3.1） |
| 目标无关 | Behavior 3.0–8.0%；Agent Action 13.0–18.0% | Behavior 0.0–6.0%；Agent Action 6.0–17.0% |

- 在 ChatGPT 与 Claude 生产界面上人工抽查，24/25 例复现成功。

### 防御有效但不均衡

- 提示级加固（朴素、GEPA 优化、极端 spotlighting 及组合）可以把部分模型的注入压到近零——尤其是 Sonnet-4.6 与 Gemini-3.1。论文明确验证了朴素与 GEPA 加固不损害合法记忆写入。
- 但对其他模型与自适应攻击者依然脆弱：GEPA 加固下，自适应变体把 Kimi-K2.6 的注入率从 6.2% 拉回 64.6%。
- 检测是互补信号：激活探测用 ~125 篇文档即可超过 0.95 AUROC；LLM 文档扫描器定位载荷 >0.96。

### 机制分析

- 注入样本在中间层激活上高度可分：单层探测 0.93–0.99 AUROC，融合多层探测 >0.95。
- 成功攻击对攻击载荷分配了显著更多的注意力。
- 该信号部分跨模型共享：朴素 PCA 对齐不能迁移（~0.48–0.54 AUROC），Procrustes 对齐可恢复大量跨模型迁移（~0.74–0.85）。

## 威胁模型与指标

我们假设**黑盒、外部内容型攻击者**：攻击者无法访问模型权重、系统提示词、记忆写入机制、检索机制或记忆库内容，只能控制用户之后要求助手处理的一段外部内容。

一次攻击成功必须连贯通过三个阶段，框架对每阶段单独打分：

| 阶段 | 问题 | 指标 |
|-------|----------|--------|
| **注入（Injection）** | 处理恶意内容是否会让助手写入一条与攻击者目标一致的记忆？ | **注入率（IR）** |
| **检索（Retrieval）** | 在之后的独立会话中，这条被投毒的记忆是否被取回上下文？ | **检索率（RR）** |
| **恶意使用（Adversarial Usage）** | 毒记忆进入上下文后，助手是否真的按攻击者意图行事？ | **恶意使用率（AUR）** |

框架评估两种记忆管理机制（真实助手在"记什么"上的差异）：

- **工具型机制（Tool-based）**——目标模型通过原生记忆工具自己写记忆（ChatGPT、Claude、Gemini 等助手的模式）。
- **外部管理器机制（External-manager / mem0）**——独立 LLM 记忆管理器观察对话并决定留存什么。这里是**模拟**的：不运行 [Mem0](https://github.com/mem0ai/mem0) 服务，而是用 Mem0 公开的记忆抽取提示词驱动一个管理器模型（副本在 `sleeper_eval/prompts/mem0_manager/`），管理器行为显式、可审计。配置/脚本中的 `mem0` 字样即指该模拟管理器。

后注入探测使用两类查询：**目标邻近（goal-adjacent）**——与毒记忆主题相关；**目标无关（goal-distant）**——与毒记忆无关，用来检验毒记忆是否泄漏进无关会话。

## 仓库内容

**源码区**（已提交，手写文件）：

- `datasets/` —— 数据表面：发布的源数据集、smoke 数据集、数据集构建器、来源/分类文件。
- `goals_generation/` —— 对抗行为目标生成管线的源码与流程。固定发布的 700 条目标库在 `datasets/released/behavior_goals/`。
- `prompts/` —— **稳定的提示词附件面**：论文附录引用的每个提示词工件（攻击、防御、判分、管理器、提供商）。
- `sleeper_eval/` —— 评估运行时：任务、攻击、记忆后端、防御、判分器与战役编排器。
- `scripts/` —— 战役运行器、mem0 回放/判分工具、分析辅助脚本。
- `experiments/` —— 特定消融的保留模块：GEPA 防御优化、提供商风格互换、良性保存、外部管理器模型消融。

**生成区**（不入库，可删可重建）：

- `.artifacts/` —— 评测日志、bundle 包、分析输出的默认目录。
- `logs/` —— Inspect 运行日志与回放管线工作目录。

## 导航速查

| 路径 | 用途 |
|------|------|
| `README.md` | 本文件（英文原版） |
| `datasets/README.md` | 数据表面与构建器：准备或重建数据集 |
| `goals_generation/README.md` | 对抗目标管线：理解或重新生成目标库 |
| `prompts/README.md` | 提示词附件总览 |
| `scripts/README.md` | 运行器与工具索引 |
| `scripts/configs/paper/README.md` | 官方配置与运行手册：跑某个论文实验 |
| `experiments/*/README.md` | 各消融备注 |
| `sleeper_eval/eval_campaign/README.md` | 战役内部机制 |
| `sleeper_eval/prompts/document_representation/` | 提供商文档渲染研究：审计各提供商如何建模附件文档 |

## 环境搭建

项目依赖 [`uv`](https://github.com/astral-sh/uv) 与 [`inspect-ai`](https://inspect.aisi.org.uk/)，要求 Python 3.11+：

```bash
uv venv
uv pip install -e '.[dev]'
cp .env.example .env
```

然后编辑 `.env`。`.env.example` 列出全部变量，只需填入所用配置涉及的提供商凭证（目标模型，以及外部管理器机制下的 mem0 管理器）：

| 提供商 / 路由 | 所需变量 | 使用方 |
|------------------|---------|--------|
| OpenRouter | `OPENROUTER_API_KEY` | 多数目标模型（Gemini-pro、Kimi）及 OpenRouter 路由的 mem0 管理器 |
| OpenAI（直连） | `OPENAI_API_KEY` | GPT 目标；默认判分器/judge |
| Anthropic（直连） | `ANTHROPIC_API_KEY` | Claude 目标/管理器 |
| Google Gemini（原生 API） | `GOOGLE_API_KEY` | Gemini 目标/管理器（原生路径） |
| DeepSeek（直连） | `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` | DeepSeek 目标与 DeepSeek mem0 管理器 |

Inspect 运行时变量（`INSPECT_LOG_DIR`、`INSPECT_TRACE_FILE`、`HOME`、`INSPECT_EVAL_MAX_CONNECTIONS`）也在 `.env.example` 中。不必猜测配置需要哪些 key：`--dry-run` 预检会精确报告该配置选择的模型缺哪些环境变量。外部管理器 mem0 路径还需 `uv pip install -e '.[mem0_eval]'`。

> **本机实测补充**：本项目实际使用 OpenAI 兼容端点 `https://model.shouxu.tech/v1`（`.env` 中 `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` 指向它），subject 与 grader 统一用 `openai/<model>` 路由。注意：**不要在 Windows 上用 `uv run` 直接跑**（会在默认 PyPI 源上 sync 卡死）；请用本地启动器 `repro_local/run_campaign.py`（自动加载 .env 并打 Windows fsspec 路径补丁）。依赖安装建议用国内镜像（如清华源）加速。

## 准备数据集

行为、代理与多语言数据集以**混合记忆源 bundle** 形式提交在 `datasets/released/`。论文配置消费的"有记忆/无记忆"精确拆分是**本地生成**的——由已提交的源派生，并非直接下发。运行实验前先执行两步构建：

```bash
# 1. 物化论文面向的记忆拆分到 datasets/released/generated/
uv run python datasets/builders/materialize_released_splits.py

# 2. 物化与论文对齐的良性保存切片
uv run python scripts/create_benign_save_dataset.py --paper-true-opt-proportional --seed 42
```

数据文件说明与"如何构建新文档/查询语料"见 [`datasets/README.md`](datasets/README.md)。

> 本机注意：`uv run` 可能卡住，请直接用 `.venv/Scripts/python.exe` 执行上述脚本（等价）。

## 实验分类

每个论文实验对应一个官方配置与一个匹配的 smoke 配置，全部通过统一入口 `scripts/run_eval_campaign.py` 运行：

| 论文实验 | 机制 | 官方配置 | 冒烟配置 |
|------------------|--------|------------------|--------------|
| 主注入评估 | 工具型 | `scripts/configs/paper/tool_based_main.yaml` | `smoke/tool_based_main.smoke.yaml`、`smoke/tool_based_main.limit1.smoke.yaml` |
| 多语言 / 非英语 OOD 注入 | 工具型 | `scripts/configs/paper/tool_based_multilingual.yaml` | `smoke/tool_based_multilingual.smoke.yaml` |
| 主注入评估 | 外部管理器 | `scripts/configs/paper/external_manager_main.yaml` | `smoke/external_manager_main.smoke.yaml` |
| 良性保存保持消融 | 工具型 | `scripts/configs/paper/benign_save_ablation.yaml` | `smoke/benign_save_ablation.smoke.yaml` |
| 提供商风格互换消融 | 工具型 | `scripts/configs/paper/provider_setup_swap_ablation.yaml` | `smoke/provider_setup_swap_ablation.smoke.yaml` |
| 管理器模型消融 | 外部管理器 | `scripts/configs/paper/external_manager_model_ablation/all/*.yaml` | （用受限回放运行，见下） |

提示级防御（朴素提示加固、GEPA 提示加固、极端 spotlighting / 不可信内容标记、加固+spotlighting）不是一个独立配置族——它们在上述配置内用 `defenses` 字段选择。防御提示词工件在 [`prompts/defense/`](prompts/defense/)，GEPA 后缀优化源码在 [`experiments/gepa_defense_optimization/`](experiments/gepa_defense_optimization/)。

## 运行实验

所有战役配置遵循同样的两步契约：先用 `--dry-run` 预览，再 `--yes` 执行：

```bash
# 1. 预检：校验配置并估算成本，不调用任何模型
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_main.yaml --dry-run

# 2. 快速冒烟（小单元格限额）——确认端到端布线
uv run python scripts/run_eval_campaign.py scripts/configs/paper/smoke/tool_based_main.smoke.yaml --yes

# 3. 全量运行
uv run python scripts/run_eval_campaign.py scripts/configs/paper/tool_based_main.yaml --yes
```

工具型主实验另有一个**全矩阵**冒烟配置 `smoke/tool_based_main.limit1.smoke.yaml`：每个 模型×攻击×防御 单元格各测 1 条，适合在全量运行前抓矩阵布线问题。

> 本机注意：把上面的 `uv run python` 全部替换为 `./.venv/Scripts/python.exe repro_local/run_campaign.py`（本地启动器会自动加载 .env、打 Windows 补丁）。已写好并验证过的三个本地配置：`repro_local/configs/tool_based_smoke.yaml`、`tool_based_subset.yaml`、`defense_subset.yaml`。

### 外部管理器（mem0）——走哪条路径？

外部管理器机制分两步：生成**目标模型对话记录（transcripts）**，再把 transcripts **回放（replay）** 到一个或多个记忆管理器。支持三种方式：

| 你的目标 | 使用 | 命令 |
|-----------|------|------|
| 一次跑完整主矩阵 | 标准战役运行器 | `run_eval_campaign.py scripts/configs/paper/external_manager_main.yaml --yes` |
| 单管理器、最干净的端到端 | 单管理器管线 | `run_mem0_replay_pipeline.py scripts/configs/paper/external_manager_model_ablation/all/replay_gemini31flashlite.yaml --output-dir logs/extmgr_replay_gemini31flashlite --score-semantic --yes` |
| 共享 transcripts、对比多个管理器 | 套件管线 | `run_mem0_replay_suite_pipeline.py scripts/configs/paper/external_manager_model_ablation/all/subject_transcript.yaml --output-dir logs/extmgr_suite --score-semantic --max-parallel-managers 3 --yes` |

## 提示词工件

仓库有两棵提示词树，区分是有意的：

- **`prompts/`** 是**稳定的提示词附件面**：论文附录引用的每个提示词都冻结在这里，读者能找到论文所指的确切内容。
- **`sleeper_eval/prompts/`** 是**运行时提示词区**，供代码使用，可能随实现演进，不是引用面。其中的 `document_representation/` 子目录值得注意：存放实证的提供商研究（Claude/GPT/Gemini 各两轮），支撑了框架对每家提供商文档渲染的选择——16k 字符长文档阈值、Claude `antml:` 命名空间、csv/xlsx/xml 排除等。

编辑时请保持这个区分：不要把论文附件引用指向运行时文件，也不要把运行时提示词当作发布工件。

## 输出与分析

战役配置默认把生成的日志、bundle 与分析写到 `.artifacts/`（通常 `.artifacts/paper/<实验族>/`）；mem0 回放管线写工作目录到 `logs/`。源码目录始终保持纯源码。

保留的公开"无写入失败分类器"从 `.eval` 日志重算确定性的 成功/错误写入/未写入 结果，并把 `no_write` 行细分为 `refused`、`ignored`、`accepted_but_not_stored`、`ambiguous`：

```bash
uv run python scripts/analysis/report_injection_failure_types.py --help
```

默认 judge 模型是 `gpt-5.4-nano`，其他辅助脚本见 [`scripts/analysis/README.md`](scripts/analysis/README.md) 与 [`scripts/support/README.md`](scripts/support/README.md)。

## 当前范围（未收录部分）

本发布未作为一等模块收录的论文邻接组件：

- LLM 扫描器工件
- 激活探测 / 机制分析工件
- 保留注入工作流之外的措辞敏感性消融资源

位置消融所需注入框架已保留，但没有独立的消融模块。

## 引用

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

## 许可

本项目的代码、提示词、对抗目标与基准标注在此发布。部分捆绑的评测数据派生自自带许可或使用条款的第三方语料；上游条款继续约束底层源材料及数据集中任何源派生部分。

---

*本地复现详情（环境、命令、进度、结果）见 [`repro_local/REPRO_GUIDE.md`](repro_local/REPRO_GUIDE.md)，复现结果分析见 [`repro_local/结果分析.md`](repro_local/结果分析.md)，代码讲解见 [`repro_local/代码说明.md`](repro_local/代码说明.md)。*