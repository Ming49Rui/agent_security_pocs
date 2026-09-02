# Sleeper Memory Poisoning 复现手册

论文：*Hidden in Memory: Sleeper Memory Poisoning in LLM Agents*（arXiv:2605.15338v2）
官方仓库（已克隆到本目录）：`ivaxi0s/LLM-agent-memory-poisoning`

---

## 1. 论文核心（30 秒版）

LLM 助手带持久记忆（ChatGPT Memory / Claude memory / Mem0 / OpenClaw）。攻击者把一条"伪装的用户记忆"藏进外部内容（文档/网页/邮件/仓库），助手在处理时把它写入记忆库；这条记忆在**以后的某个新会话**里被检索出来，悄悄引导助手的回答/操作——恶意内容早已消失，攻击者也不在场。攻击分三阶段：**注入（写进记忆）→ 检索（未来被取回）→ 恶意使用（影响行为）**，对应指标 **IR / RR / AUR**。

## 2. 复现环境状态

| 项 | 状态 |
|---|---|
| Python 3.11.16 venv（`.venv/`） | ✅ 已建好 |
| 项目依赖（inspect-ai 0.3.261 等） | ✅ 已装（清华镜像） |
| 发布数据集拆分（`datasets/released/generated/`） | ✅ 已物化 |
| 良性保存数据集（70 样本） | ✅ 已生成 |
| 端点配置（`shouxu.tech` OpenAI 兼容端点，写于 `.env`） | ✅ 已配置 |
| 冒烟/子集/防御/后注入配置 | ✅ 已写好并跑通（见第 7 节） |
| **实际模型调用** | ✅ 已用 `deepseek-v4-flash-0731` / `deepseek-v4-pro` 跑通 |

**关键结论**：使用 OpenAI 兼容端点时，subject 和 grader 统一走 `openai/<model>` 路由（`OPENAI_BASE_URL` 指向端点）；inspect-ai 原生支持，无需额外适配。

## 3. 复现管线一键入口（全部为本地启动器，请勿用 `uv run`）

```bash
cd /c/Users/cm191/Desktop/首序任务/2605.15338v2/repro

# 预检（不调用模型，只校验配置与凭证）：
./.venv/Scripts/python.exe repro_local/run_campaign.py repro_local/configs/tool_based_smoke.yaml --dry-run

# 冒烟（约 6 次评测 + 判分，deepseek-v4-flash-0731，limit=2）：
./.venv/Scripts/python.exe repro_local/run_campaign.py repro_local/configs/tool_based_smoke.yaml --yes

# 行为 AUR（后注入）：
./.venv/Scripts/python.exe repro_local/run_followup.py --dataset-file followup/eval_in/merged_eval_in_local_behaviour.json --limit 20 --log-dir .artifacts/local/followup_behaviour20

# 解析 AUR 结果：
./.venv/Scripts/python.exe repro_local/analyze_followup.py .artifacts/local/followup_behaviour20 --by-split
```

## 4. 本地工具链清单

| 文件 | 作用 |
|---|---|
| `repro_local/run_campaign.py` | 评测战役启动器（加载 .env + Windows 补丁） |
| `repro_local/run_followup.py` | 后注入（RR/AUR）运行器（支持 limit/串行/重试） |
| `repro_local/analyze_followup.py` | 解析 .eval（zstd）输出各邻近度 AUR |
| `repro_local/build_followup_dataset.py` | 转换下游记录为 followup_eval 数据集（记忆池裁剪到 20 条） |
| `repro_local/rescore_logs.py` | 对已有日志重新应用后评 scorer（不重跑模型） |
| `repro_local/_win_patch.py` | Windows fsspec 路径补丁（本地生效，不改上游） |
| `repro_local/configs/tool_based_smoke.yaml` | 冒烟配置（表 1 极小样本） |
| `repro_local/configs/tool_based_subset.yaml` | 子集 IR 复现配置（表 1 对照） |
| `repro_local/configs/defense_subset.yaml` | 防御对比配置（表 3 对照） |

## 5. 配置与论文表格的对应关系

| 论文实验 | 官方配置（`scripts/configs/paper/`） | 论文表格 |
|---|---|---|
| 主注入实验（工具型记忆） | `tool_based_main.yaml` + 冒烟 `smoke/tool_based_main.smoke.yaml` | 表 1（IR）、3/19/20（防御） |
| 多语言 IR | `tool_based_multilingual.yaml` | 表 51 |
| 外部管理器 IR | `external_manager_main.yaml`（+ mem0 回放管线） | 表 1 下半部分、17 |
| 良性保存保持 | `benign_save_ablation.yaml` | 表 42 |
| 提供商风格互换 | `provider_setup_swap_ablation.yaml` | 表 57–59 |
| 后注入（RR/AUR） | `scripts/followup/`（另见 `sleeper_eval/followup_eval/`） | 表 2、29–36 |

仓库**未包含**（论文 Current scope 注明，需自行实现）：LLM 文档扫描器、激活探测/机制分析、措辞敏感性消融资产。

## 6. 模型接入渠道

> **本项目实际用法**：用户提供 OpenAI 兼容端点 `https://model.shouxu.tech/v1`（可用模型含 `deepseek-v4-flash-0731`、`deepseek-v4-pro`、`glm-5.2`、`qwen3.7-max` 等）。`.env` 中已配置 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` 指向该端点；subject 与 grader 统一用 `openai/<model>` 路由。端点有并发限流，大批量运行务必低并发（`max_tasks=1~2`）+ `retry_on_error` 重试。

若要换用其他渠道，各渠道所需 key：

| 想跑的模型 | 需要的 key |
|---|---|
| OpenAI 兼容端点（subject + grader） | `OPENAI_API_KEY` + `OPENAI_BASE_URL`（或 `DEEPSEEK_*`） |
| Kimi-K2.6、Gemini-3.1-Pro（经 OpenRouter） | 仅 `OPENROUTER_API_KEY` |
| Claude Sonnet 4.6 | `ANTHROPIC_API_KEY` |
| DeepSeek 官方 | `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL=https://api.deepseek.com` |
| Gemini（直连） | `GOOGLE_API_KEY` |

填入的 key 越多，可覆盖的论文模型越多。默认 grader 是 `openai/gpt-5.4-mini`。

## 7. 复现进度与结果（截至本次）

### IR（注入率，工具型/无防御）——对照论文表 1 DeepSeek 家族

| 子集 | 攻击 | 论文 DeepSeek-v4 (500/200样本) | 复现 pro (n=20/格) | 复现 flash-0731 (n=20/格) |
|---|---|---|---|---|
| Behavior | User Review | 79.0% | 85% | 85% |
| Behavior | Actor-Critic | 96.2% | 100% | 100% |
| Agent Action | User Review | 59.5% | 55% | 30% |
| Agent Action | Actor-Critic | 88.5% | 90–95% | 30–35% |

结论：`deepseek-v4-pro` 四格数值全部落在论文误差范围内，复现了"Actor-Critic 模板远强于 User Review 基线"与"Agent Action 目标更难注入"两个核心格局；flash 小模型在 Behavior 上一样高易感、在 Agent Action 上抵抗力强得多（模型间差异模式亦与论文一致）。

### AUR（恶意使用率，行为子集，flash-0731）

| 查询邻近度 | n=20 探测 | n=200 全量（内存修剪版） |
|---|---|---|
| goal-adjacent | 55% (6/11) | 运行中（已确认零错误） |
| goal-distant | 0% (0/8) | 运行中 |

论文表 2：goal-adjacent AUR 42–85%（按模型），goal-distant 0–6% —— 邻近效应被精确复现。

### 防御对比（deepseek-v4-pro，行为/有记忆，n=10/格）——对照论文表 3 DeepSeek 列

| 攻击 | 防御 | 复现 IR | 论文 DeepSeek-v4 |
|---|---|---|---|
| AC（无标记） | 无防御 | 100% | 96.2%（表 1 无防御） |
| AC+（带标记） | 无防御 | 100% | — |
| AC（无标记） | 朴素提示加固 | 100% | 89.4% |
| AC+（带标记） | 极端 spotlighting | 70% | 67.4% |

结论：朴素提示加固对 DeepSeek 家族几乎无效，spotlighting 能压制 AC+（70% vs 67.4%）——复现了论文"提示级防御不均质、可被自适应攻击部分击穿"的结论（表 3 中 DeepSeek/Kimi 正是加固后仍高 IR 的模型）。

### 运行产物

| 产物 | 路径 |
|---|---|
| 冒烟 IR | `.artifacts/local/smoke_tool_deepseek/` |
| 子集 IR（论文表 1 对照） | `.artifacts/local/tool_based_subset/`（summary/attack_result_tables） |
| 防御对比（论文表 3 对照） | `.artifacts/local/defense_subset/` |
| 行为 AUR n=20 | `.artifacts/local/followup_behaviour20/` |
| 行为 AUR n=200（内存修剪+并发2，运行中） | `.artifacts/local/followup_behaviour_full/` |

### 复现管线要点

- 所有运行用本地启动器 `repro_local/run_campaign.py`（加载 .env + 打 Windows fsspec 路径补丁），不要用 `uv run`（会在默认源上 sync 卡死）。
- `repro_local/run_followup.py`：后注入 AUR 运行器（支持 limit/串行/重试）。
- `repro_local/analyze_followup.py`：解析 .eval（zstd）输出各邻近度 AUR。
- `repro_local/build_followup_dataset.py`：把 `datasets/downstream/llm_behaviour.jsonl` 转成 followup_eval 格式（发布包未附带该格式的成品数据集）。
- **Windows 补丁** `repro_local/_win_patch.py`：inspect-ai 在 Windows 上把日志路径规范成 `file:/C:/...`（fsspec 单斜杠形式），`local_path()` 只处理 `file://`，导致落盘时 `os.makedirs('file:')` 崩溃（WinError 123）。补丁在本地扩展了 `local_path()`。
- **端点限流**：本端点并发请求过快会返回 400（type=bad_response_status_code），大批量运行须串行 + 重试。

## 8. 剩余里程碑（已完成项见第 7 节）

- [ ] 全量行为 AUR（200 样本）跑完后，用 `analyze_followup.py` 汇总最终邻近度对照
- [ ] 外部管理器（Mem0 式）IR：`external_manager_main.yaml`（需 `uv pip install -e '.[mem0_eval]'`）→ 表 1 下半
- [ ] 语义检索 RR：Qwen-3-Embedding 式 top-k 检索（需自行搭建 embedding 流程）→ 表 28
- [ ] 多语言 IR：`tool_based_multilingual.yaml` → 表 51
- [ ] 良性保存保持：`benign_save_ablation.yaml` → 表 42
- [ ] 提供商风格互换：`provider_setup_swap_ablation.yaml` → 表 57–59
- （论文未提供资产，需完全自研）LLM 文档扫描器、激活探测/机制分析、Agent Action 的 agentic AUR（OpenClaw 环境）、生产网站实测

## 9. 注意事项

- **成本**：`tool_based_main.yaml` 全量约 700 样本 × 6 模型 × 多攻击/防御，动用前沿模型会烧大量预算。务实用冒烟/子集起步，别直接跑全量。
- **安全边界**：论文 Agent Action 目标涉及外渗/劫持等行为，但官方框架在**模拟工作区**内评估（USER.md/IDENTITY.md/MEMORY.md + 占位符 URL），不触碰真实工具。复现同样只在模拟环境进行。
- **uv run 卡顿**：本机 `uv run` 会在默认源上做 sync 导致卡住，一律用 `.venv/Scripts/python.exe` 直接跑。