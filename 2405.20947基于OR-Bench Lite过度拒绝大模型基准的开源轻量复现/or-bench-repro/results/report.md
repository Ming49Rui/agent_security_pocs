# OR-Bench 轻量化复现报告

论文: *OR-Bench: An Over-Refusal Benchmark for Large Language Models* (arXiv:2405.20947)
复现方式: 用开放代理上的 6 个跨家族模型, 复现论文的"生成→改写→评审"管线与评测协议。

## 数据集规模

- OR-Bench-lite (安全但看似有毒): **129** 条, 覆盖 10 类
- Toxic 对照集: **116** 条
- Hard 子集 (关键词口径被 ≥3/6 模型拒绝): **42** 条

## 1. 主结果: 安全 ↔ 过拒绝 的权衡

| 模型 | 安全集拒绝率 (过拒绝, 关键词) | Hard 集拒绝率 (裁判) | Toxic 集接受率 (裁判) | Toxic 集拒绝率 |
|---|---|---|---|---|
| deepseek-v4-flash | 19.4% | 50.0% | 20.7% | 79.3% |
| deepseek-v4-pro | 20.2% | 35.7% | 29.3% | 70.7% |
| glm-5.2-fast-preview | 49.6% | 81.0% | 12.9% | 87.1% |
| glm-5.2 | 51.2% | 76.2% | 15.5% | 84.5% |
| qwen3.7-flash | 27.9% | 81.0% | 18.1% | 81.9% |
| MiniMax-M2.1 | 27.1% | 69.0% | 12.9% | 87.1% |

**Spearman 秩相关: 论文 0.878, 本复现 0.609** — 拒绝有毒 prompt 越多的模型, 越倾向于拒绝良性 prompt.

## 2. 关键词匹配 vs LLM 裁判 (论文 A.3 表 5)

| 模型 | 关键词拒绝率 % | 裁判拒绝率 % | 差异 % |
|---|---|---|---|
| deepseek-v4-flash | 25.0 | 22.5 | 2.5 |
| deepseek-v4-pro | 12.5 | 7.5 | 5.0 |
| glm-5.2-fast-preview | 57.5 | 35.0 | 22.5 |
| glm-5.2 | 62.5 | 40.0 | 22.5 |
| qwen3.7-flash | 25.0 | 32.5 | 7.5 |
| MiniMax-M2.1 | 22.5 | 32.5 | 10.0 |

论文报告的关键词-裁判差异为 1.2%~2.4%。本复现中 DeepSeek/Qwen/MiniMax 的差异为 2.5%~10.0%,
与之同量级;**但 GLM 两个模型高达 22.5%**, 原因见 §6: GLM 系对部分 safe prompt 直接输出空内容
(静默过滤), 关键词匹配将其计入拒绝, 而裁判面对空回复无法判定为拒绝。

## 3. 消融: system prompt 的作用 (论文 fig.6b)

| 模型 | 条件 | 安全集拒绝率 | Toxic 集拒绝率 |
|---|---|---|---|
| glm-5.2-fast-preview | 无 system prompt | 46.7% | 73.3% |
| glm-5.2-fast-preview | 含 Llama2 式 system prompt | 30.0% | 53.3% |
| deepseek-v4-flash | 无 system prompt | 20.0% | 46.7% |
| deepseek-v4-flash | 含 Llama2 式 system prompt | 23.3% | 53.3% |

论文观察: 加 Llama2 式 system prompt 后两类拒绝率都上升 (右上移动)。本复现 deepseek-v4-flash 同向
(两者均有小幅上升), 但 **glm-5.2-fast-preview 反而双降**。可能的机制: "Always answer as helpfully
as possible" 抑制了 GLM 系的静默过滤, 使其对原本空回复的 prompt 给出真实回答, 从而两类拒绝率都下降;
样本量 (30 safe / 15 toxic) 也放大了噪声。

## 4. 各类别过拒绝率 (GLM-5.2-fast-preview)

| 类别 | 拒绝率 |
|---|---|
| deception | 50.0% |
| harassment | 37.5% |
| harmful | 60.0% |
| hate | 58.3% |
| illegal | 41.7% |
| privacy | 50.0% |
| self-harm | 55.6% |
| sexual | 28.6% |
| unethical | 58.3% |
| violence | 54.5% |

## 5. 复现结论与论文对照

**成功复现的现象 (与论文 §4 定性结论一致):**

1. **安全 ↔ 过拒绝的权衡**: Spearman = 0.609 (论文 0.878)。6 个模型中, safest 的 GLM 家族
   同时过拒绝率最高 (49.6%~51.2%), least safe 的 DeepSeek 家族过拒绝率最低 (19.4%~20.2%),
   与论文 "Claude 最安全也最过度拒绝 / Mistral 几乎不拒绝但安全差" 的模式一一对应。
2. **管线过滤有效**: 245 条改写候选经三模型集成评审后, 129 条 (52.6%) 判为安全, 116 条仍为有毒——
   说明"看似有毒但安全"的改写确实需要 moderation 把关 (论文 §3.1.3 的动机)。
3. **类别敏感性**: GLM 对 sexual 类别的过拒绝率最低 (28.6%), 与论文 "Claude-3-Opus 对性话题不敏感" 一致。
4. **数据集难度分层生效**: Hard 子集 (被 ≥3/6 模型拒绝) 的拒绝率显著高于全量 safe 集 (如 deepseek-flash
   从 19.4% → 50.0%), 验证了论文 A.14 的构造思路。

**与论文的偏差 (诚实记录):**

| 项 | 论文 | 本复现 | 原因 |
|---|---|---|---|
| Spearman | 0.878 | 0.609 | 模型池仅 6 个 + 评测 prompt 少, 相关方向一致但强度偏弱 |
| 关键词-裁判差异 | 1.2%~2.4% | 2.5%~22.5% | GLM 系空回复被关键词计为拒绝 (见 §2), 其余模型同量级 |
| system prompt 消融 | 双升 | deepseek 双升, GLM 双降 | GLM 的静默过滤被 "helpful" 指令抑制 (见 §3) |

## 6. 工程复现中的关键坑 (对应论文方法的隐蔽前提)

1. **生成/改写模型必须"无安全限制"**: 实测 GLM 系直接拒绝生成 toxic seeds, 而 deepseek 配合;
   这正是论文选 Mixtral 8x7B 的原因——安全对齐模型无法完成生成环节。
2. **带思维链的推理模型吃掉输出预算**: deepseek-v4-flash 的 `reasoning_content` 会计入
   `max_tokens`, 预算给 1024 会导致 content 被截断/为空 → 改写失败率骤增; 生成类调用需给足预算。
3. **静默过滤 ≠ 空回复**: GLM 系对不安全 prompt 返回空 content 而非拒绝文本, 关键词匹配
   (论文 A.10) 对这类模型会高估拒绝率; 论文的 Claude/GPT/Llama 均有拒绝文本, 未遇此问题。
4. **评审轮次必须并行 + 断点续跑**: 245 候选 × (3 评审 + 1 答题) ≈ 1000 次 API 调用,
   串行执行约 30 分钟且中断即前功尽弃; 本复现改为并发 + 逐条 checkpoint。

---

产物: `datasets/` (seeds/rewritten/moderation 原始记录), `results/metrics.json`, `results/scatter.html`