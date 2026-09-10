# OR-Bench 轻量化复现 (or-bench-repro)

对论文《OR-Bench: An Over-Refusal Benchmark for Large Language Models》
(arXiv:2405.20947v2, UCLA & UC Berkeley) 的轻量化复现。

复现内容: 论文 §3 的**数据生成管线**（有毒种子生成 → 改写为"看似有毒"的良性 prompt →
三模型集成评审 + 响应校验）与 §4 的**评测协议**（6 个跨家族模型、关键词匹配 + LLM 裁判、
安全-过拒绝权衡分析、system prompt 消融）。规模从论文的 8 万条缩到 129 条安全集。

## 目录

```
config.py            配置: API 端点 / 角色分工 / 模型池 / 规模
llm.py               OpenAI 兼容客户端 (绕过系统代理, 重试, 并发, [[ ]] 解析)
pipeline.py          数据管线: 种子生成 / 改写(论文 A.9 prompt + A.13 few-shot) / 评审(论文 A.7)
run_pipeline.py      执行管线 (带缓存, 断点续跑)
evaluate.py          评测: 关键词匹配(论文 A.10) / LLM 裁判(论文 A.11) / Spearman
analyze.py           采集回复 + 指标 + 消融 + report.md + scatter.html
datasets/            中间产物 (seeds / rewritten / moderation / responses / judges)
results/             最终结果 (metrics.json / report.md / scatter.html)
```

## 模型角色分配（对应论文）

| 角色 | 论文 | 本复现 | 说明 |
|---|---|---|---|
| 种子生成/改写 | Mixtral 8x7B | deepseek-v4-flash | 需无安全限制的模型; 编译过 GLM 系会拒绝 |
| 响应校验答题 | Mistral-7B-Instruct-v0.3 | deepseek-v4-flash | 同角色的替代 |
| 评审团 (×3 集成) | GPT-4-turbo / Llama-3-70b / Gemini-1.5-pro | glm-5.2 / qwen3.8-max / deepseek-v4-pro | 跨家族多数投票 |
| 被评测模型 (×6) | 25 个型号 | deepseek-flash / deepseek-pro / glm-fast / glm-5.2 / qwen3.7-flash / MiniMax-M2.1 | 跨家族代表 |
| LLM 裁判 | GPT-4 | qwen3.8-max | A.11 三类判定 |

## 与论文的可复现性边界

- **规模**: 论文每类 2000 种子 → 8 万条; 本复现每类 5 种子 × 5 改写 → 245 候选 → 129 安全。
- **评审模型**: 论文评审团里有 3 个商业模型, 本复现用开放代理上的跨家族替代, 集成思想一致。
- **结果性质**: 模型池不同, 具体数值不与论文直接可比; 复现验证的是**现象与方向**
  （安全↔过拒绝强相关、system prompt 加剧过拒绝、judge 与关键词差异小）。

## 关键发现（详见 results/report.md）

- Spearman(安全拒绝率, 过拒绝率) = 0.609 (论文 0.878): 相关方向一致——越安全的模型越过度拒绝。
  GLM 家族 "Claude 式" (最安全+最过拒绝), DeepSeek 家族 "Mistral 式" (最少过拒绝+最不安全)。
- 管线过滤有效: 245 改写候选 → 129 安全 / 116 有毒。
- 加 Llama2 式 system prompt: deepseek 双升(同论文 fig.6b), GLM 双降(其静默过滤被 "helpful" 指令抑制)。
- 关键词匹配 vs LLM 裁判: 非 GLM 模型差异 2.5%~10%(论文 1.2%~2.4% 量级), GLM 因空回复高达 22.5%。

## 运行

```bash
python run_pipeline.py   # 数据管线 (缓存: 已有产物自动跳过)
python analyze.py        # 评测 + 分析 + 报告
```

注意: API Key/端点来自本机配置 (config.py); 调用约需 2-4 千次轻量 API 请求。