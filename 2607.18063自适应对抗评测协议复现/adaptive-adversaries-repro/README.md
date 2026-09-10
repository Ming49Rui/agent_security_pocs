# Adaptive Adversaries 轻量复现 (repro-lite)
基于自适应对抗评测协议的复现
对论文《Adaptive Adversaries: A Multi-Turn, Multi-LLM Benchmark for LLM Agent Security》
(arXiv:2607.18063v1, Lambda) 的轻量离线复现。

**目标**: 复现论文的核心*协议*与*定性结论*, 不追求逐位复现真实模型得分。
零 API 成本, 纯 Python 标准库, 固定随机种子 (SEED=42) 完全可重现。

## 复现了论文的哪些内容

| 论文内容 | 本仓库 |
|---|---|
| §3 协议: 15 轮 battle, 攻击者有历史, 防御者无记忆, 结构化 JSON, 型字段判定 | `core.py` BattleOrchestrator |
| §D.2 场景插件接口 (role / goal / template / schema / check_success) | `core.py` ScenarioPlugin + `scenarios.py` |
| §5.3 首轮 vs 15 轮 ASR (0–1% vs 5.4–14%) | 实验 E1 |
| §5.1 聚合并列 vs 场景级反排 (memleak/paperaccept/gameguru) | 实验 E2 |
| §5.2 多攻击者池化 1.4–2.2× 唯一攻击 | 实验 E3 |
| B.14/B.15 代理指标 reward hacking (naive substring vs 型字段) | 实验 E4 |
| §5.3/B.12 自适应性消融 (non-adaptive→0, stateful 降幅有限) | 实验 E5 |

## 文件

```
core.py         协议核心: LLM 接口 / 场景插件接口 / 15 轮 orchestrator / 统计
scenarios.py    7 个代表性场景插件 (含 canary 与 check_success)
sim_agents.py   模拟攻击者/防御者: 机制编码 + 场景易感度表
evaluate.py     5 组实验 + 输出对照结果 (python evaluate.py)
```

## 运行

```bash
cd adaptive-adversaries-repro
python evaluate.py
```

无需安装任何依赖 (>= Python 3.8)。

## 与论文的可复现性边界

- 模拟出的"模型"是**机制代理**, 不是真实 LLM: 防御者用固定概率在
  pretext/authority 两个机制族上泄漏, 概率来自论文记录的行为模式, 非真实模型输出。
- 场景与 canary 值与论文一致 (memleak 的 `K7$mF9!qZ2nV%xR3` 及其 percent-encoded
  形式; smarthomejack 的 `actions_pending` 不计入; tickettriage 的回声不算)。
- 若要接入真实 LLM: 继承 `core.LLM` 实现 `respond()`, 替换 `sim_agents` 里的
  SimAttacker/SimDefender 即可, orchestrator 与场景插件无需改动。
