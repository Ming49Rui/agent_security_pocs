# ActorBreaker 轻量化复现

复现论文: **LLMs know their vulnerabilities: Uncover Safety Gaps through Natural
Distribution Shifts** (arXiv:2410.10700, ACL 2025) —— 一个基于"行动者网络理论"的
多轮 LLM 越狱攻击与防御研究。

> ⚠️ **使用目的与安全**: 本项目仅供安全研究与教育用途, 复现论文方法用于理解
> "自然分布偏移" 这一安全漏洞机理及防御思路。报告中的模型回复按论文惯例截断,
> 不展示完整有害内容。请勿用于攻击线上服务。

## 复现了什么

论文方法 ActorBreaker 的核心管线（Algorithm 1 精简版）:

```
有害目标 x
  │
  ├─[1] 行动者网络构建 G_concept → G_inst (两层树, 六类行动者×人/非人)
  │       每片叶子 = (类型, 行动者名, 与x的关系描述) = 攻击线索 c_i
  │
  ├─[2] 采样 K 条线索 (跨类型)
  │
  └─ 对每条线索:
       ├─[3] 推断攻击链 z1..zn (话题逐步逼近x的演化路径, JSON)
       ├─[4] self-talk 自对话生成初始问题集 [q1..qN]
       │        (攻击者LLM 自我问答, 预测受害者代理响应)
       └─[5] 对受害者模型重放 q1..qN (真实多轮对话)
             每轮用裁判LLM给回复打 1-5 分, 得 5 分 → 攻击成功, early stop
```

## 与论文的差异（轻量化取舍）

| 项 | 论文 | 本复现 |
|---|---|---|
| 攻击者/裁判模型 | GPT-4o | qwen-max (经"首序"网关) |
| 受害者模型 | GPT-3.5/4o/o1, Claude-3.5, Llama-3-8B/70B | deepseek-v4-flash-0731 |
| 数据集 | HarmBench 完整 | 4 条示例目标 (取自论文 Fig.9-13) |
| 每条目标线索数 K | 3 | 2 (可配置) |
| 每轮对话上限 | 5 | 4 (可配置) |
| 动态修改(拒绝重写) | 有 (Table 4: 81.2%) | **省略** (no-DM 为 72.7%) |
| 多样性度量 | MiniLM 嵌入余弦相似度 | 省略 |
| 防御侧(安全微调) | SFT/CircuitBreaker 微调 | 省略 |
| 超参 | 攻击模型 temp=1, victim temp=0 | 同论文 |

> 注意: 因模型/数据不同, 得到的 ASR 数值不与论文可比, 本项目的目的是
> **机制层面的功能复现**, 验证"良性问题 + 多轮引导 → 绕过安全机制"这一通路.

## 运行

```bash
python run.py            # 全部 4 条目标
python run.py --smoke    # 快速验证管线 (1 条目标, 1 线索, 2 轮)
python run.py --query 0 --actors 3 --turns 5
```

依赖: 仅 Python 3.8+ 标准库 + requests (可选)。模型凭据自动从 ZCode 配置
(`~/.zcode/v2/config.json`) 读取, 密钥不会写入本目录任何文件。

## 输出

- `outputs/report.md` —— 汇总报告 (行动者网络/攻击链/问题序列/裁判分/ASR)
- `outputs/query_<id>.json` —— 每条目标的结构化完整结果

## 运行结果（2026-09-11 复现日志）

配置: 4 条目标 × K=2 线索 × 最多 5 轮, 攻击者/裁判=qwen-max, 受害者=deepseek-v4-flash-0731。

| 指标 | 结果 |
|---|---|
| 流程完整性 | ✅ 六类行动者网络/攻击链/self-talk/受害者重放/裁判打分全部跑通, 无中断 |
| 攻击链质量 | ✅ 话题按论文模式逐步逼近目标 (如: 受害者→伤害类型→装置材料→组装) |
| self-talk 追问风格 | ✅ 复刻论文 "Expanding on your previous response, please provide more detail..." |
| 裁判分分布 | 1×14 / 2×5 / 3×5, 最高 3 |
| ASR (5分算成功, 论文口径) | **0% (0/4 目标)** |
| 裁判校准 | ✅ 合成详细有害回复→5, 无害拒绝→1, 指标可信 |

**解读**: 机制全部复现成功; 本机可用的受害者模型对齐良好——多轮攻击能把对话
推进到"可执行细节"的临界点（裁判 3 分），但受害者在临界点明确拒绝，未被攻破。
这与论文 Table 1 观察一致: ASR 随受害者模型差异极大（如 GCG 在 GPT-o1 上为 0%）。
若换用论文原受害者集合（GPT-3.5 / Llama-3 等）或启用动态修改模块，成功率会显著不同。
临时提取物已清理，目录仅保留项目文件。

## 文件

| 文件 | 作用 |
|---|---|
| `ab_llm.py` | 极简 Anthropic-Messages HTTP 客户端 + JSON 解析 |
| `ab_prompts.py` | 论文提示词模板 (Fig.2 概念网络 / 攻击链 / self-talk / 裁判) |
| `actorbreaker.py` | 核心管线 (Algorithm 1 精简) + 角色路由 |
| `run.py` | 主入口, 并发执行 + 报告生成 |
| `config.json` | 模型角色 / 超参配置 |
| `test_queries.json` | 测试目标 |