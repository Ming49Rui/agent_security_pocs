# -*- coding: utf-8 -*-
"""
main.py — GhostWriter + AM-Sentry 复现实验编排
================================================
用法:
  python main.py            # 完整模式: 三组实验全部跑(攻击/防御/效用)
  python main.py --quick    # 快速: 场景减半、重复 1 次、防御/效用配置精简
  python main.py --tiny     # 超快速: 快照少 7 事件 + 代表性配置, 几十分钟出全链路
  python main.py --attack   # 只跑攻击实验(无防御下的 P1/P2)
  python main.py --defense  # 只跑防御实验(AM-Sentry 各配置 vs 攻击)
  python main.py --utility  # 只跑效用实验(AM-Sentry 对 Agent 效用的影响)

实验流程(与论文 §7 对齐):
  1) 准备工作周快照: 每个 Agent 消化工作周事件 → 存干净记忆快照
  2) 攻击实验: 每个场景 注入(P1) → 激活(P2),  统计注入率/检索率/激活率
  3) 防御实验: 各 AM-Sentry 配置(S1/S2/S3 × 检索屏 R ± A-MAC 基线)
     下重新生成快照, 重放攻击, 统计端到端攻击成功率
  4) 效用实验: 干净快照上跑回回忆/工具任务, 对比 baseline 与各配置的
     F1 / LLM裁判分 / 工具准确率
结果写入 results/: attack.json / defense.json / utility.json
"""
import argparse
import json
import os
import statistics

import config
import corpus
from agents.fact_agent import FactAgent
from agents.summary_agent import SummaryAgent
from attack import ghostwriter
from defense.am_sentry import AMSentry
from eval import utility as util

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 论文是 8 场景 × 2 变体 × 5 次重复; 默认 2 次重复(轻量版)
REPEATS = 2
# 场景全集(8 个 = 4 类目标 × 2 种投递)
ALL_SCENARIOS = corpus.scenario_ids()

# 防御配置集(完整): (策略名, 是否启用检索屏) — 论文图 8/9 同款对比,
# 3 策略 × 2 屏设置 + A-MAC 基线 + 无防御
DEFENSE_CONFIGS = [
    ("S1", False), ("S2", False), ("S3", False),
    ("S1", True), ("S2", True), ("S3", True),
    ("AMAC", False), (None, False),
]
# 防御配置集(--tiny): 只保留最关键的对照(无防御 / S2 准入 / S3 / S3+R)
DEFENSE_CONFIGS_TINY = [
    (None, False),          # 无防御对照(等同攻击实验基线)
    ("S2", False),          # 中等准入策略
    ("S3", False),          # 最强准入
    ("S3", True),           # 最强组合 S3 + 检索屏
]
# 效用配置(完整): baseline / A-MAC / S1..S3 / S3+R
UTILITY_CONFIGS = [("baseline", None, None), ("A-MAC", "AMAC", False),
                   ("S1", "S1", False), ("S2", "S2", False),
                   ("S3", "S3", False), ("S3+R", "S3", True)]
# 效用配置(--tiny): baseline 与最强防御的效用对比
UTILITY_CONFIGS_TINY = [("baseline", None, None), ("S3+R", "S3", True)]


def make_agents(defense=None):
    """构建两个代表性 Agent(事实型 / 摘要型), 可挂同一个防御。"""
    return [FactAgent(defense), SummaryAgent(defense)]


def build_snapshot(agent, n_events: int = None) -> list:
    """论文 §6.2: 让 Agent 消化工作周事件, 返回干净记忆快照。

    n_events=None 用完整 14 条事件; --tiny 模式用前 7 条加速。
    """
    agent.reset()
    events = corpus.WORKWEEK if n_events is None else corpus.WORKWEEK[:n_events]
    for ev in events:
        agent.ingest(ev)
    return agent.save_snapshot()


def load_clean(n_events: int = None) -> dict:
    """为两类 Agent 各建一份干净快照(无防御), 返回 {name: snapshot}。"""
    snaps = {}
    for ag in make_agents():
        snaps[ag.name] = build_snapshot(ag, n_events)
    return snaps


# ---------------------------------------------------------------------------
# 1) 攻击实验(无防御): 注入率 / 检索率 / 激活率
# ---------------------------------------------------------------------------

def run_attack(mode: str) -> dict:
    print("\n[1/3] 攻击实验 —— GhostWriter (无防御, 论文 §7.2)")
    n_events = 7 if mode == "tiny" else None
    snaps = load_clean(n_events)
    n_scen = {"tiny": 2, "quick": 4}.get(mode, len(ALL_SCENARIOS))
    scenarios = ALL_SCENARIOS[:n_scen]
    repeats = {"tiny": 1, "quick": 1}.get(mode, REPEATS)
    results = {}
    for ag in make_agents():
        rows = []
        for sid in scenarios:
            sc = corpus.get_scenario(sid)
            for variant in ("directive", "descriptive"):
                for _ in range(repeats):
                    ag.reset()
                    ag.load_snapshot(snaps[ag.name])
                    r = ghostwriter.run_scenario(ag, sc, variant)
                    ag.reset()
                    rows.append(r)
        results[ag.name] = summarize_rows(rows)
        print(f"  [{ag.name}] {results[ag.name]}")
    with open(os.path.join(RESULTS, "attack.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": scenarios, "repeats": repeats,
                   "results": results}, f, ensure_ascii=False, indent=2)
    return results


def summarize_rows(rows: list) -> dict:
    """rows: [{'scenario','variant','p1','p2_retrieved','p2_activated'}, ...]"""
    n = len(rows) or 1
    return {"injection_rate": round(sum(r["p1"] for r in rows) / n, 3),
            "retrieval_rate": round(sum(r["p2_retrieved"] for r in rows) / n, 3),
            "activation_rate": round(sum(r["p2_activated"] for r in rows) / n, 3),
            "n": n}


# ---------------------------------------------------------------------------
# 2) 防御实验: 各 AM-Sentry 配置下的端到端攻击成功率
# ---------------------------------------------------------------------------

def run_defense(mode: str) -> dict:
    print("\n[2/3] 防御实验 —— AM-Sentry 各配置 (论文 §7.4)")
    n_scen = {"tiny": 2, "quick": 4}.get(mode, len(ALL_SCENARIOS))
    scenarios = ALL_SCENARIOS[:n_scen]
    configs = DEFENSE_CONFIGS_TINY if mode == "tiny" else DEFENSE_CONFIGS
    n_events = 7 if mode == "tiny" else None
    # 论文 §7.4: 聚焦最难检测的 descriptive 载荷
    variant = "descriptive"
    out = {}
    for policy, with_screen in configs:
        policy_name = policy or "none"
        key = f"{policy_name} + screen:{int(with_screen)}"
        defense = AMSentry(policy=policy, with_screen=with_screen)
        per_agent = {}
        for ag in make_agents(defense):
            # 论文 §7.4: 每个配置重新生成快照(策略影响哪些记忆入库)
            snap = build_snapshot(ag, n_events)
            # 攻击端到端成功率: 注入成功 且 激活成功
            succ = []
            for sid in scenarios:
                sc = corpus.get_scenario(sid)
                ag.reset()
                ag.load_snapshot(snap)
                r = ghostwriter.run_scenario(ag, sc, variant)
                succ.append(1 if (r["p1"] and r["p2_activated"]) else 0)
                ag.reset()
            per_agent[ag.name] = {"attack_success_rate":
                                  round(sum(succ) / len(succ), 3)}
        attack_rates = [v["attack_success_rate"] for v in per_agent.values()]
        out[key] = {
            "per_agent": per_agent,
            "avg_attack_success_rate": round(
                sum(attack_rates) / len(attack_rates), 3),
        }
        print(f"  [{key}] avg attack success = "
              f"{out[key]['avg_attack_success_rate']}")
    with open(os.path.join(RESULTS, "defense.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ---------------------------------------------------------------------------
# 3) 效用实验: AM-Sentry 对 Agent 基础效能的影响(论文 §7.5)
# ---------------------------------------------------------------------------

def run_utility(mode: str) -> dict:
    print("\n[3/3] 效用实验 —— AM-Sentry 对效用的影响 (论文 §7.5)")
    configs = UTILITY_CONFIGS_TINY if mode == "tiny" else UTILITY_CONFIGS
    n_events = 7 if mode == "tiny" else None
    out = {}
    for label, policy, with_screen in configs:
        defense = AMSentry(policy=policy, with_screen=with_screen) \
            if policy else None
        per_agent = {}
        for ag in make_agents(defense):
            snap = build_snapshot(ag, n_events)
            ag.reset()
            ag.load_snapshot(snap)
            per_agent[ag.name] = util.run_utility_suite(ag)
            print(f"  [{label}/{ag.name}] {per_agent[ag.name]}")
        out[label] = per_agent
    with open(os.path.join(RESULTS, "utility.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ---------------------------------------------------------------------------
# 汇总打印(对照论文数字)
# ---------------------------------------------------------------------------

def print_summary(attack, defense, utility, mode: str) -> None:
    print("\n" + "=" * 62)
    print(f"复现结果汇总 (模型: {config.MODEL}, 模式: {mode})")
    print("=" * 62)
    print("\nGhostWriter 攻击(论文: 注入≈98% / 检索≈94% / 激活≈60%):")
    for name, v in attack.items():
        print(f"  {name:<9} 注入 {v['injection_rate']*100:5.1f}% | "
              f"检索 {v['retrieval_rate']*100:5.1f}% | "
              f"激活 {v['activation_rate']*100:5.1f}%")
    print("\nAM-Sentry 端到端攻击成功率(论文: S3 平均≈15%, S3+R <12%):")
    for key, v in defense.items():
        print(f"  {key:<20} {v['avg_attack_success_rate']*100:5.1f}%")
    print("\n效用(论文: S3 相比 baseline 掉 0.01~0.04):")
    for label, per in utility.items():
        f1m = statistics.mean([x["f1_mean"] for x in per.values()])
        print(f"  {label:<8} 平均 F1 = {f1m:.3f}")
    print("\n结果文件: results/attack.json, defense.json, utility.json")
    print(f"(注: {mode} 模式, 实验规模与论文不同: 2 种记忆架构、"
          f"重复次数 {('1' if mode != 'full' else REPEATS)} 等)")


def main() -> None:
    parser = argparse.ArgumentParser(description="GhostWriter 复现实验")
    parser.add_argument("--attack", action="store_true", help="只跑攻击实验")
    parser.add_argument("--defense", action="store_true", help="只跑防御实验")
    parser.add_argument("--utility", action="store_true", help="只跑效用实验")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 减场景/重复次数")
    parser.add_argument("--tiny", action="store_true",
                        help="超快速模式: 精简快照 + 代表性配置")
    args = parser.parse_args()

    config.check()
    os.makedirs(RESULTS, exist_ok=True)

    mode = "tiny" if args.tiny else ("quick" if args.quick else "full")
    only = args.attack or args.defense or args.utility
    attack = run_attack(mode) if (args.attack or not only) else None
    defense = run_defense(mode) if (args.defense or not only) else None
    utility = run_utility(mode) if (args.utility or not only) else None
    if not only:
        print_summary(attack, defense, utility, mode)


if __name__ == "__main__":
    main()