# -*- coding: utf-8 -*-
"""三组对比实验 + 消融：
  baseline     : 系统提示 + 恶意目标原文（论文中"被 s 拦截的任务"）
  direct_inject: 系统提示 + 显式覆盖型注入指令（集中式显式注入）
  eclipse      : 合成 prompt(SATS) + SWE 工具描述 + DTC 逐轮纠偏
  prompt_only  : 仅合成 prompt（无 SWE 无 DTC）—— 消融 Baseline
  swe_only     : 合成 prompt + SWE，无 DTC
  dtc_only     : 合成 prompt + DTC，无 SWE
输出 JSON + 控制台摘要；每条轨迹保留 transcript 供回看。
"""
import argparse
import copy
import json
import os
import sys

from agent import ReActAgent
from config import ATTACK_GOAL
from eclipse import DIRECT_INJECT_PROMPT, DTCMonitor, SYNTHETIC_PROMPT
from llm import LLMClient
from tools_env import ToolEnvironment


def run_once(llm, group, seed_env=None, verbose=False):
    """跑单次实验，返回 (result, env_snapshot)。
    group: baseline | direct_inject | eclipse | prompt_only | swe_only | dtc_only
      - eclipse    : 合成 prompt + SWE 描述 + DTC（完整攻击）
      - prompt_only: 仅合成 prompt（无 SWE 无 DTC）
      - swe_only   : 合成 prompt + SWE 描述，无 DTC
      - dtc_only   : 合成 prompt，无 SWE 描述，有 DTC
    """
    env = seed_env if seed_env is not None else ToolEnvironment()
    dtc_enabled = group in ("eclipse", "dtc_only")
    swe_enabled = group in ("eclipse", "swe_only")
    recorder = DTCMonitor(verbose=verbose) if dtc_enabled else None
    agent = ReActAgent(llm=llm, env=env, swe=swe_enabled, recorder=recorder)

    if group == "baseline":
        prompt = ATTACK_GOAL
    elif group == "direct_inject":
        prompt = DIRECT_INJECT_PROMPT
    else:
        prompt = SYNTHETIC_PROMPT

    result = agent.run(prompt, allow_patches=True)
    result["group"] = group
    if recorder is not None:
        result["dtc"] = recorder.summary()
    return result, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1, help="每组的试验次数")
    ap.add_argument("--groups", nargs="*", default=["baseline", "direct_inject", "eclipse"],
                    help="要跑的组 (baseline|direct_inject|eclipse|swe_only|dtc_only)")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    llm = LLMClient()
    all_results = []
    for group in args.groups:
        for t in range(1, args.trials + 1):
            print(f"--- [{group}] trial {t}/{args.trials} ---", flush=True)
            res, env = run_once(llm, group, verbose=args.verbose)
            ok = res["fa_achieved"]
            seq = " -> ".join(res["executed"]) if res["executed"] else "(none)"
            print(f"  result: fa_achieved={ok} reason={res['reason']} calls={res['num_calls']}")
            print(f"  executed: {seq}", flush=True)
            # 精简保存：transcript 保留但截断大输出
            slim = copy.deepcopy(res)
            for e in slim.get("transcript", []):
                if "output" in e and len(e["output"]) > 500:
                    e["output"] = e["output"][:500] + "..."
            all_results.append(slim)
            # 每轮独立环境
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()