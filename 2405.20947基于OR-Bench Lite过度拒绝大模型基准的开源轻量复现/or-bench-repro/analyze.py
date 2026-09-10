# -*- coding: utf-8 -*-
# 评测 + 分析 + 报告: 加载管线产物 -> 采集回复 -> 指标 -> 消融 -> report.md + HTML 散点图
import html
import json
import os
import random
import time

from config import SUBJECTS, SAFE_CACHE_DIR, RESULTS_DIR, HARD_MIN_REJECTORS, PARALLEL_WORKERS, MAX_TOKENS
from llm import LLMClient, dump_json, load_json
from evaluate import (
    SYSTEM_PROMPT, is_refusal_keyword, judge_label, ask_subject,
    collect_responses, refusal_stats, spearman,
)

random.seed(42)


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = LLMClient()
    safe = load_json(os.path.join(SAFE_CACHE_DIR, "safe_prompts.json"))
    toxic = load_json(os.path.join(SAFE_CACHE_DIR, "toxic_prompts.json"))
    print(f"[analyze] safe={len(safe)} toxic={len(toxic)}")

    # 1) 采集 6 个被测模型的回复 (safe + toxic, temp 0.0, 无 system prompt)
    store = collect_responses(safe, toxic, SUBJECTS)

    # 2) 安全集: 关键词匹配拒绝率 (过拒绝)
    safe_rej = refusal_stats(store, safe, SUBJECTS)

    # 3) toxic 集: LLM 裁判 (论文表 3 即 LLM 裁判口径)
    print("[analyze] toxic 集 LLM 裁判 (A.11)...")
    judge_cache = os.path.join(SAFE_CACHE_DIR, "judge_toxic.json")
    jtoxic = load_json(judge_cache) if os.path.exists(judge_cache) else {}
    todo = [(m, p) for m in SUBJECTS for p in toxic if f"{m}\x1f{p['prompt']}" not in jtoxic]
    if todo:
        outs = client.map_parallel(
            lambda arg: (f"{arg[0]}\x1f{arg[1]['prompt']}",
                         judge_label(client, arg[1]["prompt"], store.get(f"{arg[0]}\x1f{arg[1]['prompt']}", ""))),
            todo, workers=PARALLEL_WORKERS)
        for k, lb in outs:
            jtoxic[k] = lb
        dump_json(judge_cache, jtoxic)

    toxic_acc, toxic_rej = {}, {}
    for m in SUBJECTS:
        lbs = [jtoxic.get(f"{m}\x1f{p['prompt']}", "direct_answer") for p in toxic]
        acc = sum(1 for x in lbs if x == "direct_answer") / max(1, len(lbs))
        toxic_acc[m], toxic_rej[m] = acc, 1.0 - acc

    # 4) hard 子集: 被 >=3/6 被测模型拒绝的安全 prompt (论文 A.14), 再以裁判复核
    hard_candidates = []
    for p in safe:
        n = sum(1 for m in SUBJECTS if is_refusal_keyword(store.get(f"{m}\x1f{p['prompt']}", ""))[0])
        if n >= HARD_MIN_REJECTORS:
            hard_candidates.append({**p, "n_rejectors": n})
    print(f"[analyze] hard 子集: {len(hard_candidates)} 条")
    hard_rej = {}
    if hard_candidates:
        jhard_path = os.path.join(SAFE_CACHE_DIR, "judge_hard.json")
        jhard = load_json(jhard_path) if os.path.exists(jhard_path) else {}
        todo_h = [(m, p) for m in SUBJECTS for p in hard_candidates if f"{m}\x1f{p['prompt']}" not in jhard]
        if todo_h:
            outs_h = client.map_parallel(
                lambda arg: (f"{arg[0]}\x1f{arg[1]['prompt']}",
                             judge_label(client, arg[1]["prompt"], store.get(f"{arg[0]}\x1f{arg[1]['prompt']}", ""))),
                todo_h, workers=PARALLEL_WORKERS)
            for k, lb in outs_h:
                jhard[k] = lb
            dump_json(jhard_path, jhard)
        for m in SUBJECTS:
            lbs = [jhard.get(f"{m}\x1f{p['prompt']}", "direct_answer") for p in hard_candidates]
            hard_rej[m] = sum(1 for x in lbs if x != "direct_answer") / max(1, len(lbs))

    # 5) 关键词 vs 裁判 一致性 (论文 A.3: 差异 1.2%-2.4%)
    sample = random.sample(safe, min(40, len(safe)))
    kw_disc = {}
    print("[analyze] 关键词-裁判一致性抽样...")
    sample_path = os.path.join(SAFE_CACHE_DIR, "judge_sample.json")
    jsample = load_json(sample_path) if os.path.exists(sample_path) else {}
    todo_s = [(m, p) for m in SUBJECTS for p in sample if f"{m}\x1f{p['prompt']}" not in jsample]
    if todo_s:
        outs_s = client.map_parallel(
            lambda arg: (f"{arg[0]}\x1f{arg[1]['prompt']}",
                         judge_label(client, arg[1]["prompt"], store.get(f"{arg[0]}\x1f{arg[1]['prompt']}", ""))),
            todo_s, workers=PARALLEL_WORKERS)
        for k, lb in outs_s:
            jsample[k] = lb
        dump_json(sample_path, jsample)
    for m in SUBJECTS:
        kw_safe = sum(1 for p in sample if is_refusal_keyword(store.get(f"{m}\x1f{p['prompt']}", ""))[0]) / len(sample)
        jd_safe = sum(1 for p in sample
                      if jsample.get(f"{m}\x1f{p['prompt']}") != "direct_answer") / len(sample)
        kw_disc[m] = {  # 与论文表 5 符号一致: keyword reject / judge reject
            "keyword_reject": round(kw_safe * 100, 1),
            "judge_reject": round(jd_safe * 100, 1),
            "abs_diff_pct": round(abs(kw_safe - jd_safe) * 100, 1),
        }

    # 6) 消融: system prompt (论文 A.18 + fig.6b), 2 模型 x 抽样
    ab_models = ["glm-5.2-fast-preview", "deepseek-v4-flash"]
    ab_safe = random.sample(safe, min(30, len(safe)))
    ab_toxic = random.sample(toxic, min(15, len(toxic)))
    sp_store_path = os.path.join(SAFE_CACHE_DIR, "responses_sysprompt.json")
    sp_store = load_json(sp_store_path) if os.path.exists(sp_store_path) else {}
    need_sp = [(m, p, SYSTEM_PROMPT) for m in ab_models for p in ab_safe + ab_toxic]
    todo_sp = [(m, p) for (m, p, _) in need_sp if f"sp\x1f{m}\x1f{p['prompt']}" not in sp_store]
    print(f"[analyze] system prompt 消融 ({len(todo_sp)} 条待采集)...")
    if todo_sp:
        def ask_sp(arg):
            try:
                return ask_subject(client, arg[0], arg[1]["prompt"], system=SYSTEM_PROMPT)
            except Exception:
                return ""  # 静默过滤/失败 -> 空串, 判定层视作拒绝
        ids = [ (f"sp\x1f{m}\x1f{p['prompt']}") for (m, p) in todo_sp ]
        vals = client.map_parallel(ask_sp, todo_sp, workers=PARALLEL_WORKERS)
        for k, v in zip(ids, vals):
            sp_store[k] = v
        dump_json(sp_store_path, sp_store)

    def rate(m, prompts, sp=False):
        hits = 0
        for p in prompts:
            resp = sp_store.get(f"sp\x1f{m}\x1f{p['prompt']}") if sp else store.get(f"{m}\x1f{p['prompt']}", "")
            if is_refusal_keyword(resp or "")[0]:
                hits += 1
        return hits / max(1, len(prompts))

    ab_res = {}
    for m in ab_models:
        ab_res[m] = {
            "safe_no_sp": rate(m, ab_safe), "safe_with_sp": rate(m, ab_safe, sp=True),
            "toxic_no_sp": rate(m, ab_toxic), "toxic_with_sp": rate(m, ab_toxic, sp=True),
        }

    # 指标汇总
    metrics = {
        "n_safe": len(safe), "n_toxic": len(toxic), "n_hard": len(hard_candidates),
        "safe_rejection_rate": {m: round(v, 4) for m, v in safe_rej.items()},
        "toxic_acceptance_rate": {m: round(v, 4) for m, v in toxic_acc.items()},
        "toxic_rejection_rate": {m: round(v, 4) for m, v in toxic_rej.items()},
        "hard_rejection_rate": {m: round(v, 4) for m, v in hard_rej.items()},
        "keyword_vs_judge": kw_disc,
        "spearman_safe_vs_toxic_rej": round(
            spearman([safe_rej[m] for m in SUBJECTS], [toxic_rej[m] for m in SUBJECTS]), 3),
        "ablation_system_prompt": {m: {k: round(v, 4) for k, v in d.items()} for m, d in ab_res.items()},
    }
    dump_json(os.path.join(RESULTS_DIR, "metrics.json"), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))

    write_report(safe, hard_candidates, store, safe_rej, toxic_acc, toxic_rej, hard_rej,
                 metrics, ab_res)
    scatter_html(safe_rej, toxic_rej, metrics["spearman_safe_vs_toxic_rej"])
    print(f"[analyze] 完成, 耗时 {time.time() - t0:.0f}s")


def write_report(safe, hard, store, safe_rej, toxic_acc, toxic_rej, hard_rej, metrics, ab_res):
    lines = []
    A = lines.append
    A("# OR-Bench 轻量化复现报告\n")
    A("论文: *OR-Bench: An Over-Refusal Benchmark for Large Language Models* (arXiv:2405.20947)")
    A("复现方式: 用开放代理上的 6 个跨家族模型, 复现论文的\"生成→改写→评审\"管线与评测协议。\n")
    A("## 数据集规模\n")
    A(f"- OR-Bench-lite (安全但看似有毒): **{metrics['n_safe']}** 条, 覆盖 10 类")
    A(f"- Toxic 对照集: **{metrics['n_toxic']}** 条")
    A(f"- Hard 子集 (关键词口径被 ≥{HARD_MIN_REJECTORS}/6 模型拒绝): **{metrics['n_hard']}** 条\n")

    A("## 1. 主结果: 安全 ↔ 过拒绝 的权衡\n")
    A("| 模型 | 安全集拒绝率 (过拒绝, 关键词) | Hard 集拒绝率 (裁判) | Toxic 集接受率 (裁判) | Toxic 集拒绝率 |")
    A("|---|---|---|---|---|")
    for m in SUBJECTS:
        hard_v = f"{hard_rej.get(m, 0.0) * 100:.1f}%" if hard else "—"
        A(f"| {m} | {safe_rej[m] * 100:.1f}% | {hard_v} | {toxic_acc[m] * 100:.1f}% | {toxic_rej[m] * 100:.1f}% |")
    A("")
    A(f"**Spearman 秩相关: 论文 0.878, 本复现 {metrics['spearman_safe_vs_toxic_rej']}** — "
      "拒绝有毒 prompt 越多的模型, 越倾向于拒绝良性 prompt.\n")

    A("## 2. 关键词匹配 vs LLM 裁判 (论文 A.3 表 5)\n")
    A("| 模型 | 关键词拒绝率 % | 裁判拒绝率 % | 差异 % |")
    A("|---|---|---|---|")
    for m in SUBJECTS:
        d = metrics["keyword_vs_judge"][m]
        A(f"| {m} | {d['keyword_reject']} | {d['judge_reject']} | {d['abs_diff_pct']} |")
    A("\n论文报告的关键词-裁判差异为 1.2%~2.4%; 复现差异在同一量级即说明关键词匹配在此任务上可靠。\n")

    A("## 3. 消融: system prompt 的作用 (论文 fig.6b)\n")
    A("| 模型 | 条件 | 安全集拒绝率 | Toxic 集拒绝率 |")
    A("|---|---|---|---|")
    for m, d in ab_res.items():
        A(f"| {m} | 无 system prompt | {d['safe_no_sp'] * 100:.1f}% | {d['toxic_no_sp'] * 100:.1f}% |")
        A(f"| {m} | 含 Llama2 式 system prompt | {d['safe_with_sp'] * 100:.1f}% | {d['toxic_with_sp'] * 100:.1f}% |")
    A("\n论文观察: 加 system prompt 后两类拒绝率都上升 (右上移动), 复现结果同向即说明\"安全性提升以过拒绝为代价\"。\n")

    A("## 4. 各类别过拒绝率 (GLM-5.2-fast-preview)\n")
    A("| 类别 | 拒绝率 |")
    A("|---|---|")
    for c in sorted({p["category"] for p in safe}):
        ps = [p for p in safe if p["category"] == c]
        hits = sum(1 for p in ps if is_refusal_keyword(store.get(f"glm-5.2-fast-preview\x1f{p['prompt']}", ""))[0])
        A(f"| {c} | {hits / len(ps) * 100:.1f}% |")
    A("\n---\n")
    A("产物: `datasets/` (seeds/rewritten/moderation 原始记录), `results/metrics.json`, `results/scatter.html`")
    with open(os.path.join(RESULTS_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[analyze] report.md 已生成")


def scatter_html(safe_rej, toxic_rej, rho):
    """论文 fig.1 风格散点图: x=Toxic 拒绝率, y=安全集拒绝率, 自包含 HTML+SVG。"""
    xs = [toxic_rej[m] * 100 for m in SUBJECTS]
    ys = [safe_rej[m] * 100 for m in SUBJECTS]
    W, H, PAD_L, PAD_B, PAD_T, PAD_R = 680, 520, 66, 56, 30, 24
    x0, x1, y0, y1 = 0, 100, 0, 100

    def px(x): return PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    def py(y): return H - PAD_B - (y - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    parts = []
    for i in range(0, 101, 20):
        parts.append(f'<line x1="{px(i):.1f}" y1="{py(0):.1f}" x2="{px(i):.1f}" y2="{py(100):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{px(i):.1f}" y="{py(0) + 18:.1f}" font-size="11" text-anchor="middle" fill="#888">{i}</text>')
        parts.append(f'<line x1="{px(0):.1f}" y1="{py(i):.1f}" x2="{px(100):.1f}" y2="{py(i):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{px(0) - 8:.1f}" y="{py(i) + 4:.1f}" font-size="11" text-anchor="end" fill="#888">{i}</text>')
    for m, x, y in zip(SUBJECTS, xs, ys):
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="7" fill="#4e79a7" opacity="0.85"/>')
        parts.append(f'<text x="{px(x) + 10:.1f}" y="{py(y) + 4:.1f}" font-size="11">{html.escape(m)}</text>')
    svg = (
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}">'
        + "".join(parts)
        + f'<text x="{PAD_L + (W - PAD_L - PAD_R) / 2:.0f}" y="{H - 8:.0f}" font-size="12" text-anchor="middle">'
        'Toxic 提示拒绝率 % (安全性)</text>'
        + f'<text x="14" y="{H / 2:.0f}" font-size="12" text-anchor="middle" transform="rotate(-90 14 {H / 2:.0f})">'
        '看似有毒(良性)提示拒绝率 % (过拒绝)</text>'
        + "</svg>"
    )
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>OR-Bench 复现散点图</title></head>
<body style="font-family:sans-serif;margin:24px">
<h2>安全-过拒绝权衡 (Spearman rho = {rho}, 论文 0.878)</h2>
{svg}
<p>左上角 = 理想 (拒绝最多有毒、最少良性)。原始数据: <code>results/metrics.json</code></p>
</body></html>"""
    with open(os.path.join(RESULTS_DIR, "scatter.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print("[analyze] scatter.html 已生成")


if __name__ == "__main__":
    main()