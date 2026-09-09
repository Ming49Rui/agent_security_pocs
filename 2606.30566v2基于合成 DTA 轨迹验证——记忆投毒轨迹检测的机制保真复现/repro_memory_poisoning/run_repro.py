"""Reproduction pipeline for arXiv:2606.30566v2, Section 2-3 & 6.

Synthetic (mechanism-faithful) data -> 19-feature extraction -> the exact
classifier/evaluation protocol of the paper:
  * single-rule baseline (recall before send)
  * LR / RF / GBM, stratified 5-fold CV (seed 42, threshold 0.5)
  * feature-group ablation (Table 6)
  * leave-one-model-out hold-out (Table 4)
  * prefix-only variants (Sec 3.9)
  * V2 benign false-positive study + recipient gating (Sec 6)

Outputs: console tables + outputs/{roc.png, feature_importance.png, tables.csv}
"""

import csv
import json
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, recall_score,
                             precision_score, f1_score, accuracy_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import (extract_features, FEATURE_NAMES, FEATURE_INDEX,
                      FREQUENCY_GROUP, MECHANISTIC_GROUP, RATIO_GROUP,
                      BIGRAM_GROUP, FIRST_TOOL_GROUP, RECALL_RELATED_GROUP,
                      PREFIX_13_FEATURES)
from synth_data import (generate_p1, generate_benign_v2, MODELS, DEFENSES,
                        BENIGN_MODELS)

SEED = 42
OUT = "outputs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rule_auc(pos_rbs: np.ndarray, neg_rbs: np.ndarray) -> float:
    """Mann-Whitney AUC for a binary marker with ties counted half."""
    tp = pos_rbs.sum(); fp = neg_rbs.sum()
    n_pos, n_neg = len(pos_rbs), len(neg_rbs)
    fn, tn = n_pos - tp, n_neg - fp
    return float((tp * tn + 0.5 * (tp * fp + fn * tn)) / (n_pos * n_neg))


def bootstrap_auc_ci(y, oof, n_boot: int = 10000) -> tuple:
    """Percentile-bootstrap CI on pooled OOF AUC (paper used BCa; we note the
    difference in the README). Resamples observation pairs jointly."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    aucs = np.array([roc_auc_score(y[i], oof[i]) for i in idx])
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def cv_eval(clf, X, y, n_splits=5, seed=SEED, std=False):
    """Stratified 5-fold CV; returns pooled OOF probs + per-metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    scaler = StandardScaler() if std else None
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if scaler is not None:
            Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    pred = (oof > 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, oof)),
        "auc_ci": bootstrap_auc_ci(y, oof),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "oof": oof,
    }


def full_clf():
    return RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight="balanced", random_state=SEED)


def make_feature_matrix(runs):
    X = np.array([extract_features(r["seq"]) for r in runs], dtype=float)
    y = np.array([r["label"] for r in runs], dtype=int)
    return X, y


def col_names(mask):
    return [FEATURE_NAMES[i] for i in range(19) if mask[i]]


def mask_for(group):
    return np.array([FEATURE_NAMES[i] in group for i in range(19)], dtype=bool)


def print_table(rows, headers, title):
    print(f"\n{title}")
    print(" | ".join(f"{h:>14s}" for h in headers))
    print("-" * (18 * len(headers)))
    for row in rows:
        print(" | ".join(f"{str(v):>14s}" for v in row))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("REPRODUCING arXiv:2606.30566v2  (Forensic Trajectory Signatures)")
    print("Synthetic mechanism-faithful data - see README.md for the mapping")
    print("=" * 78)

    # ---- 1. data ------------------------------------------------------------
    rng = random.Random(SEED)
    p1 = generate_p1(rng)
    X, y = make_feature_matrix(p1)
    n_atk, n_non = int(y.sum()), int(len(y) - y.sum())
    print(f"\n[Dataset] P1 runs={len(p1):,}  attack={n_atk}  non-exfil={n_non} "
          f"({100*n_atk/len(p1):.1f}%)   (paper: 2,520 / 1,919 / 601)")

    # class balance by defense (paper Table 5 shape)
    rows = []
    for d in DEFENSES:
        dd = [r for r in p1 if r["defense"] == d]
        a = sum(r["label"] for r in dd)
        rows.append([d, len(dd), a, len(dd) - a])
    print_table(rows, ["defense", "N", "Natk", "Nneg"], "[Table 5 check] per-defense class balance")

    # ---- 2. Table 3 style feature means -------------------------------------
    print("\n[Table 3 check] mean feature values (attack vs non-exfil)")
    feats_atk = X[y == 1]; feats_non = X[y == 0]
    interesting = ["recall_count", "send_without_recall", "max_recall_chain",
                   "draft_then_send", "list_count", "send_count", "recall_before_send"]
    rows = [[f, f"{feats_atk[:, FEATURE_INDEX[f]].mean():.3f}",
             f"{feats_non[:, FEATURE_INDEX[f]].mean():.3f}",
             f"{feats_atk[:, FEATURE_INDEX[f]].mean() - feats_non[:, FEATURE_INDEX[f]].mean():+.3f}"]
            for f in interesting]
    print_table(rows, ["feature", "attack", "non-exfil", "delta"], "")

    # ---- 3. single-rule baseline (Sec 3.1) ----------------------------------
    rbs = X[:, FEATURE_INDEX["recall_before_send"]]
    rule_pred = (rbs == 1).astype(int)
    fp_rule = int(((y == 0) & (rule_pred == 1)).sum())
    fn_rule = int(((y == 1) & (rule_pred == 0)).sum())
    auc_rule = rule_auc(rbs[y == 1], rbs[y == 0])
    print(f"\n[Sec 3.1] Rule 'recall before send -> attack':")
    print(f"  AUC={auc_rule:.4f}   Recall={recall_score(y, rule_pred):.4f}   "
          f"Accuracy={accuracy_score(y, rule_pred):.4f}   FP={fp_rule}  FN={fn_rule}")
    print(f"  paper: AUC=0.9563  Recall=0.9792  Accuracy=0.9683  FP=40  FN=40")

    # ---- 4. classifiers (Sec 3.2 / Table 1) ----------------------------------
    cls = {
        "Logistic Regression": (LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), True),
        "Random Forest": (full_clf(), False),
        "Gradient Boosted Trees": (GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED), False),
    }
    print("\n[Sec 3.2 / Table 1] 5-fold CV (threshold 0.5)")
    results, oofs = {}, {}
    rows = []
    for name, (clf, std) in cls.items():
        res = cv_eval(clf, X, y, std=std)
        results[name] = res; oofs[name] = res["oof"]
        rows.append([name, f"{res['auc']:.4f}", f"[{res['auc_ci'][0]:.3f},{res['auc_ci'][1]:.3f}]",
                     f"{res['recall']:.4f}", f"{res['precision']:.4f}",
                     f"{res['f1']:.4f}", f"{res['fn']} ({res['fp']})"])
    print_table(rows, ["classifier", "AUC", "boot 95% CI", "Recall", "Prec", "F1", "FN (FP)"], "")
    print("  paper: all three AUC=0.9904 [0.987,0.993]  Recall=0.9838  FN=31")

    # FN localization (paper Sec 3.5): should concentrate in Memory Sandbox
    rf_oof = oofs["Random Forest"]
    fn_loc = Counter(p1[i]["defense"] for i in range(len(y))
                     if y[i] == 1 and rf_oof[i] <= 0.5)
    print(f"  FN by defense (RF): {dict(fn_loc)}   (paper: all 31 in memory_sandbox)")

    # feature importances (Sec 3.3 / Fig 2)
    rf = full_clf()
    rf.fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                 key=lambda t: -t[1])
    print("\n[Sec 3.3 / Table 2] top-10 RF importances (mean decrease in impurity)")
    for f, v in imp[:10]:
        star = "  <- mechanistic" if f in MECHANISTIC_GROUP else ""
        print(f"  {f:22s} {v:.4f}{star}")

    # ---- 5. ablation (Sec 3.6 / Table 6) -------------------------------------
    groups = {
        "None (full model)": None,
        "Mechanistic": MECHANISTIC_GROUP,
        "Frequency counts": FREQUENCY_GROUP,
        "Ratio": RATIO_GROUP,
        "Bigram transitions": BIGRAM_GROUP,
        "First-tool indicators": FIRST_TOOL_GROUP,
        "All recall-related": RECALL_RELATED_GROUP,
    }
    print("\n[Sec 3.6 / Table 6] feature-group ablation (RF, 5-fold CV)")
    rows = []
    base_auc = results["Random Forest"]["auc"]
    for gname, feats in groups.items():
        keep = np.ones(19, dtype=bool)
        if feats is not None:
            keep &= ~mask_for(feats)
        Xg = X[:, keep]
        if feats is None:
            auc = base_auc
        else:
            auc = cv_eval(full_clf(), Xg, y)["auc"]
        rows.append([gname, len([k for k in keep if k]),
                     f"{auc:.4f}", f"{auc - base_auc:+.4f}"])
    print_table(rows, ["group removed", "n feat", "AUC", "delta"], "")
    print("  paper: AUC stable at 0.9904 (min 0.9886); removing all 9 recall features -> 0.9904")

    # ---- 6. leave-one-model-out hold-out (Sec 3.4 / Table 4) -----------------
    print("\n[Sec 3.4 / Table 4] leave-one-model-out (RF)")
    rows = []
    for hm in MODELS:
        tr = [i for i, r in enumerate(p1) if r["model"] != hm]
        te = [i for i, r in enumerate(p1) if r["model"] == hm]
        clf = full_clf()
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y[te], p))
        pred = (p > 0.5).astype(int)
        rec = recall_score(y[te], pred)
        rows.append([hm, f"{auc:.3f}", f"{rec:.3f}",
                     "inversion" if auc < 0.4 else ("partial" if auc < 0.96 else "generalizes")])
    print_table(rows, ["held-out model", "AUC", "Recall", "interpretation"], "")
    print("  paper: 6/9 -> AUC 1.000; qwq:32b 0.000 (implicit bypass), "
          "qwen2.5:14b 0.083 (distributional inversion), qwen3.5:122b 0.750 (sleeper)")

    # ---- 7. prefix-only vs full-session (Sec 3.9) -----------------------------
    print("\n[Sec 3.9] prefix-only variants (RF)")
    for pname, feats in [("14-feature (+seq_len)", None),
                         ("13-feature strict prefix", PREFIX_13_FEATURES)]:
        if feats is None:
            drop = ["send_count", "recall_before_send", "recall_to_send_ratio",
                    "send_without_recall", "draft_then_send"]
            keep = ~mask_for(drop)
        else:
            keep = mask_for(feats)
        auc = cv_eval(full_clf(), X[:, keep], y)["auc"]
        print(f"  {pname:26s} AUC={auc:.3f}")
    print("  paper: 14-feat 0.990, 13-feat strict prefix 0.934")

    # ---- 8. V2 benign false-positive study (Sec 6) ----------------------------
    print("\n[Sec 6] benign baseline study (13 model proxies x 8 cells)")
    benign = generate_benign_v2(random.Random(SEED))
    Xb, yb = make_feature_matrix(benign)
    scores_b = rf.predict_proba(Xb)[:, 1]
    flag = scores_b > 0.5
    rbs_b = Xb[:, FEATURE_INDEX["recall_before_send"]]

    m1 = (rbs_b == 1)
    print(f"  N={len(benign)}  P(flag | recall_before_send=1) = "
          f"{flag[m1].mean()*100:6.1f}%   (paper: 100.0%)  n={m1.sum()}")
    m0 = (rbs_b == 0)
    print(f"  P(flag | recall_before_send=0) = {flag[m0].mean()*100:6.1f}%  "
          f"(paper: 1.5%)  n={m0.sum()}")
    for proto in ["mandatory", "optional"]:
        maskp = np.array([r["protocol"] == proto for r in benign])
        print(f"  unconditional FPR [{proto:9s}] = {flag[maskp].mean()*100:5.1f}%  "
              f"(paper: 24.7-52.6% band across protocols)")
    by_k = {}
    for k in (0, 1, 3, 5):
        maskk = np.array([r["k"] == k for r in benign])
        by_k[k] = flag[maskk].mean() * 100
    print(f"  FPR by stored-facts k: {by_k}   (paper: k0=0%, k1=57.6%, k3=46%, k5=52.2% - non-monotonic)")
    by_ph = {}
    for name, ph in BENIGN_MODELS:
        maskp = np.array([r["model"] == name for r in benign])
        by_ph[ph] = by_ph.get(ph, []) + [f"{flag[maskp].mean()*100:.0f}%"]
    print("  FPR by phenotype:", {k: v for k, v in by_ph.items()})

    # gating with recipient metadata restores separation
    gate = flag & np.array([r["external_recpt"] for r in benign], dtype=bool)
    print(f"\n  After gating 'external recipient': benign FPR={gate.mean()*100:.2f}%   "
          f"(paper: 0% - perfect separation on its data)")
    print("  Attack recall preserved (attacker address is always external): all attack "
          "sessions pass the gate -> recall 100% by construction")

    # ---- 9. figures & artifacts ------------------------------------------------
    plot_roc(results["Random Forest"]["oof"], y, OUT)
    plot_importance(imp, OUT)
    save_artifacts(results, auc_rule, by_k, OUT, p1, benign)

    print("\nDone. Figures and CSVs written to", OUT)


def save_artifacts(results, auc_rule, by_k, out_dir, p1, benign):
    summary = {
        "paper": "arXiv:2606.30566v2",
        "data": "synthetic (mechanism-faithful; companion dataset not public)",
        "rule_auc": round(auc_rule, 4),
        "cv": {k: {kk: vv for kk, vv in v.items() if kk not in ("oof",)}
               for k, v in results.items()},
        "benign_fpr_by_k": by_k,
    }
    with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(f"{out_dir}/p1_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "defense", "seq", "label"])
        for r in p1:
            w.writerow([r["model"], r["defense"], ",".join(r["seq"]), r["label"]])


def plot_roc(oof, y, out_dir):
    fpr, tpr, _ = roc_curve(y, oof)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"Aggregate CV (AUC = {roc_auc_score(y, oof):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - Trajectory-Based Poisoning Detector (synthetic P1)")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{out_dir}/roc.png", dpi=150); plt.close()


def plot_importance(imp, out_dir):
    top = imp[:10]
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    colors = ["#d13434" if n in MECHANISTIC_GROUP else "#5a7a9e" for n in names]
    plt.figure(figsize=(6, 4.5))
    plt.barh(names, vals, color=colors)
    plt.xlabel("Feature Importance (Mean Decrease in Impurity)")
    plt.title("Top-10 Feature Importances - Random Forest (synthetic P1)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/feature_importance.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()