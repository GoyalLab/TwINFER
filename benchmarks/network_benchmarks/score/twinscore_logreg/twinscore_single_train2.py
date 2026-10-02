# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_all_nofilter_combined import (true_edges_from_topo, TOPO_ROOT, OLD_NETWORKS,
    OLD_TWINFER_NOFILTER_DIR, NEW_DATASETS, NEW_BASE_TOPO, NEW_TWINFER_NOFILTER_DIR,
    auprc_from_scored_pairs, f1_topk)

BASE = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het", "z_gamma"]
FEATURES = BASE + ["abs_z_het", "abs_z_gamma", "abs_z_div"]

def load(jf):
    d = json.load(open(jf))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    df["abs_z_het"] = df["z_het"].abs()
    df["abs_z_gamma"] = df["z_gamma"].abs()
    df["abs_z_div"] = df["z_div"].abs()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# preload EVERYTHING once
store = {}   # name -> list of (X_array, pairs_list, true_edges, possible_edges)
train_pool = {}  # name -> (X_all, y_all) pooled across replicates
for name in list(OLD_NETWORKS) + NEW_DATASETS:
    if name in OLD_NETWORKS:
        cfg = OLD_NETWORKS[name]
        t, p, _ = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
        jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    else:
        base = name.split("_")[0]
        t, p, _ = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])
        jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{name}_rep_*_all_results.json")))
    reps, Xs, ys = [], [], []
    for jf in jfs:
        df = load(jf)
        m = df[FEATURES].notna().all(axis=1)
        df = df[m]
        pairs = list(zip(df.gene_1, df.gene_2))
        Xarr = df[FEATURES].to_numpy(float)
        yarr = np.array([(g1, g2) in t for g1, g2 in pairs], dtype=int)
        reps.append((Xarr, pairs, t, p))
        Xs.append(Xarr); ys.append(yarr)
    store[name] = reps
    train_pool[name] = (np.vstack(Xs), np.concatenate(ys))

names = list(store)

def eval_on(clf, name):
    aucs, f1s = [], []
    for Xarr, pairs, t, p in store[name]:
        proba = clf.predict_proba(Xarr)[:, 1]
        scores = {pr: pv for pr, pv in zip(pairs, proba)}
        aucs.append(auprc_from_scored_pairs(scores, t, p))
        f1s.append(f1_topk(scores, t, p))
    return np.mean(aucs), np.mean(f1s)

print(f"{'train_on':20s} {'AUPRC(others)':>14s} {'F1(others)':>12s}")
rows = []
for tn in names:
    X, y = train_pool[tn]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    res = [eval_on(clf, n) for n in names if n != tn]
    rows.append((tn, np.mean([r[0] for r in res]), np.mean([r[1] for r in res])))
for tn, a, f in sorted(rows, key=lambda r: -r[1]):
    print(f"{tn:20s} {a:14.4f} {f:12.4f}")

best = max(rows, key=lambda r: r[1])
print(f"\nbest single-network trainer: {best[0]}  -> AUPRC {best[1]:.4f}, F1 {best[2]:.4f}")
print(f"worst: {min(rows, key=lambda r: r[1])[0]} -> AUPRC {min(rows, key=lambda r: r[1])[1]:.4f}")
print(f"mean across all single trainers: AUPRC {np.mean([r[1] for r in rows]):.4f}, F1 {np.mean([r[2] for r in rows]):.4f}")
print(f"reference -- LOO-CV(train on 15): AUPRC~0.487 ; original twinScore: 0.409 ; z_het-fixed: 0.450")
