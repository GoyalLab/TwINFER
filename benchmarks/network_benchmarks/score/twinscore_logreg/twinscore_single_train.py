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
    return df

meta, frames = {}, {}
for net, cfg in OLD_NETWORKS.items():
    t, p, _ = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    meta[net] = (t, p, jfs)
for ds in NEW_DATASETS:
    base = ds.split("_")[0]
    t, p, _ = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])
    jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{ds}_rep_*_all_results.json")))
    meta[ds] = (t, p, jfs)

for name, (t, p, jfs) in meta.items():
    fl = []
    for jf in jfs:
        df = load(jf)
        df["is_true"] = [(g1, g2) in t for g1, g2 in zip(df.gene_1, df.gene_2)]
        fl.append(df)
    frames[name] = pd.concat(fl, ignore_index=True)

names = list(meta.keys())

def fit(train_df):
    X = train_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    v = X.notna().all(axis=1)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[v], train_df.loc[v, "is_true"].astype(int))
    return clf

def eval_on(clf, name):
    t, p, jfs = meta[name]
    aucs, f1s = [], []
    for jf in jfs:
        df = load(jf)
        X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        m = X.notna().all(axis=1)
        proba = clf.predict_proba(X[m])[:, 1]
        scores = {(g1, g2): pr for (g1, g2), pr in zip(zip(df.gene_1[m], df.gene_2[m]), proba)}
        aucs.append(auprc_from_scored_pairs(scores, t, p))
        f1s.append(f1_topk(scores, t, p))
    return np.mean(aucs), np.mean(f1s)

# train on ONE dataset, evaluate on all OTHERS
print(f"{'train_on':20s} {'mean AUPRC (others)':>20s} {'mean F1 (others)':>18s}")
rows = []
for train_name in names:
    clf = fit(frames[train_name])
    a = [eval_on(clf, n)[0] for n in names if n != train_name]
    f = [eval_on(clf, n)[1] for n in names if n != train_name]
    rows.append((train_name, np.mean(a), np.mean(f)))
for tn, a, f in sorted(rows, key=lambda r: -r[1]):
    print(f"{tn:20s} {a:20.4f} {f:18.4f}")

# reference: LOO-CV (train on 15) and original twinScore
print("\nreference:")
print(f"  LOO-CV (train on 15 others)   AUPRC~0.487  F1~0.462")
print(f"  original twinScore            AUPRC~0.409  F1~0.407")
print(f"  z_het-fixed formula (no train) AUPRC~0.450  F1~0.435")
