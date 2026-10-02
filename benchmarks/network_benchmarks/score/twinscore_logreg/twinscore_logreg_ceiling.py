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

FEATURES = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het", "z_gamma"]

def load(jf):
    d = json.load(open(jf))
    red = d["ranked_edges"]
    return pd.DataFrame(red["data"], columns=red["columns"])

# pool one representative replicate per dataset (first found) with true-edge labels
datasets = {}
for net, cfg in OLD_NETWORKS.items():
    true_edges, possible_edges, _ = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    frames = []
    for jf in jfs:
        df = load(jf)
        df["is_true"] = [(g1, g2) in true_edges for g1, g2 in zip(df.gene_1, df.gene_2)]
        frames.append(df)
    datasets[net] = (pd.concat(frames, ignore_index=True), true_edges, possible_edges, jfs)

for dataset_id in NEW_DATASETS:
    base = dataset_id.split("_")[0]
    true_edges, possible_edges, _ = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])
    jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{dataset_id}_rep_*_all_results.json")))
    frames = []
    for jf in jfs:
        df = load(jf)
        df["is_true"] = [(g1, g2) in true_edges for g1, g2 in zip(df.gene_1, df.gene_2)]
        frames.append(df)
    datasets[dataset_id] = (pd.concat(frames, ignore_index=True), true_edges, possible_edges, jfs)

names = list(datasets.keys())
loo_scores = {"auprc": [], "f1": []}
for held_out in names:
    train_frames = [datasets[n][0] for n in names if n != held_out]
    train = pd.concat(train_frames, ignore_index=True)
    X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = X_train.notna().all(axis=1)
    X_train, y_train = X_train[valid], train.loc[valid, "is_true"].astype(int)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    test_df, true_edges, possible_edges, jfs = datasets[held_out]
    for jf in jfs:
        df = load(jf)
        X_test = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        mask = X_test.notna().all(axis=1)
        proba = clf.predict_proba(X_test[mask])[:, 1]
        scores = {(g1, g2): p for (g1, g2), p in zip(zip(df.gene_1[mask], df.gene_2[mask]), proba)}
        loo_scores["auprc"].append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
        loo_scores["f1"].append(f1_topk(scores, true_edges, possible_edges))

print(f"Leave-one-dataset-out logistic regression on {FEATURES}:")
print(f"  mean AUPRC = {np.mean(loo_scores['auprc']):.4f}")
print(f"  mean F1    = {np.mean(loo_scores['f1']):.4f}")
print(f"  (for reference: original twinScore=0.4088/0.4071, zhet_fixed=0.4495/0.4348)")

print("\n--- final fit on ALL 16 datasets pooled (for interpretation only, not the CV score above) ---")
all_frames = [datasets[n][0] for n in names]
allpooled = pd.concat(all_frames, ignore_index=True)
X = allpooled[FEATURES].apply(pd.to_numeric, errors="coerce")
valid = X.notna().all(axis=1)
X, y = X[valid], allpooled.loc[valid, "is_true"].astype(int)
# standardize for interpretable coefficient magnitudes
Xz = (X - X.mean()) / X.std()
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xz, y)
for f, c in sorted(zip(FEATURES, clf.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"  {f:20s} {c:+.4f}")
