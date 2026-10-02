# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_all_nofilter_combined import (true_edges_from_topo, TOPO_ROOT, OLD_NETWORKS,
    OLD_TWINFER_NOFILTER_DIR, NEW_DATASETS, NEW_BASE_TOPO, NEW_TWINFER_NOFILTER_DIR)

BASE = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het", "z_gamma"]
FEATURES = BASE + ["abs_z_het", "abs_z_gamma", "abs_z_div"]

def load(jf):
    d = json.load(open(jf)); red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    df["abs_z_het"] = df["z_het"].abs(); df["abs_z_gamma"] = df["z_gamma"].abs(); df["abs_z_div"] = df["z_div"].abs()
    for c in FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

frames = {}
for name in list(OLD_NETWORKS) + NEW_DATASETS:
    if name in OLD_NETWORKS:
        cfg = OLD_NETWORKS[name]; t,p,_ = true_edges_from_topo(TOPO_ROOT/cfg["topo"])
        jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR/f"{cfg['json_token']}_rep_*_all_results.json")))
    else:
        base = name.split("_")[0]; t,p,_ = true_edges_from_topo(TOPO_ROOT/NEW_BASE_TOPO[base])
        jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR/f"{name}_rep_*_all_results.json")))
    fl = []
    for jf in jfs:
        df = load(jf); df["is_true"] = [(g1,g2) in t for g1,g2 in zip(df.gene_1, df.gene_2)]; fl.append(df)
    frames[name] = pd.concat(fl, ignore_index=True)

names = list(frames)
allp = pd.concat(frames.values(), ignore_index=True)
# global standardization params (so per-fold coefs are comparable)
Xall = allp[FEATURES].apply(pd.to_numeric, errors="coerce")
v = Xall.notna().all(axis=1)
mu, sd = Xall[v].mean(), Xall[v].std()

def fit(train):
    X = train[FEATURES].apply(pd.to_numeric, errors="coerce"); m = X.notna().all(axis=1)
    Xz = (X[m]-mu)/sd
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xz, train.loc[m,"is_true"].astype(int))
    return clf.coef_[0], clf.intercept_[0]

# pooled fit (all 16)
cpool, ipool = fit(allp)
# per-LOO-fold fits
coefs = np.array([fit(pd.concat([frames[n] for n in names if n!=h], ignore_index=True))[0] for h in names])

print("standardized logistic-regression weights (feature already z-scored):\n")
print(f"{'feature':18s} {'pooled':>9s}  {'LOO mean':>9s} {'LOO std':>8s}  {'sign stable?':>12s}")
order = np.argsort(-np.abs(cpool))
for i in order:
    stable = "yes" if np.all(np.sign(coefs[:,i]) == np.sign(cpool[i])) else "NO"
    print(f"{FEATURES[i]:18s} {cpool[i]:+9.3f}  {coefs[:,i].mean():+9.3f} {coefs[:,i].std():8.3f}  {stable:>12s}")
print(f"\nintercept (pooled): {ipool:+.3f}")
print(f"\n(positive weight = larger value pushes toward 'is a true edge'.")
print(f" features are panel-z-scored per replicate, then globally standardized here for comparability.)")
