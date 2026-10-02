from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
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

BASE_FEATURES = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het", "z_gamma"]

def load(jf):
    d = json.load(open(jf))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    df["abs_z_het"] = df["z_het"].abs()
    df["abs_z_gamma"] = df["z_gamma"].abs()
    df["abs_z_div"] = df["z_div"].abs()
    return df

FEATURES = BASE_FEATURES + ["abs_z_het", "abs_z_gamma", "abs_z_div"]

def panel_z(s):
    x = s.dropna()
    mu, sd = x.mean(), x.std(ddof=0)
    return (s - mu) / sd if sd else s * 0

meta = {}
for net, cfg in OLD_NETWORKS.items():
    true_edges, possible_edges, _ = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    meta[net] = (true_edges, possible_edges, jfs)
for dataset_id in NEW_DATASETS:
    base = dataset_id.split("_")[0]
    true_edges, possible_edges, _ = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])
    jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{dataset_id}_rep_*_all_results.json")))
    meta[dataset_id] = (true_edges, possible_edges, jfs)

frames = {}
for name, (true_edges, possible_edges, jfs) in meta.items():
    fl = []
    for jf in jfs:
        df = load(jf)
        df["is_true"] = [(g1, g2) in true_edges for g1, g2 in zip(df.gene_1, df.gene_2)]
        fl.append(df)
    frames[name] = pd.concat(fl, ignore_index=True)

names = list(meta.keys())
rows = []
for held_out in names:
    true_edges, possible_edges, jfs = meta[held_out]

    # original twinScore
    aucs_o, f1s_o = [], []
    aucs_z, f1s_z = [], []
    for jf in jfs:
        df = load(jf)
        scores_o = {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, df.twinScore) if pd.notna(v)}
        aucs_o.append(auprc_from_scored_pairs(scores_o, true_edges, possible_edges))
        f1s_o.append(f1_topk(scores_o, true_edges, possible_edges))
        vals_z = df.twinScore + df.heterogeneity_penalty + panel_z(df.z_het.abs())
        scores_z = {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, vals_z) if pd.notna(v)}
        aucs_z.append(auprc_from_scored_pairs(scores_z, true_edges, possible_edges))
        f1s_z.append(f1_topk(scores_z, true_edges, possible_edges))

    # logreg ceiling, trained on all OTHER datasets
    train = pd.concat([frames[n] for n in names if n != held_out], ignore_index=True)
    X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = X_train.notna().all(axis=1)
    X_train, y_train = X_train[valid], train.loc[valid, "is_true"].astype(int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    aucs_l, f1s_l = [], []
    for jf in jfs:
        df = load(jf)
        X_test = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        mask = X_test.notna().all(axis=1)
        proba = clf.predict_proba(X_test[mask])[:, 1]
        scores_l = {(g1, g2): p for (g1, g2), p in zip(zip(df.gene_1[mask], df.gene_2[mask]), proba)}
        aucs_l.append(auprc_from_scored_pairs(scores_l, true_edges, possible_edges))
        f1s_l.append(f1_topk(scores_l, true_edges, possible_edges))

    rows.append(dict(dataset=held_out,
                      auprc_original=np.mean(aucs_o), auprc_zhet_fixed=np.mean(aucs_z), auprc_logreg=np.mean(aucs_l),
                      f1_original=np.mean(f1s_o), f1_zhet_fixed=np.mean(f1s_z), f1_logreg=np.mean(f1s_l)))

res = pd.DataFrame(rows).set_index("dataset")
pd.set_option("display.width", 160)
print("=== AUPRC per dataset ===")
print(res[["auprc_original", "auprc_zhet_fixed", "auprc_logreg"]].round(3).to_string())
print("\n=== F1 per dataset ===")
print(res[["f1_original", "f1_zhet_fixed", "f1_logreg"]].round(3).to_string())
print("\n=== means ===")
print(res.mean().round(4).to_string())

out_path = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinscore_improvement_per_dataset.csv'
res.to_csv(out_path)
print(f"\nwrote {out_path}")
