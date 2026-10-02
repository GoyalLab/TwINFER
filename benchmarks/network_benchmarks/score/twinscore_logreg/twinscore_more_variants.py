# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, sys
import numpy as np
import pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_all_nofilter_combined import (true_edges_from_topo, TOPO_ROOT, OLD_NETWORKS,
    OLD_TWINFER_NOFILTER_DIR, NEW_DATASETS, NEW_BASE_TOPO, NEW_TWINFER_NOFILTER_DIR,
    auprc_from_scored_pairs, f1_topk)

def panel_z(s):
    x = s.dropna()
    mu, sd = x.mean(), x.std(ddof=0)
    return (s - mu) / sd if sd else s * 0

VARIANTS = {
    "original":                lambda df: df.twinScore,
    "zhet_fixed":              lambda df: df.twinScore + df.heterogeneity_penalty + panel_z(df.z_het.abs()),
    "zhet_fixed + drop_change": lambda df: (df.twinScore + df.heterogeneity_penalty + panel_z(df.z_het.abs())
                                             + df.z_abs_rho_change),
    "zhet_fixed + zgamma_unconditional": lambda df: (df.twinScore + df.heterogeneity_penalty
                                                       + panel_z(df.z_het.abs()) - df.gamma_bonus
                                                       + panel_z(df.z_gamma.abs())),
    "zhet_fixed + zdiv_unconditional": lambda df: (df.twinScore + df.heterogeneity_penalty
                                                     + panel_z(df.z_het.abs()) - df.divergence_penalty
                                                     + panel_z(df.z_div.abs())),
    "kitchen_sink (all 3 fixes)": lambda df: (df.z_abs_rho_t1 + df.z_abs_rho_t2
                                               + panel_z(df.z_het.abs())
                                               + panel_z(df.z_gamma.abs())
                                               + panel_z(df.z_div.abs())),
}

def load(jf):
    d = json.load(open(jf))
    red = d["ranked_edges"]
    return pd.DataFrame(red["data"], columns=red["columns"])

rows = []
for net, cfg in OLD_NETWORKS.items():
    true_edges, possible_edges, _ = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    for variant, fn in VARIANTS.items():
        aucs, f1s = [], []
        for jf in jfs:
            df = load(jf)
            vals = fn(df)
            scores = {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, vals) if pd.notna(v)}
            aucs.append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
            f1s.append(f1_topk(scores, true_edges, possible_edges))
        rows.append(dict(dataset=net, variant=variant, auprc=np.mean(aucs), f1=np.mean(f1s)))

for dataset_id in NEW_DATASETS:
    base = dataset_id.split("_")[0]
    true_edges, possible_edges, _ = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])
    jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{dataset_id}_rep_*_all_results.json")))
    for variant, fn in VARIANTS.items():
        aucs, f1s = [], []
        for jf in jfs:
            df = load(jf)
            vals = fn(df)
            scores = {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, vals) if pd.notna(v)}
            aucs.append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
            f1s.append(f1_topk(scores, true_edges, possible_edges))
        rows.append(dict(dataset=dataset_id, variant=variant, auprc=np.mean(aucs), f1=np.mean(f1s)))

res = pd.DataFrame(rows)
print(res.groupby("variant")[["auprc", "f1"]].mean().sort_values("auprc", ascending=False).round(4).to_string())
