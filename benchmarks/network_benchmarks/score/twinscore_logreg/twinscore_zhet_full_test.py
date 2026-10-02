from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
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

def score_variant(df, variant):
    if variant == "original":
        vals = df.twinScore
    elif variant == "zhet_fixed":
        vals = df.twinScore + df.heterogeneity_penalty + panel_z(df.z_het.abs())
    scores = {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, vals) if pd.notna(v)}
    return scores

rows = []

# ---- old track ----
for net, cfg in OLD_NETWORKS.items():
    true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    for variant in ["original", "zhet_fixed"]:
        aucs, f1s = [], []
        for jf in jfs:
            d = json.load(open(jf))
            red = d["ranked_edges"]
            df = pd.DataFrame(red["data"], columns=red["columns"])
            scores = score_variant(df, variant)
            aucs.append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
            f1s.append(f1_topk(scores, true_edges, possible_edges))
        rows.append(dict(track="old", dataset=net, variant=variant, auprc=np.mean(aucs), f1=np.mean(f1s), n=len(jfs)))

# ---- new track ----
for dataset_id in NEW_DATASETS:
    base = dataset_id.split("_")[0]
    true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])
    jfs = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{dataset_id}_rep_*_all_results.json")))
    for variant in ["original", "zhet_fixed"]:
        aucs, f1s = [], []
        for jf in jfs:
            d = json.load(open(jf))
            red = d["ranked_edges"]
            df = pd.DataFrame(red["data"], columns=red["columns"])
            scores = score_variant(df, variant)
            aucs.append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
            f1s.append(f1_topk(scores, true_edges, possible_edges))
        rows.append(dict(track="new", dataset=dataset_id, variant=variant, auprc=np.mean(aucs), f1=np.mean(f1s), n=len(jfs)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 160)
piv_auprc = res.pivot(index="dataset", columns="variant", values="auprc")
piv_f1 = res.pivot(index="dataset", columns="variant", values="f1")
print("=== AUPRC ===")
print(piv_auprc.round(4).to_string())
print("\n=== F1 ===")
print(piv_f1.round(4).to_string())
print("\n=== overall means ===")
print(res.groupby("variant")[["auprc", "f1"]].mean().round(4).to_string())

out_path = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinscore_zhet_fix_scores.csv'
res.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")
