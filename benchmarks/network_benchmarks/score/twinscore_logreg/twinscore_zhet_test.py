# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, sys
import numpy as np
import pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_all_nofilter_combined import (true_edges_from_topo, TOPO_ROOT, OLD_NETWORKS,
    OLD_TWINFER_NOFILTER_DIR, auprc_from_scored_pairs, f1_topk)

def panel_z(s):
    x = s.dropna()
    mu, sd = x.mean(), x.std(ddof=0)
    return (s - mu) / sd if sd else s * 0

VARIANTS = {
    "twinScore (original)": lambda df: df.twinScore,
    "+ |z_het| panel-z, unconditional": lambda df: df.twinScore + panel_z(df.z_het.abs()),
    "+ |z_het| panel-z, drop old heterogeneity_penalty": lambda df: (
        df.twinScore + df.heterogeneity_penalty + panel_z(df.z_het.abs())),
}

results = {v: {"auprc": [], "f1": []} for v in VARIANTS}
for net, cfg in OLD_NETWORKS.items():
    true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    for jf in jfs:
        d = json.load(open(jf))
        red = d["ranked_edges"]
        df = pd.DataFrame(red["data"], columns=red["columns"])
        for vname, fn in VARIANTS.items():
            vals = fn(df)
            scores = {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, vals) if pd.notna(v)}
            results[vname]["auprc"].append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
            results[vname]["f1"].append(f1_topk(scores, true_edges, possible_edges))

print(f"{'variant':50s} {'mean AUPRC':>10s} {'mean F1':>10s}")
for vname, d in results.items():
    print(f"{vname:50s} {np.mean(d['auprc']):10.4f} {np.mean(d['f1']):10.4f}")
