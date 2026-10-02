# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, sys
import numpy as np
import pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_all_nofilter_combined import (true_edges_from_topo, TOPO_ROOT, OLD_NETWORKS,
    OLD_TWINFER_NOFILTER_DIR, auprc_from_scored_pairs, f1_topk)

VARIANTS = {
    "twinScore (original)": lambda r: r.twinScore,
    "drop divergence_penalty": lambda r: r.twinScore + r.divergence_penalty,
    "drop z_abs_rho_change": lambda r: r.twinScore + r.z_abs_rho_change,
    "drop both": lambda r: r.twinScore + r.divergence_penalty + r.z_abs_rho_change,
    "z_gamma + gamma_bonus + z_abs_rho_t1/t2 only": lambda r: r.z_abs_rho_t1 + r.z_abs_rho_t2 + r.z_gamma + r.gamma_bonus,
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
            scores = {}
            for r in df.itertuples():
                v = fn(r)
                if pd.notna(v):
                    scores[(r.gene_1, r.gene_2)] = abs(float(v))
            results[vname]["auprc"].append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
            results[vname]["f1"].append(f1_topk(scores, true_edges, possible_edges))

print(f"{'variant':50s} {'mean AUPRC':>10s} {'mean F1':>10s}")
for vname, d in results.items():
    print(f"{vname:50s} {np.mean(d['auprc']):10.4f} {np.mean(d['f1']):10.4f}")
