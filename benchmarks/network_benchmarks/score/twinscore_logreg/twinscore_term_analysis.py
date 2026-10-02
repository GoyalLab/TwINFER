# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, sys
import numpy as np
import pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_all_nofilter_combined import true_edges_from_topo, TOPO_ROOT, OLD_NETWORKS, OLD_TWINFER_NOFILTER_DIR

TERMS = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_gamma", "gamma_bonus",
         "divergence_penalty", "heterogeneity_penalty", "z_div", "z_het", "z_d_het",
         "twinScore"]

rows = []
for net, cfg in OLD_NETWORKS.items():
    true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
    jfs = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
    for jf in jfs:
        d = json.load(open(jf))
        red = d["ranked_edges"]
        df = pd.DataFrame(red["data"], columns=red["columns"])
        df["network"] = net
        df["is_true_edge"] = df.apply(lambda r: (r.gene_1, r.gene_2) in true_edges, axis=1)
        rows.append(df)

full = pd.concat(rows, ignore_index=True)
print(f"total scored directed pairs pooled: {len(full)}  (true={full['is_true_edge'].sum()}, false={(~full['is_true_edge']).sum()})")
print()

results = []
for term in TERMS:
    if term not in full.columns:
        continue
    vals = pd.to_numeric(full[term], errors="coerce")
    mask = vals.notna()
    y = full.loc[mask, "is_true_edge"].astype(int)
    x = vals[mask]
    if x.nunique() <= 1 or y.nunique() <= 1:
        continue
    r = np.corrcoef(x, y)[0, 1]
    mean_true = x[y == 1].mean()
    mean_false = x[y == 0].mean()
    results.append(dict(term=term, corr_with_true_edge=r, mean_true_edges=mean_true,
                         mean_non_edges=mean_false, n=mask.sum()))

res_df = pd.DataFrame(results).sort_values("corr_with_true_edge", ascending=False)
pd.set_option("display.width", 160)
print(res_df.round(4).to_string(index=False))

print("\n--- is the twin-difference signal ~universal (saturating) rather than pair-specific? ---")
for term in ["z_het", "z_div", "z_d_het"]:
    vals = pd.to_numeric(full[term], errors="coerce")
    mask = vals.notna()
    x = vals[mask]
    y = full.loc[mask, "is_true_edge"]
    frac_sig_true = (x[y].abs() > 2.5).mean()
    frac_sig_false = (x[~y].abs() > 2.5).mean()
    print(f"{term:10s}  overall |{term}|>2.5 rate: true_edges={frac_sig_true:.3f}  non_edges={frac_sig_false:.3f}  "
          f"(n_true={y.sum()}, n_false={(~y).sum()})  std_true={x[y].std():.2f} std_false={x[~y].std():.2f}")
