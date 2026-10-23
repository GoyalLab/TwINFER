"""Paper-flowchart scenario (Fig. 2a, infer.py clone-weighted statistics) for the FM01 top-100 pairs by |C| and by |U|.
Single timepoint (pooled FM01, t1=t2) -> Stage III is infer.py's single-timepoint branch: z_reg_gated > norm.ppf(1-alpha_stage3) (alpha_stage3 = 0.01 -> 2.326).
  Stage I   rho (clone-weighted Spearman, all cells incl. singletons) vs clone-derangement null: |step1_z| > 2.576 (two-sided p=0.01) else scenario 1
  Stage II  z_het (twin rho_Delta vs fresh random-pair rho_Delta): |z_het| > 5 (infer.py: abs(z) > abs(threshold); paper uses 10) = multiple states, else scenario 2
  Stage III z_reg_gated > 2.326: scenario 3, else 4   (infer.py, one-sided);  two-sided variant |z_reg_gated| > 2.576 (= |rho_same - lambda*rho_cross| vs |null|) saved as well
usage: python 23_flowchart_scenarios_top100_C_U.py OUT_CSV [--n-shuffles 5000] [--only PAIR,PAIR] [--chunk i/n]
Reuses step functions of simulations/drift_multiple_state/run_6scenario_zscores.py (module globals GENES/PAIR set per pair)."""
import argparse, os, sys, time
import numpy as np, pandas as pd, anndata as ad, numba
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from twinfer.utils.paths import get_data_root
ROOT = get_data_root().parent
sys.path.insert(0, f"{ROOT}/code/TwINFER/simulations/drift_multiple_state")
import run_6scenario_zscores as R
from twinfer.inference.correlation_functions import assign_twin_id, calculate_gated_regulation_statistic

MAX_CLONE_SIZE = 50
ap = argparse.ArgumentParser(); ap.add_argument("out_csv"); ap.add_argument("--n-shuffles", type=int, default=5000)
ap.add_argument("--only", default=None); ap.add_argument("--chunk", default="0/1"); ap.add_argument("--n-cores", type=int, default=1)
a = ap.parse_args()
R.N_SHUFFLES = a.n_shuffles
T = f"{ROOT}/analysis_data/fm01/data/top100_C_U_scenarios"
pairs = pd.concat([pd.read_csv(f"{T}/top100_by_absC_scenarios.csv").assign(list="C"), pd.read_csv(f"{T}/top100_by_absU_scenarios.csv").assign(list="U")])
up = pairs.drop_duplicates("pair")[["pair"]].reset_index(drop=True)
if a.only: up = up[up.pair.isin(a.only.split(","))]
i, n = map(int, a.chunk.split("/")); up = up.iloc[i::n]

A = ad.read_h5ad(f"{ROOT}/finalized_data/FM01_data/FM01_integrated.h5ad")
clone = A.obs["fatemap_clone_singletcode"].astype(str)
keep = (clone.str.len() > 0).to_numpy()
vc = clone[keep].value_counts(); drop = set(vc[vc > MAX_CLONE_SIZE].index)
keep &= ~clone.isin(drop).to_numpy()
print(f"{keep.sum()} cells kept, {len(drop)} oversized clones dropped, {len(up)} pairs", flush=True)
rows = []
for k, pr in enumerate(up.pair):
    tf, g = pr.split("-", 1); t0 = time.time()
    X = A.raw[:, [tf, g]].X; X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    df = pd.DataFrame({"clone_id": clone[keep].to_numpy(), "cell_id": A.obs_names[keep], "time_step": 0,
                       "gene_1_mRNA": X[keep, 0], "gene_2_mRNA": X[keep, 1]})
    R.GENES = ["gene_1", "gene_2"]; R.PAIR = ("gene_1", "gene_2"); numba.set_num_threads(1)
    with threadpool_limits(limits=1):
        twins = assign_twin_id(df[df.clone_id.map(df.clone_id.value_counts()) >= 2]).reset_index(drop=True)
        s1, _ = R._step1(df, 101011)
        s2, _, _ = R._step2(df, twins, 101010 + 271829, 101010 + 271832, a.n_cores)
        g3 = calculate_gated_regulation_statistic(twins, "gene_1", "gene_2", s2["z_het"], n_shuffles=a.n_shuffles, random_state=101010 + 271840)
    zc = norm.ppf(1 - 0.01)
    rows.append(dict(pair=pr, rho_infer=s1["rho"], step1_z=s1["z"], z_het=s2["z_het"], z_div=s2["z_div"], rho_delta=s2["rho_delta"],
                     rho_delta_random=s2["rho_delta_random"], z_reg_gated=g3["z_reg_gated"], lambda_gated=g3["lambda"], z_crit_stage3=zc,
                     rho_same_2n=g3["rho_gene_gene_2n"], rho_cross_2n=g3["rho_cross_2n"], rho_reg_gated=g3["rho_reg_gated"],
                     gated_null_mean=g3["null_mean"], gated_null_std=g3["null_std"], z_crit_two_sided=norm.ppf(1 - 0.01 / 2)))
    print(f"[{k+1}/{len(up)}] {pr} rho={s1['rho']:.3f} z1={s1['z']:.1f} zhet={s2['z_het']:.1f} zgated={g3['z_reg_gated']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
