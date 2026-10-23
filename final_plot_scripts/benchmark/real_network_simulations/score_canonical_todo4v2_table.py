"""Canonical TODO4v2 (nogate) AUPRC table for the 7 real networks, with a PPCOR-style prototype column.

Date/time of this version: 2026-10-04 (run time is stamped into the output filename and header).

Formulas
--------
Baseline score (TODO4v2 nogate, todo4v2_sim_scoring.todo4v2_score):
  score = z_abs_rho_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s(|z_dagger|)
Prototype score (PPCOR-style hub term, W = 2.0):
  score_ppcor = score + W * s(|pc(x,y)|)
  pc = -P_xy / sqrt(P_xx P_yy), P = Ledoit-Wolf precision of rank-transformed mRNA (last timepoint, twin A).
Ranked edges: sign(rho_cross_xy) * (score - min(score)); AUPRC via score_mixed_network_sweep.compute_auprc.
AUPRC-x = raw AUPRC / base_rate, base_rate = n_true_directed / (n*(n-1)).
Ground truth: input_data/real_world_networks/<net>.txt (directed, unsigned).
Replicates: every twinfer_inference_allpairs/<net>_rep_*_all_results.json; per-replicate AUPRC averaged.

Caveats
-------
- pc is computed from ONE canonical Gillespie file per network and reused for every replicate of that
  network (approximation; replicate sim files not matched by hash).
- GSD.txt in input_data/real_world_networks is alphabetically ordered; gene_N ids are matched to rows
  positionally, the same way the canonical heatmap pipeline does it.
"""
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf

REPO = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
sys.path.insert(0, f"{REPO}/code/TwINFER/package")
sys.path.insert(0, f"{REPO}/code/TwINFER/benchmarks/network_benchmarks/score")
import todo4v2_sim_scoring as T
import score_mixed_network_sweep as M

RW = f"{REPO}/input_data/real_world_networks"
RAW = f"{REPO}/simulation_data/real_data"
INF = f"{REPO}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs"
OUT_DIR = f"{REPO}/analysis_data/boolode_sims_real_networks/diagnostics"
W = 2.0

NETS = {
    "B_cell_activation": dict(topo=f"{RW}/B_cell.txt", n=10, json="B_cell_activation",
        raw=f"{REPO}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv"),
    "EMT": dict(topo=f"{RW}/EMT.txt", n=17, json="EMT",
        raw=f"{REPO}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv"),
    "GSD": dict(topo=f"{RW}/GSD.txt", n=19, json="GSD",
        raw=f"{RAW}/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv"),
    "HSC": dict(topo=f"{RW}/HSC.txt", n=11, json="HSC",
        raw=f"{RAW}/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv"),
    "Pluripotent": dict(topo=f"{RW}/Pluripotent.txt", n=36, json="Pluripotent",
        raw=f"{REPO}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv"),
    "VSC": dict(topo=f"{RW}/VSC.txt", n=8, json="VSC",
        raw=f"{RAW}/df_rows_0_0_0_0_0_0_0_0_12072026_005406_ncells_6000_VSC_2_0_2befd3f9.csv"),
    "mCAD": dict(topo=f"{RW}/mCAD.txt", n=5, json="mCAD",
        raw=f"{RAW}/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv"),
}


def s(v):
    return T.s(v)


def partial_corr(raw_path, n):
    cols = [f"gene_{i+1}_mRNA" for i in range(n)]
    df = pd.read_csv(raw_path, usecols=["time_step", "clone_id", "replicate"] + cols)
    last = df[df.time_step == df.time_step.max()]
    a = last[last.replicate == 1].sort_values("clone_id")[cols].to_numpy(float)
    keep = a.std(axis=0) > 1e-9
    X = np.column_stack([rankdata(a[:, j]) for j in range(n)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    P = np.linalg.pinv(LedoitWolf().fit(X).covariance_)
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    pc[:, ~keep] = 0.0
    pc[~keep, :] = 0.0
    return pc


def auprc(dd, sc, gt_df):
    shifted = sc - np.nanmin(sc[np.isfinite(sc)])
    sign = np.sign(dd.rho_cross_xy.to_numpy())
    sign[sign == 0] = 1.0
    ranked = pd.DataFrame({"Gene1": dd.gene_1, "Gene2": dd.gene_2, "EdgeWeight": sign * shifted})
    return M.compute_auprc(ranked, gt_df)


rows = []
for net, cfg in NETS.items():
    n = cfg["n"]
    topo = np.loadtxt(cfg["topo"], delimiter=",")
    gn = [f"gene_{i+1}" for i in range(n)]
    recs = [(gn[i], gn[j], "+" if topo[i, j] > 0 else "-")
            for i in range(n) for j in range(n) if i != j and topo[i, j] != 0]
    gt_df = pd.DataFrame(recs, columns=["Gene1", "Gene2", "Type"])
    base_rate = len(recs) / (n * (n - 1))
    pc = partial_corr(cfg["raw"], n)

    base_vals, ppcor_vals = [], []
    for f in sorted(glob.glob(f"{INF}/{cfg['json']}_rep_*_all_results.json")):
        d = json.load(open(f))
        tsi = d["twin_score_inputs"]
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) < 3:
            continue
        U = list(zip(dd.gene_1, dd.gene_2))
        base, _ = T.todo4v2_score(dd, U, d["gated_regulation"]["z_reg_gated"], "nogate")
        idx = [(int(a.split("_")[1]) - 1, int(b.split("_")[1]) - 1) for a, b in U]
        term = s(np.array([abs(pc[i, j]) for i, j in idx]))
        base_vals.append(auprc(dd, base, gt_df))
        ppcor_vals.append(auprc(dd, base + W * term, gt_df))

    b_raw, p_raw = float(np.mean(base_vals)), float(np.mean(ppcor_vals))
    rows.append(dict(network=net, n_replicates=len(base_vals), base_rate=round(base_rate, 4),
                     todo4v2_raw_auprc=round(b_raw, 4), todo4v2_auprc_x=round(b_raw / base_rate, 3),
                     ppcor_proto_raw_auprc=round(p_raw, 4), ppcor_proto_auprc_x=round(p_raw / base_rate, 3)))

stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
df = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
out = f"{OUT_DIR}/canonical_todo4v2_table_{stamp}.csv"
with open(out, "w") as fh:
    fh.write(f"# generated {stamp}  script=code/TwINFER/final_plot_scripts/benchmark/real_network_simulations/score_canonical_todo4v2_table.py\n")
    fh.write("# baseline = TODO4v2 nogate; prototype = + 2.0 * s(|partial corr|); AUPRC-x = raw / base_rate\n")
    df.to_csv(fh, index=False)
print(df.to_string(index=False))
print("wrote", out)
