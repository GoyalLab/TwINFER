"""
All TwINFER z-scores + their correlations for the gene_1-gene_2 pair, as a
function of twin time, for one (kind x network) group of the low->mid drift
rebuild.  Mean over the replicate sim files.  Step 3 / Step 4 are two-timepoint
statistics referenced to a fixed t1 = --ref-t1.

  Step 1 : gene-gene rho(g1,g2)                 -> z_gene
  Step 2 : twin hat{rho}_Delta, random-pair rho_Delta,
           z_het, z_regulation (divergence)
  Step 3 : d = rho_D(t) - rho_D(t1)             -> z_d
  Step 4 : cross rho (g1->g2, g2->g1) t1->t     -> z_1to2, z_2to1, z_gamma

Usage:
  python score_lowmid_zscores_vs_time.py --sim-dir DIR --out-dir DIR \
      --type-tag A_to_B_lowmid_K_frozen --tmax 300 --every 10 \
      --ref-t1 1 --n-shuffles 5000 --jobs 3
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

ap = argparse.ArgumentParser()
ap.add_argument("--sim-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--type-tag", required=True, help="e.g. A_to_B_lowmid_K_frozen")
ap.add_argument("--tmax", type=int, default=300)
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--ref-t1", type=int, default=1)
ap.add_argument("--n-shuffles", type=int, default=5000)
ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--n-cores", type=int, default=4)
A = ap.parse_args()

from simulations.drift_multiple_state import run_6scenario_zscores as R
R.N_SHUFFLES = A.n_shuffles
from simulations.drift_multiple_state.run_6scenario_zscores import _step1, _step2, _step4, SEED, GENES, PAIR, _pv
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_twin_random_correlations, identify_reg_if_multiple_states)

TIMES = list(range(A.every, A.tmax + 1, A.every))
if A.ref_t1 not in TIMES:
    TIMES = [A.ref_t1] + TIMES


def _rep(f):
    return int(re.search(r"df_rows_0_0_(\d+)_", os.path.basename(f)).group(1))


def one_file(path):
    df = pd.read_csv(path)
    rep = _rep(path)
    reps = sorted(df["replicate"].unique())
    r1 = df[df["time_step"] == A.ref_t1].reset_index(drop=True)
    tw1 = assign_twin_id(r1).reset_index(drop=True)
    tm1, rm1 = calculate_twin_random_correlations(r1, tw1, GENES, random_state=SEED + 271829, unit="clone")
    ac_left = df[(df["time_step"] == A.ref_t1) & (df["replicate"] == reps[0])].reset_index(drop=True)

    rows = []
    for t in [x for x in TIMES if (df["time_step"] == x).any()]:
        raw = df[df["time_step"] == t].reset_index(drop=True)
        tw = assign_twin_id(raw).reset_index(drop=True)
        s1, _ = _step1(raw, SEED + 1)
        s2, tm, rm = _step2(raw, tw, SEED + 271829, SEED + 271832, n_cores=A.n_cores)
        _, _, st3 = identify_reg_if_multiple_states(
            tm1, tm, rm1, rm, [PAIR], GENES, tw1, tw, t1_raw=r1, t2_raw=raw,
            alpha=0.01, n_shuffles=A.n_shuffles, shuffle_seed=SEED + 271831,
            unit="clone", n_cores_to_use=A.n_cores)
        s3 = _pv(st3) or {}
        acr = df[(df["time_step"] == t) & (df["replicate"] == reps[1])].reset_index(drop=True)
        try:
            s4 = _step4(ac_left, acr, n_cores=A.n_cores)
        except Exception:  # noqa: BLE001
            s4 = {}
        rows.append({
            "rep": rep, "time": t,
            "rho_gene": s1["rho"], "z_gene": s1["z"],
            "rho_delta": s2["rho_delta"], "rho_delta_random": s2["rho_delta_random"],
            "z_het": s2["z_het"], "z_reg": s2["z_div"],
            "step3_d": s3.get("d"), "z_d": s3.get("z_d"),
            "rho_cross_1to2": s4.get("step4_rho_cross_1to2"),
            "rho_cross_2to1": s4.get("step4_rho_cross_2to1"),
            "z_1to2": s4.get("step4_z_1to2"), "z_2to1": s4.get("step4_z_2to1"),
            "z_gamma": s4.get("z_gamma_1to2"),
        })
    print(f"  [{A.type_tag} rep {rep}] {len(rows)} timepoints", flush=True)
    return rows


Z_SPECS = [("z_gene", "Step 1  z (gene-gene rho)"), ("z_het", "Step 2  z_het"),
           ("z_reg", "Step 2  z_regulation"), ("z_d", "Step 3  z_d"),
           ("z_1to2", "Step 4  z  g1->g2"), ("z_2to1", "Step 4  z  g2->g1"),
           ("z_gamma", "Step 4  z_gamma")]
C_SPECS = [("rho_gene", "gene-gene rho"), ("rho_delta", "twin rho_Delta"),
           ("rho_delta_random", "random-pair rho_Delta"), ("step3_d", "d = rho_D(t)-rho_D(t1)"),
           ("rho_cross_1to2", "cross rho g1->g2"), ("rho_cross_2to1", "cross rho g2->g1")]


def fig(agg, specs, ylab, title, stem, thr=None):
    n = len(specs); ncol = 3; nrow = int(np.ceil(n / ncol))
    f, ax = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 3.3 * nrow), squeeze=False, sharex=True)
    for a, (col, name) in zip(ax.flat, specs):
        m, s = agg[f"{col}_mean"], agg[f"{col}_std"]
        a.plot(agg.time, m, "-o", ms=3, color="#0072B2")
        a.fill_between(agg.time, m - s, m + s, color="#0072B2", alpha=0.15)
        a.axhline(0, color="0.7", lw=0.8)
        if thr:
            for y in (thr, -thr):
                a.axhline(y, ls="--", lw=1, color="0.5")
        a.set_title(name, fontsize=10); a.set_ylabel(ylab); a.grid(alpha=0.25)
    for a in ax.flat[n:]:
        a.set_visible(False)
    for a in ax[-1]:
        a.set_xlabel("twin time after division (h)")
    f.suptitle(title, fontsize=12)
    f.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(A.out_dir, stem)
    f.savefig(p, dpi=160, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    os.makedirs(A.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(A.sim_dir, f"df_rows_0_0_*_{A.type_tag}_*.csv")), key=_rep)
    print(f"{len(files)} files  x {len([x for x in TIMES])} timepoints  "
          f"(every {A.every}h to {A.tmax}h, ref t1={A.ref_t1}, N={A.n_shuffles})", flush=True)
    res = Parallel(n_jobs=min(A.jobs, max(1, len(files))), backend="loky")(
        delayed(one_file)(f) for f in files)
    df = pd.DataFrame([r for sub in res for r in sub])
    df.insert(0, "type_tag", A.type_tag)
    df.to_csv(os.path.join(A.out_dir, f"{A.type_tag}_zscores_vs_time_perrep.csv"), index=False)

    vcols = [c for c in df.columns if c not in ("type_tag", "rep", "time")]
    agg = df.groupby("time")[vcols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("time")
    agg.insert(0, "type_tag", A.type_tag)
    agg.to_csv(os.path.join(A.out_dir, f"{A.type_tag}_zscores_vs_time.csv"), index=False)

    ttl = f"{A.type_tag}  -  {len(files)} reps  -  ref t1={A.ref_t1}h, N={A.n_shuffles}"
    fig(agg, Z_SPECS, "z", ttl + "  -  z-scores vs time", f"{A.type_tag}_zscores_vs_time.png", thr=2.33)
    fig(agg, C_SPECS, r"$\rho$", ttl + "  -  correlations vs time", f"{A.type_tag}_correlations_vs_time.png")
