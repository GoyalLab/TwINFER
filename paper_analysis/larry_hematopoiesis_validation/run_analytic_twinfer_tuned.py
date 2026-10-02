"""Analytic TwINFER with "everything except the panel": the CURRENT gene sets, but
    - log1p-CP10k normalized input (twinfer_input_cp10k)
    - UNWEIGHTED correlations (use_clone=False, unit="twin")
    - yscher's tuned TwinScore composition (helpers/twinscore_current.py) + the z_reg_gated gate

Tuned score (twinscore_current.py `TS`), with s() = panel z-standardization. Because s() is
shift/scale invariant and every analytic z is an affine function of the underlying correlation,
s(z_type) == s(rho_quantity), so the raw correlation quantities are used directly:

  E1  = s(|rho_t1|)
  CHG = s(|rho_t2 - rho_t1|)
  RD1 = s(|rho_Delta(t1)|)
  DR  = s(|rho_Delta_random(t2) - rho_Delta_random(t1)|)
  XS  = s(min(|rho_cross_xy|, |rho_cross_yx|))
  GAM = s(|gamma|),  gamma = |rho_cross_xy| - |rho_cross_yx|
  ZDD = s(rho_cross_xy - rho_cross_yx)                 # signed z_Ddag
  REG[g] = standardized (across genes) mean of (rho_cross(g,w) - rho_cross(w,g)) over partners w
  TS  = E1 - CHG - RD1 - DR - XS - GAM + 0.5*ZDD + (REG[x] - REG[y])
  gate: |z_reg_gated| > 1.645 ,  z_reg_gated = (rho_same - lambda*rho_cross_2n) / SD_HET_T1,
        lambda = min(1, |z_het| / 2.33) ;  failing pairs score = -E1

Writes resources/analytic_infer_tuned/{gs}/ranked_edges.csv with a `twinScore` column so
benchmark_methods.ipynb picks it up.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    split_twins,
)
from twinfer.scoring.analytic_zscores import SD_HET_T1

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INPUT_DIR = os.environ.get("INPUT_DIR", f"{HERE}/resources/twinfer_input_cp10k")
INPUT_DIR = os.environ.get("INPUT_DIR", f"{RES_HERE}/resources/twinfer_input_cp10k")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.environ.get("OUT_DIR", f"{HERE}/resources/analytic_infer_tuned")
OUT_DIR = os.environ.get("OUT_DIR", f"{RES_HERE}/resources/analytic_infer_tuned")
N_RAND = int(os.environ.get("N_RAND", "40"))
T1, T2, SEED = 2, 4, 0
GATE_Z = 1.645
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ALL = list(json.load(open(f"{HERE}/resources/gene_sets.json")))
ALL = list(json.load(open(f"{RES_HERE}/resources/gene_sets.json")))
GENE_SETS = os.environ.get("GENE_SETS", ",".join(ALL)).split(",")
os.makedirs(OUT_DIR, exist_ok=True)


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def run(gs):
    df = pd.read_csv(f"{INPUT_DIR}/{gs}.csv")
    # dict_to_matrix in the use_clone=False path splits gene keys on "-"; sanitize hyphenated
    # symbols (e.g. H2-Q7) and restore on write.
    orig = [c[:-5] for c in df.columns if c.endswith("_mRNA")]
    san = {g: g.replace("-", "_") for g in orig}
    inv = {v: k for k, v in san.items()}
    df = df.rename(columns={f"{g}_mRNA": f"{san[g]}_mRNA" for g in orig})
    genes = [san[g] for g in orig]
    gcols = [f"{g}_mRNA" for g in genes]
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    t1_tw = assign_twin_id(t1_raw).reset_index(drop=True)
    t2_tw = assign_twin_id(t2_raw).reset_index(drop=True)

    r1 = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, genes, use_clone=False)
    r2 = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, genes, use_clone=False)
    tw1, _ = calculate_twin_random_correlations(t1_raw, t1_tw, genes, random_state=SEED, unit="twin")
    rr1 = np.mean([calculate_twin_random_correlations(t1_raw, t1_tw, genes, random_state=SEED + 100 + i, unit="twin")[1].to_numpy()
                   for i in range(N_RAND)], axis=0)
    rr2 = np.mean([calculate_twin_random_correlations(t2_raw, t2_tw, genes, random_state=SEED + 500 + i, unit="twin")[1].to_numpy()
                   for i in range(N_RAND)], axis=0)
    rr1 = pd.DataFrame(rr1, index=genes, columns=genes)
    rr2 = pd.DataFrame(rr2, index=genes, columns=genes)
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in genes for b in genes if a != b]
    xc = get_cross_correlations(at1, at2, gene_pairs=ordered + [(g, g) for g in genes], unit="twin")

    # rho_same / rho_cross on the pooled 2n twin-member rows (for the gate)
    a0, b0 = split_twins(t1_tw)
    Xa = a0[gcols].to_numpy(float); Xb = b0[gcols].to_numpy(float)
    Xp = np.vstack([Xa, Xb])
    Rk = np.apply_along_axis(rankdata, 0, Xp); Rk -= Rk.mean(0)
    S = np.sqrt((Rk ** 2).sum(0)); S[S < 1e-12] = 1
    rho_same = (Rk.T @ Rk) / np.outer(S, S)
    Yc = np.vstack([Xb, Xa])
    RkY = np.apply_along_axis(rankdata, 0, Yc); RkY -= RkY.mean(0)
    SY = np.sqrt((RkY ** 2).sum(0)); SY[SY < 1e-12] = 1
    rho_cross_2n = (Rk.T @ RkY) / np.outer(S, SY)
    gx = {g: i for i, g in enumerate(genes)}

    rows = []
    for a, b in ordered:
        rt1, rt2 = float(r1.loc[a, b]), float(r2.loc[a, b])
        rd1 = float(tw1.loc[a, b])
        drr = float(rr2.loc[a, b] - rr1.loc[a, b])
        rxy, ryx = float(xc.loc[a, b]), float(xc.loc[b, a])
        het = (rd1 - float(rr1.loc[a, b])) / SD_HET_T1
        lam = min(1.0, abs(het) / 2.33)
        rreg = rho_same[gx[a], gx[b]] - lam * rho_cross_2n[gx[a], gx[b]]
        rows.append(dict(gene_1=a, gene_2=b, rho_t1=rt1, rho_t2=rt2, rho_delta_t1=rd1,
                         drr=drr, rxy=rxy, ryx=ryx, gamma=abs(rxy) - abs(ryx),
                         z_reg_gated=rreg / SD_HET_T1))
    D = pd.DataFrame(rows)

    reg_raw = D.assign(asym=D.rxy - D.ryx).groupby("gene_1")["asym"].mean()
    rv = reg_raw.to_numpy()
    REG = {g: ((reg_raw.get(g, np.nan) - rv.mean()) / max(rv.std(ddof=1), 1e-12)) for g in genes}

    E1 = s(np.abs(D.rho_t1))
    TS = (E1
          - s(np.abs(D.rho_t2 - D.rho_t1))
          - s(np.abs(D.rho_delta_t1))
          - s(np.abs(D.drr))
          - s(np.minimum(np.abs(D.rxy), np.abs(D.ryx)))
          - s(np.abs(D.gamma))
          + 0.5 * s(D.rxy - D.ryx)
          + np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(D.gene_1, D.gene_2)]))
    gate = np.abs(D.z_reg_gated.to_numpy()) > GATE_Z
    # SUPERSEDED 2026-09-09: gate-fail -> -E1 was meant to sink confounded high-|rho| pairs, but
    # downstream scorers take |twinScore|, which FLIPS a very-negative -E1 (weak / NaN rho_t1)
    # back to the top and lets degenerate genes (e.g. HSC CEBPa/Gata2, rho undefined) dominate.
    # D["twinScore"] = np.where(gate, TS, -E1)
    # Gate-fail -> NaN so every gate-failing pair is dropped by the loaders and floored to the
    # bottom of the ranking (a hard filter: failures are non-edges, ranked last, tied).
    D["twinScore"] = np.where(gate, TS, np.nan)
    D["u_abs_rho_t1"] = np.abs(D.rho_t1); D["u_abs_rho_t2"] = np.abs(D.rho_t2)  # for downstream tools

    D["gene_1"] = D["gene_1"].map(lambda g: inv.get(g, g))
    D["gene_2"] = D["gene_2"].map(lambda g: inv.get(g, g))
    d = f"{OUT_DIR}/{gs}"; os.makedirs(d, exist_ok=True)
    D.sort_values("twinScore", ascending=False).to_csv(f"{d}/ranked_edges.csv", index=False)
    return len(D), int(gate.sum())


if __name__ == "__main__":
    print(f"input {INPUT_DIR}  ->  {OUT_DIR}   (unweighted + tuned + gate, N_RAND={N_RAND})")
    for gs in GENE_SETS:
        t = time.time()
        n, ng = run(gs)
        print(f"  {gs:18s} {n:5d} directed pairs, {ng} pass the z_reg_gated gate   ({time.time()-t:.0f}s)", flush=True)
