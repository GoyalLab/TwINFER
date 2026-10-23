"""TwinScore §4–6 statistics per gene pair and the four-scenario test, network_sweep (main families, e13/e17 excluded).
Data: simulation_data/network_sweep_final df_*.csv, time t1 = 1, twin pairs = the two cells (replicate 1 and 2) of each clone.
Values v = 1e4 * mRNA / sum(mRNA) per cell. Ranks = average ranks over the cells of B (weights are 1/2 per cell).
Stage I:  rho = corr(R_x, R_y);  sigma_rho^2 = (1/N1^2) sum_c [1/nc + (1 - 1/nc) hx hy];  z_rho = rho/sigma_rho.
Within-sample twin stats (weights w_p = 1/P_c = 1 for nc = 2; N_t = #clones; sums divided by N_t so S, C are correlations):
  S, C, U = S - C, hx = C(x,x), rho_delta = U/sqrt((1-hx)(1-hy)), sigma_C^2 = [1 + S^2 + hx hy]/(2 N_t),
  sigma_delta^2 = 1/N_t,  z_C = C/sigma_C (shared part),  z_U = rho_delta/sigma_delta (twin difference).
Scenario: 1 nothing; 2 regulation (z_rho sig, shared not); 3 regulation + heterogeneity (shared sig, z_U sig);
4 heterogeneity (shared sig, z_U not sig).  Truth: unordered pair with an edge in either direction (sweep matrix).
Date: 2026-10-06
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
SIM = ROOT / "simulation_data" / "network_sweep_final"
GT = ROOT / "input_data" / "network_sweep_final"
OUT = ROOT / "analysis_data" / "synthetic_network_simulations"
THR = 1.96
T1 = 1
MAIN = ["grn_n6_e5_pos100_density", "grn_n6_e9_pos100_center", "grn_n6_e9_pos0_sign_ratio", "grn_n6_e9_pos50_sign_ratio"]


def sig(z):
    return np.isfinite(z) and abs(z) > THR


def stats_one(path):
    d = pd.read_csv(path)
    gcols = [c for c in d.columns if c.endswith("_mRNA")]
    G = len(gcols)
    s1 = d[(d.time_step == T1) & (d.replicate == 1)].sort_values("clone_id")
    s2 = d[(d.time_step == T1) & (d.replicate == 2)].sort_values("clone_id")
    s1 = s1[s1.clone_id.isin(s2.clone_id)]
    s2 = s2[s2.clone_id.isin(s1.clone_id)]
    X = d[(d.time_step == T1)]
    tot = X[gcols].sum(axis=1)
    bad = X.clone_id[tot == 0].unique()  # drop clones with an empty cell so twins stay paired
    X = X[~X.clone_id.isin(bad)]
    V = X[gcols].to_numpy(float)
    V = 1e4 * V / V.sum(axis=1, keepdims=True)
    R = np.column_stack([rankdata(V[:, g]) for g in range(G)])  # average ranks over all cells of B
    Ra = R[X.replicate.to_numpy() == 1][np.argsort(X.clone_id.to_numpy()[X.replicate.to_numpy() == 1])]
    Rb = R[X.replicate.to_numpy() == 2][np.argsort(X.clone_id.to_numpy()[X.replicate.to_numpy() == 2])]
    Nt = Ra.shape[0]
    mu = (Ra + Rb).mean(axis=0) / 2
    ra, rb = Ra - mu, Rb - mu
    Mg = ((ra ** 2 + rb ** 2) / 2).mean(axis=0)
    S = ((ra[:, :, None] * ra[:, None, :] + rb[:, :, None] * rb[:, None, :]) / 2).mean(axis=0) / np.sqrt(np.outer(Mg, Mg))
    C = ((ra[:, :, None] * rb[:, None, :] + rb[:, :, None] * ra[:, None, :]) / 2).mean(axis=0) / np.sqrt(np.outer(Mg, Mg))
    U = S - C
    hx = np.clip(np.diag(C), 0, None)
    hh = np.outer(hx, hx)
    rho_delta_num = np.zeros((G, G))
    dA = ra - rb
    rho_delta = (dA.T @ dA) / Nt / np.sqrt(np.outer((dA ** 2).mean(axis=0), (dA ** 2).mean(axis=0)))
    Rall = R
    rho = np.corrcoef(Rall.T)
    N1 = Nt  # clones; all clones have nc = 2
    sig_rho = np.sqrt((0.5 + 0.5 * hh) / N1)
    z_rho = rho / sig_rho
    sig_C = np.sqrt((1 + S ** 2 + hh) / (2 * Nt))
    z_C = C / sig_C
    sig_delta = np.sqrt(1 / Nt)
    z_U = rho_delta / sig_delta
    return gcols, z_rho, z_C, z_U


def truth_matrix(stem, G):
    M = np.loadtxt(GT / f"{stem}.txt", delimiter=",")
    return (M != 0) | (M.T != 0)


rows = []
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
per_family = {}
for path in sorted(glob.glob(str(SIM / "df_*.csv"))):
    name = Path(path).name
    stem = re.sub(r"^df_", "", name)
    stem = re.sub(r"_\d{2}\d{4}\d{4}.*$", "", stem)  # strip timestamp and hash
    stem = re.sub(r"_ncells.*$", "", stem)
    stem = re.sub(r"_rep\d+_rep\d+$", "", stem)
    if stem not in MAIN and not any(stem.startswith(m) for m in MAIN):
        continue
    key = next(m for m in MAIN if stem.startswith(m))
    if LIMIT is not None and per_family.get(key, 0) >= LIMIT:
        continue
    per_family[key] = per_family.get(key, 0) + 1
    gt_file = GT / f"{key}_rep0.txt"
    if not gt_file.exists():
        continue
    gcols, z_rho, z_C, z_U = stats_one(path)
    G = len(gcols)
    truth = np.zeros((G, G), bool)
    Mx = np.loadtxt(GT / f"{key}.txt" if (GT / f"{key}.txt").exists() else gt_file, delimiter=",")
    truth = (Mx != 0) | (Mx.T != 0)
    for i in range(G):
        for j in range(i + 1, G):
            zr, zc, zu = z_rho[i, j], 0.5 * (z_C[i, j] + z_C[j, i]), 0.5 * (z_U[i, j] + z_U[j, i])
            if sig(zc):
                sc = 3 if sig(zu) else 4
            else:
                sc = 2 if sig(zr) else 1
            rows.append(dict(file=name, family=key, pair=f"{gcols[i]}|{gcols[j]}", z_rho=zr, z_C=zc, z_U=zu,
                             scenario=sc, truth=bool(truth[i, j])))
df = pd.DataFrame(rows)
df.to_csv(OUT / "twinscore_scenario_network_sweep_pairs.csv", index=False)
print("files", df.file.nunique(), "pairs", len(df))
print(pd.crosstab(df.scenario, df.truth, margins=True).to_string())
print("\nP(scenario | true edge):", df[df.truth].scenario.value_counts(normalize=True).sort_index().round(3).to_dict())
print("P(scenario | non-edge):", df[~df.truth].scenario.value_counts(normalize=True).sort_index().round(3).to_dict())
