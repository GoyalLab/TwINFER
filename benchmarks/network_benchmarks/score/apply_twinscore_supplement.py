"""First test of the 2026-09-17 TwinScore supplement's continuous formula
(handoff/... TwinScore_supplement.pdf) on the e13_pos100 network_sweep benchmark (3 topologies,
30 replicates, t1=1/t2=20 -- the standard measurement point already used everywhere else in this
benchmark family).

TwinScore(x->y) = w*s(phi_x) + s(PAIR) + kappa_gamma * 1[gate] * log(q(x->y))
PAIR = s(D) + g*s(R) + g*v*s(Wz)

Mapping onto quantities ALREADY computed by infer_with_twinfer (twin_score_inputs /
gated_regulation, see analytic_zscores.py and correlation_functions.py):
  R   = CLR(z_reg_gated)   -- z_reg_gated is exactly the paper's raw zreg=(S-lambda*C)/sdreg
                              (calculate_gated_regulation_statistic); CLR calibration is applied
                              here, freshly, over the panel -- it was NOT already CLR'd upstream.
  z_dagger(x->y), z_dagger(y->x) = z_signed(rho_cross_xy/yx, ...) -- exactly the paper's z^dagger.
  gamma, kappa_gamma, q, direction term -- computed fresh from z_dagger per the formulas in
                              section 3.3 (r0=0.69 fixed, not fitted, as instructed by the paper).

Two pieces are NOT in any existing pipeline and are approximated here, DISCLOSED, not literal:
  D (dependence, "PIDC at the larger sample; any snapshot method fits") -- no isolated t2-only
    PIDC score is available without rerunning BEELINE differently, so this uses
    s(z_abs_rho_t2) (the already-computed |rho(t2)| z-score) as the snapshot-dependence proxy.
    The paper explicitly licenses substituting the snapshot method ("any snapshot method fits").
  g (twin-signal gate, "0 where twins carry no information (hPSC), ~1 elsewhere") -- this is a
    Gillespie simulation with real division-linked heterogeneity, not the flat hPSC case the gate
    exists to guard against, so g is set to 1 (fully on) rather than computed from a fresh zC.

phi_x, w (persistence, reliability) ARE computed properly here, from the raw simulation CSV
directly (clone_id, cell_id, time_step, gene_i_mRNA columns) -- these are the genuinely new
per-gene quantities the supplement introduces; nothing upstream computes them. Weights are
uniform (every clone has exactly 2 cells, one twin pair, at every timestep in this benchmark),
so weighted Spearman collapses to ordinary Spearman.

Wz (twin-layer partial correlation) is OMITTED -- the paper itself notes "dropping it changes
little" and it requires a split-half re-simulation of count noise this benchmark's saved data
doesn't carry.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import auc, precision_recall_curve

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

ROOT = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results'
JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs"
BEELINE_CSV = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv"

R0 = 0.69
Z_THR = 2.576


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def signal_share(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return 0.0
    return max(0.0, 1.0 - 1.0 / z.var(ddof=1))


def clr_calibrate(M, genes):
    """CLR calibration of a symmetric pair matrix, section 2: ux(x,y) standardised over the row
    of x, uy over the row of y, c(x,y) = sqrt(max(0,ux)^2 + max(0,uy)^2)."""
    n = len(genes)
    row_mean = {}
    row_sd = {}
    for g in genes:
        vals = np.array([M[g][h] for h in genes if h != g and np.isfinite(M[g].get(h, np.nan))])
        row_mean[g] = vals.mean() if len(vals) else 0.0
        row_sd[g] = vals.std(ddof=1) if len(vals) > 1 else 1.0

    def c(x, y):
        mxy = M[x].get(y, np.nan)
        if not np.isfinite(mxy):
            return np.nan
        ux = (mxy - row_mean[x]) / max(row_sd[x], 1e-12)
        uy = (mxy - row_mean[y]) / max(row_sd[y], 1e-12)
        return np.sqrt(max(0.0, ux) ** 2 + max(0.0, uy) ** 2)

    return c


# ---------------------------------------------------------------------------
# Persistence / heritability, computed fresh from the raw simulation CSV
# ---------------------------------------------------------------------------
def compute_persistence(raw_csv_path, gene_names, t1, t2):
    df = pd.read_csv(raw_csv_path, usecols=["cell_id", "time_step", "clone_id"] +
                      [f"{g}_mRNA" for g in gene_names])
    d1 = df[df.time_step == t1]
    d2 = df[df.time_step == t2]

    def twin_cols(d, col):
        """Returns (twin_a, twin_b) Series indexed by clone_id, for one gene column."""
        g = d.sort_values(["clone_id", "cell_id"]).groupby("clone_id")
        wide = g[col].apply(list)
        wide = wide[wide.apply(len) == 2]
        a = wide.apply(lambda v: v[0])
        b = wide.apply(lambda v: v[1])
        return a, b

    phi = {}
    h_t1 = {}
    h_t2 = {}
    rho_dagger = {}
    for gene in gene_names:
        col = f"{gene}_mRNA"
        # heritability at t1, t2: within-sample twin self-correlation
        x1a, x1b = twin_cols(d1, col)
        common1 = x1a.index.intersection(x1b.index)
        h1, _ = spearmanr(x1a.loc[common1], x1b.loc[common1])

        x2a, x2b = twin_cols(d2, col)
        common2 = x2a.index.intersection(x2b.index)
        h2, _ = spearmanr(x2a.loc[common2], x2b.loc[common2])

        # cross-sample self-correlation, both orderings pooled (clone-mate at t1 vs t2)
        common_clones = sorted(set(common1) & set(common2))
        vals_t1_a = x1a.loc[common_clones].to_numpy()
        vals_t1_b = x1b.loc[common_clones].to_numpy()
        vals_t2_a = x2a.loc[common_clones].to_numpy()
        vals_t2_b = x2b.loc[common_clones].to_numpy()
        pooled_t1 = np.concatenate([vals_t1_a, vals_t1_b])
        pooled_t2 = np.concatenate([vals_t2_b, vals_t2_a])  # cross to the OTHER sibling
        rho_dag, _ = spearmanr(pooled_t1, pooled_t2)

        h_t1[gene], h_t2[gene], rho_dagger[gene] = h1, h2, rho_dag
        n_null = max(len(common_clones) - 1, 2)
        null_sd = 1.0 / np.sqrt(n_null)
        # "if h_g below 2 null sd at either sample, phi_g set to panel median" -- flagged, filled after loop
        denom = np.sqrt(max(h1, 1e-9) * max(h2, 1e-9)) if (h1 > 0 and h2 > 0) else np.nan
        phi[gene] = rho_dag / denom if (np.isfinite(denom) and denom > 0 and
                                          h1 > 2 * null_sd and h2 > 2 * null_sd) else np.nan

    finite_phi = [v for v in phi.values() if np.isfinite(v)]
    panel_median = np.median(finite_phi) if finite_phi else 0.0  # every gene failed the h>2*null_sd gate
    phi = {g: (v if np.isfinite(v) else panel_median) for g, v in phi.items()}

    # reliability of persistence w = 1 - floor/var_genes(phi)
    phi_vals = np.array(list(phi.values()))
    var_phi = phi_vals.var(ddof=1) if len(phi_vals) > 1 else np.nan
    # simplified delta-method noise floor: approximate per-gene noise var via null sd of rho_dagger
    noise_vars = []
    for gene in gene_names:
        n_null = max(len(common_clones) - 1, 2)
        var_rho = 1.0 / n_null
        h1, h2 = h_t1[gene], h_t2[gene]
        if h1 > 0 and h2 > 0 and np.isfinite(rho_dagger[gene]):
            v = phi[gene] ** 2 * (var_rho / max(rho_dagger[gene], 1e-6) ** 2 +
                                    var_rho / (4 * h1 ** 2) + var_rho / (4 * h2 ** 2))
            noise_vars.append(v)
    floor = np.median(noise_vars) if noise_vars else 0.0
    w = max(0.0, 1.0 - floor / var_phi) if (var_phi and np.isfinite(var_phi) and var_phi > 0) else 0.0

    return phi, w, h_t1, h_t2


def full_report(sc, y):
    sc = np.asarray(sc, float)
    if not np.isfinite(sc).any():
        return dict(auprc_x=float("nan"), degenerate=True)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                degenerate=False)


def score_one(json_path):
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    gr = d.get("gated_regulation")
    if tsi is None or gr is None or gr.get("z_reg_gated") is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) < 3:
        return None

    gene_names = d["gene_names"]
    t1, t2 = d["settings"]["t1"], d["settings"]["t2"]
    z_reg_map = gr["z_reg_gated"]

    # --- R = CLR(z_reg_gated) ---
    M = {g: {} for g in gene_names}
    for key, val in z_reg_map.items():
        a, b = key.split("__")
        M[a][b] = val
        M[b][a] = val
    c_fn = clr_calibrate(M, gene_names)

    # --- persistence / heritability, fresh from raw sim data ---
    phi, w, h_t1, h_t2 = compute_persistence(d["source_file"], gene_names, t1, t2)

    U = list(zip(dd.gene_1, dd.gene_2))
    z_dagger_xy = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_dagger_yx = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    gamma_raw = (np.abs(z_dagger_xy) - np.abs(z_dagger_yx)) / np.sqrt(2 * (1 - 2 / np.pi))
    kappa_gamma = signal_share(gamma_raw)
    q = 0.5 + (R0 - 0.5) * (2 * norm.cdf(np.abs(gamma_raw)) - 1) * np.sign(gamma_raw)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    gate = (np.maximum(np.abs(z_dagger_xy), np.abs(z_dagger_yx)) > Z_THR).astype(float)
    direction_term = kappa_gamma * gate * np.log(q)

    D = dd.z_abs_rho_t2.to_numpy()  # snapshot-dependence proxy, disclosed above
    R = np.array([c_fn(a, b) for a, b in U])
    g_gate = 1.0  # twin-signal gate, disclosed above

    PAIR = s(D) + g_gate * s(R)
    fallback = np.median(list(phi.values())) if phi else 0.0
    phi_x = np.array([phi.get(a, fallback) for a, b in U])
    score = w * s(phi_x) + s(PAIR) + direction_term

    # build directed universe (each unordered pair scored is entered once here; TwinScore is
    # inherently directional but this benchmark's ground truth/competitors are scored on the
    # x->y direction present in twin_score_inputs, matching todo4v2_sim_scoring.py's convention)
    scored_pairs = set(U)
    U_full = [(a, b) for a in gene_names for b in gene_names if a != b and (a, b) in scored_pairs]
    score_map = dict(zip(U, score))
    gt_path = d.get("ground_truth_matrix")
    if gt_path is None or not os.path.exists(gt_path):
        return None
    Mgt = np.loadtxt(gt_path, delimiter=",")
    true_edges = {(gene_names[i], gene_names[j]) for i in range(len(gene_names))
                  for j in range(len(gene_names)) if i != j and Mgt[i, j] != 0}
    y_full = np.array([1 if p in true_edges else 0 for p in U_full])
    if y_full.sum() < 1 or y_full.sum() == len(y_full):
        return None
    sc_full = np.array([score_map[p] for p in U_full])
    rep = full_report(sc_full, y_full)

    return dict(dataset_id=d.get("dataset_id"), n_pairs=len(U_full), n_true=int(y_full.sum()),
                w=w, twinscore_auprc_x=rep["auprc_x"], degenerate=rep["degenerate"])


def main():
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"e13_pos100: {len(files)} files")
    rows = [r for r in (score_one(f) for f in files) if r]
    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{HERE}/twinscore_supplement_e13_pos100_results.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/twinscore_supplement_e13_pos100_results.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nmean twinscore_auprc_x = {df.twinscore_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable, mean w={df.w.mean():.3f})")

    b = pd.read_csv(BEELINE_CSV)
    means = b.groupby("algorithm")["auprc"].mean().sort_values(ascending=False)
    rand = (df.n_true / df.n_pairs).mean()
    print("\nBEELINE competitors (auprc_x = mean auprc / mean random baseline):")
    print((means / rand).round(3).to_string())


if __name__ == "__main__":
    main()
