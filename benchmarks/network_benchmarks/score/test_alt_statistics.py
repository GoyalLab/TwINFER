# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Compare 4 alternative Step-1 test statistics against the current Spearman
baseline, on the SAME cell partition the pipeline actually uses for
forced_step1_z_t1 (rho_t1 = t1_twins_raw + ac_left, ~6000 cells), across 10
A_rep_B replicates.

1. Spearman (current baseline) -- clone-block permutation null, 2000 shuffles.
2. Kendall's tau-b -- same permutation null, 2000 shuffles.
3. Hurdle (two-part): Part A = Mann-Whitney U of gene_1 level between
   gene_2==0 vs gene_2>0 cells; Part B = Spearman among gene_2>0 cells only;
   combined via Fisher's method (chi2, 4 df) -> z. Same permutation null,
   2000 shuffles.
4. Negative-binomial GLM (gene_2 ~ NB(mean=f(gene_1))): Wald z-stat on the
   gene_1 coefficient. NOTE: uses the GLM's own analytic null (NOT the
   clone-permutation null) -- too slow to refit per shuffle. Flagged.
5. Distance correlation: subsampled to 1500 cells (O(n^2) cost), clone-block
   permutation null, 500 shuffles. Flagged as subsampled.

All z-scores are signed to match Spearman's sign convention (negative =
repression) where the underlying statistic doesn't naturally carry a sign
(distance correlation, hurdle-combined).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
import glob
import re
import time
from scipy.stats import spearmanr, kendalltau, mannwhitneyu, chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf

T1, T2 = 1, 20
SEED = 101010
N_SHUFFLES = 2000
N_SHUFFLES_DCOR = 500
DCOR_SUBSAMPLE = 1500

rng_master = np.random.default_rng(0)


def partition_rho_t1(simulation, seed=SEED):
    rng = np.random.default_rng(seed)
    clone_ids = simulation["clone_id"].drop_duplicates().to_numpy()
    sh = rng.permutation(clone_ids)
    n = len(sh) // 4
    t1c, ac = sh[:n], sh[2 * n:]
    reps = sorted(simulation["replicate"].unique())
    t1_twins_raw = simulation[simulation["clone_id"].isin(t1c) & (simulation["time_step"] == T1)].copy()
    ac_left = simulation[simulation["clone_id"].isin(ac) & (simulation["time_step"] == T1)
                          & (simulation["replicate"] == reps[0])].copy()
    return pd.concat([t1_twins_raw, ac_left], ignore_index=True)


def clone_permute_labels(clone_ids, n_shuffles, seed):
    """Return n_shuffles permutations of gene_2 clone-block-consistent shuffle indices."""
    rng = np.random.default_rng(seed)
    unique_clones = np.unique(clone_ids)
    clone_to_idx = {c: np.where(clone_ids == c)[0] for c in unique_clones}
    n = len(clone_ids)
    perms = np.empty((n_shuffles, n), dtype=np.int64)
    for s in range(n_shuffles):
        shuffled_clones = rng.permutation(unique_clones)
        order = np.concatenate([clone_to_idx[c] for c in shuffled_clones])
        # build an index array that reassigns gene_2 values according to clone-block shuffle
        perms[s] = order
    return perms


def distance_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).sum() / (n * n)
    dvarx2 = (A * A).sum() / (n * n)
    dvary2 = (B * B).sum() / (n * n)
    denom = np.sqrt(dvarx2 * dvary2)
    return np.sqrt(dcov2) / np.sqrt(denom) if denom > 0 else 0.0


def hurdle_stat(x, y):
    """x = gene_1 (predictor), y = gene_2 (response, zero-inflated).
    Part A: Mann-Whitney U z-stat, gene_1 in y==0 vs y>0 groups.
    Part B: Spearman among y>0 only.
    Combined via Fisher's method -> chi2(4) -> signed z (sign from Part B rho,
    or from Part A direction if Part B has too few points)."""
    zero_mask = y == 0
    x0, x1 = x[zero_mask], x[~zero_mask]
    if len(x0) < 5 or len(x1) < 5:
        return np.nan, np.nan
    try:
        _, pA = mannwhitneyu(x1, x0, alternative="two-sided")
    except ValueError:
        pA = 1.0
    dirA = np.sign(np.median(x1) - np.median(x0)) if len(x1) and len(x0) else 0

    if (~zero_mask).sum() >= 10:
        rhoB, pB = spearmanr(x[~zero_mask], y[~zero_mask])
        dirB = np.sign(rhoB) if not np.isnan(rhoB) else dirA
    else:
        pB = 1.0
        dirB = dirA

    pA = min(max(pA, 1e-300), 1.0)
    pB = min(max(pB, 1e-300), 1.0)
    combined_stat = -2 * (np.log(pA) + np.log(pB))
    combined_p = chi2.sf(combined_stat, df=4)
    sign = dirB if dirB != 0 else dirA
    return combined_p, sign


def p_to_z(p, sign=1):
    from scipy.stats import norm
    p = min(max(p, 1e-300), 1 - 1e-16)
    return sign * abs(norm.isf(p / 2))


files = sorted(glob.glob(
    f'{TWINFER_PROJECT_ROOT}/simulation_data/figure_3_1k/A_rep_B/df_*.csv'))[:10]

results = []
for f in files:
    m = re.search(r"rep_(\d+)", f)
    rep_id = int(m.group(1))
    df = pd.read_csv(f)
    sub = partition_rho_t1(df)
    x = sub["gene_1_mRNA"].to_numpy(dtype=float)
    y = sub["gene_2_mRNA"].to_numpy(dtype=float)
    clone_ids = sub["clone_id"].to_numpy()
    n = len(x)
    t0 = time.time()

    # ---- observed statistics ----
    obs_spear, _ = spearmanr(x, y)
    obs_kendall, _ = kendalltau(x, y)
    obs_hurdle_p, obs_hurdle_sign = hurdle_stat(x, y)
    obs_hurdle_z_raw = -np.log(max(obs_hurdle_p, 1e-300))  # bigger = more extreme, sign applied after null calib

    # ---- clone-block permutation null (shared draws) ----
    perms = clone_permute_labels(clone_ids, N_SHUFFLES, seed=42)
    null_spear = np.empty(N_SHUFFLES)
    null_kendall = np.empty(N_SHUFFLES)
    null_hurdle = np.empty(N_SHUFFLES)
    for s in range(N_SHUFFLES):
        y_shuf = y[perms[s]]
        null_spear[s], _ = spearmanr(x, y_shuf)
        null_kendall[s], _ = kendalltau(x, y_shuf)
        p_h, sign_h = hurdle_stat(x, y_shuf)
        null_hurdle[s] = -np.log(max(p_h, 1e-300)) * (sign_h if sign_h != 0 else 1)

    z_spear = (obs_spear - null_spear.mean()) / null_spear.std()
    z_kendall = (obs_kendall - null_kendall.mean()) / null_kendall.std()
    obs_hurdle_signed = obs_hurdle_z_raw * (obs_hurdle_sign if obs_hurdle_sign != 0 else 1)
    z_hurdle = (obs_hurdle_signed - null_hurdle.mean()) / null_hurdle.std()

    # ---- NB GLM (analytic Wald z, NOT permutation-calibrated) ----
    try:
        nb_df = pd.DataFrame({"y": y.astype(int), "x": x})
        model = smf.glm("y ~ x", data=nb_df, family=sm.families.NegativeBinomial()).fit()
        z_nb = model.tvalues["x"]
    except Exception as e:
        z_nb = np.nan

    # ---- distance correlation (subsampled, own permutation null) ----
    sub_idx = rng_master.choice(n, size=min(DCOR_SUBSAMPLE, n), replace=False)
    xs, ys = x[sub_idx], y[sub_idx]
    clone_s = clone_ids[sub_idx]
    obs_dcor = distance_correlation(xs, ys)
    perms_d = clone_permute_labels(clone_s, N_SHUFFLES_DCOR, seed=43)
    # perms_d indexes into the subsample array directly since clone_s has its own ordering
    null_dcor = np.empty(N_SHUFFLES_DCOR)
    for s in range(N_SHUFFLES_DCOR):
        null_dcor[s] = distance_correlation(xs, ys[perms_d[s] % len(ys)])
    z_dcor_mag = (obs_dcor - null_dcor.mean()) / null_dcor.std()
    z_dcor = z_dcor_mag * np.sign(obs_spear)  # distance correlation has no sign; borrow Spearman's

    elapsed = time.time() - t0
    results.append((rep_id, z_spear, z_kendall, z_hurdle, z_nb, z_dcor, elapsed))
    print(f"rep {rep_id}: spearman_z={z_spear:.2f}  kendall_z={z_kendall:.2f}  "
          f"hurdle_z={z_hurdle:.2f}  nb_wald_z={z_nb:.2f}  dcor_z={z_dcor:.2f}  ({elapsed:.0f}s)",
          flush=True)

res = pd.DataFrame(results, columns=["rep_id", "spearman_z", "kendall_z", "hurdle_z", "nb_z", "dcor_z", "elapsed_s"])
print()
print(res.describe())
print()
THRESH = 2.33
for col in ["spearman_z", "kendall_z", "hurdle_z", "nb_z", "dcor_z"]:
    n_pass = (res[col] < -THRESH).sum()
    print(f"{col}: {n_pass}/10 crossing -{THRESH}, mean={res[col].mean():.2f}")
