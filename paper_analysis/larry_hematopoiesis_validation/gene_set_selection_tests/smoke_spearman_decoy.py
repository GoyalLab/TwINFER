# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(0)
n_cells = 800

# synthetic universe: TF0, TF1 as TFs; T0..T39 as targets, varying in detection level
n_tf, n_tgt = 2, 40
hvg_genes = [f'TF{i}' for i in range(n_tf)] + [f'T{j}' for j in range(n_tgt)]
hvg_index = {g: i for i, g in enumerate(hvg_genes)}

# base counts: give targets a range of detection rates (some sparse, some dense) via zero-inflation
base = rng.normal(size=(n_cells, len(hvg_genes)))
detection_rate = np.linspace(0.1, 0.95, n_tgt)  # T0 sparse .. T39 dense
for j, g in enumerate([f'T{k}' for k in range(n_tgt)]):
    mask = rng.random(n_cells) > detection_rate[j]
    base[mask, hvg_index[g]] = 0.0  # zero-inflate to control detection fraction

Xh = base.copy()
def set_corr(a_idx, b_idx, r):
    Xh[:, b_idx] = r * Xh[:, a_idx] + np.sqrt(max(1 - r**2, 0)) * rng.normal(size=n_cells)

# TF0 -curated- targets: T5 (real strong), T35 (real, similar detection decile as decoys should match)
# collectri_tf_targets: TF0 -> {T5, T35}; TF1 -> {T10}
collectri_tf_targets = {'TF0': {'T5', 'T35'}, 'TF1': {'T10'}}
set_corr(hvg_index['TF0'], hvg_index['T5'], 0.6)
set_corr(hvg_index['TF0'], hvg_index['T35'], 0.5)
set_corr(hvg_index['TF1'], hvg_index['T10'], 0.55)

candidate_tfs = list(collectri_tf_targets.keys())
hvg_set = set(hvg_genes)
tf_curated_targets = {tf: sorted((collectri_tf_targets[tf] & hvg_set) - {tf}) for tf in candidate_tfs}
candidate_targets = sorted(set().union(*tf_curated_targets.values()))
tf_idx = np.array([hvg_index[g] for g in candidate_tfs])
tgt_idx = np.array([hvg_index[g] for g in candidate_targets])
target_index = {g: j for j, g in enumerate(candidate_targets)}
curated_mask = np.zeros((len(candidate_tfs), len(candidate_targets)), dtype=bool)
for i, tf in enumerate(candidate_tfs):
    for tgt in tf_curated_targets[tf]:
        curated_mask[i, target_index[tgt]] = True

def spearman_matrix(mat, rows_idx, cols_idx):
    from scipy.stats import rankdata
    Rk = np.apply_along_axis(rankdata, 0, mat).astype(np.float64)
    Rk = (Rk - Rk.mean(axis=0)) / np.maximum(Rk.std(axis=0), 1e-9)
    Rr, Rc = Rk[:, rows_idx], Rk[:, cols_idx]
    return (Rr.T @ Rc) / Rr.shape[0]

R = spearman_matrix(Xh, tf_idx, tgt_idx)
# sanity: Spearman between TF0,T5 should be close to injected 0.6 (rank-based, a bit attenuated by
# zero-inflation ties, but should clearly recover a strong positive value)
assert R[0, target_index['T5']] > 0.3, R[0, target_index['T5']]
print(f"Spearman TF0-T5 (injected r=0.6, zero-inflated): {R[0, target_index['T5']]:.3f} -- PASS (recovers signal)")

# detection fractions for candidate_targets (needed for decile matching)
frac_expr = np.asarray((Xh > 0).mean(axis=0)).ravel()
det_of = pd.Series(frac_expr, index=hvg_genes)
hvg_det = det_of.loc[candidate_targets].to_numpy()
decile_edges = np.quantile(hvg_det, np.linspace(0, 1, 11)[1:-1])
decile = np.digitize(hvg_det, decile_edges)
by_decile = {d: np.flatnonzero(decile == d) for d in range(10)}

real_pairs = {(i, j) for i in range(len(candidate_tfs)) for j in range(len(candidate_targets)) if curated_mask[i, j]}
obs_absr, dec_absr = [], []
for (i, j) in real_pairs:
    obs_absr.append(abs(R[i, j]))
    pool = by_decile[decile[j]]
    for _ in range(20):
        c = int(rng.choice(pool))
        if c != j and (i, c) not in real_pairs:
            dec_absr.append(abs(R[i, c]))
            break
obs_absr, dec_absr = np.array(obs_absr), np.array(dec_absr)
assert len(obs_absr) == 3  # TF0-T5, TF0-T35, TF1-T10
assert len(dec_absr) == 3
print(f"curated |rho| values: {obs_absr.round(3)}")
print(f"decoy   |rho| values: {dec_absr.round(3)}  -- PASS (decoys drawn, matched-decile pool used)")

u = stats.mannwhitneyu(obs_absr, dec_absr, alternative="greater")
print(f"Mann-Whitney U ran without error: statistic={u.statistic}, p={u.pvalue:.3f} -- PASS")

print("\nALL SPEARMAN+DECOY SMOKE ASSERTIONS PASSED")
