# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_cells = 500

# synthetic universe: TF0..TF4, targets T0..T19
# CollecTRI curation (synthetic): TF0 -> T0,T1,T2,T5 (4 curated targets)
#                                  TF1 -> T3,T4       (2 curated targets)
# T5 is curated for TF0 but will be given a WEAK true correlation with TF0 -- to check ranking works
# T6..T19 are NOT curated for any TF -- must never appear as a target in results
collectri_tf_targets = {"TF0": {"T0", "T1", "T2", "T5"}, "TF1": {"T3", "T4"}}
hvg_genes = [f"TF{i}" for i in range(5)] + [f"T{j}" for j in range(20)]
hvg_set = set(hvg_genes)

candidate_tfs = [g for g in hvg_genes if g in collectri_tf_targets]
tf_curated_targets = {tf: sorted((collectri_tf_targets[tf] & hvg_set) - {tf}) for tf in candidate_tfs}
candidate_tfs = [tf for tf in candidate_tfs if len(tf_curated_targets[tf]) > 0]
candidate_targets = sorted(set().union(*tf_curated_targets.values()))
assert candidate_tfs == ["TF0", "TF1"], candidate_tfs
assert candidate_targets == sorted({"T0", "T1", "T2", "T3", "T4", "T5"}), candidate_targets
print("candidate_tfs/candidate_targets construction: PASS")

hvg_index = {g: i for i, g in enumerate(hvg_genes)}
tf_idx = np.array([hvg_index[g] for g in candidate_tfs])
tgt_idx = np.array([hvg_index[g] for g in candidate_targets])

target_index = {g: j for j, g in enumerate(candidate_targets)}
curated_mask = np.zeros((len(candidate_tfs), len(candidate_targets)), dtype=bool)
for i, tf in enumerate(candidate_tfs):
    for tgt in tf_curated_targets[tf]:
        curated_mask[i, target_index[tgt]] = True
# TF0 row should have True at T0,T1,T2,T5 and False at T3,T4; TF1 row: True at T3,T4 only
assert curated_mask[0].tolist() == [target_index[t] in [target_index["T0"], target_index["T1"], target_index["T2"], target_index["T5"]] for t in candidate_targets]
print("curated_mask construction: PASS")

# build actual expression: TF0 strongly drives T0,T1,T2 (|r|~0.9), weakly drives T5 (|r|~0.3, still real,
# should still be found as the weakest of TF0's curated targets), T3/T4 driven by TF1 (~0.8).
# Give T6 (NOT curated for TF0) an even STRONGER correlation with TF0 (~0.95) -- must be EXCLUDED anyway.
Xh = rng.normal(size=(n_cells, len(hvg_genes)))
def set_corr(col_driver, col_target, r):
    Xh[:, col_target] = r * Xh[:, col_driver] + np.sqrt(1 - r**2) * rng.normal(size=n_cells)

i_tf0, i_tf1 = hvg_index["TF0"], hvg_index["TF1"]
set_corr(i_tf0, hvg_index["T0"], 0.9)
set_corr(i_tf0, hvg_index["T1"], 0.85)
set_corr(i_tf0, hvg_index["T2"], 0.8)
set_corr(i_tf0, hvg_index["T5"], 0.3)   # weak but real, still TF0's curated target
set_corr(i_tf0, hvg_index["T6"], 0.95)  # NOT curated -- must never appear in TF0's picked targets
set_corr(i_tf1, hvg_index["T3"], 0.8)
set_corr(i_tf1, hvg_index["T4"], 0.75)

def corr_matrix(mat, rows_idx, cols_idx):
    Xr = mat[:, rows_idx]; Xc = mat[:, cols_idx]
    Xr = (Xr - Xr.mean(axis=0)) / (Xr.std(axis=0) + 1e-12)
    Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-12)
    return (Xr.T @ Xc) / Xr.shape[0]

R = corr_matrix(Xh, tf_idx, tgt_idx)
R_masked = np.where(curated_mask, np.abs(R), np.nan)

N_TARGETS_PER_TF = 3
i = candidate_tfs.index("TF0")
row = pd.Series(R_masked[i], index=candidate_targets).dropna().sort_values(ascending=False)
top_targets = row.index[:N_TARGETS_PER_TF].tolist()
print("TF0's top-3 curated targets picked:", top_targets, row.head(3).round(3).tolist())
assert set(top_targets) == {"T0", "T1", "T2"}, top_targets  # T5 (weak) excluded from top-3, T6 never eligible
assert "T6" not in row.index  # T6 must not even be a candidate for TF0 despite its high raw correlation
print("per-TF curated-only top-N selection (excludes non-curated T6 despite higher |r|): PASS")

tf_mean_absr = pd.Series(np.nanmean(R_masked, axis=1), index=candidate_tfs).sort_values(ascending=False)
assert tf_mean_absr.index[0] == "TF0"
print("TF ranking by mean curated |r|: PASS")

print("\nALL COLLECTRI-LOGIC SMOKE ASSERTIONS PASSED")
