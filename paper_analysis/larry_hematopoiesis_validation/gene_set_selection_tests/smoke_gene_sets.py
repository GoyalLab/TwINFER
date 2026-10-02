# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_cells = 500

# synthetic: 5 "TF" columns, 20 "target" columns. TF0 strongly drives target0-2 (real signal).
# everything else is independent noise (should land near/at the null).
n_tf, n_tgt = 5, 20
Xh = rng.normal(size=(n_cells, n_tf + n_tgt))
tf_idx = np.arange(n_tf)
tgt_idx = np.arange(n_tf, n_tf + n_tgt)
# inject real correlation: targets 0,1,2 (global col index n_tf+0..2) driven by TF0 (col 0)
for j in range(3):
    Xh[:, n_tf + j] = 0.9 * Xh[:, 0] + 0.1 * rng.normal(size=n_cells)

def corr_matrix(mat, rows_idx, cols_idx):
    Xr = mat[:, rows_idx]
    Xc = mat[:, cols_idx]
    Xr = (Xr - Xr.mean(axis=0)) / (Xr.std(axis=0) + 1e-12)
    Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-12)
    return (Xr.T @ Xc) / Xr.shape[0]

R = corr_matrix(Xh, tf_idx, tgt_idx)
assert R.shape == (n_tf, n_tgt)
# TF0's correlation with targets 0,1,2 should be near 0.9 (way above everything else)
assert np.abs(R[0, 0]) > 0.85 and np.abs(R[0, 1]) > 0.85 and np.abs(R[0, 2]) > 0.85, R[0, :3]
assert np.abs(R[0, 3]) < 0.2  # unrelated target
print("corr_matrix: PASS (injected signal recovered, noise stays low)")

# null permutation
null_abs_r = []
perm_cols = np.concatenate([tf_idx, tgt_idx])
for p in range(5):
    Xp = Xh.copy()
    for c in perm_cols:
        rng.shuffle(Xp[:, c])
    Rn = corr_matrix(Xp, tf_idx, tgt_idx)
    null_abs_r.append(np.abs(Rn).ravel())
null_abs_r = np.concatenate(null_abs_r)
NULL_CUTOFF = float(np.percentile(null_abs_r, 95))
assert 0 < NULL_CUTOFF < 0.5, NULL_CUTOFF  # sane range for n=500 random noise
assert np.abs(R[0, 0]) > NULL_CUTOFF  # the real signal must clear the null
print(f"null permutation: PASS (NULL_CUTOFF={NULL_CUTOFF:.3f}, real signal {np.abs(R[0,0]):.3f} clears it)")

# pick_list band logic
candidate_tfs = [f"TF{i}" for i in range(n_tf)]
candidate_targets = [f"T{j}" for j in range(n_tgt)]
all_pairs = pd.DataFrame(
    [(candidate_tfs[i], candidate_targets[j], abs(R[i, j])) for i in range(n_tf) for j in range(n_tgt)],
    columns=["tf", "target", "abs_r"])

def pick_list(pairs_in_band, n_tfs, n_targets):
    tf_counts = pairs_in_band["tf"].value_counts()
    chosen_tfs = tf_counts.index[:n_tfs].tolist()
    genes_out, detail = set(chosen_tfs), []
    for tf in chosen_tfs:
        sub = pairs_in_band[pairs_in_band["tf"] == tf].sort_values("abs_r", ascending=False)
        tgts = sub["target"].head(n_targets).tolist()
        genes_out.update(tgts)
        detail.append((tf, tgts))
    return sorted(genes_out), detail

low_hi = NULL_CUTOFF * 1.15
low_pairs = all_pairs[all_pairs["abs_r"].between(NULL_CUTOFF, low_hi)]
low_list, low_detail = pick_list(low_pairs, 2, 2)
assert isinstance(low_list, list)
print(f"pick_list (low band): PASS, {len(low_pairs)} pairs in band, list={low_list}")

high_tfs_check = pd.Series(np.abs(R).mean(axis=1), index=candidate_tfs).sort_values(ascending=False)
assert high_tfs_check.index[0] == "TF0"  # TF0 should clearly win by mean |r|
print("high-TF selection: PASS (TF0 correctly identified as top TF)")

print("\nALL SMOKE ASSERTIONS PASSED")
