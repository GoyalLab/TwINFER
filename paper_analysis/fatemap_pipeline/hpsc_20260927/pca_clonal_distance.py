#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""50-PC Euclidean distance comparison: clonal (twin/sister, same clone_id) cell pairs vs
non-clonal (random cross-clone) cell pairs, on hPSC_20260927's QC-filtered matrix.
Standard log1pCP10k -> top-2000-HVG -> scale -> PCA(50) recipe (same N_HVG/N_PC convention
as fatemap_integration_utils.py, no batch_key needed here since this is single-sample).
Run separately per timepoint (endo_T0, more twin pairs) to avoid the two-timepoint
transcriptomic shift dominating the distance comparison.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import anndata as ad

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
N_HVG = 2000
N_PC = 50
SEED = 0
N_PAIRS_SAMPLE = 20000  # subsample pairs for speed if more are available


def run_for_sample(sample_label, time_step):
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    cells = np.array(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    mask = (obs["time_step"] == time_step).to_numpy()
    Xs = X[:, mask].T.tocsr()  # cells x genes
    obs_s = obs.loc[mask].copy()
    clone = obs_s["fatemap_clone_singletcode"].astype(str).to_numpy()

    A = ad.AnnData(X=Xs, obs=pd.DataFrame(index=obs_s.index), var=pd.DataFrame(index=genes))
    A.obs["clone_id"] = clone
    print(f"[{sample_label}] {A.n_obs} cells x {A.n_vars} genes, {pd.Series(clone).nunique()} clones", flush=True)

    A.X = A.X.astype(np.float32)
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=N_HVG)
    Ahvg = A[:, A.var["highly_variable"]].copy()
    sc.pp.scale(Ahvg, max_value=10)
    sc.tl.pca(Ahvg, n_comps=min(N_PC, Ahvg.n_vars - 1, Ahvg.n_obs - 1), svd_solver="arpack", random_state=SEED)
    pcs = Ahvg.obsm["X_pca"]
    print(f"[{sample_label}] PCA done: {pcs.shape}", flush=True)

    rng = np.random.default_rng(SEED)
    clone_to_idx = {}
    for i, c in enumerate(clone):
        clone_to_idx.setdefault(c, []).append(i)

    clonal_pairs = []
    for c, idxs in clone_to_idx.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                clonal_pairs.append((idxs[i], idxs[j]))
    clonal_pairs = np.array(clonal_pairs)
    if len(clonal_pairs) > N_PAIRS_SAMPLE:
        sel = rng.choice(len(clonal_pairs), N_PAIRS_SAMPLE, replace=False)
        clonal_pairs = clonal_pairs[sel]
    print(f"[{sample_label}] clonal (twin) pairs: {len(clonal_pairs)}", flush=True)

    n_nc = len(clonal_pairs)
    nonclonal_pairs = []
    n_cells = A.n_obs
    tries = 0
    while len(nonclonal_pairs) < n_nc and tries < n_nc * 20:
        i, j = rng.integers(0, n_cells, 2)
        tries += 1
        if i == j or clone[i] == clone[j]:
            continue
        nonclonal_pairs.append((i, j))
    nonclonal_pairs = np.array(nonclonal_pairs)
    print(f"[{sample_label}] non-clonal (random cross-clone) pairs: {len(nonclonal_pairs)}", flush=True)

    d_clonal = np.linalg.norm(pcs[clonal_pairs[:, 0]] - pcs[clonal_pairs[:, 1]], axis=1)
    d_nonclonal = np.linalg.norm(pcs[nonclonal_pairs[:, 0]] - pcs[nonclonal_pairs[:, 1]], axis=1)

    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(d_clonal, d_nonclonal, alternative="less")

    print(f"\n[{sample_label}] clonal distance:     mean={d_clonal.mean():.3f}  median={np.median(d_clonal):.3f}  n={len(d_clonal)}")
    print(f"[{sample_label}] non-clonal distance: mean={d_nonclonal.mean():.3f}  median={np.median(d_nonclonal):.3f}  n={len(d_nonclonal)}")
    print(f"[{sample_label}] Mann-Whitney U (clonal < non-clonal): p={p:.3e}")
    print(f"[{sample_label}] effect: clonal pairs are {100*(1-d_clonal.mean()/d_nonclonal.mean()):.1f}% closer on average\n")

    return d_clonal, d_nonclonal


if __name__ == "__main__":
    d_clonal_t0, d_nonclonal_t0 = run_for_sample("endo_T0", 0)
    d_clonal_t1, d_nonclonal_t1 = run_for_sample("endo_T1", 1)

    out = {
        "endo_T0": dict(d_clonal=d_clonal_t0.tolist(), d_nonclonal=d_nonclonal_t0.tolist()),
        "endo_T1": dict(d_clonal=d_clonal_t1.tolist(), d_nonclonal=d_nonclonal_t1.tolist()),
    }
    import json
    json.dump(out, open(f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data/pca50_clonal_distance.json', "w"))
    print("wrote pca50_clonal_distance.json")
