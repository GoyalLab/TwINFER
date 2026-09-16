#!/usr/bin/env python3
"""Builds the production final matrix with singletCode's criterion 4 (cross-sample
barcode-combination rescue) completed, on top of the existing whitelist(v2)/mito/
gene-floor chain -- i.e. the same pipeline as larry_preprocessing.ipynb, but with
apply_cross_sample_rescue() applied to singletCode's output before deriving the final
Singlet mask and larry_clone_singletcode column.

Writes to a NEW directory (qc_filtered_criterion4/), not overwriting the original
qc_filtered/ (33,047 cells, criteria 1-3 only) -- both are kept for comparison.
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import singletCode

sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code")
from singletcode_cross_sample_rescue import apply_cross_sample_rescue

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
OUT_DIR = PROCESSED + "/qc_filtered_criterion4"
K = 0.19349593495847234


def knee_threshold(counts):
    c = np.sort(counts)[::-1]
    c = c[c > 0]
    x = np.log10(np.arange(1, len(c) + 1))
    y = np.log10(c)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = p2 - p1
    line_norm = line / np.linalg.norm(line)
    vecs = np.stack([x - p1[0], y - p1[1]], axis=1)
    proj_len = vecs @ line_norm
    proj = np.outer(proj_len, line_norm) + p1
    dist = np.linalg.norm(vecs - (proj - p1), axis=1)
    return float(c[np.argmax(dist)])


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[build] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    genes = pd.Series(open(f"{PROCESSED}/genes.txt").read().split())
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes))
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)

    mito_mask = genes.str.lower().str.startswith("mt-").to_numpy()
    assert mito_mask.sum() > 0
    mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel().astype(float)
    pct_counts_mt = np.divide(100.0 * mito_counts, total_counts,
                               out=np.zeros_like(total_counts), where=total_counts > 0)
    pass_mito = pct_counts_mt <= 20.0

    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    pass_genes = n_genes_per_cell >= 200

    libs = sorted(obs["library"].unique())
    lib_masks = {lib: (obs["library"] == lib).to_numpy() for lib in libs}
    whitelist_mask = np.zeros(len(obs), dtype=bool)
    for lib in libs:
        m = lib_masks[lib]
        thr = knee_threshold(total_counts[m]) * K
        whitelist_mask |= m & (total_counts >= thr)
    n0 = int(whitelist_mask.sum())
    print(f"[build] whitelist (knee v2): {n0} cells", flush=True)

    wl_lib = obs.loc[whitelist_mask, "library"].to_numpy()
    wl_bc = cells[whitelist_mask].str.split(":", n=1).str[1].to_numpy()
    whitelist_keys = set(zip(wl_lib, wl_bc))

    umi_table_full = pd.read_csv(f"{PROCESSED}/umi_table.csv")
    keep_row = [(s, c) in whitelist_keys for s, c in
                zip(umi_table_full["sample"].to_numpy(), umi_table_full["cellID"].to_numpy())]
    umi_table = umi_table_full[keep_row].reset_index(drop=True)
    print(f"[build] umi_table restricted to whitelist: {len(umi_table)} rows", flush=True)

    singletCode.check_sample_sheet(umi_table)
    good_data, singlet_stats = singletCode.get_singlets(umi_table, dataset_name="LARRY_criterion4_final")
    good_data, rescued_cellids = apply_cross_sample_rescue(good_data)
    print(f"[build] cross-sample rescue: {len(set(rescued_cellids))} distinct cellIDs rescued", flush=True)

    singlets = good_data[good_data["label"] == "Singlet"]
    singlet_keys = set(zip(singlets["sample"], singlets["cellID"]))

    obs_library = obs["library"].to_numpy()
    obs_barcode = cells.str.split(":", n=1).str[1].to_numpy()
    is_singlet = np.array([(lib, bc) in singlet_keys for lib, bc in zip(obs_library, obs_barcode)])

    # the clone barcode(s) singletCode kept per singlet cell, AFTER rescue -- same
    # construction as larry_preprocessing.ipynb's own final "combine and save" cell
    singlet_clone = (
        singlets.groupby(["sample", "cellID"])["barcode"]
        .apply(lambda s: "".join(sorted(set(s))))
    )
    singlet_clone.index = [f"{lib}:{bc}" for lib, bc in singlet_clone.index]

    after_mito = whitelist_mask & pass_mito
    after_singlet = after_mito & is_singlet
    after_genes = after_singlet & pass_genes

    n1, n2, n3 = int(after_mito.sum()), int(after_singlet.sum()), int(after_genes.sum())
    print(f"[build] after whitelist={n0}  after mito={n1}  after singletCode(+rescue)={n2}  "
          f"after gene floor={n3}", flush=True)

    keep = after_genes
    keep_idx = np.flatnonzero(keep)
    X_qc = X[keep_idx]
    kept_cells = cells[keep].reset_index(drop=True)

    obs_qc = obs[keep].copy()
    obs_qc["pct_counts_mt"] = pct_counts_mt[keep]
    obs_qc["n_genes"] = n_genes_per_cell[keep]
    obs_qc["singlet_label"] = np.where(is_singlet[keep], "Singlet", "Multiplet")
    obs_qc["larry_clone_singletcode"] = obs_qc.index.map(singlet_clone).fillna("")
    rescued_set = set(rescued_cellids)
    obs_qc["criterion4_rescued"] = kept_cells.str.split(":", n=1).str[1].isin(rescued_set).to_numpy()

    assert X_qc.shape[0] == len(obs_qc) == len(kept_cells)
    assert obs_qc.index.is_unique

    sio.mmwrite(f"{OUT_DIR}/larry_qc_counts.mtx", X_qc)
    (open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(genes) + "\n"))
    (open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(kept_cells) + "\n"))
    obs_qc.to_csv(f"{OUT_DIR}/obs_metadata.csv")

    summary = {
        "n_raw_cells": len(obs),
        "n_after_whitelist": n0,
        "n_after_mito": n1,
        "n_after_singletcode_with_criterion4": n2,
        "n_after_gene_floor": n3,
        "whitelist_target": 85735,
        "whitelist_shared_multiplier_k": K,
        "n_distinct_cellids_rescued_by_criterion4": len(rescued_set),
    }
    (open(f"{OUT_DIR}/preprocessing_summary.json", "w")).write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[build] wrote {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
