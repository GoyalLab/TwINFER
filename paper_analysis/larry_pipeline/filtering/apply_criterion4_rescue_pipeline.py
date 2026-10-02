#!/usr/bin/env python3
"""Runs the current production whitelist (knee v2) through singletCode exactly as the
production pipeline does, then applies the criterion-4 cross-sample-combination rescue
(singletcode_cross_sample_rescue.apply_cross_sample_rescue) on top, and reports the
before/after impact at both the singletCode stage and the full final-matrix stage
(mito -> singletCode -> gene floor), so we know the actual final-cell-count effect,
not just the raw Singlet-count effect.
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import singletCode

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code")
from paper_analysis.larry_hematopoiesis_validation.preprocessing.singletcode_cross_sample_rescue import apply_cross_sample_rescue

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROCESSED = str(get_larry_dataset_dir() / "processed")
K = 0.19349593495847234  # the shared multiplier chosen for the v2 (knee) whitelist


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
    print("[criterion4] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    genes = pd.Series(open(f"{PROCESSED}/genes.txt").read().split())
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes))
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
    knee_mask = np.zeros(len(obs), dtype=bool)
    for lib in libs:
        m = lib_masks[lib]
        thr = knee_threshold(total_counts[m]) * K
        knee_mask |= m & (total_counts >= thr)
    print(f"[criterion4] whitelist (knee v2): {int(knee_mask.sum())} cells", flush=True)

    wl_lib = obs.loc[knee_mask, "library"].to_numpy()
    wl_bc = cells[knee_mask].str.split(":", n=1).str[1].to_numpy()
    whitelist_keys = set(zip(wl_lib, wl_bc))

    umi_table_full = pd.read_csv(f"{PROCESSED}/umi_table.csv")
    keep_row = [(s, c) in whitelist_keys for s, c in
                zip(umi_table_full["sample"].to_numpy(), umi_table_full["cellID"].to_numpy())]
    umi_table = umi_table_full[keep_row].reset_index(drop=True)
    print(f"[criterion4] umi_table restricted to whitelist: {len(umi_table)} rows", flush=True)

    singletCode.check_sample_sheet(umi_table)
    good_data, singlet_stats = singletCode.get_singlets(umi_table, dataset_name="LARRY_criterion4")

    singlets_before = set(zip(good_data.loc[good_data["label"] == "Singlet", "sample"],
                              good_data.loc[good_data["label"] == "Singlet", "cellID"]))
    n_singlet_before = len(singlets_before)
    print(f"[criterion4] BEFORE rescue: {n_singlet_before} singlet (sample,cellID) pairs", flush=True)

    good_data_rescued, rescued_cellids = apply_cross_sample_rescue(good_data)
    singlets_after = set(zip(good_data_rescued.loc[good_data_rescued["label"] == "Singlet", "sample"],
                             good_data_rescued.loc[good_data_rescued["label"] == "Singlet", "cellID"]))
    n_singlet_after = len(singlets_after)
    print(f"[criterion4] AFTER rescue: {n_singlet_after} singlet (sample,cellID) pairs "
          f"(+{n_singlet_after - n_singlet_before})", flush=True)
    print(f"[criterion4] distinct cellIDs rescued: {len(set(rescued_cellids))}", flush=True)

    def final_count(singlet_keys):
        obs_library = obs["library"].to_numpy()
        obs_barcode = cells.str.split(":", n=1).str[1].to_numpy()
        is_singlet = np.array([(lib, bc) in singlet_keys for lib, bc in zip(obs_library, obs_barcode)])
        after_mito = knee_mask & pass_mito
        after_singlet = after_mito & is_singlet
        after_genes = after_singlet & pass_genes
        return {
            "n_after_mito": int(after_mito.sum()),
            "n_after_singletcode": int(after_singlet.sum()),
            "n_after_gene_floor": int(after_genes.sum()),
        }

    result_before = final_count(singlets_before)
    result_after = final_count(singlets_after)
    print(f"[criterion4] FINAL before: {json.dumps(result_before)}", flush=True)
    print(f"[criterion4] FINAL after:  {json.dumps(result_after)}", flush=True)

    out = {
        "whitelist_total": int(knee_mask.sum()),
        "n_singlet_before": n_singlet_before,
        "n_singlet_after": n_singlet_after,
        "n_distinct_cellids_rescued": len(set(rescued_cellids)),
        "final_before": result_before,
        "final_after": result_after,
    }
    with open(f"{PROCESSED}/criterion4_rescue_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[criterion4] wrote {PROCESSED}/criterion4_rescue_comparison.json", flush=True)


if __name__ == "__main__":
    main()
