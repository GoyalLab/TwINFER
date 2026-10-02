#!/usr/bin/env python3
"""Update the knee-plot export: show the ACTUAL threshold we use (the v2
whitelist's per-library scaled threshold, knee_thr * k) rather than the raw
automatic knee, and mark where the h5ad-barcoded-but-excluded cells
("B only" in the barcoded-cells Venn) sit relative to it -- concentrated in
LSK_d6_1_1 among these three example libraries.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROCESSED = str(get_larry_dataset_dir() / "processed")
EXAMPLE_LIBS = ["LSK_d2_1", "LSK_d4_1_1", "LSK_d6_1_1"]
K = 0.19349593495847234  # the shared multiplier chosen for the v2 whitelist


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
    knee_idx = np.argmax(dist)
    return int(np.arange(1, len(c) + 1)[knee_idx]), float(c[knee_idx])


def downsample_curve(c, n_points=300):
    n = len(c)
    idx = np.unique(np.logspace(0, np.log10(n), n_points).astype(int) - 1)
    idx = idx[(idx >= 0) & (idx < n)]
    ranks = idx + 1
    return ranks.tolist(), c[idx].tolist()


def main():
    print("[knee-plot-v2] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    print("[knee-plot-v2] loading h5ad obs (backed)", flush=True)
    # [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] a = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad', backed="r")
    a = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/clean_data/real_data/LSK_d2_d4_d6.h5ad', backed="r")
    h5_cl = a.obs["clone_id"].astype(str)
    h5_has = ((h5_cl != "-1") & (h5_cl != "nan")).to_numpy()
    h5_keys = pd.Series(a.obs["Library"].astype(str).to_numpy() + ":" + a.obs_names.astype(str).to_numpy())
    h5_barcoded_keys = set(h5_keys[h5_has])

    out = {"k": K}
    for lib in EXAMPLE_LIBS:
        m = (obs["library"] == lib).to_numpy()
        c_full = total_counts[m]
        c = np.sort(c_full)[::-1]
        c = c[c > 0]
        knee_rank, knee_count = knee_threshold(total_counts[m])
        scaled_count = knee_count * K
        n_pass_scaled = int((c_full >= scaled_count).sum())
        ranks, counts = downsample_curve(c)

        # .to_numpy() before concatenating -- obs has a "library:barcode" string index while
        # cells has a plain RangeIndex, so a pandas "+" between the two Series aligns by index
        # (disjoint labels) rather than by position, silently doubling the row count instead of
        # raising -- caught only downstream as a length mismatch against the numpy boolean masks.
        lib_arr = obs.loc[m, "library"].to_numpy()
        bc_arr = cells[m].str.split(":", n=1).str[1].to_numpy()
        lib_keys = lib_arr.astype(str) + ":" + bc_arr.astype(str)
        excluded_in_h5ad = np.array([k in h5_barcoded_keys for k in lib_keys])
        below_scaled = c_full < scaled_count
        boundary_mask = excluded_in_h5ad & below_scaled
        boundary_counts = sorted(c_full[boundary_mask].tolist(), reverse=True)

        out[lib] = {
            "ranks": ranks, "counts": counts,
            "knee_rank": knee_rank, "knee_count": knee_count,
            "scaled_count": scaled_count, "n_pass_scaled": n_pass_scaled,
            "n_total": int(m.sum()),
            "n_h5ad_barcoded_excluded_by_scaled_threshold": int(boundary_mask.sum()),
            "boundary_cell_counts": boundary_counts,
        }
        print(f"[knee-plot-v2] {lib}: raw knee={knee_count:.0f} -> scaled={scaled_count:.0f} "
              f"({n_pass_scaled} pass) | {int(boundary_mask.sum())} h5ad-barcoded cells excluded "
              f"below the scaled threshold", flush=True)

    with open(f"{PROCESSED}/knee_plot_data_v2.json", "w") as f:
        json.dump(out, f)
    print(f"[knee-plot-v2] wrote {PROCESSED}/knee_plot_data_v2.json", flush=True)


if __name__ == "__main__":
    main()
