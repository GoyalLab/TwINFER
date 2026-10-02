#!/usr/bin/env python3
"""Compare two per-library cell-calling thresholds -- the scaled-knee
threshold currently in production (v2 whitelist) vs. a local-minimum
("valley") threshold in the bimodal transcript-count histogram -- and run
the SAME downstream QC chain (mito filter -> singletCode multiplet removal
-> 200-gene floor) under each, so the comparison covers the whole pipeline's
final cell count, not just the whitelist stage.

Valley detection: histogram on 50 log-spaced bins from 1 to 1e6 (same spec
used in export_histogram_data.py), 3-bin moving-average smoothed, valley =
the first local minimum left of the main peak whose preceding bump is at
least 1.2x taller (>=15 counts) -- the same prominence rule established
interactively while reviewing LSK_d2_1/d4_1_1/d6_1_1's histograms. Falls
back to the scaled-knee threshold for any library with no such valley
(as for LSK_d4_1_1), since that library has no second mode to cut at.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio
import singletCode

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROCESSED = str(get_larry_dataset_dir() / "processed")
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
    return float(c[np.argmax(dist)])


def valley_threshold(counts, knee_fallback):
    """(threshold, is_real_valley). Falls back to knee_fallback when no
    valley with sufficient prominence is found left of the main peak."""
    edges = np.logspace(0, 6, 50)
    hist, _ = np.histogram(counts, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    hs = np.convolve(hist.astype(float), np.ones(3) / 3, mode="same")
    main_peak = int(np.argmax(hs))
    left = hs[:main_peak]
    for i in range(1, len(left) - 1):
        if left[i] < left[i - 1] and left[i] <= left[i + 1]:
            pre_max_idx = int(np.argmax(hs[:i + 1]))
            if hs[pre_max_idx] > 0 and hs[pre_max_idx] >= 1.2 * left[i] and hs[pre_max_idx] >= 15:
                return float(centers[i]), True
    return float(knee_fallback), False


def build_whitelist_mask(obs, lib_masks, total_counts, thr_by_lib):
    mask = np.zeros(len(obs), dtype=bool)
    for lib, m in lib_masks.items():
        mask |= m & (total_counts >= thr_by_lib[lib])
    return mask


def run_downstream(strategy_name, whitelist_mask, obs, cells, umi_table_full, pass_mito, pass_genes):
    """mito filter -> singletCode -> gene floor, restricted to this whitelist."""
    after_mito = whitelist_mask & pass_mito

    # .to_numpy() throughout -- obs has a "library:barcode" string index, cells/umi_table_full
    # have plain RangeIndexes; mixing them in a pandas "+"/zip without converting first is exactly
    # the index-alignment bug that broke export_knee_plot_data_v2.py earlier this session.
    wl_lib = obs.loc[whitelist_mask, "library"].to_numpy()
    wl_bc = cells[whitelist_mask].str.split(":", n=1).str[1].to_numpy()
    whitelist_keys = set(zip(wl_lib, wl_bc))

    keep_row = [(s, c) in whitelist_keys for s, c in
                zip(umi_table_full["sample"].to_numpy(), umi_table_full["cellID"].to_numpy())]
    umi_table = umi_table_full[keep_row].reset_index(drop=True)
    print(f"[{strategy_name}] umi_table restricted to whitelist: {len(umi_table)} rows", flush=True)

    singletCode.check_sample_sheet(umi_table)
    good_data, singlet_stats = singletCode.get_singlets(umi_table, dataset_name=f"LARRY_{strategy_name}")
    singlets = good_data[good_data["label"] == "Singlet"]
    singlet_keys = set(zip(singlets["sample"].to_numpy(), singlets["cellID"].to_numpy()))

    obs_library = obs["library"].to_numpy()
    obs_barcode = cells.str.split(":", n=1).str[1].to_numpy()
    is_singlet = np.array([(lib, bc) in singlet_keys for lib, bc in zip(obs_library, obs_barcode)])

    after_singlet = after_mito & is_singlet
    after_genes = after_singlet & pass_genes

    return {
        "strategy": strategy_name,
        "n_whitelist": int(whitelist_mask.sum()),
        "n_after_mito": int(after_mito.sum()),
        "n_after_singletcode": int(after_singlet.sum()),
        "n_after_gene_floor": int(after_genes.sum()),
    }


def main():
    print("[compare] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    genes = pd.Series(open(f"{PROCESSED}/genes.txt").read().split())
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes))
    # .astype(float) -- X holds integer counts, so a raw X.sum(axis=1) is int64; my local dry run
    # used fake float counts and didn't catch that np.divide(..., out=np.zeros_like(total_counts))
    # then fails casting a float64 result into an int64 output array.
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)

    mito_mask = genes.str.lower().str.startswith("mt-").to_numpy()
    assert mito_mask.sum() > 0, "no mitochondrial genes found -- refusing a no-op filter"
    mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel().astype(float)
    pct_counts_mt = np.divide(100.0 * mito_counts, total_counts,
                               out=np.zeros_like(total_counts), where=total_counts > 0)
    pass_mito = pct_counts_mt <= 20.0

    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    pass_genes = n_genes_per_cell >= 200

    umi_table_full = pd.read_csv(f"{PROCESSED}/umi_table.csv")  # cellID, barcode, sample -- unrestricted

    libs = sorted(obs["library"].unique())
    lib_masks = {lib: (obs["library"] == lib).to_numpy() for lib in libs}

    per_lib = []
    knee_scaled, valley_thr = {}, {}
    for lib in libs:
        c = total_counts[lib_masks[lib]]
        kn = knee_threshold(c)
        ks = kn * K
        vt, found = valley_threshold(c, knee_fallback=ks)
        knee_scaled[lib] = ks
        valley_thr[lib] = vt
        n_knee = int((c >= ks).sum())
        n_valley = int((c >= vt).sum())
        per_lib.append({
            "library": lib, "n_total": int(len(c)),
            "knee_threshold": ks, "valley_threshold": vt, "valley_is_real": found,
            "n_pass_knee": n_knee, "n_pass_valley": n_valley, "delta": n_valley - n_knee,
        })
        print(f"[compare] {lib}: knee={ks:.0f} ({n_knee} pass) | valley={vt:.0f}"
              f"{'(real)' if found else '(fallback=knee)'} ({n_valley} pass) | delta={n_valley - n_knee:+d}",
              flush=True)

    knee_mask = build_whitelist_mask(obs, lib_masks, total_counts, knee_scaled)
    valley_mask = build_whitelist_mask(obs, lib_masks, total_counts, valley_thr)
    print(f"[compare] TOTAL at whitelist stage: knee={int(knee_mask.sum())}  "
          f"valley={int(valley_mask.sum())}  delta={int(valley_mask.sum()) - int(knee_mask.sum()):+d}",
          flush=True)

    downstream = []
    for name, wl_mask in [("knee_v2", knee_mask), ("valley", valley_mask)]:
        res = run_downstream(name, wl_mask, obs, cells, umi_table_full, pass_mito, pass_genes)
        downstream.append(res)
        print(f"[compare] {json.dumps(res)}", flush=True)

    out = {
        "k_shared_multiplier": K,
        "per_library": per_lib,
        "whitelist_totals": {"knee_v2": int(knee_mask.sum()), "valley": int(valley_mask.sum())},
        "downstream": downstream,
    }
    with open(f"{PROCESSED}/valley_vs_knee_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[compare] wrote {PROCESSED}/valley_vs_knee_comparison.json", flush=True)


if __name__ == "__main__":
    main()
