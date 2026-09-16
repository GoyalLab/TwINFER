#!/usr/bin/env python3
"""Cell-calling whitelist via Hartigan's dip test + Otsu's threshold, on the ACTUAL
per-cell total_counts (not a binned-histogram approximation -- an earlier attempt using
np.repeat() on histogram bin centers created an artificial comb-like distribution that
made every library look spuriously bimodal to the dip test).

Per library:
  1. Hartigan's dip test (diptest package) on log10(total_counts) decides whether the
     distribution is significantly non-unimodal (p < 0.05).
  2. If significant: Otsu's threshold (maximize between-class variance) on the same
     log10-count values gives the cut point; keep barcodes above it.
  3. If not significant: no real second mode -- keep every nonzero-count barcode.

No shared multiplier, no external target cell count anywhere in this decision.
"""
import json

import diptest
import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"


def otsu_threshold(values):
    """Standard Otsu's method on a continuous 1D sample: sort once, then for every
    candidate split point maximize between-class variance w0*w1*(m0-m1)^2."""
    v = np.sort(values)
    n = len(v)
    cumsum = np.cumsum(v)
    total_sum = cumsum[-1]
    best_t, best_var = v[0], -1.0
    # candidate splits at every distinct value transition (cheap enough at these n)
    for i in range(1, n):
        if v[i] == v[i - 1]:
            continue
        w0, w1 = i, n - i
        m0 = cumsum[i - 1] / w0
        m1 = (total_sum - cumsum[i - 1]) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, v[i]
    return best_t


def fit_library(counts):
    c = counts[counts > 0]
    log_c = np.log10(c)
    dip, pval = diptest.diptest(log_c)
    bimodal = pval < 0.05
    if bimodal:
        thr_log = otsu_threshold(log_c)
        keep = log_c >= thr_log
    else:
        thr_log = log_c.min()
        keep = np.ones(len(c), dtype=bool)
    mask = np.zeros(len(counts), dtype=bool)
    mask[np.flatnonzero(counts > 0)[keep]] = True
    return mask, float(10 ** thr_log), {"dip": float(dip), "pval": float(pval), "bimodal": bool(bimodal)}


def main():
    print("[diptest-otsu] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)

    libs = sorted(obs["library"].unique())
    whitelist_mask = np.zeros(len(obs), dtype=bool)
    per_lib = {}
    for lib in libs:
        m = (obs["library"] == lib).to_numpy()
        c = total_counts[m]
        keep_local, thr, diag = fit_library(c)
        whitelist_mask[np.flatnonzero(m)] = keep_local
        n_pass = int(keep_local.sum())
        per_lib[lib] = {"n_total": int(m.sum()), "n_pass": n_pass, "threshold": thr, **diag}
        print(f"[diptest-otsu] {lib}: {n_pass}/{int(m.sum())} pass, threshold~{thr:.0f}, "
              f"dip={diag['dip']:.4f} p={diag['pval']:.4f} bimodal={diag['bimodal']}", flush=True)

    total = int(whitelist_mask.sum())
    print(f"[diptest-otsu] TOTAL: {total} / {len(obs)} cells (no external target used)", flush=True)

    whitelist_cells = pd.DataFrame({
        "library": obs["library"].to_numpy()[whitelist_mask],
        "cell_id": cells[whitelist_mask].to_numpy(),
    })
    whitelist_cells["barcode"] = whitelist_cells["cell_id"].str.split(":", n=1).str[1]
    whitelist_cells[["library", "barcode"]].to_csv(f"{PROCESSED}/diptest_otsu_whitelist.csv", index=False)

    summary = {"total_cells": total, "n_raw": len(obs), "per_library": per_lib}
    json.dump(summary, open(f"{PROCESSED}/diptest_otsu_whitelist_summary.json", "w"), indent=2)
    print(f"[diptest-otsu] wrote {PROCESSED}/diptest_otsu_whitelist.csv and summary", flush=True)


if __name__ == "__main__":
    main()
