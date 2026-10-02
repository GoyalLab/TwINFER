#!/usr/bin/env python3
"""Fully automatic, target-free cell-calling whitelist: per library, fit a 2-component
Gaussian mixture to log10(total transcript counts) and keep any barcode whose posterior
favors the higher-mean ("real cell") component. No shared multiplier, no external target
cell count -- each library's threshold is determined entirely from its own distribution.

Degenerate case: if a library has no real second mode (as found for several libraries in
the earlier valley-detection investigation -- e.g. LSK_d4_1_1), the two fitted components
may end up close together or one may collapse to near-zero weight. Flagged and reported
rather than silently trusted, same spirit as the valley-detector's `valley_is_real` flag.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.mixture import GaussianMixture

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROCESSED = str(get_larry_dataset_dir() / "processed")


def fit_library(counts):
    """Returns (mask, threshold, diagnostic dict).

    Model selection via BIC decides WHETHER to split at all: a 2-component GMM forced onto
    a genuinely unimodal distribution still splits it roughly in half (verified on synthetic
    unimodal data -- ~51% passed, not ~100%), which would wrongly halve libraries with no
    real ambient population (e.g. LSK_d4_1_1 in the valley-detection investigation). Only
    apply the 2-component split when it meaningfully beats a 1-component fit; otherwise every
    nonzero barcode passes (a single real population, nothing to separate out).
    """
    c = counts[counts > 0]
    log_c = np.log10(c).reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(log_c)
    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(log_c)
    bic1, bic2 = gmm1.bic(log_c), gmm2.bic(log_c)
    two_component_justified = bic2 < bic1 - 10  # standard "strong evidence" BIC margin

    means = gmm2.means_.ravel()
    weights = gmm2.weights_.ravel()
    means_gap = float(abs(means[1] - means[0]))

    if two_component_justified:
        real_component = int(np.argmax(means))
        post = gmm2.predict_proba(log_c)[:, real_component]
        keep_nonzero = post > 0.5
    else:
        keep_nonzero = np.ones(len(c), dtype=bool)  # no real 2nd mode -- keep everything nonzero

    kept_counts = c[keep_nonzero]
    threshold = float(kept_counts.min()) if len(kept_counts) else float("inf")

    diagnostic = {
        "bic_1_component": float(bic1), "bic_2_component": float(bic2),
        "two_component_justified": bool(two_component_justified),
        "component_means_log10": means.tolist(), "component_weights": weights.tolist(),
        "means_gap_log10": means_gap,
    }
    mask = np.zeros(len(counts), dtype=bool)
    mask[np.flatnonzero(counts > 0)[keep_nonzero]] = True
    return mask, threshold, diagnostic


def main():
    print("[gmm-whitelist] loading combined matrix", flush=True)
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
        flag = "" if diag["two_component_justified"] else " *** UNIMODAL (no split applied, all kept) ***"
        print(f"[gmm-whitelist] {lib}: {n_pass}/{int(m.sum())} pass, threshold~{thr:.0f}, "
              f"means gap (log10)={diag['means_gap_log10']:.2f}{flag}", flush=True)

    total = int(whitelist_mask.sum())
    print(f"[gmm-whitelist] TOTAL: {total} / {len(obs)} cells (no external target used)", flush=True)

    whitelist_cells = pd.DataFrame({
        "library": obs["library"].to_numpy()[whitelist_mask],
        "cell_id": cells[whitelist_mask].to_numpy(),
    })
    whitelist_cells["barcode"] = whitelist_cells["cell_id"].str.split(":", n=1).str[1]
    whitelist_cells[["library", "barcode"]].to_csv(f"{PROCESSED}/gmm_whitelist.csv", index=False)

    summary = {"total_cells": total, "n_raw": len(obs), "per_library": per_lib}
    json.dump(summary, open(f"{PROCESSED}/gmm_whitelist_summary.json", "w"), indent=2)
    print(f"[gmm-whitelist] wrote {PROCESSED}/gmm_whitelist.csv and gmm_whitelist_summary.json", flush=True)


if __name__ == "__main__":
    main()
