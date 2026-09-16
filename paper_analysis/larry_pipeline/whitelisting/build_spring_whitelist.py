#!/usr/bin/env python3
"""SPRING-style per-library cell-calling whitelist: a knee-point threshold on
total transcript counts per library, the same principle
data_prep/spring_example_HPCs.ipynb (AllonKleinLab/SPRING_dev, same lab as
LARRY) uses to decide which barcodes are real cells vs empty-droplet/ambient
noise before ever feeding into LARRY's clonal_annotation.ipynb -- that
notebook's cell_bcs_flat.txt/samp_id_flat.txt inputs are exactly this SPRING
data_prep pipeline's output. Their example manually eyeballs a threshold per
sample (700-1000, for their own unrelated demo dataset); this uses an
automatic knee-detector instead (max-distance-from-the-chord on the log-log
rank/count curve, the same principle behind CellRanger/dropletUtils' knee
calls) so it's objective and reproducible on our own 15 libraries.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"


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
    return c[np.argmax(dist)]


def main():
    print("[whitelist] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    thresholds = {}
    whitelist_mask = np.zeros(len(obs), dtype=bool)
    for lib in sorted(obs["library"].unique()):
        m = (obs["library"] == lib).to_numpy()
        thr = knee_threshold(total_counts[m])
        thresholds[lib] = float(thr)
        whitelist_mask |= m & (total_counts >= thr)
        n_lib = m.sum()
        n_pass = (m & whitelist_mask).sum()
        print(f"[whitelist] {lib}: threshold={thr:.0f} total counts, {n_pass}/{n_lib} cells pass", flush=True)

    print(f"[whitelist] total cells {len(obs)}, whitelisted {int(whitelist_mask.sum())}", flush=True)

    whitelist_cells = pd.DataFrame({
        "library": obs["library"].to_numpy()[whitelist_mask],
        "cell_id": cells[whitelist_mask].to_numpy(),
    })
    whitelist_cells["barcode"] = whitelist_cells["cell_id"].str.split(":", n=1).str[1]
    whitelist_cells[["library", "barcode"]].to_csv(f"{PROCESSED}/spring_style_whitelist.csv", index=False)
    print(f"[whitelist] wrote {PROCESSED}/spring_style_whitelist.csv", flush=True)


if __name__ == "__main__":
    main()
