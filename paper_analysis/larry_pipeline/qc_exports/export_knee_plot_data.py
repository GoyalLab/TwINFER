#!/usr/bin/env python3
"""Export barcode-rank curve data (rank vs total counts, log-log, downsampled)
plus the knee point, for a few representative libraries, to make the knee
concept visible in a chart.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
EXAMPLE_LIBS = ["LSK_d2_1", "LSK_d4_1_1", "LSK_d6_1_1"]


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
    # log-spaced rank sampling so the plot is small but the shape (especially
    # the steep knee region) is preserved.
    n = len(c)
    idx = np.unique(np.logspace(0, np.log10(n), n_points).astype(int) - 1)
    idx = idx[(idx >= 0) & (idx < n)]
    ranks = idx + 1
    return ranks.tolist(), c[idx].tolist()


def main():
    print("[knee-plot] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    out = {}
    for lib in EXAMPLE_LIBS:
        m = (obs["library"] == lib).to_numpy()
        c = np.sort(total_counts[m])[::-1]
        c = c[c > 0]
        knee_rank, knee_count = knee_threshold(total_counts[m])
        ranks, counts = downsample_curve(c)
        out[lib] = {
            "ranks": ranks,
            "counts": counts,
            "knee_rank": knee_rank,
            "knee_count": knee_count,
            "n_total": int(m.sum()),
            "n_pass": int((total_counts[m] >= knee_count).sum()),
        }
        print(f"[knee-plot] {lib}: knee at rank {knee_rank}, count {knee_count:.0f}", flush=True)

    with open(f"{PROCESSED}/knee_plot_data.json", "w") as f:
        json.dump(out, f)
    print(f"[knee-plot] wrote {PROCESSED}/knee_plot_data.json", flush=True)


if __name__ == "__main__":
    main()
