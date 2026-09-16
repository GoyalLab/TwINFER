#!/usr/bin/env python3
"""Export a SPRING-style histogram of total transcripts per barcode, per
example library -- log-spaced bins, one panel per library, our actual
(scaled) threshold marked as a vertical line -- matching the plotting style
of SPRING_dev's spring_example_HPCs.ipynb (hist of D[s]['total_counts'] on
log-spaced bins, D[s]['meta']['min_tot'] drawn as a vertical line), rather
than the rank-vs-count knee-curve style used previously.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
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
    return float(c[np.argmax(dist)])


def main():
    print("[hist] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    # np.logspace(0, 6, 50) -- exactly the bin spec in the pasted SPRING code (1 to 1e6, 50 edges)
    bin_edges = np.logspace(0, 6, 50)

    out = {"bin_edges": bin_edges.tolist(), "k": K}
    for lib in EXAMPLE_LIBS:
        m = (obs["library"] == lib).to_numpy()
        c = total_counts[m]
        hist, _ = np.histogram(c, bins=bin_edges)

        raw_knee = knee_threshold(c)
        min_tot = raw_knee * K  # our threshold, playing the role of SPRING's manually-set min_tot

        ix = c >= min_tot
        n_pass = int(ix.sum())
        n_total = int(len(c))
        median_pass = float(np.median(c[ix])) if n_pass else None
        mean_pass = float(np.mean(c[ix])) if n_pass else None

        out[lib] = {
            "hist": hist.tolist(),
            "n_total": n_total,
            "raw_knee": raw_knee,
            "min_tot": min_tot,
            "n_pass": n_pass,
            "median_pass": median_pass,
            "mean_pass": mean_pass,
        }
        print(f"[hist] {lib}: {n_pass} / {n_total}  median={median_pass:.0f}  mean={mean_pass:.1f}  "
              f"(min_tot={min_tot:.1f}, raw_knee={raw_knee:.1f})", flush=True)

    with open(f"{PROCESSED}/histogram_data.json", "w") as f:
        json.dump(out, f)
    print(f"[hist] wrote {PROCESSED}/histogram_data.json", flush=True)


if __name__ == "__main__":
    main()
