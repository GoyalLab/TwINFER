#!/usr/bin/env python3
"""Same histogram + valley-detection logic as export_histogram_data.py and
compare_valley_vs_knee_pipeline.py, but for all 15 libraries (not just the
3 examples), and recording the near-miss diagnostics (best candidate dip's
bump/valley ratio even when it doesn't clear the prominence bar) so we can
tell "genuinely unimodal" apart from "real but too-weak-to-call" bimodality.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
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


def analyze(counts, edges, centers):
    hist, _ = np.histogram(counts, bins=edges)
    hs = np.convolve(hist.astype(float), np.ones(3) / 3, mode="same")
    main_peak = int(np.argmax(hs))
    left = hs[:main_peak]

    best = None  # best candidate dip even if it doesn't clear the bar, for diagnostics
    valley_idx, bump_idx = None, None
    for i in range(1, len(left) - 1):
        if left[i] < left[i - 1] and left[i] <= left[i + 1]:
            pre_max_idx = int(np.argmax(hs[:i + 1]))
            bump_v, valley_v = hs[pre_max_idx], left[i]
            ratio = bump_v / valley_v if valley_v > 0 else np.inf
            if best is None or ratio > best[2]:
                best = (pre_max_idx, i, ratio)
            if bump_v > 0 and ratio >= 1.2 and bump_v >= 15:
                valley_idx, bump_idx = i, pre_max_idx

    return {
        "hist": hist.tolist(),
        "valley_found": valley_idx is not None,
        "valley_count": float(centers[valley_idx]) if valley_idx is not None else None,
        "bump_count": float(centers[bump_idx]) if bump_idx is not None else None,
        "best_candidate_ratio": float(best[2]) if best is not None else None,
        "best_candidate_bump_count": float(centers[best[0]]) if best is not None else None,
        "best_candidate_valley_count": float(centers[best[1]]) if best is not None else None,
        "best_candidate_bump_n": float(hs[best[0]]) if best is not None else None,
    }


def main():
    print("[hist-all] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)

    edges = np.logspace(0, 6, 50)
    centers = np.sqrt(edges[:-1] * edges[1:])

    libs = sorted(obs["library"].unique())
    out = {"bin_edges": edges.tolist(), "k": K}
    for lib in libs:
        m = (obs["library"] == lib).to_numpy()
        c = total_counts[m]
        raw_knee = knee_threshold(c)
        entry = analyze(c, edges, centers)
        entry["n_total"] = int(len(c))
        entry["raw_knee"] = raw_knee
        entry["min_tot"] = raw_knee * K
        out[lib] = entry
        tag = "REAL VALLEY" if entry["valley_found"] else "no valley"
        near = (f"best candidate: bump={entry['best_candidate_bump_count']:.0f} "
                f"(n={entry['best_candidate_bump_n']:.0f}) -> valley={entry['best_candidate_valley_count']:.0f} "
                f"ratio={entry['best_candidate_ratio']:.2f}"
                if entry["best_candidate_ratio"] is not None else "no dip candidate at all")
        print(f"[hist-all] {lib}: {tag} | {near}", flush=True)

    with open(f"{PROCESSED}/histogram_data_all15.json", "w") as f:
        json.dump(out, f)
    print(f"[hist-all] wrote {PROCESSED}/histogram_data_all15.json", flush=True)


if __name__ == "__main__":
    main()
