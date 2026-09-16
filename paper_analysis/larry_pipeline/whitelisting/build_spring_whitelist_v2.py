#!/usr/bin/env python3
"""SPRING-style whitelist, tuned to a target total cell count instead of the
raw automatic knee.

Target derivation (per user): PROJECT_SUMMARY_27082026.pdf's own rebuilt
LARRY input matrix holds 83,735 cells -- but that run was missing cells in
LSK_d2_1 and one other sample, so 83,735 is itself an undercount of what a
complete rebuild should have. Target: 83,735 + ~2,000 = 85,735. (A rougher
cross-check: the released, already-scrublet'd h5ad has 72,946 cells; our
whitelist sits *before* our own singletCode multiplet removal, so it should
legitimately be somewhat larger than a post-doublet-removal count too --
85,735 is ~17.5% more than 72,946, consistent with that direction.)

Method: compute each library's own automatic knee threshold as before (same
as build_spring_whitelist.py), then binary-search a single multiplier k on
all 15 thresholds at once (candidate_threshold_lib = knee_thr_lib * k) until
the aggregate pass count hits the target -- one shared "how generous" knob
rather than 15 independently re-tuned ones.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
TARGET_TOTAL = 83735 + 2000


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
    print("[whitelist-v2] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    obs["total_counts"] = total_counts

    libs = sorted(obs["library"].unique())
    lib_masks = {lib: (obs["library"] == lib).to_numpy() for lib in libs}
    knee_thr = {lib: float(knee_threshold(total_counts[lib_masks[lib]])) for lib in libs}
    print(f"[whitelist-v2] target total cells: {TARGET_TOTAL} (72946 * 1.10)", flush=True)
    for lib in libs:
        print(f"[whitelist-v2] {lib}: raw knee threshold = {knee_thr[lib]:.0f}", flush=True)

    def n_pass(k):
        return sum(int((total_counts[lib_masks[lib]] >= knee_thr[lib] * k).sum()) for lib in libs)

    # n_pass(k) is monotonically non-increasing in k; k=1 is the plain knee (too strict here).
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if n_pass(mid) >= TARGET_TOTAL:
            lo = mid
        else:
            hi = mid
    k = lo
    final_n = n_pass(k)
    print(f"[whitelist-v2] chosen multiplier k={k:.5f} -> {final_n} cells "
          f"(target {TARGET_TOTAL}, off by {final_n - TARGET_TOTAL})", flush=True)

    whitelist_mask = np.zeros(len(obs), dtype=bool)
    for lib in libs:
        m = lib_masks[lib]
        thr = knee_thr[lib] * k
        pass_mask = m & (total_counts >= thr)
        whitelist_mask |= pass_mask
        print(f"[whitelist-v2] {lib}: scaled threshold={thr:.0f}, {int(pass_mask.sum())}/{int(m.sum())} pass", flush=True)

    whitelist_cells = pd.DataFrame({
        "library": obs["library"].to_numpy()[whitelist_mask],
        "cell_id": cells[whitelist_mask].to_numpy(),
    })
    whitelist_cells["barcode"] = whitelist_cells["cell_id"].str.split(":", n=1).str[1]
    whitelist_cells[["library", "barcode"]].to_csv(f"{PROCESSED}/spring_style_whitelist_v2.csv", index=False)
    print(f"[whitelist-v2] wrote {PROCESSED}/spring_style_whitelist_v2.csv "
          f"({int(whitelist_mask.sum())} cells)", flush=True)


if __name__ == "__main__":
    main()
