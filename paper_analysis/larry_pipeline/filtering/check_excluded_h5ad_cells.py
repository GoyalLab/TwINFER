#!/usr/bin/env python3
"""Where do the 1,577 h5ad cells excluded by whitelist v2's threshold actually
sit relative to that threshold -- right at the boundary (expected/benign) or
scattered lower down (would suggest something systematic)?

Also persists total_counts per cell to obs_metadata.csv so future checks
don't need to reload the full sparse matrix again.
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"

# per-library scaled thresholds from the whitelist-v2 run (k=0.19350 * raw knee)
THRESHOLDS = {
    "LSK_d2_1": 296, "LSK_d2_2": 337, "LSK_d2_3": 292,
    "LSK_d4_1_1": 418, "LSK_d4_1_2": 395, "LSK_d4_1_3": 351,
    "LSK_d4_2_1": 311, "LSK_d4_2_2": 316, "LSK_d4_2_3": 325,
    "LSK_d6_1_1": 582, "LSK_d6_1_2": 422, "LSK_d6_1_3": 486,
    "LSK_d6_2_1": 328, "LSK_d6_2_2": 357, "LSK_d6_2_3": 380,
}


def main():
    print("[check] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    obs["total_counts"] = total_counts
    obs.to_csv(f"{PROCESSED}/obs_metadata.csv")  # persist total_counts for reuse
    print("[check] persisted total_counts into obs_metadata.csv", flush=True)

    print("[check] loading h5ad (backed)", flush=True)
    a = ad.read_h5ad(
        "/home/gzu5140/TwINFER_KA/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad", backed="r"
    )
    h5_keys = set(f"{lib}:{bc}" for lib, bc in zip(a.obs["Library"].astype(str), a.obs_names.astype(str)))

    wl2 = pd.read_csv(f"{PROCESSED}/spring_style_whitelist_v2.csv")
    wl2_keys = set(wl2["library"] + ":" + wl2["barcode"])

    cell_to_idx = pd.Series(np.arange(len(cells)), index=cells.to_numpy())
    excluded = sorted(h5_keys - wl2_keys)
    print(f"[check] {len(excluded)} h5ad cells excluded by whitelist v2", flush=True)

    lib_of = [c.split(":", 1)[0] for c in excluded]
    idx = cell_to_idx.loc[excluded].to_numpy()
    tc = total_counts[idx]
    thr = np.array([THRESHOLDS[l] for l in lib_of])
    frac_of_threshold = tc / thr

    df = pd.DataFrame({
        "cell_id": excluded, "library": lib_of, "total_counts": tc,
        "threshold": thr, "frac_of_threshold": frac_of_threshold,
    })
    df.to_csv(f"{PROCESSED}/excluded_h5ad_cells_check.csv", index=False)

    print(f"[check] total_counts distribution for excluded h5ad cells "
          f"(as a fraction of their library's threshold):", flush=True)
    print(df["frac_of_threshold"].describe(), flush=True)
    for lo, hi in [(0.9, 1.0), (0.75, 0.9), (0.5, 0.75), (0.0, 0.5)]:
        n = ((frac_of_threshold >= lo) & (frac_of_threshold < hi)).sum()
        print(f"[check]   {lo:.2f}-{hi:.2f} x threshold: {n} cells", flush=True)
    print(f"[check] wrote {PROCESSED}/excluded_h5ad_cells_check.csv", flush=True)


if __name__ == "__main__":
    main()
