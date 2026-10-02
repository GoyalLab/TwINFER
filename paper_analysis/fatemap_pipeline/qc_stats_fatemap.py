#!/usr/bin/env python3
"""Per-cell QC distributions (mito %, n_genes) for FM01/FM06/FM08 raw 10x matrices, plus the
percentile at which FM06/FM08's chosen cutoffs sit, to pick FM01's cutoffs consistently."""
import numpy as np
import scipy.sparse as sp
from paper_analysis.fatemap_pipeline.fatemap_qc_utils import apply_qc_filters, load_10x_replicate

DATASETS = {
    "FM06": (["GSM7434419_FM06_A_WM989Naive", "GSM7434420_FM06_B_WM989Naive"], 21, 5000),
    "FM08": (["GSM7434423_FM08_A_PrimaryMelanocytes", "GSM7434424_FM08_B_PrimaryMelanocytes"], 26, 7200),
    "FM01": (["GSM7434407_FM01_A_1uMPLX", "GSM7434408_FM01_B_1uMPLX"], None, None),
}
for name, (reps, mc, mx) in DATASETS.items():
    ads = [load_10x_replicate(p, r) for r, p in zip("AB", reps)]
    X = sp.vstack([a.X for a in ads]).tocsr()
    q = apply_qc_filters(X, ads[0].var_names, 100, 0, 10**9)
    m, g = q["pct_counts_mt"], q["n_genes"]
    print(name, X.shape, "mt% q50/90/95/98/99:", np.percentile(m, [50, 90, 95, 98, 99]).round(1),
          "| n_genes q1/2/50/98/99/99.5:", np.percentile(g, [1, 2, 50, 98, 99, 99.5]).astype(int),
          "| min/max", g.min(), g.max(), flush=True)
    if mc:
        print(f"   chosen cutoffs sit at: mt<={mc}: {(m <= mc).mean():.3f}  genes<={mx}: {(g <= mx).mean():.3f}  "
              f"genes>=200: {(g >= 200).mean():.3f}", flush=True)
    del X, ads
