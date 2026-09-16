#!/usr/bin/env python3
"""Apply the two documented anti-artifact filters from Weinreb et al. 2020 Methods 3.3
to our final QC'd matrix, removing cells whose clone call is flagged as a likely droplet-
emulsion artifact rather than a real clone:

  (ii)  a (Cell-BC, Lineage-BC) pair seen in more than one sequencing library is discarded
        (statistically can only arise from emulsion instability/cross-contamination).
  (iii) a clone that is statistically over-abundant in one library relative to its frequency
        in the other libraries of the same sample is discarded (same reasoning).

Does NOT apply singleton-clone exclusion -- that's a different, separate finding (h5ad's own
`clone_id` convention), not one of the two documented filters, and wasn't asked for here.

Recomputed on the FINAL 33,047-cell population itself (not the pre-mito-filter Singlet set
used during the investigation), since that's the population actually being filtered.

Writes a NEW directory (qc_filtered_paperfiltered/) rather than overwriting qc_filtered/,
so the original final matrix is preserved for comparison.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import chisquare

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
QC_DIR = PROCESSED + "/qc_filtered"
OUT_DIR = PROCESSED + "/qc_filtered_paperfiltered"

SAMPLE_GROUPS = {
    "day2":       ["LSK_d2_1", "LSK_d2_2", "LSK_d2_3"],
    "day4_repl1": ["LSK_d4_1_1", "LSK_d4_1_2", "LSK_d4_1_3"],
    "day4_repl2": ["LSK_d4_2_1", "LSK_d4_2_2", "LSK_d4_2_3"],
    "day6_repl1": ["LSK_d6_1_1", "LSK_d6_1_2", "LSK_d6_1_3"],
    "day6_repl2": ["LSK_d6_2_1", "LSK_d6_2_2", "LSK_d6_2_3"],
}
LIB_TO_GRP = {lib: s for s, libs in SAMPLE_GROUPS.items() for lib in libs}


def main():
    print("[filter] loading final QC'd matrix", flush=True)
    X = sio.mmread(f"{QC_DIR}/larry_qc_counts.mtx").tocsr()
    genes = pd.Series(open(f"{QC_DIR}/genes.txt").read().split())
    cells = pd.Series(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes))

    cl = obs["larry_clone_singletcode"].astype(str)
    assert (cl.notna() & (cl != "") & (cl != "nan")).all(), "expected every final cell to have a clone"

    df = pd.DataFrame({
        "clone": cl.to_numpy(),
        "library": obs["library"].to_numpy(),
        "cell_bc": cells.str.split(":", n=1).str[1].to_numpy(),
    })
    df["grp"] = df["library"].map(LIB_TO_GRP)
    assert df["grp"].notna().all(), "every library must map to a sample group"

    # rule (ii): (cell_bc, clone) pair spanning >1 distinct library
    pair_lib_counts = df.groupby(["cell_bc", "clone"])["library"].nunique()
    flagged_pairs = set(pair_lib_counts[pair_lib_counts > 1].index)
    rule_ii = np.array([(cb, cl) in flagged_pairs for cb, cl in zip(df["cell_bc"], df["clone"])])

    # rule (iii): clone statistically over-abundant in one library vs. the rest of its sample group
    lib_totals = df.groupby(["grp", "library"]).size()
    rule_iii = np.zeros(len(df), dtype=bool)
    df["_idx"] = np.arange(len(df))
    for grp, libs in SAMPLE_GROUPS.items():
        sub = df[df["grp"] == grp]
        exp_frac = (lib_totals[grp] / lib_totals[grp].sum()).reindex(libs).fillna(0).to_numpy()
        for clone, g in sub.groupby("clone"):
            if len(g) < 2:
                continue
            obs_counts = g["library"].value_counts().reindex(libs, fill_value=0).to_numpy()
            n = obs_counts.sum()
            exp_counts = np.where(exp_frac * n == 0, 1e-9, exp_frac * n)
            _, p = chisquare(obs_counts, exp_counts)
            if p < 0.001:
                rule_iii[g["_idx"].to_numpy()] = True

    flagged = rule_ii | rule_iii
    keep = ~flagged
    print(f"[filter] rule (ii) flagged: {int(rule_ii.sum())}", flush=True)
    print(f"[filter] rule (iii) flagged: {int(rule_iii.sum())}", flush=True)
    print(f"[filter] rule (ii) or (iii): {int(flagged.sum())}", flush=True)
    print(f"[filter] before: {len(df)}  after: {int(keep.sum())}", flush=True)

    keep_idx = np.flatnonzero(keep)
    X_out = X[keep_idx]
    cells_out = cells[keep].reset_index(drop=True)
    obs_out = obs[keep].copy()
    obs_out["rule_ii_flagged"] = rule_ii[keep_idx]  # all False here, kept for provenance/debugging
    obs_out["rule_iii_flagged"] = rule_iii[keep_idx]

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    sio.mmwrite(f"{OUT_DIR}/larry_qc_counts.mtx", X_out)
    (genes.pipe(lambda s: open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(s) + "\n")))
    open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(cells_out) + "\n")
    obs_out.to_csv(f"{OUT_DIR}/obs_metadata.csv")

    summary = {
        "n_before": int(len(df)),
        "n_rule_ii_flagged": int(rule_ii.sum()),
        "n_rule_iii_flagged": int(rule_iii.sum()),
        "n_flagged_total": int(flagged.sum()),
        "n_after": int(keep.sum()),
    }
    json.dump(summary, open(f"{OUT_DIR}/filter_summary.json", "w"), indent=2)
    print(f"[filter] wrote {OUT_DIR}/ ({json.dumps(summary)})", flush=True)


if __name__ == "__main__":
    main()
