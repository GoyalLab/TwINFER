#!/usr/bin/env python3
"""THE CURRENT PRODUCTION DEFAULT (writes to qc_filtered/).

Whitelist: dip-test + Otsu (build_diptest_otsu_whitelist.py's method), not the earlier
knee-based one. Per library: Hartigan's dip test on log10(total counts) decides whether
there's a real second (ambient) mode at all (p<0.05); if so, Otsu's threshold cuts there;
if not, every nonzero-count barcode is kept. No shared multiplier, no external target cell
count anywhere in this decision -- unlike the knee method (scaled by a k tuned to hit a
target derived from the PDF's own cell count), this is fully self-contained per library.
Landed within ~5% of that same target without ever looking at it (89,813 vs 85,735).

singletCode: min_umi_cutoff=3 (matching h5ad's own documented threshold, Methods 3.3)
plus our own completed cross-sample rescue (singletcode_cross_sample_rescue.py, criterion
4 -- the one singletCode's own author left as an unintegrated TODO).

Earlier variants are kept, not deleted, for comparison:
  qc_filtered_legacy_knee_umi3_criterion4/  32,486 cells, KNEE whitelist, UMI floor 3, criteria 1-4
  qc_filtered_legacy_umi2_criteria1to3/     33,047 cells, knee whitelist, UMI floor 2, criteria 1-3 (original)
  qc_filtered_criterion4/                   34,060 cells, knee whitelist, UMI floor 2, criteria 1-4
  qc_filtered_paperfiltered/                + Weinreb rule (ii)/(iii) on top of whatever qc_filtered/ was at build time
"""
import json
import sys

import diptest
import numpy as np
import pandas as pd
import scipy.io as sio
import singletCode
from scipy.stats import chisquare

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code")
from paper_analysis.larry_hematopoiesis_validation.preprocessing.singletcode_cross_sample_rescue import apply_cross_sample_rescue

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROCESSED = str(get_larry_dataset_dir() / "processed")
OUT_DIR = PROCESSED + "/qc_filtered"  # the production default -- see module docstring

# Step 5 (Weinreb et al. 2020 Methods 3.3, rules (ii)/(iii)) -- mandatory, not optional.
# Applied in-memory below rather than as a separate downstream script/directory, so
# qc_filtered/ is genuinely the full Steps 1-5 output in one build.
SAMPLE_GROUPS = {
    "day2":       ["LSK_d2_1", "LSK_d2_2", "LSK_d2_3"],
    "day4_repl1": ["LSK_d4_1_1", "LSK_d4_1_2", "LSK_d4_1_3"],
    "day4_repl2": ["LSK_d4_2_1", "LSK_d4_2_2", "LSK_d4_2_3"],
    "day6_repl1": ["LSK_d6_1_1", "LSK_d6_1_2", "LSK_d6_1_3"],
    "day6_repl2": ["LSK_d6_2_1", "LSK_d6_2_2", "LSK_d6_2_3"],
}
LIB_TO_GRP = {lib: s for s, libs in SAMPLE_GROUPS.items() for lib in libs}


def apply_weinreb_rule_ii_iii(obs_qc, kept_cells):
    """(rule_ii, rule_iii) boolean arrays, aligned to obs_qc/kept_cells row order.
    Same logic as apply_paper_clone_filters.py, applied in-memory instead of re-reading
    an already-written matrix from disk."""
    cl = obs_qc["larry_clone_singletcode"].astype(str)
    assert (cl.notna() & (cl != "") & (cl != "nan")).all(), "expected every cell to have a clone"

    df = pd.DataFrame({
        "clone": cl.to_numpy(),
        "library": obs_qc["library"].to_numpy(),
        "cell_bc": kept_cells.str.split(":", n=1).str[1].to_numpy(),
    })
    df["grp"] = df["library"].map(LIB_TO_GRP)
    assert df["grp"].notna().all(), "every library must map to a sample group"

    # rule (ii): (cell_bc, clone) pair spanning >1 distinct library
    pair_lib_counts = df.groupby(["cell_bc", "clone"])["library"].nunique()
    flagged_pairs = set(pair_lib_counts[pair_lib_counts > 1].index)
    rule_ii = np.array([(cb, cln) in flagged_pairs for cb, cln in zip(df["cell_bc"], df["clone"])])

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
    return rule_ii, rule_iii

# --- superseded: the knee-based per-library threshold, scaled by a shared multiplier k
# tuned to an external target cell count. Kept for reference/comparison only -- see
# build_diptest_otsu_whitelist.py and compare_diptest_vs_knee_pipeline.py for why the
# dip-test+Otsu method below replaced it as the production default.
# K = 0.19349593495847234
# def knee_threshold(counts):
#     c = np.sort(counts)[::-1]
#     c = c[c > 0]
#     x = np.log10(np.arange(1, len(c) + 1))
#     y = np.log10(c)
#     p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
#     line = p2 - p1
#     line_norm = line / np.linalg.norm(line)
#     vecs = np.stack([x - p1[0], y - p1[1]], axis=1)
#     proj_len = vecs @ line_norm
#     proj = np.outer(proj_len, line_norm) + p1
#     dist = np.linalg.norm(vecs - (proj - p1), axis=1)
#     return float(c[np.argmax(dist)])


def otsu_threshold(values):
    """Standard Otsu's method on a continuous 1D sample (log10 counts here). 
    Maximizes the between-class variance of the two groups split at each unique value."""
    v = np.sort(values)
    n = len(v)
    cumsum = np.cumsum(v)
    total_sum = cumsum[-1]
    best_t, best_var = v[0], -1.0
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


def diptest_otsu_threshold(counts):
    """(mask_over_nonzero, threshold, bimodal_flag) for one library's total_counts."""
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
    return keep, float(10 ** thr_log), bimodal, float(pval)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[build] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    genes = pd.Series(open(f"{PROCESSED}/genes.txt").read().split())
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes))
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)

    mito_mask = genes.str.lower().str.startswith("mt-").to_numpy()
    assert mito_mask.sum() > 0
    mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel().astype(float)
    pct_counts_mt = np.divide(100.0 * mito_counts, total_counts,
                               out=np.zeros_like(total_counts), where=total_counts > 0)
    pass_mito = pct_counts_mt <= 20.0

    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    pass_genes = n_genes_per_cell >= 200

    libs = sorted(obs["library"].unique())
    lib_masks = {lib: (obs["library"] == lib).to_numpy() for lib in libs}
    whitelist_mask = np.zeros(len(obs), dtype=bool)
    whitelist_diag = {}
    for lib in libs:
        m = lib_masks[lib]
        keep_local, thr, bimodal, pval = diptest_otsu_threshold(total_counts[m])
        lib_mask = np.zeros(len(obs), dtype=bool)
        lib_mask[np.flatnonzero(m)[total_counts[m] > 0]] = keep_local
        whitelist_mask |= lib_mask
        whitelist_diag[lib] = {"threshold": thr, "bimodal": bimodal, "pval": pval,
                                "n_pass": int(keep_local.sum())}
        print(f"[build] {lib}: threshold~{thr:.0f} bimodal={bimodal} p={pval:.4f} "
              f"n_pass={int(keep_local.sum())}", flush=True)
    n0 = int(whitelist_mask.sum())
    print(f"[build] whitelist (dip-test+Otsu, no external target): {n0} cells", flush=True)

    wl_lib = obs.loc[whitelist_mask, "library"].to_numpy()
    wl_bc = cells[whitelist_mask].str.split(":", n=1).str[1].to_numpy()
    whitelist_keys = set(zip(wl_lib, wl_bc))

    umi_table_full = pd.read_csv(f"{PROCESSED}/umi_table.csv")
    keep_row = [(s, c) in whitelist_keys for s, c in
                zip(umi_table_full["sample"].to_numpy(), umi_table_full["cellID"].to_numpy())]
    umi_table = umi_table_full[keep_row].reset_index(drop=True)
    print(f"[build] umi_table restricted to whitelist: {len(umi_table)} rows", flush=True)

    singletCode.check_sample_sheet(umi_table)
    good_data, singlet_stats = singletCode.get_singlets(
        umi_table, dataset_name="LARRY_umi3_final", min_umi_cutoff=3)
    good_data, rescued_cellids = apply_cross_sample_rescue(good_data)
    print(f"[build] cross-sample rescue: {len(set(rescued_cellids))} distinct cellIDs rescued", flush=True)

    singlets = good_data[good_data["label"] == "Singlet"]
    singlet_keys = set(zip(singlets["sample"], singlets["cellID"]))

    obs_library = obs["library"].to_numpy()
    obs_barcode = cells.str.split(":", n=1).str[1].to_numpy()
    is_singlet = np.array([(lib, bc) in singlet_keys for lib, bc in zip(obs_library, obs_barcode)])

    # the clone barcode(s) singletCode kept per singlet cell, AFTER rescue -- same
    # construction as larry_preprocessing.ipynb's own final "combine and save" cell
    singlet_clone = (
        singlets.groupby(["sample", "cellID"])["barcode"]
        .apply(lambda s: "".join(sorted(set(s))))
    )
    singlet_clone.index = [f"{lib}:{bc}" for lib, bc in singlet_clone.index]

    after_mito = whitelist_mask & pass_mito
    after_singlet = after_mito & is_singlet
    after_genes = after_singlet & pass_genes

    n1, n2, n3 = int(after_mito.sum()), int(after_singlet.sum()), int(after_genes.sum())
    print(f"[build] after whitelist={n0}  after mito={n1}  after singletCode(+rescue)={n2}  "
          f"after gene floor={n3}", flush=True)

    keep = after_genes
    keep_idx = np.flatnonzero(keep)
    X_qc = X[keep_idx]
    kept_cells = cells[keep].reset_index(drop=True)

    obs_qc = obs[keep].copy()
    obs_qc["pct_counts_mt"] = pct_counts_mt[keep]
    obs_qc["n_genes"] = n_genes_per_cell[keep]
    obs_qc["singlet_label"] = np.where(is_singlet[keep], "Singlet", "Multiplet")
    obs_qc["larry_clone_singletcode"] = obs_qc.index.map(singlet_clone).fillna("")
    rescued_set = set(rescued_cellids)
    obs_qc["criterion4_rescued"] = kept_cells.str.split(":", n=1).str[1].isin(rescued_set).to_numpy()

    assert X_qc.shape[0] == len(obs_qc) == len(kept_cells)
    assert obs_qc.index.is_unique

    # Step 5 (mandatory): Weinreb et al. 2020 Methods 3.3 rules (ii)/(iii) --
    # cross-library (Cell-BC, Lineage-BC) repeats and statistically over-abundant
    # clones, both droplet-emulsion-instability artifacts.
    rule_ii, rule_iii = apply_weinreb_rule_ii_iii(obs_qc, kept_cells)
    n4_before = len(obs_qc)
    step5_keep = ~(rule_ii | rule_iii)
    print(f"[build] Step 5: rule(ii)={int(rule_ii.sum())} rule(iii)={int(rule_iii.sum())} "
          f"flagged={int((~step5_keep).sum())}  before={n4_before} after={int(step5_keep.sum())}",
          flush=True)

    step5_idx = np.flatnonzero(step5_keep)
    X_qc = X_qc[step5_idx]
    kept_cells = kept_cells[step5_keep].reset_index(drop=True)
    obs_qc = obs_qc[step5_keep].copy()
    obs_qc["rule_ii_flagged"] = rule_ii[step5_idx]  # all False here, kept for provenance
    obs_qc["rule_iii_flagged"] = rule_iii[step5_idx]
    n4 = int(step5_keep.sum())

    assert X_qc.shape[0] == len(obs_qc) == len(kept_cells)
    assert obs_qc.index.is_unique

    sio.mmwrite(f"{OUT_DIR}/larry_qc_counts.mtx", X_qc)
    (open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(genes) + "\n"))
    (open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(kept_cells) + "\n"))
    obs_qc.to_csv(f"{OUT_DIR}/obs_metadata.csv")

    summary = {
        "n_raw_cells": len(obs),
        "n_after_whitelist": n0,
        "n_after_mito": n1,
        "n_after_singletcode_umi3_plus_criterion4": n2,
        "n_after_gene_floor": n3,
        "n_after_weinreb_rule_ii_iii": n4,
        "whitelist_method": "diptest_otsu_per_library_no_external_target",
        "whitelist_per_library": whitelist_diag,
        "singletcode_min_umi_cutoff": 3,
        "n_distinct_cellids_rescued_by_criterion4": len(rescued_set),
        "n_rule_ii_flagged": int(rule_ii.sum()),
        "n_rule_iii_flagged": int(rule_iii.sum()),
    }
    (open(f"{OUT_DIR}/preprocessing_summary.json", "w")).write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[build] wrote {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
