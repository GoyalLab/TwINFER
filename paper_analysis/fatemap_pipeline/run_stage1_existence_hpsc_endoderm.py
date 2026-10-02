#!/usr/bin/env python3
"""Stage I existence funnel for hPSC_20260927, using the REAL package function exactly
as yscher's own code calls it (confirmed by subagent read of helpers/twinscore_script_only.py):
calculate_pairwise_gene_gene_correlation_matrix + check_gene_gene_correlation_threshold,
use_clone=True, n_shuffles=10000 (the clone-block-derangement permutation null, not a
jackknife -- the PDF's jackknife description does not match any code found in her readable
tree; this permutation-null function is the actual, confirmed implementation reachable from
her own JSON-producing script).

All pairs among the 489-gene ChIP-Atlas+Perturb-seq panel (twinfer_input_hpsc_20260927_
chipatlas_perturbseq_panel.csv) are tested -- C(489,2)=119,316 pairs, all-to-all (per
instruction: no star-topology/TF-only restriction). BH-corrected at q=0.05. The resulting
"Called" survivors (and their induced gene set) are what Stage II-IV (twin S/C/U, direction)
then run on via infer_with_twinfer -- NOT the full 489-gene panel, avoiding the OOM seen
when Stage II's differentiate_single_state_reg_and_multiple_states was run on all 119,316
pairs directly.

Timing: benchmarked at n_shuffles=20 -> 8.7s; linear scaling predicts ~73min at the real
n_shuffles=10000 default. Run as its own SLURM job (not on a login node).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    calculate_pairwise_gene_gene_correlation_matrix,
    check_gene_gene_correlation_threshold,
)

LABEL = "hpsc_20260927"
GENE_SET = "chipatlas_perturbseq_panel"
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
BH_Q = 0.05


def run_one_timepoint(df_all, timepoint, genes, n_shuffles):
    df = df_all[df_all.time_step == timepoint].reset_index(drop=True)
    print(f"[hPSC_20260927] Stage I on timepoint={timepoint}: {len(df):,} cells, "
          f"{len(genes)} genes, {df.clone_id.nunique():,} clones, n_shuffles={n_shuffles}",
          flush=True)

    t0 = time.time()
    R = calculate_pairwise_gene_gene_correlation_matrix(df, genes, use_clone=True)
    print(f"[hPSC_20260927] t{timepoint} correlation matrix: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    no_reg, pot_reg, null_stats, z_scores = check_gene_gene_correlation_threshold(
        df, R, genes, use_clone=True, n_shuffles=n_shuffles, n_cores_to_use=8,
        verbose=False,
    )
    print(f"[hPSC_20260927] t{timepoint} Step I null ({n_shuffles} shuffles): {time.time()-t0:.1f}s", flush=True)

    rows = []
    for (gi, gj), z in z_scores.items():
        rho = float(R.loc[gi, gj])
        rows.append({"gene_1": gi, "gene_2": gj, "rho": rho, "z_rho": z})
    out = pd.DataFrame(rows)
    out["p_val"] = 2 * norm.sf(np.abs(out["z_rho"].to_numpy()))
    finite = out["p_val"].notna()
    reject = np.zeros(len(out), dtype=bool)
    qvals = np.full(len(out), np.nan)
    reject[finite.to_numpy()], qvals[finite.to_numpy()], _, _ = multipletests(
        out.loc[finite, "p_val"], alpha=BH_Q, method="fdr_bh")
    out["q_val"] = qvals
    out["called"] = reject
    out["timepoint"] = timepoint
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=10000)
    args = ap.parse_args()

    input_csv = f"{DATA_DIR}/twinfer_input_{LABEL}_{GENE_SET}.csv"
    df_all = pd.read_csv(input_csv)
    genes = [c[:-5] for c in df_all.columns if c.endswith("_mRNA")]

    out_t0 = run_one_timepoint(df_all, 0, genes, args.n_shuffles)
    out_t1 = run_one_timepoint(df_all, 1, genes, args.n_shuffles)

    out_t0.to_csv(f"{DATA_DIR}/hpsc_endoderm_stage1_t0_all.csv", index=False)
    out_t1.to_csv(f"{DATA_DIR}/hpsc_endoderm_stage1_t1_all.csv", index=False)

    # PDF/yscher scenario convention: stage1 = max(|z_rho_t1|, |z_rho_t2|) significant --
    # a pair is "Called" if BH-significant (q<=0.05) at EITHER timepoint (union), matching
    # apply_twinscore_supplement_larry.py's stage1 = max(|z_rho_t1|,|z_rho_t2|) > gate logic.
    m0 = out_t0.set_index(["gene_1", "gene_2"])["called"]
    m1 = out_t1.set_index(["gene_1", "gene_2"])["called"]
    union_called = (m0 | m1)
    called_pairs = union_called[union_called].index.tolist()

    merged = out_t0[["gene_1", "gene_2", "rho", "z_rho", "q_val"]].rename(
        columns={"rho": "rho_t0", "z_rho": "z_rho_t0", "q_val": "q_val_t0"})
    merged = merged.merge(
        out_t1[["gene_1", "gene_2", "rho", "z_rho", "q_val"]].rename(
            columns={"rho": "rho_t1", "z_rho": "z_rho_t1", "q_val": "q_val_t1"}),
        on=["gene_1", "gene_2"])
    merged["called"] = union_called.to_numpy()
    merged.to_csv(f"{DATA_DIR}/hpsc_endoderm_stage1_merged.csv", index=False)

    called = merged[merged.called].copy()
    called["max_abs_z"] = called[["z_rho_t0", "z_rho_t1"]].abs().max(axis=1)
    called = called.sort_values("max_abs_z", ascending=False)
    called.to_csv(f"{DATA_DIR}/hpsc_endoderm_stage1_called.csv", index=False)

    called_genes = sorted(set(called.gene_1) | set(called.gene_2))
    import json
    json.dump(called_genes, open(f"{DATA_DIR}/hpsc_endoderm_stage1_called_genes.json", "w"))

    print(f"\n=== SUMMARY (union of t0, t1) ===", flush=True)
    print(f"total pairs tested (each timepoint): {len(merged)}", flush=True)
    print(f"BH-called (q<={BH_Q}) at t0: {int(out_t0.called.sum())} "
          f"({100*out_t0.called.mean():.2f}%)", flush=True)
    print(f"BH-called (q<={BH_Q}) at t1: {int(out_t1.called.sum())} "
          f"({100*out_t1.called.mean():.2f}%)", flush=True)
    print(f"BH-called union: {len(called)} ({100*len(called)/len(merged):.2f}%)", flush=True)
    print(f"induced gene set from called pairs: {len(called_genes)}", flush=True)
    print(f"wrote {DATA_DIR}/hpsc_endoderm_stage1_merged.csv", flush=True)
    print(f"wrote {DATA_DIR}/hpsc_endoderm_stage1_called.csv", flush=True)
    print(f"wrote {DATA_DIR}/hpsc_endoderm_stage1_called_genes.json", flush=True)


if __name__ == "__main__":
    main()
