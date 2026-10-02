#!/usr/bin/env python3
"""Stage II-IV (twin S/C/U existence-of-regulation, z_het; cross-time direction z_dagger/
gamma) for hPSC_20260927, restricted to EXACTLY the 4,219 pairs that survived Stage I's
BH-corrected existence screen (hpsc_endoderm_stage1_called.csv) -- NOT all C(477,2)
combinations among the induced gene set, which is what OOM-killed the earlier attempt.
The underlying package functions (differentiate_single_state_reg_and_multiple_states,
identify_actual_directed_edges) both accept an explicit list of pairs, so restricting to
the Stage-I survivors here is the funnel doing its job.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys
import time

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id,
    calculate_twin_random_correlations,
    differentiate_single_state_reg_and_multiple_states,
    _build_cross_time_twins,
    get_cross_correlations,
    identify_actual_directed_edges,
)

LABEL = "hpsc_20260927"
GENE_SET = "chipatlas_perturbseq_panel"
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'


def log(m):
    print(m, flush=True)


def main():
    df_all = pd.read_csv(f"{DATA_DIR}/twinfer_input_{LABEL}_{GENE_SET}.csv")
    genes = [c[:-5] for c in df_all.columns if c.endswith("_mRNA")]
    called = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_stage1_called.csv")
    pairs = list(zip(called.gene_1, called.gene_2))
    log(f"[hPSC_20260927] {len(pairs)} Stage-I-called pairs, "
        f"{len(set(called.gene_1) | set(called.gene_2))} induced genes")

    t1_df = df_all[df_all.time_step == 0].reset_index(drop=True)  # endo_T0
    t2_df = df_all[df_all.time_step == 1].reset_index(drop=True)  # endo_T1

    # ---- Stage II: twin vs random-pair correlation, z_het (existence of regulation) ----
    log("[hPSC_20260927] Stage II: twin/random correlations + z_het, at endo_T0")
    tw = assign_twin_id(t1_df)
    tw_corr, rand_corr = calculate_twin_random_correlations(t1_df, tw, genes, random_state=0, unit="clone")

    t0 = time.time()
    het_out = differentiate_single_state_reg_and_multiple_states(
        t1_df, pairs, tw_corr, rand_corr, genes, z_score_threshold=1e9,
        verbose=False, unit="clone", n_cores_to_use=8,
    )
    log(f"[hPSC_20260927] Stage II done: {time.time()-t0:.1f}s")
    z_het_dict = het_out[3]

    # ---- Stage IV: cross-time direction, z_dagger + gamma ----
    log("[hPSC_20260927] Stage IV: cross-time twins + direction (z_dagger, gamma)")
    at1, at2 = _build_cross_time_twins(t1_df, t2_df)
    ordered_pairs = []
    for a, b in pairs:
        ordered_pairs.append((a, b))
        ordered_pairs.append((b, a))
    t0 = time.time()
    Cr = get_cross_correlations(at1, at2, gene_pairs=ordered_pairs, unit="clone")
    log(f"[hPSC_20260927] cross correlations: {time.time()-t0:.1f}s")

    t0 = time.time()
    significant_edges, z_score_calc = identify_actual_directed_edges(
        at1, at2, Cr, ordered_pairs, verbose=False, n_cores_to_use=8, unit="clone",
        return_z_scores=True,
    )
    log(f"[hPSC_20260927] direction stage: {time.time()-t0:.1f}s")
    significant_set = set(significant_edges)

    rows = []
    for a, b in pairs:
        z_het = z_het_dict.get((a, b), z_het_dict.get((b, a), np.nan))
        z_ab = z_score_calc.get((a, b), np.nan)
        z_ba = z_score_calc.get((b, a), np.nan)
        a_sig, b_sig = (a, b) in significant_set, (b, a) in significant_set
        direction = ("a->b" if a_sig and not b_sig else
                     "b->a" if b_sig and not a_sig else
                     "symmetric" if a_sig and b_sig else "uncoupled")
        rows.append(dict(
            gene_1=a, gene_2=b, z_het=z_het,
            z_gamma_ab=z_ab, z_gamma_ba=z_ba, direction=direction,
        ))
    out = pd.DataFrame(rows)
    out["abs_z_het"] = out["z_het"].abs()
    out = out.sort_values("abs_z_het", ascending=False)
    out_path = f"{DATA_DIR}/ranked_edges_{LABEL}_{GENE_SET}_stage234.csv"
    out.to_csv(out_path, index=False)
    log(f"wrote {out_path} ({len(out)} pairs)")
    log(f"\ntop 30 by |z_het|:\n{out.head(30).to_string(index=False)}")


if __name__ == "__main__":
    main()
