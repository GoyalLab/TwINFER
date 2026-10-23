#!/usr/bin/env python3
"""Save the top-100 lists used for the identity analyses: all TF x target pairs (no ChIP filter), Stage I (BH over all candidate pairs; for datasets stored as TF-subset parts the BH is recomputed over the merged parts),
NO gene filter, non-TF regulators excluded BEFORE ranking (AEBP1, CENPA, HMGA1, YBX1 and any regulator not in the Lambert TF list), then top 100 by |z_U| (unshared) and by |z_C| = |h| (shared).
Output: analysis_data/fm_meta_programs_comparison/top100_lists/<dataset>_{unshared,shared}.csv (TF, gene, h, z_U) and <dataset>_stage1_targets.txt (background = all targets of the Stage I pairs).
usage: build_top100_lists_alltf.py [FM01 FM06 Watermelon_naive ...]"""
import sys, glob, os
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; OUT = f"{A}/fm_meta_programs_comparison/top100_lists"; os.makedirs(OUT, exist_ok=True)
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
DIRS = {"FM01": "fm01", "FM03": "fm03", "FM05": "fm05", "FM06": "fm06", "FM07": "fm07", "Watermelon_naive": "watermelon_naive"}
cols = ["TF", "gene", "z_rho", "h", "z_U"]
for ds in (sys.argv[1:] or ["FM01", "FM06", "Watermelon_naive"]):
    D = f"{A}/{DIRS[ds]}/data/stage1_tf_ge20"; parts = sorted(glob.glob(f"{D}/part[0-9]/pdf_procedure_all_candidate_pairs.csv"))
    if parts:   # TF-subset parts: recompute BH over the merged set
        P = pd.concat([pd.read_csv(f, usecols=cols, dtype={"TF": "category", "gene": "category"}) for f in parts], ignore_index=True)
        pv = 2 * norm.sf(np.abs(P.z_rho.to_numpy(np.float64))); o = np.argsort(pv); q = pv[o] * len(pv) / (np.arange(len(pv)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); P["stage1"] = qq <= 0.05
    else: P = pd.read_csv(f"{D}/pdf_procedure_all_candidate_pairs.csv", usecols=cols + ["stage1"], dtype={"TF": "category", "gene": "category"})
    P = P[P.stage1].copy(); P["TF"] = P.TF.astype(str); P["gene"] = P.gene.astype(str)
    open(f"{OUT}/{ds}_stage1_targets.txt", "w").write("\n".join(sorted(set(P.gene))) + "\n")
    Q = P[~(P.TF.isin(four) | ~P.TF.isin(lam))]
    for lab, k in (("unshared", "z_U"), ("shared", "h")):
        Q.reindex(Q[k].abs().sort_values(ascending=False).index).head(100)[["TF", "gene", "h", "z_U"]].to_csv(f"{OUT}/{ds}_{lab}.csv", index=False)
    print(ds, f"Stage I pairs {len(P):,}; after excluding non-TF regulators {len(Q):,}; targets {P.gene.nunique():,}", flush=True)
