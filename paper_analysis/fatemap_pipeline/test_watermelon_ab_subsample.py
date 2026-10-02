#!/usr/bin/env python3
"""Test of the Watermelon per-stage A/B input with per-side subsampling (cap 100). Exits non-zero on any failure.
Checks: (1) counts match the dry run (cells/clones), (2) no side above the cap and every clone kept, (3) the inference
loader (stage h5ad) and the scoring loader (per-stage qc dir) pick the SAME cells with the same clone/time_step,
(4) their expression agrees (log1p CP10k), (5) determinism under row shuffling."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys

import numpy as np

from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb
from paper_analysis.fatemap_pipeline.run_infer_fatemap_ab_split import AB_SUBSAMPLE_PER_SIDE, build_twinfer_input_ab, subsample_per_side

EXPECT = {"naive": (6730, 195), "lag": (6076, 220), "late": (4327, 138)}  # cells, clones from the 2026-09-23 dry run
fail = []
for stage, (n_cells, n_clones) in EXPECT.items():
    ds = f"Watermelon_{stage}"
    gs_name = "correlation_high"
    genes = json.load(open(f"{TWINFER_PROJECT_ROOT}/analysis_data/{ds.lower()}/data/gene_sets_{ds.lower()}.json"))[gs_name]
    inf = build_twinfer_input_ab(ds, genes)
    sc_df, _, _, _, _ = gb.load_raw_ab(ds, gs_name)
    side = inf.groupby(["clone_id", "time_step"]).size()
    ok = {
        "cells == dry run": len(inf) == n_cells,
        "clones == dry run": inf.clone_id.nunique() == n_clones,
        "max side <= cap": int(side.max()) <= AB_SUBSAMPLE_PER_SIDE[ds],
        "same cells (inference vs scoring)": set(inf.cell_id) == set(sc_df.cell_id) and len(inf) == len(sc_df),
    }
    a = inf.set_index("cell_id").sort_index(); b = sc_df.set_index("cell_id").sort_index()
    ok["same clone/time_step"] = ok["same cells (inference vs scoring)"] and bool((a.clone_id == b.clone_id).all() and (a.time_step == b.time_step).all())
    if ok["same cells (inference vs scoring)"]:
        cols = [f"{g}_mRNA" for g in genes]
        d = float(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float)).max())
        ok[f"expression agrees (max abs diff {d:.2e})"] = d < 1e-3
    sh = inf.sample(frac=1, random_state=3).reset_index(drop=True)
    ok["deterministic under row shuffle"] = set(subsample_per_side(sh, AB_SUBSAMPLE_PER_SIDE[ds]).cell_id) == set(subsample_per_side(inf, AB_SUBSAMPLE_PER_SIDE[ds]).cell_id)
    print(f"[{stage}] cells {len(inf):,} clones {inf.clone_id.nunique()} A {(inf.time_step==0).sum():,} B {(inf.time_step==1).sum():,} "
          f"largest side {int(side.max())}", flush=True)
    for k, v in ok.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}", flush=True)
        if not v:
            fail.append(f"{stage}: {k}")
if fail:
    print("TEST FAILED:", fail, flush=True); sys.exit(1)
print("ALL TESTS PASSED", flush=True)
