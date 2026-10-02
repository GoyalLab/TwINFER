#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""TODO4v2 vs CollecTRI for HS054_Sot48h -- ports apply_todo4v2_vs_collectri.py's formula
and full_report() verbatim (dataset-agnostic once given ranked_edges/z_reg/z_dagger paths,
which for HS054 use the plain (non-_allpairs) filenames run_infer_hs054.py already wrote,
label='hs054' matching exactly)."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline.apply_todo4v2_vs_collectri import COLLECTRI_PATH, full_report, todo4v2_score

DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data'
GENE_SETS = [
    "variability_high", "variability_mid", "variability_low",
    "detection_high", "detection_mid", "detection_low",
    "correlation_high", "correlation_mid", "correlation_low",
]


def main():
    ct = pd.read_csv(COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    print(f"human CollecTRI: {len(CE):,} directed edges\n", flush=True)

    rows = []
    for gs in GENE_SETS:
        re_path = f"{DATA_DIR}/ranked_edges_hs054_{gs}.csv"
        zreg_path = f"{DATA_DIR}/z_reg_gated_hs054_{gs}.json"
        zdag_path = f"{DATA_DIR}/z_dagger_hs054_{gs}.json"
        if not (os.path.exists(re_path) and os.path.exists(zreg_path) and os.path.exists(zdag_path)):
            print(f"[{gs}] SKIP -- outputs not found yet", flush=True)
            continue

        dd = pd.read_csv(re_path)
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        if dd.empty:
            print(f"[{gs}] SKIP -- no directed edge candidates", flush=True)
            continue
        z_reg_map = json.load(open(zreg_path))
        z_dagger_map = json.load(open(zdag_path))

        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            print(f"[{gs}] SKIP -- only {int(y.sum())} true CollecTRI edges in {len(U)} candidates", flush=True)
            continue

        score, n_pass_gate = todo4v2_score(dd, z_reg_map, z_dagger_map)
        m = full_report(score, y)
        base_m = full_report(dd["twinScore"].to_numpy(), y)

        rows.append(dict(
            gene_set=gs, n_pairs=len(U), n_true=int(y.sum()), n_pass_gate=n_pass_gate,
            todo4v2_auprc_x=m["auprc_x"], todo4v2_topk_precision=m["topk_precision"],
            todo4v2_max_f1=m["max_f1"], todo4v2_topk_hits=f"{m['tp']}/{m['k']}",
            twinscore_auprc_x=base_m["auprc_x"], twinscore_topk_precision=base_m["topk_precision"],
        ))
        print(f"[{gs}] n_true={int(y.sum())}/{len(U)}  n_pass_gate={n_pass_gate}  "
              f"TODO4v2 auprc_x={m['auprc_x']:.3f}x top-k_prec={m['topk_precision']:.3f}  |  "
              f"twinScore auprc_x={base_m['auprc_x']:.3f}x top-k_prec={base_m['topk_precision']:.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    out_csv = f"{DATA_DIR}/hs054_todo4v2_vs_collectri.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}\n", flush=True)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False), flush=True)
    if len(df):
        print(f"\nmean TODO4v2 auprc_x = {df.todo4v2_auprc_x.mean():.3f}x  "
              f"mean twinScore auprc_x = {df.twinscore_auprc_x.mean():.3f}x  "
              f"({len(df)}/{len(GENE_SETS)} gene sets with >=3 true edges)", flush=True)


if __name__ == "__main__":
    main()
