#!/usr/bin/env python3
"""No-filter variant of run_infer_hpsc_endoderm_gt50.py: same 54-gene gt50 panel (6 TF, 48
target, 50 ground-truth ChIP-Atlas+Perturb-seq edges), but alpha_gene_gene_corr and
alpha_stage3 relaxed to ~1.0 and bypass_stage3_gate=True -- this bypasses Stage I's
existence significance test (the thing that normally drops most pairs into "no_regulation"
with NO twinScore computed at all) so that literally every one of the C(54,2)=1431 pairs
gets a real computed twinScore, per explicit user request ("no filters at all - so all
edges get scores"). t1=0 (endo_T0), t2=1 (endo_T1).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: replaces hardcoded project-root paths]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# [2026-10-01 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

LABEL = "hpsc_20260927"
GENE_SET = "gt50"
# DATA_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
OUT_DIR = f"{DATA_DIR}/gt50_nofilter_full_results"
T1, T2 = 0, 1


def save_matrix(obj, name):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(f"{OUT_DIR}/{name}.csv")


def save_pairdict(d, name):
    """dict keyed by (gene_1, gene_2) tuples -> scalar or nested dict/array."""
    rows = []
    for k, v in d.items():
        a, b = k if isinstance(k, tuple) else (k, None)
        if isinstance(v, dict):
            row = {"gene_1": a, "gene_2": b}
            for vk, vv in v.items():
                if isinstance(vv, (np.ndarray, list)):
                    continue  # skip raw null-draw arrays here (too big for a flat CSV)
                row[vk] = vv
            rows.append(row)
        else:
            rows.append({"gene_1": a, "gene_2": b, "value": v})
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/{name}.csv", index=False)


def save_listpairs(lst, name):
    pd.DataFrame(lst, columns=["gene_1", "gene_2"]).to_csv(f"{OUT_DIR}/{name}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    input_csv = f"{DATA_DIR}/twinfer_input_{LABEL}_{GENE_SET}.csv"
    df = pd.read_csv(input_csv)
    n_genes = sum(c.endswith("_mRNA") for c in df.columns)
    print(f"[hPSC_20260927/gt50_nofilter] TwINFER input: {len(df):,} cells, {n_genes} genes, "
          f"{df['clone_id'].nunique():,} clones, time_step counts "
          f"{df['time_step'].value_counts().to_dict()}, n_shuffles={args.n_shuffles}", flush=True)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=T1,
        t2=T2,
        check_for_steady_state=False,
        use_clone=True,
        unit="clone",
        n_shuffles_step1=args.n_shuffles,
        n_shuffles_step2=args.n_shuffles,
        n_shuffles_stage3=args.n_shuffles,
        n_shuffles_direction=args.n_shuffles,
        n_shuffles_fanout=args.n_shuffles,
        separate_fan_outs_from_mutual_regulation_flag=True,
        n_cores=args.n_cores,
        seed=101010,
        verbose=False,
        plot=False,
        alpha_gene_gene_corr=0.999999,  # "no filters" -- let essentially every pair pass
        alpha_stage3=0.999999,          # Stage I existence so it gets a real computed twinScore
        bypass_stage3_gate=True,        # also skip the multiple-states z_reg_gated sign gate
    )

    # ---- save EVERYTHING, not just ranked_edges ----
    json.dump(result["settings"], open(f"{OUT_DIR}/settings.json", "w"), default=str)

    for k, v in result["classification"].items():
        save_listpairs(v, f"classification_{k}")

    save_listpairs(result["directed_edges"], "directed_edges")

    for k, v in result["correlations"].items():
        save_matrix(v, f"correlations_{k}")

    save_pairdict(result["direction"]["z_scores"], "direction_z_scores")
    save_pairdict(result["direction"]["rho_cross_null"], "direction_rho_cross_null")
    save_pairdict(result["direction"]["gamma_details"], "direction_gamma_details")
    save_listpairs(result["direction"]["rescued_from_no_regulation"], "direction_rescued_from_no_regulation")
    save_matrix(result["direction"]["unfiltered_matrix"], "direction_unfiltered_matrix")

    save_pairdict(result["stage3"], "stage3_details") if isinstance(result["stage3"], dict) else \
        json.dump(str(result["stage3"]), open(f"{OUT_DIR}/stage3_details.json", "w"))

    save_pairdict(result["heterogeneity"]["z_het"], "heterogeneity_z_het")
    save_pairdict(result["divergence"]["z_div"], "divergence_z_div")
    save_pairdict(result["gated_regulation"]["z_reg_gated"], "gated_regulation_z_reg_gated")

    json.dump(str(result["fan_out"]), open(f"{OUT_DIR}/fan_out.json", "w"))

    if result["twin_score_inputs"] is not None:
        try:
            pd.DataFrame(result["twin_score_inputs"]).to_csv(f"{OUT_DIR}/twin_score_inputs.csv", index=False)
        except Exception as e:
            json.dump(str(result["twin_score_inputs"]), open(f"{OUT_DIR}/twin_score_inputs.json", "w"))

    ranked_edges = result["ranked_edges"]
    if ranked_edges is not None:
        ranked_edges.to_csv(f"{OUT_DIR}/ranked_edges.csv", index=False)
        print(f"[hPSC_20260927/gt50_nofilter] ranked_edges: {len(ranked_edges)} directed edge candidates", flush=True)
        print(ranked_edges[["gene_1", "gene_2", "twinScore"]]
              .sort_values("twinScore", ascending=False).head(40).to_string(index=False), flush=True)

    print(f"[hPSC_20260927/gt50_nofilter] classification: "
          f"{ {k: len(v) for k, v in result['classification'].items()} }", flush=True)
    print(f"[hPSC_20260927/gt50_nofilter] directed_edges: {len(result['directed_edges'])}", flush=True)
    print(f"[hPSC_20260927/gt50_nofilter] wrote all stage outputs to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
