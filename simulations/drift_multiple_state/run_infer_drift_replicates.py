"""
TwINFER Step 1 / Step 2 / Step 3 z-scores for the 20 drift-simulation replicates,
regulated (A_to_B) vs unregulated (A_B_no_reg) 2-state populations, gene_1-gene_2.

This mirrors the Step 1-3 sequence inside twinfer.inference.infer.infer_with_twinfer,
but (a) forces the gene_1-gene_2 pair through every step regardless of the routing
thresholds, and (b) evaluates the Step-2 twin-vs-random rho_Delta z-score at BOTH
t1 and t2 (the pipeline only classifies on t1).

  Step 1 z    : gene-gene rho vs cell-scramble null            (reported at t1 and t2)
  Step 2 z    : (rho_hat_Delta - mean rho_Delta_rand)/SD        (reported at t1 and t2)
  Step 3 z_d  : (d_obs - mean d_null)/SD,  d = rho_Delta(t2) - rho_Delta(t1)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import os
import re
import glob
import argparse
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from twinfer.inference.correlation_functions import (
    assign_twin_id,
    calculate_pairwise_gene_gene_correlation_matrix,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_cross_correlations,
    identify_actual_directed_edges,
    _build_cross_time_twins,
)

SIM_DIR = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference'

T1, T2 = 1, 20
GENES = ["gene_1", "gene_2"]
PAIR = ("gene_1", "gene_2")
# Read at import so joblib/loky workers (which re-import this module) see it too.
N_SHUFFLES = int(os.environ.get("DRIFT_N_SHUFFLES", "5000"))
OUT_TAG = os.environ.get("DRIFT_OUT_TAG", "")
SEED = 101010
UNIT = "clone"

# N_CORES_PER_JOB = 2
# N_PARALLEL_JOBS = 8
N_CORES_PER_JOB = int(os.environ.get("DRIFT_CORES_PER_JOB", "4"))
N_PARALLEL_JOBS = int(os.environ.get("DRIFT_PARALLEL_JOBS", "1"))


def _rep_index(path):
    return int(re.search(r"df_rows_0_0_(\d+)_", os.path.basename(path)).group(1))


def build_file_table(reg_only=False):
    reg, noreg = {}, {}
    for f in glob.glob(os.path.join(SIM_DIR, "df_rows_0_0_*_A_to_B_2_states_*.csv")):
        reg.setdefault(_rep_index(f), []).append(f)
    for f in glob.glob(os.path.join(SIM_DIR, "df_rows_0_0_*_A_B_no_reg_2_states_*.csv")):
        noreg.setdefault(_rep_index(f), []).append(f)
    rows = []
    # for i in range(1, 21):
    #     rows.append({"replicate": i, "condition": "reg",   "path": sorted(reg[i])[0]})
    #     rows.append({"replicate": i, "condition": "no_reg", "path": sorted(noreg[i])[0]})
    for i in sorted(reg):
        rows.append({"replicate": i, "condition": "reg", "path": sorted(reg[i])[0]})
    if not reg_only:
        for i in sorted(noreg):
            rows.append({"replicate": i, "condition": "no_reg", "path": sorted(noreg[i])[0]})
    return rows


def _fmt(x):
    return "None" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


def _pair_val(d, pair=PAIR):
    if d is None:
        return None
    for k in (pair, (pair[1], pair[0])):
        if k in d:
            return d[k]
    return None


def _step1_z(raw, tp_seed):
    mat = calculate_pairwise_gene_gene_correlation_matrix(raw, GENES, use_clone=True)
    no_reg, pot, null_stats, z_scores = check_gene_gene_correlation_threshold(
        raw, mat, GENES,
        use_scramble=True,
        z_score_threshold=2.5,          # routing only; z values are returned regardless
        verbose=False,
        use_clone=True,
        n_shuffles=N_SHUFFLES,
        base_seed=tp_seed,
    )
    return _pair_val(z_scores), float(mat.loc[PAIR[0], PAIR[1]])


def _step2(raw, twins, rand_seed):
    twin_mat, rand_mat = calculate_twin_random_correlations(
        raw, twins, GENES, random_state=rand_seed, unit=UNIT
    )
    _, _, null_stats, z_calc = differentiate_single_state_reg_and_multiple_states(
        raw, [PAIR], twin_mat, rand_mat, GENES,
        z_score_threshold=1e9,          # never route to "multi-state" here; keep z value
        verbose=False,
        unit=UNIT,
        n_shuffles=N_SHUFFLES,
        n_cores_to_use=N_CORES_PER_JOB,
    )
    return _pair_val(z_calc), float(twin_mat.loc[PAIR[0], PAIR[1]]), twin_mat, rand_mat


def _step4(df):
    """Direction inference: signed cross-correlation z-score for each direction.

    Cross-time twins pair sibling-1 at t1 with sibling-2 at t2 (per clone),
    matching infer.py's simulation construction.
    """
    reps = sorted(df["replicate"].unique())
    left = df[(df["time_step"] == T1) & (df["replicate"] == reps[0])].reset_index(drop=True)
    right = df[(df["time_step"] == T2) & (df["replicate"] == reps[1])].reset_index(drop=True)
    at1, at2 = _build_cross_time_twins(left, right)

    dpairs = [("gene_1", "gene_2"), ("gene_2", "gene_1"),
              ("gene_1", "gene_1"), ("gene_2", "gene_2")]
    dmat = get_cross_correlations(at1, at2, gene_pairs=dpairs, unit=UNIT)
    _, zc, _ = identify_actual_directed_edges(
        at1, at2, dmat, gene_pairs=dpairs,
        z_score_threshold=2.5, use_scramble=True,
        n_shuffles=N_SHUFFLES, n_cores_to_use=N_CORES_PER_JOB,
        verbose=False,
        return_z_scores=True, return_rho_cross_null=True,
        prepare_rho_cross_null=True, unit=UNIT,
    )
    return {
        "step4_z_1to2": zc.get(("gene_1", "gene_2")),
        "step4_z_2to1": zc.get(("gene_2", "gene_1")),
        "step4_rho_1to2": float(dmat.loc["gene_1", "gene_2"]),
        "step4_rho_2to1": float(dmat.loc["gene_2", "gene_1"]),
    }


def run_one(row):
    rec = {"replicate": row["replicate"], "condition": row["condition"],
           "path": os.path.basename(row["path"]), "error": ""}
    try:
        df = pd.read_csv(row["path"])
        t1_raw = df[df["time_step"] == T1].reset_index(drop=True)
        t2_raw = df[df["time_step"] == T2].reset_index(drop=True)
        t1_twins = assign_twin_id(t1_raw).reset_index(drop=True)
        t2_twins = assign_twin_id(t2_raw).reset_index(drop=True)

        s1_t1_z, rho_t1 = _step1_z(t1_raw, SEED + 1)
        s1_t2_z, rho_t2 = _step1_z(t2_raw, SEED + 2)

        s2_t1_z, rhoD_t1, twin_t1, rand_t1 = _step2(t1_raw, t1_twins, SEED + 271829)
        s2_t2_z, rhoD_t2, twin_t2, rand_t2 = _step2(t2_raw, t2_twins, SEED + 271830)

        _, _, stage3 = identify_reg_if_multiple_states(
            twin_t1, twin_t2, rand_t1, rand_t2,
            [PAIR], GENES, t1_twins, t2_twins,
            t1_raw=t1_raw, t2_raw=t2_raw,
            alpha=0.01,
            n_shuffles=N_SHUFFLES,
            shuffle_seed=SEED + 271831,
            unit=UNIT,
            n_cores_to_use=N_CORES_PER_JOB,
        )
        s3 = _pair_val(stage3) or {}
        s4 = _step4(df)

        rec.update({
            "rho_t1": rho_t1, "rho_t2": rho_t2,
            "rho_delta_t1": rhoD_t1, "rho_delta_t2": rhoD_t2,
            "step1_z_t1": s1_t1_z, "step1_z_t2": s1_t2_z,
            "step2_z_t1": s2_t1_z, "step2_z_t2": s2_t2_z,
            "step3_z_d": s3.get("z_d"),
            "step3_d": s3.get("d"),
            "step3_z_critical": s3.get("z_critical"),
            "step3_call": s3.get("call"),
            **s4,
        })
        print(f"[{row['condition']} rep{row['replicate']:>2}] "
              f"s1(t1,t2)=({s1_t1_z:+.2f},{s1_t2_z:+.2f}) "
              f"s2(t1,t2)=({s2_t1_z:+.2f},{s2_t2_z:+.2f}) "
              f"s3_zd={_fmt(s3.get('z_d'))} "
              f"s4(1>2,2>1)=({_fmt(s4['step4_z_1to2'])},{_fmt(s4['step4_z_2to1'])})", flush=True)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        print(f"[{row['condition']} rep{row['replicate']:>2}] ERROR {rec['error']}", flush=True)
        traceback.print_exc()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=N_PARALLEL_JOBS)
    ap.add_argument("--reg-only", action="store_true",
                   help="only run the A_to_B (regulated) replicates")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = build_file_table(reg_only=args.reg_only)
    print(f"Running {len(rows)} inferences ({args.jobs} parallel, "
          f"{N_CORES_PER_JOB} cores each), {N_SHUFFLES} null draws ...", flush=True)

    results = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(run_one)(r) for r in rows
    )

    out = pd.DataFrame(results).sort_values(["condition", "replicate"])
    out["n_shuffles"] = N_SHUFFLES
    csv = os.path.join(OUT_DIR, f"drift_step2_step3_zscores{OUT_TAG}.csv")
    out.to_csv(csv, index=False)
    print(f"\nSaved -> {csv}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
