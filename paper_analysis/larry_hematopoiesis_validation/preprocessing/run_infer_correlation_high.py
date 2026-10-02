"""Run infer_with_twinfer on one gene set (t1=2, t2=4), save every z-score-bearing structure, and
(since this run is expensive) the raw permutation-null draws behind each z-score via infer.py's new
return_raw_nulls=True option, as .npz.

Which gene set: GENE_SET env var, one of the 9 keys in resources/gene_sets.json (default
"correlation_high" for manual/interactive use). Output goes to
resources/infer_results/{GENE_SET}_t2_t4/.

There are 11 distinct z-score-type quantities TwINFER computes -- see the printed summary at the
end of this script's log, or infer_with_twinfer's docstring / calculate_twin_score's docstring in
correlation_functions.py for the exact formulas:
  1. Step 1 z            -- gene-gene co-expression existence test (undirected)
  2. z_het                -- heterogeneity/twin-difference-correlation (fresh random-pair null), t1
  3. z_div                -- divergence (Yuval fixed-Delta null), t1
  4. z_reg_gated          -- heterogeneity-gated regulation statistic (diagnostic)
  5. z_d_het              -- change in twin correlation t1->t2, random-pair-difference null
                             (same construction as z_het, "d" instead of within-time rho_Delta)
  6. z_d_div              -- change in twin correlation t1->t2, fixed-Delta null
  7. z_rho_cross          -- directed cross-correlation, per ordered (x,y) pair
  8. u_gamma / z_gamma    -- directional asymmetry (raw null-unit + panel-standardized)
  9. u_abs_rho_t1/z_abs_rho_t1   -- co-expression magnitude at t1
 10. u_abs_rho_t2/z_abs_rho_t2   -- co-expression magnitude at t2
 11. u_abs_rho_change/z_abs_rho_change -- change in co-expression magnitude

raw_nulls.npz null types (one 2D-flattened array per "{null_type}__{gene_a}__{gene_b}" key):
  step1                 -- Stage-I co-expression null (z, item 1)
  rho_delta_het_t1       -- z_het's null (item 2)
  rho_delta_div_t1/t2    -- z_div's null, per timepoint (item 3)
  d_het                  -- z_d_het's null: same random-pair (not twin-pair) construction as
                            rho_delta_het_t1, jointly resampled across t1/t2 for d = rho_Delta(t2)
                            - rho_Delta(t1) (item 5) -- only present for pairs that reached Stage 3
  d_div                  -- z_d_div's null, analogous fixed-Delta version (item 6)
  rho_t1/t2_for_twinscore -- Stage-I null reused for TwinScore's |rho(t1)|/|rho(t2)| terms
                             (items 9-10)
  (the direction/gamma null, item 7-8/11, is already in result["direction"]["rho_cross_null"]
  and twin_score_inputs.csv/ranked_edges.csv rather than raw_nulls.npz)

Run directly: python3 run_infer_correlation_high.py
Run a different gene set: GENE_SET=variability_low python3 run_infer_correlation_high.py
(or via the run_infer_{variability,detection,correlation}.sh array jobs -- same command, just batched)

ALL_PAIRS=1 relaxes every filter so a score is computed for the FULL pair universe instead of just
the pairs that pass TwINFER's internal significance gates (alpha_gene_gene_corr/alpha_stage3 ->
0.9999, z_score_threshold_two_states/z_score_threshold_cross_correlation -> 0). Note
corr_threshold_cross_correlation is NOT touched -- confirmed by reading
identify_actual_directed_edges directly that it's only used when use_scramble=False, which we
never set. Output goes to resources/infer_results/{GENE_SET}_t2_t4_allpairs/ so it never
overwrites the strict-threshold run's results.
ALL_PAIRS=1 GENE_SET=correlation_high python3 run_infer_correlation_high.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
GENE_SET = os.environ.get("GENE_SET", "correlation_high")
ALL_PAIRS = os.environ.get("ALL_PAIRS", "0") == "1"
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")  # extra tag so a concurrent rerun (e.g. more cores)
                                                # doesn't collide with an in-flight run's output dir
# INPUT_DIR: default twinfer_input (raw counts). INPUT_DIR=twinfer_input_cp10k -> log1p-CP10k
# normalized panels (Spearman is not library-normalization invariant -- big effect on the corr
# benchmark). Output dir is unchanged so the benchmark loaders find it.
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INPUT_CSV = os.path.join(HERE, "resources", os.environ.get("INPUT_DIR", "twinfer_input"), f"{GENE_SET}.csv")
INPUT_CSV = os.path.join(RES_HERE, "resources", os.environ.get("INPUT_DIR", "twinfer_input"), f"{GENE_SET}.csv")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.path.join(HERE, "resources", "infer_results",
OUT_DIR = os.path.join(RES_HERE, "resources", "infer_results",
                        f"{GENE_SET}_t2_t4" + ("_allpairs" if ALL_PAIRS else "") + OUT_SUFFIX)
assert os.path.exists(INPUT_CSV), f"missing input CSV for GENE_SET={GENE_SET!r}: {INPUT_CSV}"
os.makedirs(OUT_DIR, exist_ok=True)

T1, T2 = 2, 4
N_CORES = int(os.environ.get("N_CORES", "8"))
ALPHA = 0.9999 if ALL_PAIRS else 0.01
Z_THRESH = 0.0 if ALL_PAIRS else None  # None = let infer_with_twinfer use its own defaults
if ALL_PAIRS:
    print("ALL_PAIRS=1: alpha_gene_gene_corr=alpha_stage3=0.9999, "
          "z_score_threshold_two_states=z_score_threshold_cross_correlation=0 -- "
          "every pair carried through the full pipeline, not just the significant subset. "
          "This can be SLOWER than the strict-threshold run: more pairs now reach the "
          "expensive Stage-3/direction permutation stages instead of being dropped at Step 1.",
          flush=True)


def log(m):
    print(m, flush=True)


def _pairdict_to_json(d):
    """{(a,b): value, ...} -> {"a__b": value, ...}, floats made JSON-safe."""
    out = {}
    for k, v in d.items():
        key = "__".join(k) if isinstance(k, tuple) else str(k)
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif v is None or (isinstance(v, float) and not np.isfinite(v)):
            v = None
        out[key] = v
    return out


log(f"loading {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
log(f"{len(df):,} rows, {df['clone_id'].nunique():,} clones, "
    f"timepoints present {sorted(df['time_step'].unique())}")

kwargs = dict(
    data=df,
    is_simulation_data=False,
    t1=T1,
    t2=T2,
    n_cores=N_CORES,
    verbose=True,
    plot=False,
    return_diagnostics=True,
    ranked_list=True,
    return_raw_nulls=True,
    # 2026-09-08: pin every Monte-Carlo null to 5000 draws. Without this, ALL_PAIRS sets
    # alpha_gene_gene_corr=alpha_stage3=0.9999, so the conservative count
    # ceil(expected_null_rejections / alpha) = 51 falls below the 2000 default floor and every
    # null (step1, step2, stage3, direction, fan_out) silently drops to 2000 draws -- coarser
    # tails than the strict-threshold run, which gets ceil(50/0.01)=5000. No-op for strict runs.
    minimum_null_draws=5000,
)
if ALL_PAIRS:
    kwargs.update(
        alpha_gene_gene_corr=ALPHA,
        alpha_stage3=ALPHA,
        z_score_threshold_two_states=Z_THRESH,
        z_score_threshold_cross_correlation=Z_THRESH,
    )

t0 = time.time()
result = infer_with_twinfer(**kwargs)
elapsed = time.time() - t0
log(f"\ninfer_with_twinfer finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

# ---------------------------------------------------------------- z-scores: save every location
diag = result["diagnostics"]

json.dump(
    {
        "step1_z_scores": _pairdict_to_json(diag["step1"]["z_scores"]),
        "step1_alpha": diag["step1"]["alpha"],
        "step1_z_critical": diag["step1"]["z_critical"],
        "z_het": _pairdict_to_json(result["heterogeneity"]["z_het"]),
        "z_div": _pairdict_to_json(result["divergence"]["z_div"]),
        "z_reg_gated": _pairdict_to_json(result["gated_regulation"]["z_reg_gated"]),
        "direction_z_scores": _pairdict_to_json(result["direction"]["z_scores"]),
        "stage3": {
            "__".join(k) if isinstance(k, tuple) else str(k): {
                kk: (vv.item() if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
                if not isinstance(vv, (np.ndarray, pd.DataFrame))
            }
            for k, v in result["stage3"].items()
        },
        "classification": result["classification"],
        "directed_edges": result["directed_edges"],
        "settings": result["settings"],
    },
    open(os.path.join(OUT_DIR, "z_scores_by_step.json"), "w"),
    indent=2,
    default=str,
)
log(f"wrote {OUT_DIR}/z_scores_by_step.json (per-step z-score dicts, canonical locations)")

# The two comprehensive per-directed-pair tables -- between them these two CSVs contain every
# z-score/null-unit value above in one row per (gene_1 -> gene_2) direction.
if result["twin_score_inputs"] is not None:
    result["twin_score_inputs"].to_csv(os.path.join(OUT_DIR, "twin_score_inputs.csv"), index=False)
    log(f"wrote {OUT_DIR}/twin_score_inputs.csv "
        f"({len(result['twin_score_inputs'])} directed pair(s): "
        "z_het, z_div, z_d_het, u_abs_rho_t1/t2/change, u_gamma, ...)")
if result["ranked_edges"] is not None:
    result["ranked_edges"].to_csv(os.path.join(OUT_DIR, "ranked_edges.csv"), index=False)
    log(f"wrote {OUT_DIR}/ranked_edges.csv "
        f"({len(result['ranked_edges'])} directed pair(s), panel-standardized "
        "z_abs_rho_t1/t2/change, z_gamma, twinScore -- ranked)")

# Correlation matrices, for context.
# 2026-09-08: added random_delta_t1/t2 (Stage-3 random-pair reference matrices) and rho_cross
# (== direction.unfiltered_matrix, the full pre-threshold cross-correlation matrix) so every
# matrix in result["correlations"] is now on disk, not just the 5 originally listed.
#   old: for name in ("gene_t1", "gene_t2", "twin_delta_t1", "twin_delta_t2", "direction"):
for name in ("gene_t1", "gene_t2", "twin_delta_t1", "twin_delta_t2",
             "random_delta_t1", "random_delta_t2", "direction", "rho_cross"):
    mat = result["correlations"].get(name)
    if mat is not None and not mat.empty:
        mat.to_csv(os.path.join(OUT_DIR, f"corr_{name}.csv"))
log(f"wrote correlation matrices (corr_*.csv)")

# ------------------------------------------------- direction / scramble cross-correlation null
# 2026-09-08: the paired cross-clone permutation ("scramble") null behind z_rho_cross / z_gamma
# was previously only summarised into the two twin-score CSVs. Persist the raw draws (like
# raw_nulls.npz does for every other null) plus the per-direction scalar summary.
rho_cross_null = result.get("direction", {}).get("rho_cross_null", {}) or {}
if rho_cross_null:
    dir_npz = {}
    dir_summary = {}
    for (x, y), d in rho_cross_null.items():
        vals = d.get("null_values")
        if vals is not None:
            dir_npz[f"{x}__{y}"] = np.asarray(vals, dtype=float)
        dir_summary[f"{x}__{y}"] = {
            k: (v.item() if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in d.items() if k != "null_values"
        }
    np.savez_compressed(os.path.join(OUT_DIR, "direction_rho_cross_null.npz"), **dir_npz)
    json.dump(dir_summary, open(os.path.join(OUT_DIR, "direction_rho_cross_null_summary.json"), "w"),
              indent=2, default=str)
    log(f"wrote direction_rho_cross_null.npz ({len(dir_npz)} directed null array(s)) "
        f"+ direction_rho_cross_null_summary.json")
else:
    log("no direction/scramble cross-correlation null in result -- nothing to write "
        "(no pair reached the direction step)")

# ---------------------------------------------------------------- raw nulls -> npz
raw_nulls = result.get("raw_nulls", {})
npz_payload = {}
for null_type, pairs in raw_nulls.items():
    for pair_key, arr in pairs.items():
        npz_payload[f"{null_type}__{pair_key}"] = arr
npz_path = os.path.join(OUT_DIR, "raw_nulls.npz")
np.savez_compressed(npz_path, **npz_payload)
log(f"wrote {npz_path} ({len(npz_payload)} null array(s) across {len(raw_nulls)} null type(s): "
    f"{sorted(raw_nulls.keys())})")

log(f"\ndone -- all outputs in {OUT_DIR}")
