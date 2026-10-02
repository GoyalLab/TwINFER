"""
NEW, additive statistic: z-score for the ABSOLUTE change in rho_Delta between
t1 and t2, computed for BOTH the twin-based reference and the random-pair
reference, gene_1-gene_2, per replicate.

    d_obs = | rho_Delta(t2) - rho_Delta(t1) |

Two null constructions, both computed WITHIN a single replicate (never pooled
across replicates):

  Method "pool_relabel" (cell-level permutation):
    Pool all raw cells from t1 and t2 together, then randomly reassign which
    cells count as "pseudo-t1" / "pseudo-t2" (same group sizes as the real
    split). From each pseudo-group's cell pool, draw a *fresh* set of random
    pairs (same slot count/weights as the real reference) and compute
    rho_Delta. d_null = |rho_Delta(pseudo-t2) - rho_Delta(pseudo-t1)|.
    Repeated 10,000 times.

    For the TWIN statistic this collapses onto "pool_permute" below: a twin
    pair is fixed by clonal lineage (siblings snapshotted together at ONE
    timepoint), so there is no cell-level resampling operation that keeps
    "twin" meaningful after pooling cells across t1/t2 -- the smallest unit
    that can be permuted is the whole twin-clone (its already-observed
    Delta row), which is exactly what "pool_permute" does. So only ONE null
    is reported for twins; RANDOM pairs get both.

  Method "pool_permute" (pair-delta permutation):
    Keep every pair (twin-clone, or the one fixed random-pair draw used for
    the observed random-pair statistic) and its already-computed Delta row
    exactly as observed -- no cell resampling. Pool the Delta rows computed
    at t1 with those computed at t2, then randomly split the pooled set into
    two groups of the original t1/t2 sizes and compute rho_Delta from each
    synthetic group. d_null = |rho_Delta(group B) - rho_Delta(group A)|.
    Repeated 10,000 times.

z = (d_obs - mean(d_null)) / std(d_null, ddof=1)

This file only READS existing functions from correlation_functions.py /
run_6scenario_zscores.py (imported, never modified) and adds new code on top.

Usage:
    python compute_abs_drift_zscores.py --output-dir DIR [--n-draws 10000]
                                         [--jobs 1] [--only scen:rep,...]
"""
import argparse
import glob
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import numba
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from twinfer.inference.correlation_functions import (
    _prepare_observed_delta_rows,
    _prepare_random_twin_null_inputs,
    _sample_fresh_random_twin_cells,
    _build_raw_clone_index,
    get_unit_weights,
    weighted_spearman,
)

# Reuse the exact scenario/file collection + clone partition from the
# existing 6/8-scenario pipeline. Not modified, only imported.
from simulations.drift_multiple_state.run_6scenario_zscores import collect_tasks, _load, _partition, T1, T2, SEED

GENES = ["gene_1", "gene_2"]
PAIR = ("gene_1", "gene_2")
GI, GJ = GENES.index(PAIR[0]), GENES.index(PAIR[1])
N_DRAWS_DEFAULT = 10000


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------- Method "pool_permute"
def _pool_permute_null(delta_a, w_a, delta_b, w_b, n_draws, rng):
    """
    Pool two sets of already-computed Delta rows (+ weights), repeatedly
    split the pool at random into groups of the original sizes, and return
    the null array of |rho(group2) - rho(group1)|.
    """
    pooled_delta = np.concatenate([delta_a, delta_b], axis=0)
    pooled_w = np.concatenate([w_a, w_b], axis=0)
    n1, n2 = len(delta_a), len(delta_b)
    n = n1 + n2
    out = np.empty(n_draws, dtype=float)
    for k in range(n_draws):
        perm = rng.permutation(n)
        idx1, idx2 = perm[:n1], perm[n1:]
        r1 = weighted_spearman(pooled_delta[idx1, GI], pooled_delta[idx1, GJ], pooled_w[idx1])
        r2 = weighted_spearman(pooled_delta[idx2, GI], pooled_delta[idx2, GJ], pooled_w[idx2])
        out[k] = abs(r2 - r1)
    return out


# ---------------------------------------------------------------- Method "pool_relabel"
def _pool_relabel_null(rprep_t1, rprep_t2, w_t1, w_t2, n_draws, rng):
    """
    Pool the raw CELLS (not pairs) from t1 and t2, randomly relabel which
    cells are pseudo-t1/pseudo-t2 (same group sizes as real), draw fresh
    random pairs from each pseudo-group (same slot count/weights as the real
    reference), and return the null array of |rho(pseudo-t2) - rho(pseudo-t1)|.
    """
    vals = np.concatenate([rprep_t1["raw_values"], rprep_t2["raw_values"]], axis=0)
    clones = np.concatenate([rprep_t1["raw_clones"], rprep_t2["raw_clones"]], axis=0)
    n1_cells = len(rprep_t1["raw_values"])
    n2_cells = len(rprep_t2["raw_values"])
    n_cells = n1_cells + n2_cells

    n_slots_t1 = len(rprep_t1["source_clones"])
    n_slots_t2 = len(rprep_t2["source_clones"])
    slots_t1 = rprep_t1["source_clones"]   # only its length is used downstream
    slots_t2 = rprep_t2["source_clones"]

    out = np.empty(n_draws, dtype=float)
    for k in range(n_draws):
        perm = rng.permutation(n_cells)
        idx_pseudo1, idx_pseudo2 = perm[:n1_cells], perm[n1_cells:]

        vals1, clones1 = vals[idx_pseudo1], clones[idx_pseudo1]
        vals2, clones2 = vals[idx_pseudo2], clones[idx_pseudo2]
        groups1 = _build_raw_clone_index(clones1)
        groups2 = _build_raw_clone_index(clones2)
        if len(groups1) < 2 or len(groups2) < 2:
            out[k] = np.nan
            continue

        a1, b1, _, _ = _sample_fresh_random_twin_cells(
            slots_t1[:n_slots_t1], clones1, rng, raw_groups=groups1)
        a2, b2, _, _ = _sample_fresh_random_twin_cells(
            slots_t2[:n_slots_t2], clones2, rng, raw_groups=groups2)
        d1 = vals1[a1] - vals1[b1]
        d2 = vals2[a2] - vals2[b2]
        r1 = weighted_spearman(d1[:, GI], d1[:, GJ], w_t1)
        r2 = weighted_spearman(d2[:, GI], d2[:, GJ], w_t2)
        out[k] = abs(r2 - r1)
    return out


def process_replicate(path, scenario, rep_id, n_draws=N_DRAWS_DEFAULT, seed=SEED):
    numba.set_num_threads(1)
    with threadpool_limits(limits=1):
        sim = _load(path)
        P = _partition(sim, seed=seed)

        rec = {"scenario": scenario, "rep_id": rep_id, "n_draws": n_draws}

        # ---- twin-based rho_Delta ----
        tprep_t1 = _prepare_observed_delta_rows(P["t1_twins"], GENES, unit="clone")
        tprep_t2 = _prepare_observed_delta_rows(P["t2_twins"], GENES, unit="clone")
        rho_twin_t1 = weighted_spearman(tprep_t1["delta"][:, GI], tprep_t1["delta"][:, GJ], tprep_t1["weights"])
        rho_twin_t2 = weighted_spearman(tprep_t2["delta"][:, GI], tprep_t2["delta"][:, GJ], tprep_t2["weights"])
        d_obs_twin = abs(rho_twin_t2 - rho_twin_t1)

        rng_null_twin = np.random.default_rng(seed + 900001)
        null_twin = _pool_permute_null(
            tprep_t1["delta"], tprep_t1["weights"], tprep_t2["delta"], tprep_t2["weights"],
            n_draws, rng_null_twin)
        m, s = np.nanmean(null_twin), np.nanstd(null_twin, ddof=1)
        rec.update({
            "rho_delta_twin_t1": float(rho_twin_t1), "rho_delta_twin_t2": float(rho_twin_t2),
            "d_obs_twin": float(d_obs_twin),
            "twin_null_mean": float(m), "twin_null_std": float(s),
            "z_abs_drift_twin": float((d_obs_twin - m) / s) if s > 0 else None,
        })

        # ---- random-pair rho_Delta (one fixed observed draw per timepoint) ----
        rprep_t1 = _prepare_random_twin_null_inputs(P["t1_twins"], P["rho_t1"], GENES)
        rprep_t2 = _prepare_random_twin_null_inputs(P["t2_twins"], P["rho_t2"], GENES)
        w_t1 = get_unit_weights(rprep_t1["rep_0"], unit="clone")
        w_t2 = get_unit_weights(rprep_t2["rep_0"], unit="clone")

        rng_obs = np.random.default_rng(seed + 271829)
        a1, b1, _, _ = _sample_fresh_random_twin_cells(
            rprep_t1["source_clones"], rprep_t1["raw_clones"], rng_obs,
            raw_groups=rprep_t1["raw_groups"], sampling_cache=rprep_t1["random_pair_sampling_cache"])
        rng_obs2 = np.random.default_rng(seed + 271830)
        a2, b2, _, _ = _sample_fresh_random_twin_cells(
            rprep_t2["source_clones"], rprep_t2["raw_clones"], rng_obs2,
            raw_groups=rprep_t2["raw_groups"], sampling_cache=rprep_t2["random_pair_sampling_cache"])
        delta_rand_t1 = rprep_t1["raw_values"][a1] - rprep_t1["raw_values"][b1]
        delta_rand_t2 = rprep_t2["raw_values"][a2] - rprep_t2["raw_values"][b2]
        rho_rand_t1 = weighted_spearman(delta_rand_t1[:, GI], delta_rand_t1[:, GJ], w_t1)
        rho_rand_t2 = weighted_spearman(delta_rand_t2[:, GI], delta_rand_t2[:, GJ], w_t2)
        d_obs_random = abs(rho_rand_t2 - rho_rand_t1)

        rng_m2 = np.random.default_rng(seed + 900002)
        null_r_permute = _pool_permute_null(delta_rand_t1, w_t1, delta_rand_t2, w_t2, n_draws, rng_m2)
        m2, s2 = np.nanmean(null_r_permute), np.nanstd(null_r_permute, ddof=1)

        rng_m1 = np.random.default_rng(seed + 900003)
        null_r_relabel = _pool_relabel_null(rprep_t1, rprep_t2, w_t1, w_t2, n_draws, rng_m1)
        m1, s1 = np.nanmean(null_r_relabel), np.nanstd(null_r_relabel, ddof=1)

        rec.update({
            "rho_delta_random_t1": float(rho_rand_t1), "rho_delta_random_t2": float(rho_rand_t2),
            "d_obs_random": float(d_obs_random),
            "random_pool_permute_null_mean": float(m2), "random_pool_permute_null_std": float(s2),
            "z_abs_drift_random_pool_permute": float((d_obs_random - m2) / s2) if s2 > 0 else None,
            "random_pool_relabel_null_mean": float(m1), "random_pool_relabel_null_std": float(s1),
            "z_abs_drift_random_pool_relabel": float((d_obs_random - m1) / s1) if s1 > 0 else None,
        })
    return rec


def run_one(path, scenario, rep_id, out_dir, n_draws):
    out = os.path.join(out_dir, f"{scenario}_rep_{rep_id}.json")
    if os.path.exists(out):
        return out, "skipped"
    try:
        rec = process_replicate(path, scenario, rep_id, n_draws=n_draws)
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, cls=NpEnc, indent=2)
        return out, (f"OK  z_twin={rec['z_abs_drift_twin']:.2f}  "
                     f"z_rand_permute={rec['z_abs_drift_random_pool_permute']:.2f}  "
                     f"z_rand_relabel={rec['z_abs_drift_random_pool_relabel']:.2f}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return out, f"ERROR {type(e).__name__}: {e}"


def _aggregate(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*_rep_*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["scenario", "rep_id"])
    csv = os.path.join(out_dir, "abs_drift_zscores.csv")
    df.to_csv(csv, index=False)
    print(f"\naggregated {len(df)} rows -> {csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", type=str, default=None, help="scenario:rep,scenario:rep")
    ap.add_argument("--list-tasks", action="store_true")
    args = ap.parse_args()

    tasks = collect_tasks()
    if args.list_tasks:
        for s, r, _ in tasks:
            print(f"{s}:{r}")
        raise SystemExit(0)
    if args.only:
        want = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in want]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(tasks)} tasks -> {args.output_dir} ({args.jobs} parallel), "
          f"n_draws={args.n_draws}  t1={T1}h t2={T2}h", flush=True)

    def _do(idx, scen, rep, path):
        t0 = time.time()
        out, status = run_one(path, scen, rep, args.output_dir, args.n_draws)
        print(f"[{idx+1}/{len(tasks)}] {scen} rep {rep}: {status}  ({time.time()-t0:.1f}s)", flush=True)

    if args.jobs > 1:
        Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(_do)(i, s, r, p) for i, (s, r, p) in enumerate(tasks))
    else:
        for i, (s, r, p) in enumerate(tasks):
            _do(i, s, r, p)

    _aggregate(args.output_dir)
