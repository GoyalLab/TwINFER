"""
Run infer_with_twinfer (the real pipeline function) on the regenerated 2-state
drift simulations and save the diagnostics.

  - A_B_no_reg_2_states : both genes drift into up/down states, NO regulation
  - A_to_B_2_states     : both genes drift, gene_1 -> gene_2 regulation

Per file: <scenario>_rep_<i>.json (trimmed, json-safe result) +
aggregated drift_infer_summary.csv. Deduplicates by replicate index.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import json
import os
import re
import time
import traceback

import numpy as np
import pandas as pd
import numba
from threadpoolctl import threadpool_limits

from twinfer.inference.infer import infer_with_twinfer

DRIFT = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation'
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/drift_infer_with_twinfer'
T1, T2 = 1, 20
PAIR = ("gene_1", "gene_2")


def _rep(fn):
    return int(re.search(r"df_rows_0_0_(\d+)_", fn).group(1))


def collect(pattern, label):
    by = {}
    for f in sorted(glob.glob(os.path.join(DRIFT, pattern))):
        by.setdefault(_rep(os.path.basename(f)), f)
    return [(label, r, by[r]) for r in sorted(by)]


def jsonable(o):
    """Recursively coerce a nested result dict to JSON-safe types."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return {str(i): {str(c): jsonable(o.loc[i, c]) for c in o.columns} for i in o.index}
    if isinstance(o, pd.Series):
        return {str(i): jsonable(v) for i, v in o.items()}
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _pv(d, pair=PAIR):
    if not isinstance(d, dict):
        return None
    for k in (pair, (pair[1], pair[0])):
        if k in d:
            return d[k]
    return None


def _flat_pair(mat):
    """value for the gene_1-gene_2 cell of a 2x2 correlation matrix/DataFrame."""
    if mat is None:
        return None
    try:
        return float(pd.DataFrame(mat).loc[PAIR[0], PAIR[1]])
    except Exception:  # noqa: BLE001
        m = np.asarray(mat)
        return float(m[0, 1]) if m.shape == (2, 2) else None


def run_one(scenario, rep, path):
    numba.set_num_threads(1)
    rec = {"scenario": scenario, "rep": rep, "file": os.path.basename(path), "error": ""}
    try:
        with threadpool_limits(limits=1):
            res = infer_with_twinfer(
                path_to_simulation_file=path,
                is_simulation_data=True,
                match_sim_details=False,
                check_for_steady_state=False,
                t1=T1, t2=T2, seed=101010,
                plot=False, verbose=False,
                ranked_list=True, return_diagnostics=True,
                n_cores=2,
            )

        diag = res.get("diagnostics", {})
        cls = res.get("classification", {})
        pair_cls = next((k for k, v in cls.items()
                         if [list(PAIR)] == [list(x) for x in v]
                         or [list(PAIR[::-1])] == [list(x) for x in v]), "unknown")
        s3 = _pv(res.get("stage3", {})) or {}
        dir_z = _pv(res.get("direction", {}).get("z_scores"))
        xnull = _pv(res.get("direction", {}).get("rho_cross_null")) or {}
        cor = res.get("correlations", {})

        rec.update({
            "classification": pair_cls,
            "directed_edges": json.dumps([list(e) for e in res.get("directed_edges", [])]),
            "step1_z": _pv(diag.get("step1", {}).get("z_scores")),
            "step2_z_het": _pv(diag.get("step2", {}).get("z_het")
                               or diag.get("step2", {}).get("z_scores")),
            "z_div": _pv(diag.get("step2", {}).get("z_div")
                         or diag.get("step2", {}).get("z_score_div")),
            "step3_z_d": s3.get("z_d"),
            "step3_z_d_div": s3.get("z_d_div"),
            "step4_z_1to2": dir_z if dir_z is not None else xnull.get("z_rho_cross"),
            "step4_z_2to1": _pv(res.get("direction", {}).get("z_scores"), (PAIR[1], PAIR[0])),
            "rho_t1": _flat_pair(cor.get("gene_t1")),
            "rho_t2": _flat_pair(cor.get("gene_t2")),
            "rho_delta_t1": _flat_pair(cor.get("twin_delta_t1")),
            "rho_delta_t2": _flat_pair(cor.get("twin_delta_t2")),
        })

        os.makedirs(OUT, exist_ok=True)
        trimmed = {k: res.get(k) for k in
                   ("settings", "classification", "directed_edges", "correlations",
                    "direction", "stage3", "heterogeneity", "divergence",
                    "statistics", "diagnostics")}
        try:
            with open(os.path.join(OUT, f"{scenario}_rep_{rep}.json"), "w") as fh:
                json.dump(jsonable(trimmed), fh, indent=2)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"(metrics ok) json save failed: {e}"

        print(f"[{scenario} rep{rep:>2}] {pair_cls}  s1={_r(rec['step1_z'])}  "
              f"s2het={_r(rec['step2_z_het'])}  zdiv={_r(rec['z_div'])}  "
              f"s3={_r(rec['step3_z_d'])}  s4={_r(rec['step4_z_1to2'])}", flush=True)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"[{scenario} rep{rep:>2}] ERROR {rec['error']}", flush=True)
    return rec


def _r(x):
    return "None" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-only", action="store_true")
    ap.add_argument("--noreg-only", action="store_true")
    args = ap.parse_args()

    tasks = []
    if not args.reg_only:
        tasks += collect("df_rows_0_0_*_A_B_no_reg_2_states_*.csv", "drift_A_B")
    if not args.noreg_only:
        tasks += collect("df_rows_0_0_*_A_to_B_2_states_*.csv", "drift_A_to_B")

    os.makedirs(OUT, exist_ok=True)
    print(f"{len(tasks)} sims -> {OUT}", flush=True)
    rows = []
    for i, (s, r, p) in enumerate(tasks):
        t0 = time.time()
        rows.append(run_one(s, r, p))
        print(f"  ({i + 1}/{len(tasks)}, {time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["scenario", "rep"])
    csv = os.path.join(OUT, "drift_infer_summary.csv")
    df.to_csv(csv, index=False)
    print(f"\nsaved {csv}", flush=True)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df[["scenario", "rep", "classification", "step1_z", "step2_z_het",
                  "z_div", "step3_z_d", "step4_z_1to2", "step4_z_2to1"]].to_string(index=False))
