"""
Build two synthetic mixed scenarios for the 6-scenario z-score plot:

  scenario 7  mix_upDrift_A_B     = 3000 up/up pairs from drift_A_B    + 3000 pairs from non-drift A_B
  scenario 8  mix_upDrift_A_to_B  = 3000 up/up pairs from drift_A_to_B + 3000 pairs from non-drift A_to_B

Each mix = the "up" branch of a drift sim spliced 50/50 with its MATCHING
non-drift steady-state population (yscher figure_2_simulations_1000/A_B or /A_to_B).
This freezes out the time-drift dynamics and leaves a static two-population
mixture; scenario 8 additionally carries real g1->g2 regulation in both halves,
scenario 7 carries none.  Tests whether a static mixture of two populations, on
its own, trips Step 2 (heterogeneity) / Step 4 (direction).

Per replicate i (1..20):
  * drift file  df_rows_0_0_i_*_<TAG>_*.csv  -> keep rows with state == "up"
    (verified: both siblings of a clone always share state; 3000 up/up clones)
  * A_B file    df_rows_0_1_*_ncells_6000_A_B_rep_i_*.csv  -> 3000 random twin pairs
  * clone_id / cell_id of the A_B half are offset by +10_000_000 to stay disjoint
  * columns kept: cell_id, time_step, gene_1_mRNA, gene_1_protein,
                  gene_2_mRNA, gene_2_protein, clone_id, replicate, state
    (state = "up" for the drift half, "baseline" for the A_B half)

Output:
  simulation_data/drift_simulation/scenario78_mix/
      df_rows_0_0_<i>_ncells_6000_mix_upDrift_A_B_2_states_<hash>.csv
      df_rows_0_0_<i>_ncells_6000_mix_upDrift_A_to_B_2_states_<hash>.csv
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd

DRIFT = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation'
YSCHER = "/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000"
OUT = os.path.join(DRIFT, "scenario78_mix")

N_PAIRS_EACH = 3000
CLONE_OFFSET = 10_000_000
KEEP = ["cell_id", "time_step", "gene_1_mRNA", "gene_1_protein",
        "gene_2_mRNA", "gene_2_protein", "clone_id", "replicate", "state"]


def _rep(fn):
    m = re.search(r"_rep_(\d+)", fn) or re.search(r"df_rows_0_0_(\d+)_", fn)
    return int(m.group(1)) if m else None


def _by_rep(files):
    out = {}
    for f in sorted(files):
        r = _rep(os.path.basename(f))
        if r is not None:
            out.setdefault(r, f)
    return out


def load_up_drift(path, rep):
    d = pd.read_csv(path)
    d = d[d["state"] == "up"].copy()
    up_clones = np.sort(d["clone_id"].unique())
    rng = np.random.default_rng(1000 + rep)
    if len(up_clones) > N_PAIRS_EACH:
        up_clones = np.sort(rng.choice(up_clones, N_PAIRS_EACH, replace=False))
    d = d[d["clone_id"].isin(up_clones)].copy()
    d["state"] = "up"
    return d[KEEP], len(up_clones)


def load_baseline(path, rep, n_pairs):
    """A non-drift steady-state population (yscher A_B or A_to_B), 3000 twin pairs."""
    d = pd.read_csv(path)
    clones = np.sort(d["clone_id"].unique())
    rng = np.random.default_rng(2000 + rep)
    pick = np.sort(rng.choice(clones, min(n_pairs, len(clones)), replace=False))
    d = d[d["clone_id"].isin(pick)].copy()
    d["clone_id"] = d["clone_id"].astype("int64") + CLONE_OFFSET
    d["cell_id"] = d["cell_id"].astype("int64") + CLONE_OFFSET
    d["state"] = "baseline"
    return d[KEEP]


def build(scenario, rep, dpath, apath):
    up, n_up = load_up_drift(dpath, rep)
    base = load_baseline(apath, rep, n_up)
    merged = pd.concat([up, base], ignore_index=True)
    merged = merged.sort_values(["clone_id", "replicate", "time_step"]).reset_index(drop=True)

    h = hashlib.md5(f"{scenario}{rep}".encode()).hexdigest()[:8]
    fn = f"df_rows_0_0_{rep}_ncells_6000_{scenario}_2_states_{h}.csv"
    merged.to_csv(os.path.join(OUT, fn), index=False)
    n_pairs = merged.groupby("state")["clone_id"].nunique().to_dict()
    print(f"  [{scenario} rep {rep}] {fn}  ({len(merged)} rows, pairs={n_pairs})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    # each mix scenario = up-state of a drift sim  +  its MATCHING non-drift
    # steady-state baseline (A_B "up-drift A_B", A_to_B for "up-drift A_to_B").
    # The two sources are independent sims, so drift rep k is paired positionally
    # with baseline rep k; the merged file is labelled by the drift rep (1..20).
    for scenario, drift_tag, base_dir, base_pat in [
        ("mix_upDrift_A_B",   "A_B_no_reg_2_states",
         f"{YSCHER}/A_B",    "df_rows_0_1_*_ncells_6000_A_B_rep_*.csv"),
        ("mix_upDrift_A_to_B", "A_to_B_2_states",
         f"{YSCHER}/A_to_B", "df_rows_0_1_*_ncells_6000_A_to_B_rep_*.csv"),
    ]:
        base_by = _by_rep(glob.glob(os.path.join(base_dir, base_pat)))
        base_sorted = [base_by[r] for r in sorted(base_by)[:20]]
        drift_by = _by_rep(glob.glob(os.path.join(DRIFT, f"df_rows_0_0_*_{drift_tag}_*.csv")))
        drift_reps = sorted(drift_by)[:20]
        print(f"=== {scenario}  ({len(drift_reps)} drift x {len(base_sorted)} baseline reps) ===")
        for k, drep in enumerate(drift_reps):
            if k >= len(base_sorted):
                break
            build(scenario, drep, drift_by[drep], base_sorted[k])
    print("done ->", OUT)
