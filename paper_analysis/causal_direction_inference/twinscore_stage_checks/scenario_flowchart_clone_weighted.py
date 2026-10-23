"""Paper-flowchart scenario calls (Fig. 2a) from infer.py's clone-weighted statistics, plus S / C / U for the rank-overlap check.

Per replicate of A_B, A_to_B, A_B (two states), A_to_B (two states) (yscher figure_2_simulations_1000) it reuses
simulations/drift_multiple_state/run_6scenario_zscores.process_replicate (same clone partition / seeds / null draws as infer.py) and adds:
  S, C, U, h_x, h_y, rho_hat_Delta (analytic) at t1 and t2 from the within-time twin table (twin-centred ranks).
Calls (gene_1-gene_2), thresholds are arguments of the summary step, the raw numbers are all saved:
  Stage I   rho vs scramble null at t1:            |step1_z_t1| > 2.576 (two-sided p=0.01, infer.py) [paper: one-tailed p=0.01 -> 2.326]
  Stage II  z_het (twin rho_Delta(t1) vs random-pair rho_Delta):   |z_het_t1| > 5 (infer.py default) [paper: -10]
  Stage III z_d = d = rho_hat_Delta(t2)-rho_hat_Delta(t1) vs null:  z_d > 2.326 (alpha_stage3 = 0.01)
  1 no regulation | 2 single state + regulation | 3 multiple states + regulation | 4 multiple states, no regulation
Usage: python scenario_flowchart_clone_weighted.py --output-dir DIR --n-reps 100 [--only scen:rep,...] [--jobs 8]
"""
import argparse, glob, json, os, sys, time, traceback
import numpy as np, pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "simulations", "drift_multiple_state"))
import run_6scenario_zscores as R   # noqa: E402

KEEP = {"no_regulation": "A_B", "A_to_B": "A_to_B", "multistate_A_B": "A_B (two states)", "multistate_A_to_B": "A_to_B (two states)"}
G = ["gene_1", "gene_2"]


def _rank(v):
    return pd.Series(v).rank(method="average").to_numpy()


def scu(twins):
    """S, C, U, h_x, h_y, rho_hat_Delta for gene_1-gene_2 from a twin table (2 rows per twin_id; one twin = one clone, weight 1)."""
    t = twins.copy()
    t["_k"] = t.groupby("twin_id").cumcount()
    out = {}
    r = {}
    for g in G:
        col = f"{g}_mRNA"
        t["_r"] = _rank(t[col].to_numpy())
        w = t.pivot(index="twin_id", columns="_k", values="_r").to_numpy()
        r[g] = w - w.mean()
    M = {g: ((r[g] ** 2).sum(1) / 2).mean() for g in G}
    nrm = np.sqrt(M["gene_1"] * M["gene_2"])
    x, y = r["gene_1"], r["gene_2"]
    S = ((x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]) / 2).mean() / nrm
    C = ((x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]) / 2).mean() / nrm
    hx = max(0.0, (x[:, 0] * x[:, 1]).mean() / M["gene_1"]); hy = max(0.0, (y[:, 0] * y[:, 1]).mean() / M["gene_2"])
    dx, dy = x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]
    rd = (dx * dy).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    n = x.shape[0]
    return dict(S=S, C=C, U=S - C, h_x=hx, h_y=hy, rho_hat_delta=rd, n_twins=n,
                z_C=np.sign(S) * C / np.sqrt((1 + S ** 2 + hx * hy) / (2 * n)), z_U=rd * np.sqrt(n))


def run_one(path, scen, rep, out_dir, n_cores):
    out = os.path.join(out_dir, f"{scen}_rep_{rep}.json")
    if os.path.exists(out):
        return out, "skipped"
    try:
        rec = R.process_replicate(path, scen, rep, n_cores=n_cores)
        P = R._partition(R._load(path))
        for tag, key in (("t1", "t1_twins"), ("t2", "t2_twins")):
            for k, v in scu(P[key]).items():
                rec[f"{k}_{tag}"] = float(v)
            # analytic z_rho (picture table): rho (clone-weighted Spearman over all cells at this time) / sigma_rho,
            # sigma_rho^2 = (NT*(1/2 + 1/2 h_x h_y) + NSH) / NB^2  (NT twin clones weight 1/2 per cell, NSH singleton cells)
            nt = rec[f"n_twins_{tag}"]; nsh = len(P["ac_left"]); nb = nt + nsh
            sig = np.sqrt((nt * (0.5 + 0.5 * rec[f"h_x_{tag}"] * rec[f"h_y_{tag}"]) + nsh) / nb ** 2)
            rec[f"z_rho_{tag}"] = float(rec[f"rho_{tag}"] / sig)
        with open(out + ".tmp", "w") as f:
            json.dump(rec, f, cls=R.NpEnc)
        os.replace(out + ".tmp", out)
        return out, "OK"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return out, f"ERROR {type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True); ap.add_argument("--n-reps", type=int, default=20)
    ap.add_argument("--n-cores", type=int, default=1); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    R.N_REPS = a.n_reps
    tasks = [t for t in R.collect_tasks() if t[0] in KEEP]
    if a.only:
        want = set(a.only.split(",")); tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in want]
    os.makedirs(a.output_dir, exist_ok=True)
    print(f"{len(tasks)} tasks -> {a.output_dir}", flush=True)

    def _do(i, s, r, p):
        t0 = time.time(); _, st = run_one(p, s, r, a.output_dir, a.n_cores)
        print(f"[{i + 1}/{len(tasks)}] {s} rep {r}: {st} ({time.time() - t0:.0f}s)", flush=True)

    if a.jobs > 1:
        Parallel(n_jobs=a.jobs, backend="loky")(delayed(_do)(i, s, r, p) for i, (s, r, p) in enumerate(tasks))
    else:
        for i, (s, r, p) in enumerate(tasks): _do(i, s, r, p)
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.output_dir, "*_rep_*.json")))]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.output_dir, "flowchart_zscores.csv"), index=False)
