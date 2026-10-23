"""All-cells-at-time-20 benchmark inputs for the 7 real-network simulations.

For every simulation replicate (same replicate set as the existing t1=10 real-network BEELINE runs; EMT = its
10 rep files in paper_analysis/EMT/simulate/20260825_224653) keep ALL 12,000 cells (6000 clones x 2 sisters) at
time_step == 20 and write
  (a) a BEELINE run dir  code/Beeline/inputs/real_networks_t20_all/<net>/simrep<rep>_all_t20/{ExpressionData,PseudoTime}.csv
      (genes x cells, raw mRNA counts, PseudoTime = 20 for all cells) + <net>/GroundTruthNetwork.csv
  (b) z_U (= z_rho-hat-Delta) for every ordered gene pair from all 6000 twin pairs (twinscore_stage_checks.twin_stats),
      -> analysis_data/paper_analysis/real_networks/zU_t20_all/<net>_<rep>_zU.csv

    python build_t20_all_cells_inputs_and_zU.py [--jobs 4] [--limit 1]
"""
import argparse, glob, os, sys, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
sys.path.insert(0, str(REPO / "code/TwINFER/package"))
sys.path.insert(0, str(REPO / "code/TwINFER/benchmarks/network_benchmarks/infer"))
sys.path.insert(0, str(REPO / "code/TwINFER/benchmarks/network_benchmarks/to_beeline"))
sys.path.insert(0, str(REPO / "code/TwINFER/paper_analysis/causal_direction_inference/twinscore_stage_checks"))
import infer_real_network_allpairs as M
from twin_decomposition import twin_stats
from twinfer_to_boolode_format import convert_ground_truth

RW = REPO / "input_data/real_world_networks"
BEE = REPO / "code/Beeline/inputs/real_networks_t20_all"
ZOUT = REPO / "analysis_data/paper_analysis/real_networks/zU_t20_all"
EMT_DIR = REPO / "analysis_data/paper_analysis/EMT/simulate/20260825_224653"
T = 20


def tasks():
    out = [(net, rep, Path(p), M.NETWORKS[net]["n_genes"], M.NETWORKS[net]["topology"])
           for p, net, rep, _ in M.build_tasks() if net in ("GSD", "HSC", "VSC", "mCAD", "B_cell_activation", "Pluripotent")]
    for f in sorted(glob.glob(f"{EMT_DIR}/df_rows_*_ncells_6000_EMT_rep_*.csv")):
        rep = os.path.basename(f).split("_EMT_rep_")[1][:-4]
        out.append(("EMT", rep, Path(f), 17, "EMT.txt"))
    return out


def run(a):
    net, rep, path, n, topo = a
    run_dir = BEE / net / f"simrep{rep}_all_t20"
    zf = ZOUT / f"{net}_{rep}_zU.csv"
    ok = lambda q: q.exists() and q.stat().st_size > 1000   # a failed (quota) write leaves an empty file
    if ok(zf) and ok(run_dir / "ExpressionData.csv") and ok(run_dir / "PseudoTime.csv"):
        return net, rep, "skip"
    cols = [f"gene_{i+1}_mRNA" for i in range(n)]
    d = pd.read_csv(path, usecols=["cell_id", "time_step", "replicate", "clone_id"] + cols)
    d = d[d.time_step == T].sort_values(["clone_id", "replicate"]).reset_index(drop=True)
    ncl = d.clone_id.nunique()
    assert len(d) == 2 * ncl == 12000 and (d.replicate.to_numpy().reshape(ncl, 2) == [1, 2]).all(), (net, rep, len(d))
    X = {i: d[cols[i]].to_numpy(float).reshape(ncl, 2) for i in range(n)}
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if X[i].std() == 0 or X[j].std() == 0:
                continue
            o = twin_stats(X[i], X[j]); rows.append(dict(gene_1=f"gene_{i+1}", gene_2=f"gene_{j+1}", z_U=o["z_U"], z_rho=o["z_rho"], z_C=o["z_C"], S=o["S"], C=o["C"]))
    ZOUT.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(zf, index=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"cell{c}_t{T}" for c in d.cell_id]
    expr = d[cols].T; expr.columns = labels; expr.index = [f"gene_{i+1}" for i in range(n)]
    expr.to_csv(run_dir / "ExpressionData.csv")
    pd.DataFrame({"PseudoTime1": np.full(len(d), T)}, index=labels).to_csv(run_dir / "PseudoTime.csv")
    return net, rep, f"{len(rows)} pairs"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    T_ = tasks()
    if a.limit: T_ = T_[:a.limit]
    for net, _, _, _, topo in T_:
        gt = BEE / net / "GroundTruthNetwork.csv"
        if not gt.exists():
            convert_ground_truth(RW / topo, gt)
    print(len(T_), "tasks", flush=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run, T_):
            print(*r, flush=True)
