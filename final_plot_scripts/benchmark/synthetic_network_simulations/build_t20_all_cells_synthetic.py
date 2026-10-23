"""All-cells-at-time-20 BEELINE inputs + z_U (z_rho-hat-Delta, all 6000 twin pairs) for the synthetic sweeps.
Families: network_sweep (network_sweep_final + e13_pos100), mixed_network_sweep. Same recipe as
real_network_simulations/build_t20_all_cells_inputs_and_zU.py. Skips finished runs; skips (and reports) files that are
mid-write/incomplete, so it can be rerun after the refill simulations finish.
    python build_t20_all_cells_synthetic.py [--jobs 8] [--family network_sweep|mixed_network_sweep] [--limit k]
"""
import argparse, os, sys, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
for p in ("code/TwINFER", "code/TwINFER/package", "code/TwINFER/benchmarks/network_benchmarks/infer", "code/TwINFER/benchmarks/network_benchmarks/to_beeline",
          "code/TwINFER/paper_analysis/causal_direction_inference/twinscore_stage_checks"):
    sys.path.insert(0, str(REPO / p))
import infer_network_sweep_final_allpairs as NSF, infer_e13_pos100_allpairs as E13, mixed_to_beeline_format as MIX
from twin_decomposition import twin_stats
from twinfer_to_boolode_format import convert_ground_truth
BEE = REPO / "code/Beeline/inputs"; ZOUT = REPO / "analysis_data/synthetic_network_simulations/zU_t20_all"; T = 20


def tasks(family):
    """[(group_dir_in_beeline, net, rep, path, topology_txt)]"""
    out = []
    if family == "network_sweep":
        for net, rep, p in E13.discover_tasks():
            out.append(("network_sweep_t20_all", net, rep, Path(p), Path(E13.LABEL_TO_GT[net])))
        for net, rep, p in NSF.discover_tasks():
            out.append(("network_sweep_t20_all", net, rep, Path(p), Path(NSF.TOPOLOGY_DIR) / f"{net}.txt"))
    else:
        for net, rep, p in MIX.discover():
            out.append(("mixed_network_sweep_t20_all", net, rep, Path(p), MIX.TOPOLOGY_DIR / f"{net}.txt"))
    return out


def run(a):
    grp, net, rep, path, topo = a
    n = pd.read_csv(topo, header=None).shape[0]
    run_dir = BEE / grp / net / f"simrep{rep}_all_t20"; zf = ZOUT / grp / f"{net}_rep{rep}_zU.csv"
    ok = lambda q: q.exists() and q.stat().st_size > 1000
    if ok(zf) and ok(run_dir / "ExpressionData.csv") and ok(run_dir / "PseudoTime.csv"):
        return net, rep, "skip"
    cols = [f"gene_{i+1}_mRNA" for i in range(n)]
    try:
        d = pd.read_csv(path, usecols=["cell_id", "time_step", "replicate", "clone_id"] + cols)
        d = d[d.time_step == T].sort_values(["clone_id", "replicate"]).reset_index(drop=True); ncl = d.clone_id.nunique()
        assert len(d) == 2 * ncl == 12000 and (d.replicate.to_numpy().reshape(ncl, 2) == [1, 2]).all()
    except Exception as e:
        return net, rep, f"INCOMPLETE/UNREADABLE ({type(e).__name__})"
    X = {i: d[cols[i]].to_numpy(float).reshape(ncl, 2) for i in range(n)}; rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if X[i].std() == 0 or X[j].std() == 0: continue
            o = twin_stats(X[i], X[j]); rows.append(dict(gene_1=f"gene_{i+1}", gene_2=f"gene_{j+1}", z_U=o["z_U"], z_rho=o["z_rho"], z_C=o["z_C"], S=o["S"], C=o["C"]))
    zf.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(zf, index=False)
    run_dir.mkdir(parents=True, exist_ok=True); labels = [f"cell{c}_t{T}" for c in d.cell_id]
    expr = d[cols].T; expr.columns = labels; expr.index = [f"gene_{i+1}" for i in range(n)]; expr.to_csv(run_dir / "ExpressionData.csv")
    pd.DataFrame({"PseudoTime1": np.full(len(d), T)}, index=labels).to_csv(run_dir / "PseudoTime.csv")
    return net, rep, f"{len(rows)} pairs"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--family", default=None, choices=["network_sweep", "mixed_network_sweep"]); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); T_ = []
    for f in ([a.family] if a.family else ["network_sweep", "mixed_network_sweep"]): T_ += tasks(f)
    if a.limit: T_ = T_[:a.limit]
    for grp, net, _, _, topo in T_:
        gt = BEE / grp / net / "GroundTruthNetwork.csv"
        if not gt.exists(): convert_ground_truth(topo, gt)
    print(len(T_), "tasks", flush=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run, T_, chunksize=1): print(*r, flush=True)
