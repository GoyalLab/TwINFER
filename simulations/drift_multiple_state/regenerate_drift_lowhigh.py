"""
Drift 0.66 -> {0.12 (low), 1.66 (high)} over 0-10 h, with the gene_1->gene_2 Hill K tracking k_on.

Burn-in: k_on = 0.66 (the Figure-2 single-state value; K calibrated there).  The burn-in end state is
REUSED from the Figure-2 1000-replicate set (simulation_before_division_*_rep_<n>_*.csv), so replicate n
here starts from exactly the same parents as Figure-2 single-state replicate n.
At division the 6,000 parents split 50/50 into clones (twins share fate):
    low  half (clone_id 0..2999):    k_on 0.66 -> 0.12 linearly over tau = 10 h, K tracks live (K_ramp)
    high half (clone_id 3000..5999): k_on 0.66 -> 1.66 linearly over tau = 10 h, K tracks live (K_ramp)
K_ramp = gillespie_meanfield_K_all_cells (K set every dt_K h to the half's mean gene_1 protein^n).
For A_B (no gene_1->gene_2 reaction) it is a plain k_on ramp.  Twins run twin_time h after division, hourly.
state column = "low" / "high".  Output: <OUT>/df_rows_0_1_<ts>_ncells_6000_<type>_rep_<n>_<id>.csv
(file naming and columns match the Figure-2 files so the same analysis scripts read them).
"""
import argparse, glob, json, os, re, time, uuid
from datetime import datetime
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from numba import set_num_threads
from twinfer.simulation.gillespie_drift import (
    read_input_matrix, generate_reaction_network_from_matrix, generate_initial_state_from_genes,
    assign_parameters_to_genes, add_interaction_terms, setup_gillespie_params_from_reactions,
    make_lowmid_mid_update, gillespie_simulation_all_cells, gillespie_meanfield_K_all_cells, convert_samples_to_df)
from twinfer.utils.paths import get_repo_root

REPO = get_repo_root()
K_ON = {"burnin": 0.66, "low": 0.12, "high": 1.66}          # Figure-2: single-state / low_k_on / high_k_on
FIG2 = "/projects/b1255/yscher/TranscriptomicDistance/simulation_data/figure_2_simulations_1000"
NETWORKS = {"A_B": "connectivity_matrix_A_B.txt", "A_to_B": "connectivity_matrix_A_to_B.txt"}

def find_burnin(network, rep):
    pat = re.compile(rf"^simulation_before_division_df_rows_0_1_.*_ncells_6000_{network}_rep_{rep}_[0-9a-f]{{8}}\.csv$")
    d = f"{FIG2}/{network}"; fs = [f for f in os.listdir(d) if pat.match(f)]
    if not fs: raise FileNotFoundError(f"no burn-in for {network} rep {rep}")
    fs.sort(key=lambda f: os.path.getmtime(f"{d}/{f}")); return f"{d}/{fs[-1]}", len(fs)

def run_rep(rep, cfg):
    out_dir, net, ttype = cfg["out_dir"], cfg["network"], cfg["type"]
    if glob.glob(f"{out_dir}/df_rows_0_1_*_ncells_6000_{ttype}_rep_{rep}_*.csv"):
        print(f"[{ttype} rep {rep}] skip (exists)", flush=True); return None
    set_num_threads(cfg["cores_per"]); t0 = time.time()
    tau, t_div = float(cfg["tau"]), int(cfg["burnin_hours"])
    n_cells, n_half = 6000, 3000
    n_genes, conn = read_input_matrix(f"{REPO}/simulation_example_input_data/{NETWORKS[net]}")
    reactions_df, gene_list = generate_reaction_network_from_matrix(conn)
    init_states = generate_initial_state_from_genes(gene_list)
    pdict = assign_parameters_to_genes(f"{REPO}/simulation_example_input_data/median_parameter.csv", gene_list, [0, 1])
    for g in gene_list: pdict[f"{{k_on_{g}}}"] = K_ON["burnin"]
    n_matrix = np.zeros((n_genes, n_genes)); k_add_matrix = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(n_genes):
            if conn[i, j] != 0:
                e = f"{gene_list[i]}_to_{gene_list[j]}"
                n_matrix[i, j] = pdict.get(f"{{n_{e}}}", 2.0); k_add_matrix[i, j] = pdict.get(f"{{k_add_{e}}}", 6.0)
    ss, fpd = add_interaction_terms(pdict, conn, gene_list, n_matrix=n_matrix, k_add_matrix=k_add_matrix, scale_k=None)
    pop0, update_matrix, update_prop, species_index = setup_gillespie_params_from_reactions(init_states, reactions_df, fpd)
    k_on_idx = np.where(reactions_df['propensity'].str.contains("k_on")
                        & reactions_df['species1'].str.contains("gene_1|gene_2", regex=True))[0]
    _reg = np.where(reactions_df['propensity'].str.contains(r"\{k_gene_1_to_gene_2\}", regex=True))[0]
    reg_idx = int(_reg[0]) if len(_reg) else -1
    k_add_e = float(pdict.get("{k_add_gene_1_to_gene_2}", 6.0)); hill_n = float(pdict.get("{n_gene_1_to_gene_2}", 2.0))
    g1p, g2i = species_index["gene_1_protein"], species_index["gene_2_I"]
    live_K = reg_idx >= 0
    # burn-in end state (reused from Figure-2)
    bpath, nfound = find_burnin(net, rep)
    bdf = pd.read_csv(bpath, usecols=["cell_id", "time_step"] + [f"{g}_{s}" for g in gene_list for s in ("mRNA", "protein")])
    bdf = bdf[bdf["time_step"] == bdf["time_step"].max()].sort_values("cell_id")
    assert len(bdf) == n_cells, (len(bdf), bpath)
    final = np.zeros((n_cells, len(species_index)), dtype=np.int64)
    for g in gene_list:
        final[:, species_index[f"{g}_I"]] = 1
        for s in ("mRNA", "protein"): final[:, species_index[f"{g}_{s}"]] = bdf[f"{g}_{s}"].to_numpy()
    halves = {"low": final[:n_half], "high": final[n_half:]}
    rep_time = np.arange(0, cfg["twin_time"] + 1, 1)
    outs = {}
    for name, fin in halves.items():
        pop = np.concatenate([fin.T, fin.T], axis=1); factor = K_ON[name] / K_ON["burnin"]
        flags = np.zeros(2 * n_half, dtype=np.int64)
        if live_K:
            outs[name] = gillespie_meanfield_K_all_cells(update_prop, update_matrix, pop, rep_time, flags,
                np.asarray(k_on_idx, dtype=np.int64), int(reg_idx), int(g1p), int(g2i), float(k_add_e), 1.0, float(hill_n),
                float(factor), float(t_div), float(tau), float(t_div), float(cfg["dt_K"]))
        else:
            upd = make_lowmid_mid_update(update_prop, k_on_idx, reg_idx, g1p, g2i, k_add_e, 1.0, hill_n,
                                         kon_factor_final=factor, t_start=t_div, tau=tau, t_offset=t_div)
            outs[name] = gillespie_simulation_all_cells(upd, update_matrix, pop, rep_time, flags)
    df = convert_samples_to_df(np.concatenate([outs["low"], outs["high"]], axis=0), species_index)
    clone = np.concatenate([np.tile(np.arange(n_half), 2), np.tile(np.arange(n_half, n_cells), 2)])
    repl = np.concatenate([np.ones(n_half, int), np.full(n_half, 2, int), np.ones(n_half, int), np.full(n_half, 2, int)])
    state = np.array(["low"] * (2 * n_half) + ["high"] * (2 * n_half))
    df["clone_id"] = clone[df["cell_id"]]; df["replicate"] = repl[df["cell_id"]]; df["state"] = state[df["cell_id"]]
    ts = datetime.now().strftime("%d%m%Y_%H%M%S"); uid = uuid.uuid4().hex[:8]
    path = f"{out_dir}/df_rows_0_1_{ts}_ncells_6000_{ttype}_rep_{rep}_{uid}.csv"
    df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    with open(f"{out_dir}/log.jsonl", "a") as f:
        f.write(json.dumps({"id": uid, "rep": rep, "type": ttype, "timestamp": ts, "burnin_file": os.path.basename(bpath),
                            "burnin_files_found": nfound, "k_on": K_ON, "tau": tau, "live_K": live_K, "dt_K": cfg["dt_K"],
                            "twin_time": cfg["twin_time"], "minutes": round((time.time() - t0) / 60, 2)}) + "\n")
    print(f"[{ttype} rep {rep}] {(time.time()-t0)/60:.1f} min -> {path}", flush=True); return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=list(NETWORKS)); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", default="0-999", help="e.g. 0-999, 5, 10-19"); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cores-per", type=int, default=4); ap.add_argument("--twin-time", type=int, default=48)
    ap.add_argument("--tau", type=float, default=10.0); ap.add_argument("--burnin-hours", type=int, default=1000)
    ap.add_argument("--dt-K", type=float, default=0.05)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    lo, hi = a.reps.split("-") if "-" in a.reps else (a.reps, a.reps); reps = range(int(lo), int(hi) + 1)
    cfg = dict(out_dir=a.out_dir, network=a.network, type=f"{a.network}_drift_lowhigh_Kramp", cores_per=a.cores_per,
               twin_time=a.twin_time, tau=a.tau, burnin_hours=a.burnin_hours, dt_K=a.dt_K)
    print(f"{len(reps)} reps  net={a.network}  tau={a.tau}h  twin={a.twin_time}h  jobs={a.jobs} x {a.cores_per} cores -> {a.out_dir}", flush=True)
    Parallel(n_jobs=a.jobs, backend="multiprocessing")(delayed(run_rep)(i, cfg) for i in reps)
    print("done", flush=True)
