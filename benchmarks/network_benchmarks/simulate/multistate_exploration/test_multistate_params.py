# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Medium test: do the tuned params produce >1 mode in the stochastic sim?
400 cells x 800 steps, GSD (n=3,x2), EMT (n=2,x2), mCAD (n=2,x1), VSC (n=2,x1)."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, time, sys, numpy as np, pandas as pd, glob
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation")
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set, read_input_matrix
from benchmarks.network_benchmarks.simulate.real_network_multistate_sim import build_matrices, KADD_POS, KADD_NEG

set_num_threads(24)
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/69296547-e537-4dbf-b059-2e4d21364c6b/scratchpad/simtest_params"
OUT = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/69296547/simtest_params"
os.makedirs(f"{OUT}/logs", exist_ok=True)
IR = f'{TWINFER_PROJECT_ROOT}/input_data'

for net, nh, ks in [("GSD", 3.0, 2.0), ("EMT", 2.0, 2.0), ("mCAD", 2.0, 1.0), ("VSC", 2.0, 1.0)]:
    ng, M = read_input_matrix(f"{IR}/real_world_networks/{net}.txt")
    nm, km = build_matrices(M, nh, ks)
    cfg = dict(
        n_cells=400, simulation_time_before_division=800,
        twin_simulation_time_after_division=10, twin_measurement_resolution=1,
        path_to_connectivity_matrix=f"{IR}/real_world_networks/{net}.txt",
        param_csv=f"{IR}/network_sweep/parameters.csv",
        rows_to_use=[[0]*ng], output_folder=OUT, log_file=f"{OUT}/logs/{net}.log",
        type=f"{net}_mstest", combinatorial_interaction_type="additive",
        n_matrix=nm, k_add_matrix=km, use_csv_k_add=False,
        number_of_parallel_parameters=1, number_of_cores_per_parameter=24,
    )
    t0 = time.time()
    process_param_set([0]*ng, "rows_"+"_".join(["0"]*ng), cfg)
    print(f"\n### {net} (n={nh}, k_add x{ks}): {time.time()-t0:.0f}s", flush=True)
    f = sorted(glob.glob(f"{OUT}/simulation_before_division_df_*{net}_mstest*.csv"))[-1]
    d = pd.read_csv(f)
    last = d[d.time_step == d.time_step.max()]
    pcols = [c for c in d.columns if c.endswith("_protein")]
    for c in pcols:
        x = np.log1p(last[c].values)
        if x.std() < 1e-6:
            print(f"   {c:16s} flat at {last[c].mean():.0f}")
            continue
        # bimodality coefficient
        from scipy.stats import skew, kurtosis
        b, k = skew(x), kurtosis(x)
        bc = (b**2 + 1) / (k + 3)
        mult = "  <-- BIMODAL" if bc > 0.555 else ""
        print(f"   {c:16s} mean={last[c].mean():8.0f} cv={last[c].std()/(last[c].mean()+1e-9):.2f} BC={bc:.3f}{mult}")
