from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import os, time, sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation")
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set, read_input_matrix
from benchmarks.network_benchmarks.simulate.real_network_multistate_sim import build_matrices

set_num_threads(50)
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/69296547-e537-4dbf-b059-2e4d21364c6b/scratchpad/mt"
OUT = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/69296547/mt"
os.makedirs(f"{OUT}/logs", exist_ok=True)
IR = f'{TWINFER_PROJECT_ROOT}/input_data'
NC, NT = 300, 300

CASES = [("GSD", 2.0, 2.0), ("GSD", 3.0, 2.0), ("HSC", 3.0, 2.0),
         ("EMT", 2.0, 2.0), ("mCAD", 2.0, 1.0), ("VSC", 2.0, 1.0)]

for net, nh, ks in CASES:
    ng, M = read_input_matrix(f"{IR}/real_world_networks/{net}.txt")
    nm, km = build_matrices(M, nh, ks)
    cfg = dict(
        n_cells=NC, simulation_time_before_division=NT,
        twin_simulation_time_after_division=5, twin_measurement_resolution=1,
        path_to_connectivity_matrix=f"{IR}/real_world_networks/{net}.txt",
        param_csv=f"{IR}/network_sweep/parameters.csv", rows_to_use=[[0] * ng],
        output_folder=OUT, log_file=f"{OUT}/logs/{net}.log",
        type=f"{net}_mt_n{nh}_k{ks}", combinatorial_interaction_type="additive",
        n_matrix=nm, k_add_matrix=km, use_csv_k_add=False,
        number_of_parallel_parameters=1, number_of_cores_per_parameter=50,
    )
    t0 = time.time()
    process_param_set([0] * ng, "r", cfg)
    dt = time.time() - t0
    full_t = 6000 if net == "VSC" else 1500
    scale = (3000 / NC) * (full_t / NT)
    print(f"@@@ {net} n={nh} kadd_x{ks}:  {dt:.0f}s ({NC}c x {NT}t)  "
          f"-> ~{dt*scale/3600:.1f} h for 3000c x {full_t}t @ 50 cores", flush=True)
