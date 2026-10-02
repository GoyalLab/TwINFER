# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Timing test: 800 cells x 1000 steps per net, 26 threads -> extrapolate to full run."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, time
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set

THREADS = 26
set_num_threads(THREADS)
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/69296547-e537-4dbf-b059-2e4d21364c6b/scratchpad/simtime"
OUT = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/69296547/simtime"
os.makedirs(f"{OUT}/logs", exist_ok=True)
INPUT_ROOT = f'{TWINFER_PROJECT_ROOT}/input_data'

NC, NT = 800, 1000
for net, n_genes, full_nt in [("mCAD", 5, 2000), ("GSD", 19, 2000), ("VSC", 8, 6000)]:
    cfg = dict(
        n_cells=NC, simulation_time_before_division=NT,
        twin_simulation_time_after_division=48, twin_measurement_resolution=1,
        path_to_connectivity_matrix=f"{INPUT_ROOT}/real_world_networks/{net}.txt",
        param_csv=f"{INPUT_ROOT}/network_sweep/parameters.csv",
        rows_to_use=[[0] * n_genes], output_folder=OUT,
        log_file=f"{OUT}/logs/{net}.log", type=f"{net}_timetest",
        combinatorial_interaction_type="additive",
        number_of_parallel_parameters=1, number_of_cores_per_parameter=THREADS,
    )
    label = "rows_" + "_".join(map(str, cfg["rows_to_use"][0]))
    t0 = time.time()
    process_param_set(cfg["rows_to_use"][0], label, cfg)
    dt = time.time() - t0
    # before-division cost ~ n_cells * n_steps ; twin cost roughly constant here
    scale = (6000 / NC) * (full_nt / NT)
    print(f"\n### {net}: {dt:.1f}s for {NC}c x {NT}t  ->  ~{dt*scale/60:.0f} min for 6000c x {full_nt}t  (x{scale:.0f})\n", flush=True)
