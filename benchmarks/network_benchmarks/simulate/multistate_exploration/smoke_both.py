# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Tiny smoke test of BOTH drivers' internals: mCAD + HSC, 60 cells x 30 steps.
Runs on the 4-core login node -> keep it tiny."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, time, numpy as np, pandas as pd, glob
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/heterogeneity_vs_regulation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code")
from numba import set_num_threads
set_num_threads(4)
from twinfer.simulation.gillespie_simulations import process_param_set, read_input_matrix
from benchmarks.network_benchmarks.simulate.real_network_multistate_sim import NETWORKS, build_matrices
from benchmarks.network_benchmarks.simulate.real_network_seeded_sim import seeded_states, make_pop0_builder

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/69296547-e537-4dbf-b059-2e4d21364c6b/scratchpad/smoke_both_out"
OUT = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/69296547/smoke_both_out"
os.makedirs(f"{OUT}/logs", exist_ok=True)
IR = f'{TWINFER_PROJECT_ROOT}/input_data'

for net in ["mCAD", "HSC"]:
    spec = NETWORKS[net]
    ng, M = read_input_matrix(f"{IR}/real_world_networks/{net}.txt")
    nm, km = build_matrices(M, spec["n_hill"], spec["kadd_scale"])
    base = dict(
        n_cells=60, simulation_time_before_division=30,
        twin_simulation_time_after_division=5, twin_measurement_resolution=1,
        path_to_connectivity_matrix=f"{IR}/real_world_networks/{net}.txt",
        param_csv=f"{IR}/network_sweep/parameters.csv", rows_to_use=[[0]*ng],
        output_folder=OUT, log_file=f"{OUT}/logs/{net}.log",
        combinatorial_interaction_type="additive",
        n_matrix=nm, k_add_matrix=km, use_csv_k_add=False,
        number_of_parallel_parameters=1, number_of_cores_per_parameter=4,
    )
    # --- IC-unset ---
    c = dict(base, type=f"{net}_smoke_unset")
    t0 = time.time(); process_param_set([0]*ng, "r", c)
    print(f"[{net}] IC-unset OK ({time.time()-t0:.0f}s)", flush=True)

    # --- seeded ---
    mrna, prot, afrac = seeded_states(M, spec["n_hill"], spec["kadd_scale"])
    print(f"[{net}] seeded_states -> {len(prot)} stable FPs", flush=True)
    c = dict(base, type=f"{net}_smoke_seeded", pop0_mat=make_pop0_builder(mrna, prot, afrac))
    t0 = time.time(); process_param_set([0]*ng, "r", c)
    print(f"[{net}] seeded OK ({time.time()-t0:.0f}s)", flush=True)

    # check the seeded before-division file starts with distinct sub-populations
    f = sorted(glob.glob(f"{OUT}/simulation_before_division_df_*{net}_smoke_seeded*.csv"))[-1]
    d = pd.read_csv(f)
    t0d = d[d.time_step == d.time_step.min()]
    pcols = [x for x in d.columns if x.endswith("_protein")]
    print(f"[{net}] seeded t=0 protein spread (should show >1 level per gene for multi-FP nets):")
    for x in pcols[:4]:
        print(f"    {x}: unique start values = {sorted(t0d[x].unique())[:6]}")
print("\nALL SMOKE TESTS PASSED")
