from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, json, pickle, glob, traceback
import numpy as np
import pandas as pd

path_to_repo = f'{TWINFER_PROJECT_ROOT}'
path_to_code_repo = f"{path_to_repo}/code/TwINFER"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.append(path_to_code_repo)

# from TwINFER_function_scripts.infer_with_twinfer import infer_with_twinfer   # [2026-09-30 ported: TwINFER_function_scripts was merged into the twinfer package (git 58f3442); old import kept above as a comment]
from twinfer.inference.infer import infer_with_twinfer

# path_to_connectivity_matrix = f"{path_to_repo}/input_data/cycle_6_node.txt"   [2026-09-30 replaced per user: the original name no longer exists; cycle_6_node.txt -> input_data/cycle_g6.txt; simulation_data/cycle_data -> simulation_data/cyclic_6_nodes]
path_to_connectivity_matrix = f"{path_to_repo}/input_data/cycle_g6.txt"
param_csv = f"{path_to_repo}/input_data/network_sweep/parameters.csv"
out_root = f"{path_to_repo}/simulation_data/cyclic_6_nodes"

k_add_matrix = np.full((6, 6), 2)
rows_to_use = [[0] * 6]

# No new simulation is run here. We reuse the 10 already-simulated replicate
# files sitting in out_root (created previously) and only run inference on them.
simulation_files = sorted(glob.glob(f"{out_root}/df_rows_*.csv"))
N_REPLICATES = len(simulation_files)
print(f"Found {N_REPLICATES} existing simulation files to run inference on.", flush=True)

summary = []

for i, path_to_simulation_file in enumerate(simulation_files):
    rep_dir = f"{out_root}/replicate_{i}"
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(f"{rep_dir}/logs", exist_ok=True)
    print(f"\n===== Replicate {i} (inference only) =====", flush=True)
    print(f"Using existing simulation file: {path_to_simulation_file}", flush=True)
    try:
        base_config = {
            'n_cells': 6000,
            'simulation_time_before_division': 6000,
            'twin_simulation_time_after_division': 48,
            'twin_measurement_resolution': 1,
            'path_to_connectivity_matrix': path_to_connectivity_matrix,
            'param_csv': param_csv,
            'rows_to_use': rows_to_use,
            'output_folder': f"{rep_dir}/",
            'log_file': f"{rep_dir}/logs/replicate_{i}.log",
            'type': f"cyclic_6_node_rep{i}",
            'multiple_interaction_type': "additive",
            'number_of_parallel_parameters': 1,
            'number_of_cores_per_parameter': 20,
            'k_add_matrix': k_add_matrix,
            'use_csv_k_add': False,
        }

        twinfer_kwargs = {
            "path_to_simulation_file": path_to_simulation_file,
            "base_config": base_config,
            "t1": 1,
            "t2": 20,
            "check_for_steady_state": False,
            "threshold_gene_gene_corr": 0.04,
            "use_scramble": True,
            "p_val_threshold_scrambled_gene_correlation": 0.02,
            "show_scrambled_distribution_gene_correlation": 0.02,
            "z_score_threshold_two_states": 12,
            "p_value_threshold_cross_correlation": 0.01,
            "plot_correlation_matrices_as_heatmap": False,
            "have_any_output": True,
            "seed": 101010 + i,
            "merge_time_points": True,
            "n_cores": 18,
            "ranked_list": True,
            "match_sim_details": False,
        }

        correlation_matrices = infer_with_twinfer(**twinfer_kwargs)

        with open(f"{rep_dir}/correlation_matrices.pkl", "wb") as f:
            pickle.dump(correlation_matrices, f)

        final_directed_edges = correlation_matrices.get('final_directed_edges', [])
        gene_list = sorted(
            {g for pair in correlation_matrices.get('all_gene_pairs', []) for g in pair},
            key=lambda x: int(x.split("_")[1])
        )

        true_edges = {("gene_1","gene_2"),("gene_2","gene_3"),("gene_3","gene_4"),
                      ("gene_4","gene_5"),("gene_5","gene_6"),("gene_6","gene_1")}
        inferred_edges = set(tuple(e) for e in final_directed_edges)
        tp = len(inferred_edges & true_edges)
        fp = len(inferred_edges - true_edges)
        fn = len(true_edges - inferred_edges)

        rep_summary = {
            "replicate": i,
            "path_to_simulation_file": path_to_simulation_file,
            "n_inferred_edges": len(inferred_edges),
            "true_positive_edges": tp,
            "false_positive_edges": fp,
            "false_negative_edges": fn,
            "inferred_edges": sorted(list(inferred_edges)),
        }
        with open(f"{rep_dir}/summary.json", "w") as f:
            json.dump(rep_summary, f, indent=2)
        summary.append(rep_summary)
        print(f"Replicate {i}: TP={tp} FP={fp} FN={fn} (of 6 true cycle edges)", flush=True)
    except Exception as e:
        print(f"Replicate {i} FAILED: {e}", flush=True)
        traceback.print_exc()
        summary.append({"replicate": i, "error": str(e)})

with open(f"{out_root}/replicates_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll replicates done. Summary written to", f"{out_root}/replicates_summary.json")
