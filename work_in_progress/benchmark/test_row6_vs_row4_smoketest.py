"""
One-off diagnostic: does the CURRENT simulation code, run with the CURRENT
median_parameter.csv row 6 (k_add=2), reproduce the OLD figure_3_simulations/
A_rep_B data's gene_2 mRNA mean (~0.25)? Or does row 4 (k_add=0.8, matching
the new figure_3_1k data) do that instead?

Context: old data's filenames say "rows_6_6" but was generated in Aug 2025,
before rows 2-3 (k_on=0.12/1.66 variants) were apparently inserted into
median_parameter.csv, which may have shifted what "row 6" means. This script
settles it empirically with the current code rather than guessing from row
indices. 2 reps each at row 4 and row 6, n_cells=6000 (matching old/new data).

Output: prints gene_2_mRNA mean/std at t1 for both, alongside the reference
numbers already measured:
  OLD (Aug 2025, "rows_6_6"): gene_2 mean=0.251, std=0.580 (dataset-level)
  NEW (figure_3_1k, row 4):   gene_2 mean=0.361, std=0.680 (dataset-level)
"""
import os
import time
import pandas as pd
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set
from twinfer.utils.paths import get_repo_root

PATH_TO_CODE_REPO = str(get_repo_root())
PATH_TO_INPUT_DATA = f"{PATH_TO_CODE_REPO}/simulation_example_input_data"
OUTPUT_FOLDER = "/home/gzu5140/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/_row6_vs_row4_smoketest"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

N_CORES = 8
N_REPS = 2

set_num_threads(N_CORES)

for row in (4, 6):
    for rep_id in range(N_REPS):
        cfg = {
            "n_cells": 6000,
            "simulation_time_before_division": 1500,
            "twin_simulation_time_after_division": 48,
            "twin_measurement_resolution": 1,
            "path_to_connectivity_matrix": f"{PATH_TO_INPUT_DATA}/connectivity_matrix_A_rep_B.txt",
            "param_csv": f"{PATH_TO_INPUT_DATA}/median_parameter.csv",
            "rows_to_use": [[row] * 2],
            "output_folder": OUTPUT_FOLDER,
            "log_file": f"{OUTPUT_FOLDER}/row{row}.jsonl",
            "type": f"row{row}_rep_{rep_id}",
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1,
            "number_of_cores_per_parameter": N_CORES,
        }
        t0 = time.time()
        path = process_param_set([row, row], f"row{row}_{row}_rep{rep_id}", cfg)
        elapsed = time.time() - t0
        df = pd.read_csv(path)
        t1 = df[df["time_step"] == 1]
        print(f"row={row} rep={rep_id}: elapsed={elapsed:.0f}s  "
              f"gene_2_mRNA mean={t1['gene_2_mRNA'].mean():.4f} std={t1['gene_2_mRNA'].std():.4f}  "
              f"-> {path}", flush=True)

print("done")
