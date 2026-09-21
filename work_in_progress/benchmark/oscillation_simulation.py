# %%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numba
import tqdm
import scipy
import seaborn
import os
import sys
import argparse
from numba import set_num_threads, get_num_threads
import importlib
import sys
import os
#%%
#Set output path


#Path to code repo
import twinfer
from twinfer.utils.paths import get_repo_root
from twinfer.simulation.gillespie_oscillations import process_param_set
from twinfer.utils.paths import stage_dir


path_to_code_repo = get_repo_root()
#Path to output files
num_cores_available = 40
 #%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_index", type=int, default=0)
    args, _ = parser.parse_known_args()
    config_index = args.config_index
    print(f"🔧 Running base_config #{config_index}")

    # ==========================================================
    # Define base configurations (4 configs: indices 0..3, one per
    # phase/division scenario -- all four share the same connectivity
    # matrix, parameter CSV, and gene-level settings, isolating the effect
    # of the oscillator-phase / division-timing assumption alone:
    #
    #   0. in_phase        -- every cell shares Phi=0; twins copy Phi
    #                          exactly; synchronous division.
    #   1. out_of_phase     -- population Phi ~ Normal(0, sigma_phi_population),
    #                          wrapped mod 2*pi; twins copy Phi exactly;
    #                          synchronous division.
    #   2. twin_dephasing   -- same out-of-phase population as #1, but twins
    #                          each get independent Normal(0, sigma_phi_twin)
    #                          noise added to their inherited Phi at division.
    #   3. async_division   -- population Phi=0 (in-phase oscillator drive);
    #                          division itself happens at a genuinely
    #                          different simulated time per cell, drawn from
    #                          Normal(base_division_time, sigma_t_division).
    #
    # param_csv must include an 'A' column (per-gene oscillation amplitude)
    # for the oscillator term to have any effect -- see gillespie_pipeline_
    # with_oscillator.py's generate_reaction_network_from_matrix.
    # ==========================================================

    _connectivity_matrix = f"/home/gzu5140/TwINFER_KA/input_data/oscillation_test.txt"  # EDIT to your actual network
    _param_csv           = f"/home/gzu5140/TwINFER_KA/input_data/parameters_oscillations.csv"            # EDIT -- must include an 'A' column
    _n_genes              = 4  # EDIT to match your connectivity matrix
    _output_root          = f"/home/gzu5140/TwINFER_KA/simulation_data/synthetic_network/oscillator_scenarios_redo/"

    _shared = {
        'n_cells': 6000,                          # Number of cells before division (number of twin pairs)
        'simulation_time_before_division': 1000,  # Time to reach steady state before division [hours]
        'twin_simulation_time_after_division': 48,  # Time twins are simulated after division [hours]
        'twin_measurement_resolution': 1,          # Time between twin measurements [hours]
        "path_to_connectivity_matrix": _connectivity_matrix,
        "param_csv": _param_csv,
        "rows_to_use": [[0] * _n_genes],           # Row in parameters.csv per gene
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": num_cores_available,
        # oscillator defaults shared by all four scenarios unless overridden below
        "sigma_phi_population": 1.18,   # rad, from a +/-7.11h population division-time window
        "sigma_phi_twin": 0.2,          # rad, twin phase-noise magnitude
        "sigma_t_division": 7.11,         # hours, from the same +/-7.11h window
        "phi_center": 0.0,
        "phi_seed": 0,
        "division_time_seed": 0,
    }

    base_configs = [
        {
            **_shared,
            "phi_scenario": "in_phase",
            "dephase_twins": False,
            "async_division": False,
            "output_folder": f"{_output_root}/in_phase",
            "log_file": f"{_output_root}/in_phase/logs/in_phase.log",
            "type": "in_phase",
        },
        {
            **_shared,
            "phi_scenario": "out_of_phase",
            "dephase_twins": False,
            "async_division": False,
            "output_folder": f"{_output_root}/out_of_phase",
            "log_file": f"{_output_root}/out_of_phase/logs/out_of_phase.log",
            "type": "out_of_phase",
        },
        {
            **_shared,
            "phi_scenario": "out_of_phase",
            "dephase_twins": True,
            "async_division": False,
            "output_folder": f"{_output_root}/twin_dephasing",
            "log_file": f"{_output_root}/twin_dephasing/logs/twin_dephasing.log",
            "type": "twin_dephasing",
        },
        {
            **_shared,
            "phi_scenario": "in_phase",
            "dephase_twins": False,
            "async_division": True,
            "output_folder": f"{_output_root}/async_division",
            "log_file": f"{_output_root}/async_division/logs/async_division.log",
            "type": "async_division",
        },
    ]

    for config_index, base_config in enumerate(base_configs):
        print(f"\n{'='*60}")
        print(f"🔧 Running base_config #{config_index}: {base_config['type']}")
        print(f"{'='*60}")
 
        # Configure numba threads
        set_num_threads(base_config["number_of_cores_per_parameter"])
        print(f"🧠 Using {get_num_threads()} threads for config: {base_config['type']}")
 
        # Ensure output directory exists
        os.makedirs(base_config['output_folder'], exist_ok=True)
 
        # Prepare rows and labels
        rows_to_use = base_config['rows_to_use']   # e.g. [[7,7,7]]
        labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]
 
        original_type = base_config["type"]
        last_path = None
 
        # ==========================================================
        # Run 10 replicates of this config
        # ==========================================================
        for i in range(3):
            # shallow copy so we don't mutate base_config
            cfg = dict(base_config)
            cfg["type"] = f"{original_type}_rep_{i}"
            # vary the RNG seeds per replicate so replicates aren't identical draws
            cfg["phi_seed"] = base_config.get("phi_seed", 0) + i
            cfg["division_time_seed"] = base_config.get("division_time_seed", 0) + i
            print(f"▶️  Running replicate {i} with type={cfg['type']}")
            path_to_simulation_file = process_param_set(
                rows_to_use[0],
                labels[0],
                cfg
            )
            last_path = path_to_simulation_file
 
        print(f"✅ Config #{config_index} ({original_type}) last simulation file: {last_path}")
 
    print(f"\n🏁 All {len(base_configs)} configs complete.")