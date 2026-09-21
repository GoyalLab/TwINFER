# %%
"""
Vanilla (default-parameter, non-multistate) rerun of the Pluripotent real-network
sim at 6000 before-division steps, instead of the original 1000.

Why: the settling-time analysis on the existing 1000-step run
(real_networks_gene_expression_trajectories.pdf / settling_time_v2.py) found
Pluripotent's population-mean protein trajectories had NOT settled by t=999
(the last time point) for at least one gene -- the run was too short to reach
steady state. This reruns the same network/parameters, just longer, to check
whether 6000 steps is enough.

Same network/parameters as real_network_sim.py (single default parameter row
per gene, no Hill-exponent/k_add tuning -- this is NOT the multistate
variant), just:
  - topology  -> Pluripotent.txt (36 genes)
  - simulation_time_before_division -> 6000 (was 1000)
  - output written to a NEW run_tag ("6000steps") so the original 1000-step
    run is left untouched.
  - 1 replicate only (rep_0) -- that's the only replicate the trajectory /
    settling-time analysis actually uses.

Chunked run + incremental write (does NOT use process_param_set, which holds
the whole before-division trajectory in memory and only writes once at the
end): the 6000-step before-division sim is run and written in 1000-step
blocks, each block's final per-cell state seeding the next block's initial
condition. This caps peak memory at ~1 block's samples array (~6000 cells x
1000 steps x 144 species-components x 8B =~ 6.9GB) instead of the full run's
~41.5GB, and means the CSV has real data on disk after every block instead of
only at the very end of a ~24h+ job.

Output: <data_root>/paper_analysis/Pluripotent/simulate/6000steps/
  simulation_before_division_df_<prefix>.csv  (appended every CHUNK_SIZE steps)
  df_<prefix>.csv                             (post-division twin data, written once at the end)
"""
import argparse
import gc
import json
import os
import uuid
from datetime import datetime

import numpy as np

from numba import set_num_threads, get_num_threads

from twinfer.simulation.gillespie_simulations import (
    read_input_matrix,
    generate_reaction_network_from_matrix,
    generate_initial_state_from_genes,
    assign_parameters_to_genes,
    resolve_all_k_add,
    add_interaction_terms,
    setup_gillespie_params_from_reactions,
    get_promoter_indices,
    validate_simulation_inputs,
    run_simulation,
    convert_samples_to_df,
    is_steady_state,
    gillespie_simulation_all_cells,
    validate_no_stuck_cells,
)
from twinfer.utils.paths import get_repo_root, stage_dir

N_GENES = 36
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
CHUNK_SIZE = 1000
N_CHUNKS = 6  # 6 * 1000 = 6000 before-division steps

path_to_output = stage_dir("Pluripotent", "simulate", run_tag="6000steps", make_latest=False)
print(path_to_output)


def run_chunked(cfg, rows, label):
    n_cells = cfg["n_cells"]
    n_genes, connectivity_matrix = read_input_matrix(cfg["path_to_connectivity_matrix"])
    assert len(rows) >= n_genes, "The number of parameter rows entered is less than the number of genes"
    reactions_df, gene_list = generate_reaction_network_from_matrix(
        connectivity_matrix, combinatorial_interaction_type=cfg["multiple_interaction_type"]
    )
    init_states = generate_initial_state_from_genes(gene_list)
    param_dict = assign_parameters_to_genes(cfg["param_csv"], gene_list, rows)

    n_matrix = np.zeros((n_genes, n_genes))
    for j in range(n_genes):
        regulators = np.where(connectivity_matrix[:, j] != 0)[0]
        for i in regulators:
            edge = f"{gene_list[i]}_to_{gene_list[j]}"
            if n_matrix[i, j] == 0:
                n_matrix[i, j] = param_dict.get(f"{{n_{edge}}}", 2.0)

    param_dict = resolve_all_k_add(
        param_dict=param_dict, connectivity_matrix=connectivity_matrix, gene_list=gene_list,
        k_add_list=None, k_add_matrix=None, use_csv_k_add=True,
    )
    steady_state, full_param_dict = add_interaction_terms(
        param_dict=param_dict, connectivity_matrix=connectivity_matrix, gene_list=gene_list,
        n_matrix=n_matrix, scale_K=None, use_given_K=False, K_to_use=None,
    )
    pop0, update_matrix, update_prop, species_index = setup_gillespie_params_from_reactions(
        init_states, reactions_df, full_param_dict
    )
    promoter_indices = get_promoter_indices(species_index, gene_list)
    validate_simulation_inputs(connectivity_matrix, full_param_dict, pop0,
                                update_prop, update_matrix, reactions_df, species_index)

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    prefix = f"{label}_{timestamp}_ncells_{n_cells}_{cfg['type']}_{run_id}"
    before_path = f"{cfg['output_folder']}/simulation_before_division_df_{prefix}.csv"

    time_points = np.arange(CHUNK_SIZE)
    pop0_mat = None  # chunk 0: tiled default pop0
    final_states = None
    for chunk in range(N_CHUNKS):
        t0 = datetime.now()
        chunk_samples = run_simulation(
            update_prop, update_matrix, pop0, time_points, n_cells,
            promoter_indices=promoter_indices, pop0_mat=pop0_mat,
        )
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"[chunk {chunk}] steps {chunk*CHUNK_SIZE}-{(chunk+1)*CHUNK_SIZE-1} "
              f"done in {elapsed:.1f}s", flush=True)

        if chunk == N_CHUNKS - 1:
            if not is_steady_state(samples=chunk_samples, time_points=time_points,
                                    param_dict=full_param_dict,
                                    interaction_matrix=connectivity_matrix, gene_list=gene_list):
                print(f"WARNING: base simulation (basal) for {label} may not be steady "
                      f"even in the final chunk. Verify / extend further.", flush=True)
                log_folder = os.path.dirname(cfg["log_file"])
                os.makedirs(log_folder, exist_ok=True)
                with open(os.path.join(log_folder, "error_log.jsonl"), "a") as fh:
                    fh.write(json.dumps({
                        "id": run_id, "rows": rows,
                        "timestamp": datetime.now().strftime("%d%m%Y_%H%M%S"),
                        "issue": "Base simulation not steady (final chunk)", "label": label,
                    }) + "\n")

        chunk_df = convert_samples_to_df(chunk_samples, species_index)
        chunk_df["time_step"] = chunk_df["time_step"] + chunk * CHUNK_SIZE
        chunk_df.to_csv(before_path, mode="a", header=(chunk == 0), index=False)

        final_states = chunk_samples[:, -1, :]
        pop0_mat = final_states.T.copy()  # (n_species, n_cells) seed for next chunk
        del chunk_samples, chunk_df
        gc.collect()
        print(f"[chunk {chunk}] appended to {before_path}", flush=True)

    # ---- post-division twins, exactly as process_param_set --------------------
    rep_time = np.arange(0, cfg["twin_simulation_time_after_division"] + cfg["twin_measurement_resolution"],
                          cfg["twin_measurement_resolution"])
    pop0_rep = np.concatenate([final_states.T, final_states.T], axis=1)
    verbose_flags = np.zeros(2 * n_cells, dtype=np.int64)
    rep_samples = gillespie_simulation_all_cells(update_prop, update_matrix, pop0_rep, rep_time, verbose_flags)
    validate_no_stuck_cells(verbose_flags, label=label)

    df_rep = convert_samples_to_df(rep_samples, species_index)
    replicate_ids = np.repeat([1, 2], n_cells)
    clone_ids = np.tile(np.arange(n_cells), 2)
    df_rep["replicate"] = replicate_ids[df_rep["cell_id"]]
    df_rep["clone_id"] = clone_ids[df_rep["cell_id"]]
    df_rep["cell_id"] = df_rep.index // len(rep_time)
    rep_path = f"{cfg['output_folder']}/df_{prefix}.csv"
    df_rep.to_csv(rep_path, index=False)
    print(f"[twins] wrote {rep_path}", flush=True)

    return before_path, rep_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_index", type=int, default=0)
    args, _ = parser.parse_known_args()
    config_index = args.config_index
    print(f"Running base_config #{config_index}")

    base_configs = [
        {
            "n_cells": 6000,
            "twin_simulation_time_after_division": 48,
            "twin_measurement_resolution": 1,
            "path_to_connectivity_matrix": f"{get_repo_root().parent.parent}/input_data/real_world_networks/Pluripotent.txt",
            "param_csv": f"{get_repo_root().parent.parent}/input_data/network_sweep/parameters.csv",
            "rows_to_use": [[0] * N_GENES],
            "output_folder": str(path_to_output),
            "log_file": f"{path_to_output}/logs/Pluripotent.log",
            "type": "Pluripotent",
            "multiple_interaction_type": "additive",
            "number_of_parallel_parameters": 1,
            "number_of_cores_per_parameter": NUM_CORES,
        }
    ]

    if not (0 <= config_index < len(base_configs)):
        raise ValueError(f"config_index={config_index} is out of range for {len(base_configs)} configs.")

    base_config = base_configs[config_index]
    set_num_threads(base_config["number_of_cores_per_parameter"])
    print(f"Using {get_num_threads()} threads for config: {base_config['type']}")

    os.makedirs(base_config["output_folder"], exist_ok=True)
    os.makedirs(f"{base_config['output_folder']}/logs", exist_ok=True)

    rows_to_use = base_config["rows_to_use"]
    label = "rows_" + "_".join(map(str, rows_to_use[0]))

    cfg = dict(base_config)
    cfg["type"] = f"{base_config['type']}_rep_0"
    print(f"Running replicate 0 with type={cfg['type']} in {N_CHUNKS} chunks of {CHUNK_SIZE} steps")
    before_path, rep_path = run_chunked(cfg, rows_to_use[0], label)

    print(f"DONE -> {before_path}")
    print(f"DONE -> {rep_path}")
