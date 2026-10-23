"""Simulate the single-state / two-state scenarios (A_B, A_to_B and their low / high k_on versions) of the heterogeneity_vs_regulation figure.

Figure  : heterogeneity_vs_regulation.
Purpose : picks one of the six configurations defined in `simulation_configs.py` by index and simulates a range of replicates of it
          with the twinfer Gillespie simulator. It is called once per task by `run_generate_replicates_0_99.sbatch` (replicates 0-99)
          and `run_generate_replicates_100_999.sbatch` (replicates 100-999), but works equally well from a shell for a few replicates.
          The simulation settings (6,000 cells, 1,000 h burn-in, 48 h of twins, rows of median_parameter.csv, ...) come from
          `simulation_configs.py`; only the output folder, the log file and the file-name suffix `_rep_<n>` (so replicates never
          overwrite each other) are set here.

Configuration index (CONFIG_INDEX)
  0 A_B   1 A_B_low_k_on   2 A_B_high_k_on   3 A_to_B   4 A_to_B_low_k_on   5 A_to_B_high_k_on

Environment variables
  CONFIG_INDEX  configuration to run, 0-5 (default: SLURM_ARRAY_TASK_ID, or 0)
  REP_START     first replicate number (default 0)
  REP_COUNT     number of consecutive replicates (default 10)
  NCORES        numba threads for the simulator (default 60)
  OUT_DIR       output root (default final_figure_data/heterogeneity_vs_regulation/simulation_data/single_and_two_state_simulations).
                That default is a symlink to the original set, so set OUT_DIR when generating new data.
  INPUT_DIR     folder with the connectivity matrices and median_parameter.csv (default .../heterogeneity_vs_regulation/input_folder)
  TWINFER_PROJECT_ROOT  set by code/TwINFER/env.sh (default /gpfs/projects/b1255/hzhang/TwINFER_KA)

Output
  <OUT_DIR>/<config type>/   (must exist and be writable; the script stops with a message otherwise)
    simulation_before_division_df_rows_*_rep_<n>_*.csv   time course of the 6,000 cells from t = 0 until division (1,000 h burn-in);
                                                         the last time point is the state at division
    df_rows_*_rep_<n>_*.csv                              twin cells after division (hourly, 48 h)
    log.jsonl, manifest_rep<start>_<end>.json            what was run and how long it took

Example (3 replicates of A_to_B, written to a new folder)
  mkdir -p /path/to/new_set/A_to_B
  CONFIG_INDEX=3 REP_START=0 REP_COUNT=3 NCORES=20 OUT_DIR=/path/to/new_set python simulate_heterogeneity_and_regulation.py

Requires: the `twinfer` package and numba.
          REMINDER: twinfer is installed with `pip install -e package/` (repository root) for now; it will eventually be released as a pip package (`pip install twinfer`).
Cost    : in the original run one replicate took 41-456 s (median 190 s) on 20 cores and about 1.3 GB of memory.
"""
import json
import os
import time

from numba import set_num_threads, get_num_threads

# from TwINFER_function_scripts.gillespie_script_variations import process_param_set   # original import
from twinfer.simulation.gillespie_simulations import process_param_set

from simulation_configs import get_base_configs      # in the same folder as this script

# ---------------------------------------------------------------- locations
ROOT = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")                # set by env.sh
FIGURE_DATA = os.path.join(ROOT, "final_figure_data", "heterogeneity_vs_regulation")
OUT = os.environ.get("OUT_DIR") or os.path.join(FIGURE_DATA, "simulation_data", "single_and_two_state_simulations")
INPUT = os.environ.get("INPUT_DIR") or os.path.join(FIGURE_DATA, "input_folder")
# Previously (yscher's layout) the six configurations were defined in figure_2_simulations.py and were cut out of that file's text; they now live in simulation_configs.py.


def log(message):
    print(message, flush=True)


def main():
    index = int(os.environ.get("CONFIG_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    start = int(os.environ.get("REP_START", "0"))
    count = int(os.environ.get("REP_COUNT", "10"))
    cores = int(os.environ.get("NCORES", "60"))

    configs = get_base_configs(INPUT, OUT, cores)
    if not (0 <= index < len(configs)):
        raise SystemExit(f"CONFIG_INDEX {index} is out of range for {len(configs)} configurations (0-{len(configs) - 1})")
    base = configs[index]
    config_type = base["type"]

    # Check the destination before the (long) simulation starts, so a missing or read-only folder fails immediately.
    folder = os.path.join(OUT, config_type)
    if not os.path.isdir(folder):
        raise SystemExit(f"destination does not exist: {folder}\n"
                         f"create it, and give the account running the job write access to it, before submitting")
    if not os.access(folder, os.W_OK | os.X_OK):
        raise SystemExit(f"destination is not writable by {os.environ.get('USER', 'this account')}: {folder}")

    set_num_threads(cores)
    log(f"configuration {index}: {config_type}")
    for key in ("n_cells", "simulation_time_before_division", "twin_simulation_time_after_division",
                "twin_measurement_resolution", "rows_to_use", "multiple_interaction_type"):
        log(f"   {key} = {base.get(key)}")
    log(f"   numba threads {get_num_threads()};  replicates {start}..{start + count - 1} -> {folder}")

    rows = base["rows_to_use"][0]
    label = "rows_" + "_".join(map(str, rows))
    made = []
    for n in range(start, start + count):
        config = dict(base, type=f"{config_type}_rep_{n}", output_folder=folder, log_file=os.path.join(folder, "log.jsonl"))
        t0 = time.time()
        path = process_param_set(rows, label, config)
        made.append({"replicate": n, "file": os.path.basename(path), "minutes": round((time.time() - t0) / 60, 2)})
        log(f"   rep {n:2}  {(time.time() - t0) / 60:6.1f} min  {os.path.basename(path)}")
        # the manifest is rewritten after every replicate, so it is complete even if the job is stopped
        with open(os.path.join(folder, f"manifest_rep{start}_{start + count - 1}.json"), "w") as handle:
            json.dump({"config_index": index, "type": config_type,
                       "base_config": {k: str(v) for k, v in base.items()},
                       "configs": "simulation_configs.py", "replicates": made}, handle, indent=1)
    log(f"done: {len(made)} replicates in {folder}")


if __name__ == "__main__":
    main()
