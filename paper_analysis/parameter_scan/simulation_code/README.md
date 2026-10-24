# simulation_code: parameter_scan

Simulations of the parameter scan: 25,000 parameter sets for each of four two-gene networks. Inputs are in `final_figure_data/parameter_scan/input_folder/` (parameter sheets, connectivity matrices) and outputs in `final_figure_data/parameter_scan/simulation_data/`, under the project folder (`TWINFER_PROJECT_ROOT`).

| File | Role |
|---|---|
| `parameter_generation_for_simulation.ipynb` | Makes the two parameter sheets (activation and repression): 25,000 Latin-hypercube sets (seed 42) × 3 genes. Six parameters per gene (pi_on, k_on, mRNA and protein half-life, k_prod_mRNA, k_prod_protein; log-uniform within the bounds given in the notebook) and a Hill coefficient and k_add for the links 1→2, 2→1, 1→3, 2→3 |
| `run_gillespie_A_B.sh` | SLURM array, network A_B (no link), activation sheet |
| `run_gillespie_A_to_B.sh` | network A_to_B (gene 1 → gene 2), activation sheet |
| `run_gillespie_A_and_B.sh` | network A_and_B (gene 1 ↔ gene 2), repression sheet, 2,000 h burn-in |
| `run_gillespie_A_rep_B.sh` | network A_rep_B (gene 1 ⊣ gene 2), repression sheet |
| `extract_minimal_scan_data.py` | Reduces each simulation to what the analysis reads (time steps 1 and 20, 7 columns, gzip): `python extract_minimal_scan_data.py <input_dir> <output_dir>` (`--check` inspects one file) |

Settings: each launcher runs `python -m twinfer.simulation.gillespie_simulations` with 6,000 cells, a 1,500 h burn-in (2,000 h for A_and_B), and twins simulated for 48 h after division, sampled hourly. One task simulates 600 parameter sets (42 tasks for sets 0-24,999); one output file per set, named `df_row_<rows>_<timestamp>_ncells_6000_<network>_<id>.csv` (the rows of the parameter sheet; set id = first row / 3). The sheets have 3 genes per set; gene 3 has no outgoing links, so it does not affect genes 1 and 2, and only genes 1 and 2 are analysed.

Options (environment variables, see the header of each launcher): `OUT_ROOT` (output folder), `SAVE_BEFORE_DIVISION=1` (also save the burn-in), `SET_LIST_FILE` (simulate only the parameter sets listed in a file).
