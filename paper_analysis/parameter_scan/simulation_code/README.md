# simulation_code — parameter_scan

Parameter-scan simulations used by `heterogeneity_vs_regulation` and `causal_direction_inference`.
Data: `final_figure_data/parameter_scan/` (`input_folder/` = parameter sheets and connectivity matrices, `simulation_data/` = simulations).

| File | Role |
|---|---|
| `parameter_generation_for_simulation.ipynb` | Makes the two parameter sheets: 25,000 Latin-hypercube sets (seed 42) x 3 genes, 6 gene parameters per gene (pi_on, k_on, mRNA and protein half-life, k_prod_mRNA, k_prod_protein; log-uniform in the stated bounds) and a Hill n and k_add for the links 1->2, 2->1, 1->3, 2->3 |
| `run_gillespie_A_B.sh` | SLURM array, topology A_B (no regulation), activation sheet |
| `run_gillespie_A_to_B.sh` | topology A_to_B (gene_1 -> gene_2), activation sheet |
| `run_gillespie_A_and_B.sh` | topology A_and_B (gene_1 <-> gene_2), repression sheet, 2000 h burn-in |
| `run_gillespie_A_rep_B.sh` | topology A_rep_B (gene_1 -| gene_2), repression sheet |
| `extract_minimal_scan_data.py` | Reduces a full simulation file to the minimal format (time steps 1 and 20, 7 columns, gzip) that the analysis reads; this is how `parameter_scan_minimal` was made. Usage: `python extract_minimal_scan_data.py <input_dir> <output_dir>` (`--check` inspects one file) |

Each launcher runs `python -m twinfer.simulation.gillespie_simulations` (6,000 cells, 1,500 h burn-in (2,000 h for A_and_B), twins 48 h after division, hourly).
One output file per parameter set; the file name lists the sheet rows used (`df_row_<rows>_<timestamp>_ncells_6000_<topology>_<id>.csv`; set id = first row / 3).
Parameter sets are numbered 0-24,999 and `--row_to_start` / `--row_to_end` are inclusive. The sheet has 3 genes per set; the first runs (24-26 Aug 2025) simulated
3 genes (3 rows in the file name), later runs 2 genes (2 rows). Gene 3 has no outgoing links, so genes 1 and 2 are not affected.
`parameter_scan_minimal` (yscher) is the reduced version of these simulations (time steps 1 and 20, 7 columns, gzip).

Options: `OUT_ROOT`, `SAVE_BEFORE_DIVISION=1`, `SET_LIST_FILE` (re-simulate chosen parameter sets) — see the header of each script.
Missing parameter sets in `parameter_scan_minimal` (A_to_B 14, A_and_B 53, A_rep_B 3): `final_figure_data/parameter_scan/analysis_data/missing_parameter_sets_<topology>.txt`.
The original logs of the 2025 runs are not available, so the reason for the missing sets is unknown.
