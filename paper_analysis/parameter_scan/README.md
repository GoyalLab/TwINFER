# parameter_scan

How well do the three z-scores work across parameter space? For each of four two-gene networks, 25,000 parameter sets (Latin hypercube over promoter switching, production and half-lives of mRNA and protein, and the regulation strength) were simulated with 6,000 twin pairs:

| topology | network |
|---|---|
| `A_B` (X Y) | no link |
| `A_to_B` (X → Y) | gene 1 activates gene 2 |
| `A_and_B` (X ↔ Y) | gene 1 and gene 2 regulate each other |
| `A_rep_B` (X ⊣ Y) | gene 1 represses gene 2 |

Each simulation is a **single-state** system. **Two-state** systems combine 3,000 twin pairs of one simulation with 3,000 twin pairs of another simulation of the same topology whose protein levels differ (Brunner-Munzel p < 0.01 for both genes). All systems are scored 20 hours after division; the z-scores are then compared between classes with ROC and precision-recall curves:

* `z_rho`: regulation, or several states, against no regulation in a single state;
* `z_rho_s`: two states against a single state (with a random-pair control in which the twin structure is broken);
* `z_rho_delta`: regulation against no regulation, also among two-state systems.

## Code

| Folder | Script | Does |
|---|---|---|
| `simulation_code/` | `parameter_generation_for_simulation.ipynb` | the two parameter sheets (25,000 sets × 3 genes; activation and repression) |
| | `run_gillespie_A_B.sh`, `run_gillespie_A_to_B.sh`, `run_gillespie_A_and_B.sh`, `run_gillespie_A_rep_B.sh` | SLURM arrays, one simulation per parameter set |
| | `extract_minimal_scan_data.py` | reduces each simulation to time steps 1 and 20 and the 7 columns the analysis reads |
| | `README.md` | all simulation settings |
| `analysis_code/` | `twin_covariance_decomposition_z_scores.py` | z-scores of every single-state and two-state system (`--break-twins`: random-pair control) |
| | `roc_of_z_scores.py` | ROC curves and AUROC → `plot_data/roc_curves.csv`, `roc_summary.csv` |
| | `auprc_of_z_scores.py` | precision-recall curves and AUPRC per topology → `plot_data/auprc_curves.csv`, `auprc_summary.csv` |
| `plot_code/` | `plot_roc_panels.py` | `roc_panels.pdf`, `roc_two_states_regulation.pdf` |
| | `plot_auprc.py` | `z_rho_auprc.pdf`, `z_rho_s_auprc.pdf`, `z_rho_delta_auprc.pdf` |

## Run

Data are read from and written to `$TWINFER_PROJECT_ROOT/final_figure_data/parameter_scan/` (`input_folder/`, `simulation_data/`, `analysis_data/`, `plot_data/`); plots go to `$TWINFER_PROJECT_ROOT/final_plots/parameter_scan/`. Every script accepts other paths as arguments (see its header).

```bash
export TWINFER_PROJECT_ROOT=/path/to/project
export TWINFER_PYTHON=$(which python)
source env.sh                                  # from the repository root; used by the SLURM launchers
P=paper_analysis/parameter_scan

# 1. simulations (SLURM, 42 tasks of 600 parameter sets per topology), then the minimal format
sbatch $P/simulation_code/run_gillespie_A_to_B.sh                  # and the other three topologies
python $P/simulation_code/extract_minimal_scan_data.py <simulation folder> <minimal folder>

# 2. z-scores (about 8 CPU hours for all 200,000 systems; --jobs sets the number of processes)
python $P/analysis_code/twin_covariance_decomposition_z_scores.py --jobs 32
python $P/analysis_code/twin_covariance_decomposition_z_scores.py --jobs 32 --break-twins

# 3. ROC and precision-recall data, then the plots
python $P/analysis_code/roc_of_z_scores.py
python $P/analysis_code/auprc_of_z_scores.py
python $P/plot_code/plot_roc_panels.py
python $P/plot_code/plot_auprc.py
```

AUPRC is reported together with its ratio to the base rate (the fraction of positives; 1.0 = chance).
