# interaction_parameter_choice  (Extended Data figure E6, "Choice of the interaction model parameters")

Panel mapping (user, 2026-10-01): **E6c = hill_constant_effect**, **E6a and E6b = k_add_effect**.
| Panel | Source | Files here |
|---|---|---|
| E6c (Hill constant) | `hill_constant_effect/` | `simulating_multiple_hill_constant.py`, `run_simulating_multiple_hill_constant.sh` |
| E6a, E6b (k_add) | `k_add_effect/` | `parameters_for_multiple_k_add.ipynb` (sampled parameter ranges), `multiple_k_add_A_rep_B.sh`, `multiple_k_add_A_to_B.sh` |
Analysis/plot of all three panels: `../analysis.ipynb`, `../plot.ipynb`. Which of E6a / E6b each k_add script produces is not recorded here.

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| multiple_k_add_A_rep_B.sh | Run Python script with matching CLI arguments |
| multiple_k_add_A_to_B.sh | Path to parameter file generated using parameters_for_multiple_k_add.ipynb |
| parameters_for_multiple_k_add.ipynb | Creating parameters to sample ranges for testing the impact of $k_{on}$ and $k_{add}$ |
| run_simulating_multiple_hill_constant.sh |  |
| simulating_multiple_hill_constant.py | This is the script to simulate different values of Hill constant as a function of mean estimated protein level. The default is 1. |
