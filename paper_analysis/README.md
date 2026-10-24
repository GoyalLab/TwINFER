# paper_analysis

The code behind the figures of the TwINFER paper (bioRxiv [10.1101/2026.02.22.707230](https://doi.org/10.1101/2026.02.22.707230)). Every folder uses the `twinfer` package (`package/`) for the statistics and the simulations; nothing is reimplemented here.

| Folder | Shows |
|---|---|
| [`heterogeneity_vs_regulation/`](heterogeneity_vs_regulation/) | Two-gene simulations (no regulation / regulation, one state / two states / drifting states): the three z-scores separate regulation from heterogeneity; effect of the number of twin pairs. |
| [`parameter_scan/`](parameter_scan/) | 25,000 random parameter sets for each of four two-gene networks (no link, activation, mutual regulation, repression), single- and two-state: ROC and precision-recall of the three z-scores. |
| [`real_data_analysis_and_melanoma/`](real_data_analysis_and_melanoma/) | Real lineage-tracing datasets: how each raw dataset was processed into the matrices used in the figure (analysis and plots to be added). |

## Layout of a figure folder

```
<figure>/
├── README.md
├── simulation_code/     scripts that made the simulations (+ SLURM launchers, README with the settings)
├── preprocessing_code/  real data only: raw data -> final matrices
├── analysis_code/       simulations or final matrices -> analysis tables
└── plot_code/           analysis / plot tables -> PDF figures
```

## Data

Simulations and analysis tables are not in the repository (they are tens of GB). The scripts read and write them under a project folder:

```
$TWINFER_PROJECT_ROOT/
├── final_figure_data/<figure>/
│   ├── input_folder/      connectivity matrices, parameter sheets and other inputs of the simulations
│   ├── simulation_data/   simulation outputs
│   ├── analysis_data/     analysis tables
│   └── plot_data/         the exact data of each plot (when it differs from an analysis table)
└── final_plots/<figure>/  the PDF figures
```

Set the project folder before running a script:

```bash
export TWINFER_PROJECT_ROOT=/path/to/project
```

Every script starts with a header that names its inputs, outputs and an example command. The figure READMEs give the run order.
