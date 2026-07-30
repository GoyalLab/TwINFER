This is the repository for the TwINFER project.

> This repo is being reorganized into `package/` (installable library), `paper_analysis/` (per-figure simulate/analyze/plot), and `tutorials/`. See `REORG_CHECKLIST.md` at the repo root for current progress. The sections below (Installation, Paths & data location) already describe the target setup; the rest of this file still describes the pre-reorg layout until that migration finishes.

## Installation

From the repo root, inside the `twinfer-code` conda environment:

```bash
conda env create -f package/environment.yml   # first time only
conda activate twinfer-code
pip install -e package/
```

This registers `twinfer` as an editable install, so `import twinfer` works from any script or notebook in this environment. See `package/README.md` for the full module map and when to use `grnboost_env.yml` instead.

## Paths & data location

All paths in this repo are resolved through `twinfer.utils.paths` (`package/twinfer/utils/paths.py`) — nothing should hardcode an absolute path. Configure it via environment variables, e.g. in your shell profile:

| Variable | Purpose | Default |
|---|---|---|
| `TWINFER_DATA_ROOT` | Root for simulate/analyze/plot output | `<repo>/../../analysis_data` |
| `TWINFER_RUN_TAG` | Pin a specific run's simulate/analyze/plot stages to the same output folder | auto-generated timestamp per run |
| `TWINFER_BEELINE_PATH` | Location of the local BEELINE repo (benchmarking) | `<repo>/../Beeline` |
| `TWINFER_BOOLODE_PATH` | Location of the local BoolODE repo (benchmarking) | `<repo>/../BoolODE` |

Only set these if your setup differs from the defaults above.

The notebook TwINFER_simulation_and_analysis contains code to simulate a cell population with any gene underlying a regulatory network, along with parameters for each gene and regulatory interaction. It also has the code to infer the GRN from simulated data using the TwINFER framework.

Some example input data for simulation can be found in the simulation_example_input_data folder.
Example simulation output data can be found in the simulation_example_output_data folder.

The code and data used to estimate parameters from the literature are in the parameter_estimation folder.

The scripts_analyse_figure_data contains scripts to analyse simulations and generate data for the figures.

The scripts_to_plot_figure contain the code to generate the plots shown in the figures.

The specific details of code need to reproduce the exact plots in the figures of the paper, starting from simulations, analysis and making the final plot is provided on this [Google sheet](https://docs.google.com/spreadsheets/d/1dc1jYql7xb4ZE71f3lR6cSMpQq1c9PkpWk9iuyNDUtk/edit?usp=sharing). All the data can be found in this [folder](https://drive.google.com/drive/folders/1apg1QFkGD_QGuxIaUuTUs-l7r6ofpGsA?usp=sharing) and references in the sheet are with respect to this. Paths in the individual files may need to be set as necessary.
