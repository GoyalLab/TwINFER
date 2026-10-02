# TwINFER code (reorganised 2026-10)

The reorganised TwINFER code (cleanup of 2026-09/10; the previous layout is in the git history of `reorg/paper-analysis`). Data is NOT in here: it lives in the project tree next to the repo (`TWINFER_PROJECT_ROOT`, data folder `analysis_data/`, `TWINFER_DATA_ROOT`). Benchmark result tables that used to be tracked in `work_in_progress/benchmark/` are now in `analysis_data/benchmarks/network_benchmarks/score_results/`.

## Quick start
```bash
source env.sh          # sets TWINFER_PROJECT_ROOT / DATA_ROOT / REPO_ROOT / BEELINE_PATH / BOOLODE_PATH, PYTHONPATH, TWINFER_PYTHON
# the smoke tests and the static checkers (tests/smoke, dev_tools) are kept with the author's working copy, not in this repo
```
Environment: `twinfer-code` for everything (call its python directly). Exception: Beeline/BoolODE runs use the `BEELINE` env (`BEELINE_PYTHON`), see THIRD_PARTY.md. `sbatch` exports the environment, so `source env.sh` before submitting. Nothing here has been submitted by the cleanup.

## Layout
| Folder | Contents |
|---|---|
| `package/twinfer/` | the Python package (simulation, inference, plotting, utils). Not final. Editable installs in the env still point at the original tree; clean_code is used through PYTHONPATH. |
| `benchmarks/network_benchmarks/` | synthetic/real-network benchmark pipeline by stage: `generate_networks`, `simulate`, `infer`, `to_beeline`, `score` (+`score/formula_search`), `analysis_plots` |
| `benchmarks/beeline/`, `benchmarks/boolode/` | Beeline run scripts/configs/generators and BoolODE twin simulations; vendored fragments + local patches of both |
| `paper_analysis/` | figure-by-figure analyses (heterogeneity, causal direction, triplet motifs, LARRY, fate-map, real networks, parameter scan, ...) |
| `simulations/` | drift, cyclic 6-node, pulsed regulation, network dynamics checks, timing tests |
| `notebooks/` | real-data analysis notebooks and the tutorial (outputs stripped) |
| `real_data_scripts/`, `simulation_example_input_data/` | small data-side scripts; example inputs read by the examples |
| `_archive/` | superseded/legacy copies kept for the record |

## Conventions
- Replaced code is commented out with a dated note (`# [2026-09-30 ...]`); nothing was deleted.
- Files rescued from old Claude scratchpads/`/tmp` carry `# [UNREVIEWED: rescued ...]`; their placement is a best guess.
- Paths come from `twinfer.utils.paths` via `TWINFER_PROJECT_ROOT = get_data_root().parent`; scripts that used to sit in data folders reference those folders explicitly.

Working notes of the cleanup (handoffs, review log, open-items lists, provenance tables) are kept outside this repo.
