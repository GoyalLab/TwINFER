# `twinfer`

Core TwINFER library: Gillespie simulation of gene regulatory networks, twin-pair correlation inference, and network plotting. Everything else in this repository (`paper_analysis/`, `tutorials/`) imports from this package rather than reimplementing these functions.

Figure-specific code, benchmarking, and evaluation metrics are not part of this package; see `paper_analysis/` for those. `twinfer` does not import from `paper_analysis/` or `tutorials/` under any circumstance — the dependency runs one way only.

## Installation

From the repository root, inside the `twinfer-code` conda environment:

```bash
conda env create -f package/environment.yml   # first time only
conda activate twinfer-code
pip install -e package/
```

`pip install -e` performs an editable install: `package/twinfer/` is linked into the environment's `site-packages` rather than copied. `import twinfer` then works from any script or notebook running in this environment, from any directory, without modifying `sys.path`. Editing files under `twinfer/` takes effect immediately; rerun `pip install -e package/` only if `pyproject.toml`'s dependencies change.

To verify the installation:

```bash
python -c "import twinfer; print(twinfer.__file__)"
```

### Environments

- **`environment.yml`** (`twinfer-code`, Python 3.12): the environment for this package — simulation, inference, plotting. Use this unless running the GRNBoost2/arboreto baseline.
- **`grnboost_env.yml`** (`grnboost_env`, Python 3.10): required only for the GRNBoost2 comparison in `paper_analysis/benchmark/`, which depends on older numpy/pandas/scipy pins than `twinfer-code` uses.

## Module map

```
twinfer/
├── simulation/   Gillespie simulation engines
├── inference/    Twin-pair correlation inference
├── plotting/     Network visualization
└── utils/        Cross-cutting helpers (paths, JSON serialization)
```

### `twinfer.simulation`

Gillespie simulation of stochastic gene expression over a regulatory network. Several engine variants exist side by side by design; each models a distinct scenario rather than a superseded draft.

| Module | Models | Used by |
|---|---|---|
| `gillespie_simulations` | Default engine: standard promoter/expression dynamics used for most figures | `figure_2`, `figure_3`, `figure_4` simulations and most of the simulation-driver scripts |
| `gillespie_drift` | Multi-state drift dynamics | `drift_multiple_state/` analyses |
| `gillespie_pulse` | Pulsed regulatory input | `drift_multiple_state/` analyses |
| `shared_functions` | Helpers identical across every engine variant (matrix I/O, network construction, parameter assignment, steady-state checks). Imported by the engines above, not called directly | — |

### `twinfer.inference`

Given simulated or real single-cell data with twin/clone labels, infers a regulatory network from correlation structure across twin pairs.

- `infer` — main entry point (`infer_with_twinfer`-style pipeline: existence, then heterogeneity, then direction).
- `correlation_analysis_functions` — Spearman correlation, permutation/shuffle, and twin-pair-construction routines called by `infer`.

### `twinfer.plotting`

Network visualization.

### `twinfer.utils`

Cross-cutting helpers used by simulation, inference, and `paper_analysis/`: path resolution (`paths.py`, replacing hardcoded absolute paths) and JSON-safe serialization (`json_utils.py`).

## Usage

```python
import twinfer
from twinfer.simulation import gillespie_simulations
from twinfer.inference import infer

# Simulate, then infer a network from the result.
# See tutorials/01_simulate_and_infer_basics.ipynb for a complete worked example.
```

## Status

This package is being migrated from the previous flat `TwINFER_function_scripts/` layout. See `REORG_CHECKLIST.md` at the repository root for current progress. `simulation/` and `inference/` contain migrated code; `plotting/` and `utils/` have not yet been populated.