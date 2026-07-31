# paper_analysis

Simulation, inference, and figure-generation code for the TwINFER manuscript
(bioRxiv 2026.02.22.707230). Depends on `twinfer` (`package/`); does not
reimplement anything from it.

Figure-to-folder mapping: `figures_manifest.yaml`.

## Folders

| Folder | Contents |
|---|---|
| `heterogeneity_vs_regulation/` | Two-gene simulation and the heterogeneity/regulation flowchart |
| `causal_direction_inference/` | Cross-correlation direction inference, 5-gene cascade |
| `triplet_motif_discrimination/` | Fan-out, feed-forward-loop, regulated-mutual motifs |
| `larry_hematopoiesis_validation/` | LARRY dataset analysis |
| `parameter_space_scan/` | Latin-hypercube parameter sweep, used by two figures above |
| `grnboost2_comparison/` | GRNBoost2 baseline via Arboreto, used by two figures above |
| `extended_figures/` | Standalone extended-data and supplement analyses |
| `parameter_estimation/` | Literature parameter compilation (Table E1/E2 sourcing) |
| `benchmark/` | BEELINE/BoolODE evaluation — status unconfirmed, see manifest |
| `work_in_progress/` | Code not tied to any published figure |

Each folder contains its own `README.md` with a script-to-panel table.

## Layout convention

Each figure folder follows the same three stages where applicable:

```
<folder>/
├── README.md
├── simulate.py        # or preprocess.py for real-data folders
├── analysis.ipynb
└── plot.ipynb
```

## Output structure

No script hardcodes an output path. Output paths come from
`twinfer.utils.paths.stage_dir(folder_name, stage)`, which resolves to:

```
<data_root>/paper_analysis/<folder>/<stage>/<run_tag>/
```

- `data_root` — `TWINFER_DATA_ROOT` if set, otherwise `<repo_root>/../../analysis_data`.
- `run_tag` — a timestamp (`YYYYMMDD_HHMMSS`) generated per run, unless
  `TWINFER_RUN_TAG` is set (pins every stage of one pipeline invocation to the
  same tag) or an explicit `run_tag` is passed to reproduce/inspect a specific
  past run.
- `stage_dir()` repoints a `latest` symlink at the newest run for that
  folder/stage, so `stage_dir(folder, stage, run_tag="latest")` resolves the
  most recent run without knowing its timestamp.

Example, for `heterogeneity_vs_regulation/`:

```
<data_root>/paper_analysis/heterogeneity_vs_regulation/
├── simulate/
│   ├── 20260731_101530/
│   │   ├── A_B/
│   │   ├── A_to_B/
│   │   └── logs/
│   └── latest -> 20260731_101530/
├── analysis/
│   └── ...
└── plot/
    └── ...
```

A downstream stage reads its input the same way — e.g. `analysis.ipynb` resolves
its simulation input via `stage_dir("heterogeneity_vs_regulation", "simulate", run_tag="latest")`
rather than a hardcoded path.

## Running

```
./run_simulation.sh <folder> [simulate|analyze|plot|all]
./reproduce_all.sh --validate
```

`--validate` compares output against the reference values reported in the
manuscript (AUROC, F1, precision/recall) rather than only checking that scripts
ran without error.

## Status

Code has been moved into `heterogeneity_vs_regulation/`, `causal_direction_inference/`,
`triplet_motif_discrimination/`, `larry_hematopoiesis_validation/`,
`parameter_space_scan/`, and `grnboost2_comparison/`. Hardcoded paths and stale
`TwINFER_function_scripts` imports within those files are not yet fixed —
`stage_dir()` above is the target state, not yet universally in place. See
`REORG_CHECKLIST.md` at the repository root for current per-file status.
`extended_figures/`, `parameter_estimation/`, `benchmark/`, and `work_in_progress/`
have not yet had code moved in.
