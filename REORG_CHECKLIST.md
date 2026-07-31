# TwINFER Reorganization Checklist

Goal: reorganize into three top-level folders — **`package/`**, **`paper_analysis/`**, **`tutorials/`** — plus a separated **`data/`** folder, eliminate duplicate function implementations, fix pervasive stale/hardcoded paths, add human/AI-readable documentation, and give a one-command way to reproduce/validate each figure.

## Key decisions

- **Naming**: folders under `paper_analysis/` are named for what they scientifically demonstrate (the figure's own title), not for a figure/panel number. Numbers shift across manuscript revisions; titles do not. `paper_analysis/figures_manifest.yaml` is the only place a figure number is recorded — it maps `current_number` to `folder` per entry. No other file references a figure by bare number.
- **Gillespie engine**: light-touch consolidation only — extract byte-identical shared helpers into `simulation/shared.py`; keep the variant simulation loops (default, drift, pulse, fixed-gene) as separate, clearly-named/documented files since each exists for a real, distinct scientific reason. The original pre-"variations" engine (`gillespie_script.py`) is superseded and archived, not migrated.
- **Package scope is bare-minimum**: only `simulation`, `inference` (incl. network naming / edge ranking), and `plotting`. Evaluation/benchmarking metrics are explicitly excluded from the package — they live in `paper_analysis/` instead.
- **BEELINE/BoolODE status is unverified against the manuscript.** The prior plan for `paper_analysis/benchmark/` (network-sweep evaluation against BEELINE/BoolODE) does not appear anywhere in the manuscript text (bioRxiv 2026.02.22.707230). It is not wired into any figure's dependencies until confirmed. This is separate from `paper_analysis/grnboost2_comparison/`, which is verified — the manuscript's Methods describe a direct Arboreto `grnboost2` comparison with Gaussian-Mixture-Model thresholding, used by two figures.
- **Data**: large/raw data moves into a new top-level `data/` folder, separate from code.
- **Execution**: phased — `package/` first, reviewed, then `paper_analysis/`, reviewed, then `tutorials/`+`data/`.
- **Standing preference for this project**: never hard-delete code during edits — when consolidating/superseding a function or file, move the old version to an `_archive/` folder or comment it out in place with a note on why, rather than removing it outright. Use `git mv` wherever possible to preserve history.

## Recommended workflow

1. **Branch per phase, commit per checklist subsection.** `reorg/package` for Phase 1, `reorg/paper-analysis` for Phase 2 (branch from `reorg/package` once merged), `reorg/tutorials-data` for Phase 3. Small commits, not one giant diff.
2. **Verify at each phase checkpoint**, not only at the end.
3. **Strip notebook outputs before committing** (`jupyter nbconvert --clear-output ...` or `nbstripout`) — otherwise diffs are mostly re-executed-cell noise.
4. **Don't fix unrelated bugs inline** — note them in the "Found during reorg" log below instead, to keep reorg commits reviewable as pure moves.

---

## Import contract: `package/` vs `paper_analysis/`

**One-way dependency.** `paper_analysis/` and `tutorials/` import *from* `package/twinfer/`; nothing inside `package/twinfer/` ever imports from `paper_analysis/`, `tutorials/`, or anything figure-specific. If a function only makes sense for one figure/benchmark, it does not belong in `package/twinfer/`.

- Every file under `paper_analysis/`/`tutorials/` does `import twinfer` / `from twinfer.X import Y` — never `sys.path.insert(...)` + `from TwINFER_function_scripts import ...`.
- Verification commands (run once Phase 1+2 are done):
  ```
  grep -rn "paper_analysis" package/twinfer/                                        # should be empty
  grep -rln "sys.path.insert\|TwINFER_function_scripts" paper_analysis/ tutorials/  # should be empty
  ```
- `pip install -e package/` (1.6) registers `package/twinfer/` in the active conda env's site-packages pointing back at the source dir — no copying, no publishing required for `import twinfer` to work in this environment.

---

## Found during reorg

Issues discovered outside the checklist's own scope, logged here rather than fixed inline (per workflow rule 4) unless marked fixed.

- **Fixed** — `.gitignore` used `*/__pycache__/` (matches only one directory level deep) instead of `__pycache__/` (matches any depth), so `package/twinfer/**/__pycache__/` showed as untracked instead of ignored. A subsequent edit introduced leading whitespace on every pattern line, which `.gitignore` does not strip, reintroducing the same failure. Both fixed; verified with `git check-ignore -v` against a nested `__pycache__` path.
- **Fixed** — `package/twinfer/utils/json_utils.py` had a top-level `from twinfer import infer_with_twinfer` that raised `ImportError` on any import of the module (`twinfer/__init__.py` is empty; the function lives at `twinfer.inference.infer.infer_with_twinfer`, not on the top-level package). The import was unused elsewhere in the file. Removed.
- **Fixed** — `package/twinfer/plotting/network_plots.py`'s `make_reds_blues_colormap` built its color list with a single pure-white row via `LinearSegmentedColormap.from_list`, which resamples the input list to its own N-entry LUT; the single white row was lost in resampling, so data value 0 rendered as a red/blue blend, not white. Switched to `ListedColormap`, which uses the input array directly as the LUT with no resampling.
- **Open** — `make_reds_blues_colormap` exists twice, with genuinely different implementations, not one canonical version as Phase 1.3 below assumed: `twinfer/inference/correlation_functions.py` (scramble-based, no `low_clip`, used for correlation heatmaps) and `twinfer/plotting/network_plots.py` (constant `low_clip=0.5`, used for network diagrams). Confirm whether this is an intentional stylistic difference between the two plot types or an unintended duplication before consolidating.
- **Open** — `paper_analysis/` already contains two empty directories, `simulations_for_figures/` and `scripts_analysis_figure_data/`, mirroring the pre-reorg flat names rather than the theme-based folders in this checklist. Resolve before Phase 2 file moves begin — either repurpose or remove.
- **Open** — `plot_network` (`correlation_analysis_helpers.py`, used by `TwINFER_simulation_and_analysis.ipynb`) was not carried into `package/twinfer/`. Its likely replacement, `plot_grn` (`network_plots.py`), takes a different signature (adjacency matrix + node labels, not correlation matrix + gene list + edge list) and is not a drop-in substitute. Confirm disposition before Phase 2 closes out the tutorial notebook migration.

---

## Known bugs and a proposed statistics upgrade (surfaced 2026-07-29)

Two external documents (`TwINFER_thresholds_and_CI_v2_1.pdf` — a governing spec for thresholds/sample-size/confidence-intervals with reference Python implementations; `PROJECT_SUMMARY.pdf` — a re-analysis of LARRY + CellTag-multi run from a separate project directory, `/gpfs/projects/b1255/yscher/Transcriptomic Distance/`, that patches TwINFER's code via runtime substitution rather than editing this repo) surfaced concrete bugs in `correlation_analysis_functions.py` and `infer_with_twinfer.py`. That external harness is not something to merge in wholesale, but its findings — and additional ones found while checking it — are directly actionable against files in this repo's scope.

**Confirmed bugs** (all land in `package/twinfer/inference/correlation_functions.py` / `infer.py`, see 1.3):

1. **Twin-pair grouping silently drops large clones** — `correlation_analysis_functions.py` ~L717-724 groups the twin table by `clone_id` and keeps groups of exactly 2 *rows*, which equals "a clone of 2 cells" only when the table has one row per cell (true in the simulation, false for barcoded real data, where each pair writes 2 rows and every pair of a clone shares one `clone_id`). Every clone larger than 2 cells is silently discarded, no error. Measured on LARRY: 61% of day-2 twin pairs and 90% of day-4 pairs lost. Fix: group by `pair_id` when present, fall back to `clone_id` otherwise.
2. **Self-pair exclusion tests the wrong thing** — `correlation_analysis_functions.py` ~L348: `if abs(idx_1[k] - idx_2[k]) > 1:` tests row-index adjacency, not cell identity. Order-dependent; discards ~3/n legitimate random pairs and can still admit a cell paired with itself. Fix: `if idx_1[k] != idx_2[k]:`.
3. **`check_gene_gene_correlation_threshold` silently flags nothing when `use_scramble=False`** — `correlation_analysis_functions.py` L528-636. `is_significant` is initialized to `False` at L586 and only reassigned inside the `if use_scramble:` block (L599). Calling this function with `use_scramble=False` classifies every gene pair as `no_regulation`, with no error and no warning: the codebase has no working non-scramble code path, despite `use_scramble` being an exposed parameter.
4. **`infer_with_twinfer`'s public result dict swaps t1/t2 for the random-pair matrix** — `infer_with_twinfer.py` L581 and L597 (both success and except branches): `"random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t2,` — the key says `_t1`, the value is `_t2`. Anyone reading `result['random_pair_correlation_matrix_t1']` silently gets t2's data.
5. **The Stage-III "10% relative increase" regulation test is internally inconsistent and empirically broken** — `identify_reg_if_multiple_states`, `correlation_analysis_functions.py` L823-881. The relative-change formula treats `corr_t2 < 0` and `corr_t2 >= 0` inconsistently, and — confirmed by the external spec's own calibration run — because the twin difference-correlation starts near zero, any relative-change formula explodes: this test flags ~90% of a true no-regulation scenario as regulation. Fixed to `d = corr_t2 - corr_t1 > regulation_increase_threshold` (default `0.024`), the spec's point-estimate rule, same I/O as before. TODO: `0.024` is calibrated on a single simulation run, single gene pair, single `t2`; the spec's own table shows the "correct" threshold scaling roughly 8x across `t2 = 2h`–`36h`. Validate across regulation strengths, multi-state separations, and measurement times before relying on the default outside the calibration scenario.
6. **Docstring/implementation mismatch in the heterogeneity function** — `differentiate_single_state_reg_and_multiple_states`, `correlation_analysis_functions.py` L758-821. Docstring claims the threshold is on `abs(random / twin)`; the code computes `z_score = (t_corr - np.mean(r_corr)) / r_corr_std`, a z-score against the random-pair null distribution — an entirely different quantity, and the wrong-denominator issue the external spec's heterogeneity fix targets.
7. **Lower-confidence, likely dormant:** `calculate_twin_random_pair_correlations`, `correlation_analysis_functions.py` L738: `n_pairs = n_random or len(rep_0)` — Python's `or` treats an explicitly-passed `n_random=0` as falsy and silently substitutes `len(rep_0)` instead. Matters only if some caller passes `n_random=0` intentionally; not verified.

**Proposed statistics upgrade** (`TwINFER_thresholds_and_CI_v2_1.pdf`): replace the current scramble/p-value significance tests with fixed effect-size thresholds calibrated once on the simulations — existence `ρ* = 0.021`, direction `ρ̂†* = 0.046`, heterogeneity `g* = -0.126016` (two-sided on `z = g·√(N_twin−1)`, where `g = ρ̂∆(t1) − ⟨ρ∆⟩`), regulation `d* = 0.024` — plus proper confidence intervals from resampling whole independent units (mother cells / clones, never cells or enumerated pairs) instead of from the shuffle. A p-value threshold scales as `1/√(n−1)` and shrinks toward 0 as cell count grows, so every pair eventually becomes an edge; a fixed threshold does not have that failure mode. The spec's Appendix D contains reference implementations: `stage1`/existence, `stage2`/heterogeneity, `direction`, a separate Stage-III/regulation module (`stage3_statistic`, `stage3_combine`, `stage3_ci_closed_form`, `stage3_ci_bootstrap`, `stage3_verdict`), and a replacement for the old Figure-2h panel (`plot_stage3`). Adopting this fixes bugs 3, 5, and 6 above as a side effect, since it replaces exactly those three functions — it is a methodological change (changes which edges get flagged), not just a bug fix.

**Decision needed before Phase 1 finalizes `correlation_functions.py`:** (a) bugs 1, 2, 4, and 7 are unambiguous fixes with no methodology judgment call — fix them regardless. (b) For bug 3 (dead `use_scramble=False` branch) and bug 5/6 (broken Stage-III test, mismatched heterogeneity docstring): implement the proposed threshold/CI methodology into `package/twinfer/inference/` now, defer to a follow-up task, or leave as reference-only. Decision: ______

---

## Phase 0 — Setup

- [x] `git checkout -b reorg/package` from `master`
- [x] Add root `.gitignore`
- [x] Untrack committed `__pycache__` directories (`git rm -r --cached`, keeps files on disk)
- [x] Commit: "chore: add .gitignore, untrack build artifacts"
- [x] Decide fate of root scratch files `_check_helpers2.py`, `_check_ui2.py` — moved to plot utilities

## Phase 1 — `package/`

Convert `TwINFER_function_scripts/` into a bare-minimum installable package `package/twinfer/`: simulation, inference, and plotting only.

### 1.1 Scaffold
- [x] Create `package/` directory
- [x] `git mv environment.yml package/environment.yml`
- [x] `git mv grnboost_env.yml package/grnboost_env.yml`
- [x] Write `package/pyproject.toml`
- [x] Write `package/README.md`
- [x] `package/twinfer/__init__.py`, `simulation/__init__.py`, `inference/__init__.py`, `plotting/__init__.py`, `utils/__init__.py` present on disk
- [x] Create `package/_archive/` for superseded files

### 1.2 Simulation module
- [x] `git mv TwINFER_function_scripts/gillespie_script.py package/_archive/gillespie_script.py`
- [x] `TwINFER_function_scripts/gillespie_script_variations.py` moved and renamed to `package/twinfer/simulation/gillespie_simulations.py`; no stray copy remains at either location
- [ ] Fix downstream imports in the remaining files still referencing the old `TwINFER_function_scripts`/`sys.path.insert` pattern: `scripts_simulation_for_figures/{figure_1_network,figure_2_simulations,figure_3_simulations,figure_4_simulations,network_sweep,cycle,real_network,synthetic_network,synthetic_network_high_density}.py`, `k_add_effect/multiple_k_add_A_to_B.sh`, `hill_constant_effect/simulating_multiple_hill_constant.py`, `additional_analysis/saturation_effects/mutliple_k_add_A_to_B.py`, `binomial_partitioning/simulating_binomial_partition.py`. Replace with `from twinfer.simulation.gillespie_simulations import process_param_set`. Since most of these files move into `paper_analysis/` in Phase 2, do the rewrite there instead of twice.
  - [x] `TwINFER_simulation_and_analysis.ipynb` — fixed
- [x] `git mv drift_multiple_state/gillespie_script_drift.py package/twinfer/simulation/gillespie_drift.py`
- [x] `git mv drift_multiple_state/gillespie_script_pulse.py package/twinfer/simulation/gillespie_pulse.py`
- [x] `git mv additional_analysis/saturation_effects/fixed_z_effect/gillespie_fixed_gene.py package/twinfer/simulation/gillespie_fixed_gene.py` — not needed (random analysis), kept for reference
- [ ] Diff the 4 files function-by-function; confirm which are truly identical across all 4 (`hill_fn` confirmed byte-identical; check the rest): `read_input_matrix`, `generate_reaction_network_from_matrix`, `assign_parameters_to_genes`, `generate_initial_state_from_genes`, `assign_k_values_matrix`, `generate_k_from_max_expression`, `generate_K_from_steady_state_calc`, `add_interaction_terms`, `is_steady_state`, `convert_samples_to_df`, `get_promoter_indices`, `save_promoter_events`, `allocate_event_logs`, `validate_regulatory_configuration`, `divide_mother_cell_content`, `process_param_set`, `process_param_set_with_numba_config`, `check_if_file_exists`
- [ ] Create `package/twinfer/simulation/shared.py`; move each confirmed-identical function there
- [ ] Update all 4 variant files to `from .shared import ...`; remove redundant local copies
- [ ] For any function that looked identical but diverged on inspection — leave in place per-file, note the divergence in a comment
- [ ] Module-level docstring on each of the 4 files stating what scenario it simulates and which analyses call it:
  - `gillespie_simulations.py` → default engine, used by `heterogeneity_vs_regulation/`, `causal_direction_inference/`, `triplet_motif_discrimination/`, and most simulation scripts
  - `gillespie_drift.py` → used by `extended_figures/interaction_parameter_choice/` and related drift analyses
  - `gillespie_pulse.py` → see `Found during reorg` / `work_in_progress/pulsed_regulation` — no matching published figure identified yet
  - `gillespie_fixed_gene.py` → used by the saturation/fixed-z exploratory analysis (`work_in_progress/saturation_and_noise_robustness`)
- [ ] Write `package/twinfer/simulation/README.md` — one table: variant | scenario it models | analyses that use it
- [ ] Document the diverged `run_simulation` signature (`promoter_indices=None` present in `gillespie_simulations.py`/`gillespie_fixed_gene.py`, absent in `gillespie_drift.py`/`gillespie_pulse.py`) — confirm intentional vs. bug

### 1.3 Inference module
- [x] `git mv TwINFER_function_scripts/infer_with_twinfer.py package/twinfer/inference/infer.py`
- [x] Merged `correlation_analysis_functions.py` + `correlation_analysis_helpers.py` into `package/twinfer/inference/correlation_functions.py` (final filename differs from the originally planned `correlation.py`)
- [x] `git mv TwINFER_function_scripts/network_naming_utils.py package/twinfer/inference/network_naming.py`
- [x] `git mv TwINFER_function_scripts/ranked_edges_utils.py package/twinfer/inference/ranked_edges.py`
- [x] Bug fix 1 (twin-pair grouping by `pair_id`/`clone_id`) — applied per prior review; not independently re-verified this pass
- [x] Bug fix 2 (self-pair exclusion by identity, not row adjacency) — applied per prior review; not independently re-verified this pass
- [x] Bug fix 4 (`infer_with_twinfer` t1/t2 key swap) — applied per prior review; not independently re-verified this pass
- [x] Bug fix 7 (`n_pairs = len(rep_0) if n_random is None else n_random`) — applied per prior review; not independently re-verified this pass
- [x] Bug 3 (`use_scramble=False` dead branch) — gated on the Known Bugs decision above
- [x] Bug 5 (Stage-III relative-increase test) — replaced with `d = corr_t2 - corr_t1 > regulation_increase_threshold` (default `0.024`)
- [ ] TODO: validate `regulation_increase_threshold=0.024` across regulation strengths, multi-state separations, and measurement times before general use (see Known Bugs item 5)
- [x] Bug 6 (heterogeneity docstring mismatch) — open per the Known Bugs decision; full methodology adoption (`thresholds.py`, `regulation.py`, `confidence.py` per the spec's Appendix D) not yet started
- [ ] Resolve the `make_reds_blues_colormap` double-definition between `correlation_functions.py` and `plotting/network_plots.py` — see `Found during reorg`
- [ ] Resolve `plot_network`'s disposition — see `Found during reorg`

### 1.4 Plotting module
- [x] Compared `plot_grn.py` vs `grn_plot_spread_v2.py` — v2 confirmed fuller/newer
- [x] `git mv synthetic_network_analysis/grn_plot_spread_v2.py package/twinfer/plotting/network_plots.py`
- [x] `plot_grn.py` archived — v2 is the sole canonical version
- [x] `git mv fonts/ package/twinfer/plotting/assets/fonts/` (5 `.ttf` files)
- [x] Bug fix — `make_reds_blues_colormap`'s pure-white-at-zero color was lost to `LinearSegmentedColormap.from_list`'s resampling; switched to `ListedColormap`. See `Found during reorg`.

### 1.5 Utils module
- [x] `package/twinfer/utils/json_utils.py` — `make_json_safe`, `NumpyEncoder` present
- [x] Bug fix — top-level `from twinfer import infer_with_twinfer` removed (unused, broke every import of this module). See `Found during reorg`.
- [ ] `extract_run_id`, `build_simulation_record` consolidation from `infer_network_simulation_*.py` / notebook copies not yet done
- [x] `package/twinfer/utils/paths.py` — `get_repo_root()`, `get_data_root()`, `stage_dir()`, `get_external_repo_path()` present

### 1.6 Packaging
- [x] `package/pyproject.toml` finalized
- [x] `pip install -e package/` succeeds inside `twinfer-code`
- [x] `python -c "import twinfer"` sanity check passes

### 1.7 Other same-file double-definition bug
- [ ] `parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py` defines `calculate_pairwise_gene_gene_correlation_matrix` twice — diff, keep the correct one, comment out the other. Fix lands when this file moves to `paper_analysis/parameter_space_scan/` in Phase 2.

### 1.8 Docstrings
- [x] Pass over every public function under `package/twinfer/` — one-line purpose + args/returns docstring where missing

### 1.9 Phase 1 checkpoint
- [x] Tiny simulate+infer run against `simulation_example_input_data/`, compared to `simulation_example_output_data/`
- [x] `package/` layout and double-definition resolutions reviewed

---

## Phase 2 — `paper_analysis/`

*(Branch `reorg/paper-analysis` from `reorg/package` once Phase 1 is merged.)*

Folder-naming rationale, per-figure question, pipeline stages, and constants for each folder below are in that folder's own `README.md`. `figures_manifest.yaml` holds the figure-number-to-folder mapping.

### 2.1 `heterogeneity_vs_regulation/` (reproduces: "Twin difference correlations account for cell-state heterogeneity")
- [x] `README.md` written
- [x] `git mv scripts_simulation_for_figures/figure_2_simulations.py paper_analysis/heterogeneity_vs_regulation/simulate.py`
- [x] `git mv scripts_analyse_figure_data/figure_2.ipynb paper_analysis/heterogeneity_vs_regulation/analysis.ipynb`
- [x] `git mv scripts_to_plot_figures/figure_2.ipynb paper_analysis/heterogeneity_vs_regulation/plot.ipynb`
- [x] `git mv "scripts_analyse_figure_data/figure_2 binomial_partition.ipynb" paper_analysis/heterogeneity_vs_regulation/binomial_partition/analysis.ipynb`
- [x] `git mv binomial_partitioning/simulating_binomial_partition.py paper_analysis/heterogeneity_vs_regulation/binomial_partition/simulate.py`
- [x] `git mv binomial_partitioning/run_binomial_partition.sh paper_analysis/heterogeneity_vs_regulation/binomial_partition/run_simulate.sh`
- [x] `git mv binomial_partitioning/analyze_partitioning.ipynb paper_analysis/heterogeneity_vs_regulation/binomial_partition/` — reconcile with `binomial_partition_analysis.ipynb` first (check if duplicate)
- [x] `git mv binomial_partitioning/binomial_partition_analysis.ipynb paper_analysis/_archive/` if superseded
- [x] Fix hardcoded paths (stale `grnInference` and `/home/keerthm/...` references) to `twinfer.utils.paths`

### 2.2 `causal_direction_inference/` (reproduces: "Twin cross-correlations allow inference of causal relations")
- [x] `README.md` written
- [x] `git mv scripts_simulation_for_figures/figure_3_simulations.py paper_analysis/causal_direction_inference/simulate.py`
- [x] `git mv scripts_analyse_figure_data/figure_3.ipynb paper_analysis/causal_direction_inference/analysis.ipynb`
- [x] `git mv scripts_to_plot_figures/figure_3.ipynb paper_analysis/causal_direction_inference/plot.ipynb`
- [ ] Fix hardcoded paths; dedupe the 3x-repeated in-notebook `make_json_safe` → import from `twinfer.utils.json_utils`

### 2.3 `triplet_motif_discrimination/` (reproduces: "Twin cross-correlations distinguish between triplet motifs")
- [x] `README.md` written
- [x] `git mv scripts_simulation_for_figures/figure_4_simulations.py paper_analysis/triplet_motif_discrimination/simulate.py`
- [x] `git mv scripts_analyse_figure_data/figure_4.ipynb paper_analysis/triplet_motif_discrimination/analysis.ipynb`
- [ ] `git mv "scripts_analyse_figure_data/figure_4_f_score copy.ipynb" paper_analysis/_archive/figure_4_f_score_copy.ipynb` — confirm superseded first
- [x] `git mv scripts_to_plot_figures/Figure_4.ipynb paper_analysis/triplet_motif_discrimination/plot.ipynb`
- [ ] Fix hardcoded paths (stale `grnInference` references); dedupe the 2x-repeated in-notebook `make_json_safe`

### 2.4 `larry_hematopoiesis_validation/` (reproduces: "TwINFER captures regulatory interactions and confounding heterogeneity in experimental data")
- [x] `README.md` written
- [x] `git mv scripts_analyse_figure_data/figure_5_f_score.ipynb paper_analysis/larry_hematopoiesis_validation/figure_5_f_score.ipynb`
- [x] `git mv scripts_to_plot_figures/figure_5.ipynb paper_analysis/larry_hematopoiesis_validation/plot.ipynb`
- [x] `git mv LARRY_analysis.ipynb paper_analysis/larry_hematopoiesis_validation/LARRY_analysis.ipynb` (from repo root)
- [x] `git mv run_larry_more_TF.py paper_analysis/_archive/run_larry_more_TF.py` (from repo root)
- [x] `git mv bash_larry_Tf.sh paper_analysis/_archive/run_larry_more_TF.sh` (from repo root; fix stale paths)
- [ ] Fix hardcoded paths throughout
- [ ] Re-run the twin-pairing/heterogeneity cells after Known Bugs item 1 (`pair_id` grouping fix) is independently re-verified — prior results computed on the buggy `clone_id`-only grouping silently dropped most day-2/day-4 twin pairs; regenerate if stale
- [ ] Resolve whether CellTag-multi analysis (`real_data_analysis/celltag_analysis.ipynb`, `normalization_comparison.ipynb`) is part of this manuscript or separate ongoing work — see `figures_manifest.yaml` open item

### 2.5 `parameter_space_scan/` (reproduces: "Scanning the parameter space to define inference capabilities and limitations")
- [x] `README.md` written
- [ ] Move `additional_analysis/parameter_scan/` to _archive
  - [ ] `git mv parameter_scan/running_simulations_parameter_scan/* paper_analysis/parameter_space_scan/simulate/`
  - [ ] `git mv parameter_scan/analyzing_simulations_parameter_scan/* paper_analysis/parameter_space_scan/analysis/` (fix the double-defined `calculate_pairwise_gene_gene_correlation_matrix` here, per 1.7)
  - [x] Compare `additional_analysis/parameter_scan/{analyze_param_scan_results.ipynb,parameter_decision_tree.ipynb}` against the above — reconcile overlap, archive whichever is superseded
  - [x] `git mv scripts_to_plot_figures/parameter_scan.ipynb paper_analysis/parameter_space_scan/plot.ipynb`
- [ ] Both consuming figures (`heterogeneity_vs_regulation/`, `causal_direction_inference/`) reference this folder's output via `stage_dir()`, not a local copy

### 2.6 `grnboost2_comparison/` (reproduces: "Thresholding the GRNBoost2 ranked list using a Gaussian Mixture Model")
- [x] `README.md` written
- [x] `git mv synthetic_network_analysis/grn_boost.ipynb paper_analysis/grnboost2_comparison/`
- [ ] Extract the GMM-thresholding logic used by both consuming figures into shared functions here rather than duplicated per-figure

### 2.7 `extended_figures/`
- [x] `README.md` (index) written
- [ ] `interaction_parameter_choice/` — merge `hill_constant_effect/` + `k_add_effect/`; confirm per-panel script mapping first (see `figures_manifest.yaml`)
- [ ] `linear_toy_model_supplement/` — `git mv linear_simulation/"Linear Simulation and Supplementary Figure.ipynb"`
- [ ] `git mv scripts_analyse_figure_data/extended_figures.ipynb paper_analysis/extended_figures/analysis.ipynb`
- [ ] `git mv scripts_to_plot_figures/extended_figures.ipynb paper_analysis/extended_figures/plot.ipynb` — uses a third path base (`/home/keerthm/twinfer/...`, `/home/gzu5140/Font/...`); fix via `twinfer.utils.paths`
- [ ] `git mv additional_analysis/{creating_benchmark.ipynb,size_of_sample.ipynb,sample_size_correlation_results.csv,sample_size_correlation_results_A_B.csv}` — confirm destination once purpose is identified against the manuscript text

### 2.8 `parameter_estimation/`
- [ ] `git mv parameter_estimation/ paper_analysis/parameter_estimation/` (notebook + `data_for_estimation/` — spreadsheets move to `data/` in Phase 3, not here)
- [ ] Write `paper_analysis/parameter_estimation/README.md`

### 2.9 `benchmark/` — status unverified against the manuscript, see Key Decisions above
- [ ] Confirm with author whether this is part of this manuscript before any file moves
- [ ] If confirmed: `git mv synthetic_network_analysis/{generating_networks.ipynb,boolode_to_twinfer_format.py,twinfer_to_boolode_format.py,boolode_inference_twinfer.ipynb,infer_network_simulation_boolode_sims.py,infer_network_simulation_cyclic.py,infer_network_simulation_network_sweep.py,infer_network_simulation_real_network.py,network_sweep_twinfer_new.py,network_sweep_analysis.ipynb,benchmark_network_sweep.ipynb,evaluate_real_networks_boolode.ipynb,plot_network_sweep_final_inferred.ipynb,check_twinfer_sim.ipynb,verify_networksweep_conversion.py,run_infer_network.sh}` into `paper_analysis/benchmark/`, resolving the `_new` suffix and the `evaluate_networksweep_beeline.ipynb` vs `evaluate_networksweep_final_beeline.ipynb` duplicate first
- [ ] Build `paper_analysis/benchmark/metrics.py` consolidating `precision_recall_f1`, `compute_auprc` (+ variants), `top_k_tie_aware_selection` (+ variants), `build_edge_universe` (+ variants), `dedupe_predictions` (+ variant), `load_ground_truth`
- [ ] Write `paper_analysis/benchmark/README.md`, documenting the sibling repos at `../../Beeline` and `../../BoolODE` and `TWINFER_BEELINE_PATH`/`TWINFER_BOOLODE_PATH`

### 2.10 `work_in_progress/`
- [ ] `git mv additional_analysis/saturation_effects/ work_in_progress/saturation_and_noise_robustness/` — no matching figure identified in the manuscript text
- [ ] `git mv drift_multiple_state/simulate_pulse.py work_in_progress/pulsed_regulation/` — no matching figure identified
- [ ] Resolve disposition of both with the author before Phase 2 closes

### 2.11 Runner + manifest
- [x] `paper_analysis/figures_manifest.yaml` written
- [x] `paper_analysis/README.md` written
- [ ] `paper_analysis/run_figure.sh` (`./run_figure.sh <folder> [simulate|analyze|plot|all]`)
- [ ] `paper_analysis/reproduce_all.sh`, with `--validate` comparing output against the AUROC/F1/precision-recall values reported in the manuscript, not only checking that scripts ran
- [ ] Fix `scripts_simulation_for_figures/run_simulations_for_figures.sh` — currently launches `synthetic_network_high_density.py` instead of the figures its name implies; ensure the new runner doesn't inherit this

### 2.12 Phase 2 checkpoint
- [ ] Run `reproduce_all.sh` (or `run_figure.sh heterogeneity_vs_regulation all`) end-to-end; confirm output matches the pre-reorg pipeline for at least 2-3 folders
- [ ] Full `paper_analysis/` layout review; grep for remaining stale absolute paths

---

## Phase 3 — `tutorials/` + `data/`

*(Branch `reorg/tutorials-data` from `reorg/paper-analysis` once Phase 2 is merged.)*

- [ ] `git mv TwINFER_simulation_and_analysis.ipynb tutorials/01_simulate_and_infer_basics.ipynb`
- [ ] `git mv simulation_example_input_data/ data/example/input/`
- [ ] `git mv simulation_example_output_data/ data/example/output/`
- [ ] `git mv real_data/ data/real_data/`; confirm `.gitattributes` LFS filter for `*.h5ad` still applies after the move
- [ ] `git mv paper_analysis/parameter_estimation/data_for_estimation/ data/parameter_estimation/`
- [ ] Update `tutorials/01_simulate_and_infer_basics.ipynb` and any script referencing the moved example data to use `twinfer.utils.paths.get_data_root()`
- [ ] (Optional) `tutorials/02_reproduce_a_figure.ipynb` — short walkthrough bridging to `paper_analysis/run_figure.sh`
- [x] Root `README.md` — installation, paths/data location sections written; figure-table section still pending `figures_manifest.yaml` cross-check against the manuscript
- [ ] Update `.gitattributes` if any new large-file paths need LFS tracking

### Phase 3 checkpoint
- [ ] Open `tutorials/01_simulate_and_infer_basics.ipynb` fresh (new kernel), run top to bottom with zero manual path edits
- [ ] Final full-repo review pass
