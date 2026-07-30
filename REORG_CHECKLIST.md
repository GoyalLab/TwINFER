# TwINFER Reorganization Checklist

Goal: reorganize into three top-level folders — **`package/`**, **`paper_analysis/`**, **`tutorials/`** — plus a separated **`data/`** folder, eliminate duplicate function implementations, fix pervasive stale/hardcoded paths, add human/AI-readable documentation, and give a one-command way to reproduce/validate each figure.

## Key decisions baked into this checklist

- **Gillespie engine**: light-touch consolidation only — extract byte-identical shared helpers into `simulation/shared.py`; keep the variant simulation loops (default, drift, pulse, fixed-gene) as separate, clearly-named/documented files since each exists for a real, distinct scientific reason. The original pre-"variations" engine (`gillespie_script.py`) is superseded and archived, not migrated.
- **Package scope is bare-minimum**: only `simulation`, `inference` (incl. network naming / edge ranking), and `plotting`. Evaluation/benchmarking metrics are explicitly excluded from the package — they live in `paper_analysis/benchmark/` instead, since that area is still evolving.
- **Benchmarking (`paper_analysis/benchmark/`) is loosely structured on purpose** — not forced into the figure_N simulate/analyze/plot template, since it's still in flux. The user's local, edited copies of BEELINE and BoolODE live at `/home/gzu5140/Keerthana_b1042/TwINFER/code/Beeline` and `.../code/BoolODE` (sibling directories, **not** inside this repo, and not to be moved in) — only referenced via a configurable path.
- **Data**: large/raw data moves into a new top-level `data/` folder, separate from code.
- **Execution**: phased — `package/` first, reviewed, then `paper_analysis/`, reviewed, then `tutorials/`+`data/`.
- **Standing preference for this project**: never hard-delete code during edits — when consolidating/superseding a function or file, move the old version to an `_archive/` folder or comment it out in place with a note on why, rather than removing it outright. Use `git mv` wherever possible to preserve history.

## Recommended workflow

1. **Branch per phase, commit per checklist subsection.** `reorg/package` for Phase 1, `reorg/paper-analysis` for Phase 2 (branch from `reorg/package` once merged), `reorg/tutorials-data` for Phase 3. Small commits, not one giant diff.
2. **Verify at each phase checkpoint**, not only at the end.
3. **Strip notebook outputs before committing** (`jupyter nbconvert --clear-output ...` or `nbstripout`) — otherwise diffs are mostly re-executed-cell noise.
4. **Don't fix unrelated bugs inline** — note them in a running "found during reorg" list instead, to keep reorg commits reviewable as pure moves.

---

## Import contract: `package/` vs `paper_analysis/`

**One-way dependency.** `paper_analysis/` and `tutorials/` import *from* `package/twinfer/`; nothing inside `package/twinfer/` ever imports from `paper_analysis/`, `tutorials/`, or anything figure-specific. If a function only makes sense for one figure/benchmark, it does not belong in `package/twinfer/` (this is why evaluation metrics live in `paper_analysis/benchmark/metrics.py`, not the package).

- Every file under `paper_analysis/`/`tutorials/` should end up doing `import twinfer` / `from twinfer.X import Y` — never `sys.path.insert(...)` + `from TwINFER_function_scripts import ...`.
- **Verification commands** (run once Phase 1+2 are done):
  ```
  grep -rn "paper_analysis" package/twinfer/                                        # should be empty
  grep -rln "sys.path.insert\|TwINFER_function_scripts" paper_analysis/ tutorials/  # should be empty
  ```
- **No publishing required to use `import twinfer`.** `pip install -e package/` (1.6) is an editable install — it registers `package/twinfer/` in the active conda env's site-packages pointing back at the source dir, no copying, no PyPI upload. After that, `import twinfer` works from any script/notebook running in that same env, from any directory. Publishing (PyPI or similar) is a separate, optional later step only needed for people outside this environment — it has no bearing on whether `import twinfer` works for you now. It does require real `__init__.py` files in every subpackage dir to behave predictably (flagged below — currently missing on disk despite the box being checked).

---

## Known bugs and a proposed statistics upgrade (surfaced 2026-07-29)

Two external documents (`TwINFER_thresholds_and_CI_v2_1.pdf` — a governing spec for thresholds/sample-size/confidence-intervals with reference Python implementations; `PROJECT_SUMMARY.pdf` — a re-analysis of LARRY + CellTag-multi run from a separate project directory, `/gpfs/projects/b1255/yscher/Transcriptomic Distance/`, that patches TwINFER's code via runtime substitution rather than editing this repo) surfaced concrete bugs in `correlation_analysis_functions.py` and `infer_with_twinfer.py`. I independently re-read the flagged code and confirmed the following (that external harness is not something to merge in wholesale, but its findings — and the additional ones found while checking it — are directly actionable against files already in this repo's scope):

**Confirmed bugs, verified by direct code inspection** (all land in `package/twinfer/inference/correlation.py` / `infer.py`, see 1.3):

1. **Twin-pair grouping silently drops large clones** — `correlation_analysis_functions.py` ~L717-724 groups the twin table by `clone_id` and keeps groups of exactly 2 *rows*, which equals "a clone of 2 cells" only when the table has one row per cell (true in the simulation, false for barcoded real data, where each pair writes 2 rows and every pair of a clone shares one `clone_id`). Every clone larger than 2 cells is silently discarded, no error. Measured on LARRY: 61% of day-2 twin pairs and 90% of day-4 pairs lost. **Fix:** group by `pair_id` when present, fall back to `clone_id` otherwise.
2. **Self-pair exclusion tests the wrong thing** — `correlation_analysis_functions.py` ~L348: `if abs(idx_1[k] - idx_2[k]) > 1:` tests row-index adjacency, not cell identity. Order-dependent; discards ~3/n legitimate random pairs and can still admit a cell paired with itself. **Fix:** `if idx_1[k] != idx_2[k]:`.
3. **`check_gene_gene_correlation_threshold` silently flags nothing when `use_scramble=False`** — `correlation_analysis_functions.py` L528-636. `is_significant` is initialized to `False` at L586 and is only ever reassigned inside the `if use_scramble:` block (L599). Call this function with `use_scramble=False` (exactly what the proposed fixed-threshold methodology below would do) and **every single gene pair silently gets classified as `no_regulation`**, with no error and no warning — verified by reading the full control flow. This is the single highest-impact bug found: it means the codebase currently has no working non-scramble code path at all, despite `use_scramble` being an exposed parameter.
4. **`infer_with_twinfer`'s public result dict swaps t1/t2 for the random-pair matrix** — `infer_with_twinfer.py` L581 and L597 (both the success and except branches): `"random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t2,` — the key says `_t1` but the value assigned is `random_pair_correlation_matrix_t2`. Anyone reading `result['random_pair_correlation_matrix_t1']` from the public API silently gets t2's data instead. Confirmed copy-paste bug, present in both branches.
5. **The Stage-III "10% relative increase" regulation test is both internally inconsistent and empirically broken** — `identify_reg_if_multiple_states`, `correlation_analysis_functions.py` L823-881. The relative-change formula treats `corr_t2 < 0` and `corr_t2 >= 0` inconsistently (L869-872: takes `abs()` of the raw difference in one branch, signed difference in the other), and — independently confirmed by the external spec's own calibration run — because the twin difference-correlation starts near zero (`ρ̂∆(t1) ≈ 0`), any relative-change formula explodes and this test flags ~90% of a true no-regulation simulation scenario as regulation (near coin-toss). **Fixed:** `identify_reg_if_multiple_states` now tests `d = corr_t2 - corr_t1 > regulation_increase_threshold` (default `0.024`), the spec's point-estimate rule, same I/O as before. **TODO — needs extensive validation before trusting the default in general use:** `0.024` is calibrated on a single simulation run, a single gene pair, at a single `t2`; the spec's own table shows the "correct" threshold scaling roughly 8x across `t2 = 2h`–`36h`, and even at the calibration point the regulation/no-regulation `d` ranges overlap over repeated draws. Validate across a range of regulation strengths, multi-state separations, and measurement times before relying on the default outside the calibration scenario — see the docstring caveat in `identify_reg_if_multiple_states`.
6. **Docstring/implementation mismatch in the heterogeneity function** — `differentiate_single_state_reg_and_multiple_states`, `correlation_analysis_functions.py` L758-821. Docstring claims the threshold is on `abs(random / twin)`; the actual code computes `z_score = (t_corr - np.mean(r_corr)) / r_corr_std`, a z-score against the random-pair null distribution — an entirely different quantity. This is precisely the "old" `z = g/σ∆` formula the external spec identifies as having the wrong denominator (the spread of the *comparator*, not the standard error of the *statistic*) — i.e. this un-patched function is the bug the spec's heterogeneity fix (below) targets.
7. **Lower-confidence, likely dormant:** `calculate_twin_random_pair_correlations`, `correlation_analysis_functions.py` L738: `n_pairs = n_random or len(rep_0)` — Python's `or` treats an explicitly-passed `n_random=0` as falsy and silently substitutes `len(rep_0)` instead. Only matters if some caller ever passes `n_random=0` intentionally; not verified whether any does.

**A proposed statistics upgrade** (`TwINFER_thresholds_and_CI_v2_1.pdf`): replace the current scramble/p-value significance tests with fixed effect-size thresholds calibrated once on the simulations — existence `ρ* = 0.021`, direction `ρ̂†* = 0.046`, heterogeneity `g* = -0.126016` (two-sided on `z = g·√(N_twin−1)`, where `g = ρ̂∆(t1) − ⟨ρ∆⟩`), regulation `d* = 0.024` (replacing the broken relative-increase test above) — plus proper confidence intervals from resampling whole independent units (mother cells / clones, never cells or enumerated pairs) instead of from the shuffle. Rationale: a p-value threshold scales as `1/√(n−1)` and shrinks toward 0 as cell count grows, so every pair eventually becomes an edge; a fixed threshold does not have that failure mode. The spec's Appendix D contains full reference implementations: `stage1`/existence, `stage2`/heterogeneity, `direction`, a separate Stage-III/regulation module (`stage3_statistic`, `stage3_combine`, `stage3_ci_closed_form`, `stage3_ci_bootstrap`, `stage3_verdict`), and a replacement for the old Figure-2h panel (`plot_stage3`). Adopting this would also fix bugs 3, 5, and 6 above as a side effect, since it replaces exactly those three functions — but it is still a real methodological change (it changes which edges get flagged), not just a bug fix, so it needs your explicit sign-off.

**Decision needed before Phase 1 finalizes `correlation.py`:** (a) bugs 1, 2, 4, and 7 are unambiguous fixes with no methodology judgment call — fix them regardless. (b) For bug 3 (dead `use_scramble=False` branch) and bug 5/6 (broken Stage-III test, mismatched heterogeneity docstring): do you want the proposed threshold/CI methodology implemented into `package/twinfer/inference/` now as part of this reorg, deferred to a follow-up task after the reorg lands, or left as reference-only (documented but not wired in)? Note the decision here once made: ______

---

## Phase 0 — Setup

- [x] `git checkout -b reorg/package` from `master`
- [x] Add root `.gitignore`:
  ```
  __pycache__/
  *.pyc
  *.nbc
  *.nbi
  .ipynb_checkpoints/
  ```
- [x] `find . -name "__pycache__" -type d` to enumerate every committed cache dir (known: repo root, `TwINFER_function_scripts/`, `synthetic_network_analysis/`, `drift_multiple_state/`, `additional_analysis/saturation_effects/fixed_z_effect/`); `git rm -r --cached` each (keeps files on disk, only untracks)
- [x] Commit: "chore: add .gitignore, untrack build artifacts"
- [x] Decide fate of root scratch files `_check_helpers2.py`, `_check_ui2.py` (both hardcode absolute paths into `synthetic_network_analysis/`) — archive to `package/_archive/` or confirm safe to drop; note decision here: (moved it to plot utilities)

## Phase 1 — `package/`

Convert `TwINFER_function_scripts/` into a bare-minimum installable package `package/twinfer/`: simulation, inference, and plotting only.

### 1.1 Scaffold
- [x] Create `package/` directory
- [x] `git mv environment.yml package/environment.yml`
- [x] `git mv grnboost_env.yml package/grnboost_env.yml`
- [x] Write `package/pyproject.toml` (package name `twinfer`, source dir `twinfer/`, dependencies from `environment.yml`)
- [ ] Write `package/README.md`: module map, `pip install -e package/` instructions, when to use `environment.yml` vs `grnboost_env.yml`
- [x] Create `package/twinfer/__init__.py`, `simulation/__init__.py`, `inference/__init__.py`, `plotting/__init__.py`, `utils/__init__.py`
  - ⚠️ **Verified not actually present on disk** (checked via `find package -type f`): none of `package/twinfer/__init__.py`, `simulation/__init__.py`, `inference/__init__.py`, `plotting/__init__.py`, `utils/__init__.py` exist yet. Without them `twinfer` currently resolves only as an implicit Python 3 namespace package — `import twinfer` may appear to work but `from twinfer.simulation import X`-style imports and `pip install -e` packaging can behave unpredictably. Create the 5 files (empty is fine) before relying on `import twinfer` anywhere.
- [x] Create `package/_archive/` for superseded files

### 1.2 Simulation module
- [x] `git mv TwINFER_function_scripts/gillespie_script.py package/_archive/gillespie_script.py` (superseded by variations engine)
- [x] `git mv TwINFER_function_scripts/gillespie_script_variations.py package/twinfer/simulation/gillespie_simulations.py`
  - ⚠️ **Verified this was a copy, not a `git mv`, and the rename didn't happen either.** `TwINFER_function_scripts/gillespie_script_variations.py` still exists at the old location, and `package/twinfer/simulation/gillespie_script_variations.py` is a new **untracked** file (`git status` shows `??`) still under its *old* filename — there are now two copies of the same engine on disk. Fix in one step (does the move, the rename, and preserves history correctly):
    ```
    rm package/twinfer/simulation/gillespie_script_variations.py   # drop the stray untracked copy
    git mv TwINFER_function_scripts/gillespie_script_variations.py package/twinfer/simulation/gillespie_simulations.py
    ```
  - [ ] **Fix every downstream import once the rename above is done.** 14 files reference `gillespie_script_variations` by name via the same broken pattern (hardcoded `path_to_code_repo`, often the stale pre-rename `grnInference` path, `sys.path.insert`, then `from TwINFER_function_scripts import gillespie_script_variations` + `importlib.reload`): `scripts_simulation_for_figures/{figure_1_network,figure_2_simulations,figure_3_simulations,figure_4_simulations,network_sweep,cycle,real_network,synthetic_network,synthetic_network_high_density}.py`, `TwINFER_simulation_and_analysis.ipynb`, `k_add_effect/multiple_k_add_A_to_B.sh`, `hill_constant_effect/simulating_multiple_hill_constant.py`, `additional_analysis/saturation_effects/mutliple_k_add_A_to_B.py`, `binomial_partitioning/simulating_binomial_partition.py`. In each, replace the whole `path_to_code_repo`/`sys.path.insert`/`from TwINFER_function_scripts import gillespie_script_variations`/`importlib.reload(...)` block with: `from twinfer.simulation import gillespie_simulations` (or `from twinfer.simulation.gillespie_simulations import process_param_set` where that's the only thing used) — no `sys.path` hack needed once `pip install -e package/` (1.6) is done. Since most of these files move into `paper_analysis/` in Phase 2 anyway, it's fine to do this rewrite there instead of twice — just don't lose track of the list.
- [x] `git mv drift_multiple_state/gillespie_script_drift.py package/twinfer/simulation/gillespie_drift.py`
- [x] `git mv drift_multiple_state/gillespie_script_pulse.py package/twinfer/simulation/gillespie_pulse.py`
- [x] `git mv additional_analysis/saturation_effects/fixed_z_effect/gillespie_fixed_gene.py package/twinfer/simulation/gillespie_fixed_gene.py` - #COMMENT: not needed (it is a random analysis)
- [l] Diff the 4 files function-by-function; confirm which of these are truly identical across all 4 (`hill_fn` is confirmed byte-identical; check the rest): `read_input_matrix`, `generate_reaction_network_from_matrix`, `assign_parameters_to_genes`, `generate_initial_state_from_genes`, `assign_k_values_matrix`, `generate_k_from_max_expression`, `generate_K_from_steady_state_calc`, `add_interaction_terms`, `is_steady_state`, `convert_samples_to_df`, `get_promoter_indices`, `save_promoter_events`, `allocate_event_logs`, `validate_regulatory_configuration`, `divide_mother_cell_content`, `process_param_set`, `process_param_set_with_numba_config`, `check_if_file_exists`
- [l] Create `package/twinfer/simulation/shared.py`; move each confirmed-identical function there (this is a move, not a delete — content isn't lost, just centralized)
- [l] Update all 4 variant files to `from .shared import ...` instead of redefining; remove the now-redundant local copies
- [l] For any function that looked identical but actually diverged slightly on inspection — do NOT force-merge it; leave it in place per-file and note the divergence in a comment
- [l] Add a module-level docstring to each of the 4 files stating: what scenario it simulates, and which figures/analyses call it:
  - `gillespie_simulations.py` → default engine, used by figure_2/3/4 simulations and most of `scripts_simulation_for_figures/`
  - `gillespie_drift.py` → used by `drift_multiple_state/simulate_drift_multiple_states.py` → `visualize_drift_simulation.ipynb`
  - `gillespie_pulse.py` → used by `drift_multiple_state/simulate_pulse.py` → `visualize_drift_simulation.ipynb`
  - `gillespie_fixed_gene.py` → used by `additional_analysis/saturation_effects/fixed_z_effect/fixed_z_simulations.py` → `analyze_fixed_z.ipynb`
- [ ] Write `package/twinfer/simulation/README.md` — one table: variant | scenario it models | figures/analyses that use it
- [l] Document the already-diverged `run_simulation` signature (`promoter_indices=None` present in `gillespie_simulations.py`/`gillespie_fixed_gene.py`, absent in `gillespie_drift.py`/`gillespie_pulse.py`) with an explicit comment in each — confirm whether this is intentional or a bug before deciding whether to reconcile it now or later

### 1.3 Inference module
- [x] `git mv TwINFER_function_scripts/infer_with_twinfer.py package/twinfer/inference/infer.py`
- [x] Merge `TwINFER_function_scripts/correlation_analysis_functions.py` + `correlation_analysis_helpers.py` into one `package/twinfer/inference/correlation.py`
- [x] Resolve the `make_reds_blues_colormap` double-definition inside `correlation_analysis_helpers.py` (line ~276: simple two-tone version vs line ~554: zero-centered `vmin=-0.05, vmax=0.18` version) — check call sites to see which one callers actually expect, keep that as canonical in `correlation.py`, comment out the other with a note explaining which was superseded and why
- [x] `git mv TwINFER_function_scripts/network_naming_utils.py package/twinfer/inference/network_naming.py`
- [x] `git mv TwINFER_function_scripts/ranked_edges_utils.py package/twinfer/inference/ranked_edges.py`
- [x] **Bug fix (unambiguous, no methodology judgment call — see "Known bugs" above):** twin-pair grouping in `calculate_twin_random_pair_correlations`/nearby helper (~L717-724) groups by `clone_id` and filters to `len(g)==2` rows, silently discarding every clone >2 cells on barcoded data (61%/90% of LARRY day-2/day-4 pairs lost). Group by `pair_id` when present, fall back to `clone_id` otherwise.
- [x] **Bug fix:** self-pair exclusion at ~L348 tests `abs(idx_1[k] - idx_2[k]) > 1` (row adjacency) instead of `idx_1[k] != idx_2[k]` (cell identity) — fix to the identity test.
- [x] **Bug fix:** in `infer_with_twinfer.py` L581 and L597, `"random_pair_correlation_matrix_t1"` is assigned the value of `random_pair_correlation_matrix_t2` in both the success and except branches — fix to assign `random_pair_correlation_matrix_t1`.
- [x] **Bug fix, lower priority:** `n_pairs = n_random or len(rep_0)` (~L738) silently ignores an explicit `n_random=0` — change to `n_pairs = len(rep_0) if n_random is None else n_random`.
- [x] **Gated on your decision (see "Known bugs" section above):** `check_gene_gene_correlation_threshold`'s `use_scramble=False` path never sets `is_significant` (silently flags every pair as `no_regulation` — L586/L599)
- [x] `identify_reg_if_multiple_states`'s 10%-relative-increase Stage-III test — replaced with the spec's point-estimate rule (`d = corr_t2 - corr_t1 > regulation_increase_threshold`, default `0.024`), same I/O as before (light-touch fix, not the full `thresholds.py`/`regulation.py`/`confidence.py` module split described below).
- [ ] **TODO — extensive validation needed:** the `regulation_increase_threshold=0.024` default in `identify_reg_if_multiple_states` is calibrated on a single simulation run, single gene pair, single `t2` — not a validated universal constant (see docstring caveat and "Known bugs" item 5 above). Before trusting it in general use, validate across a range of regulation strengths, multi-state separations, and measurement times; consider whether the threshold should instead be derived as a function of `t2` (the spec's Appendix has a `d*(t2)` table suggesting it scales ~8x from `t2=2h` to `t2=36h`).
- [x] `differentiate_single_state_reg_and_multiple_states`'s docstring doesn't match its own z-score formula (still open — see "Known bugs" item 6). If adopting the full proposed methodology: add `package/twinfer/inference/thresholds.py` (existence/heterogeneity/direction stage functions + `verdict()` three-way reporting, replacing `check_gene_gene_correlation_threshold` and `differentiate_single_state_reg_and_multiple_states`), `package/twinfer/inference/regulation.py` (Stage III: `stage3_statistic`, `stage3_combine`, `stage3_ci_closed_form`, `stage3_ci_bootstrap`, `stage3_verdict`), and `package/twinfer/inference/confidence.py` (`unit_bootstrap`, clone-stratified bootstrap) — port from `TwINFER_thresholds_and_CI_v2_1.pdf` Appendix D, keeping the calibrated constants (`ρ*=0.021`, `ρ̂†*=0.046`, `g*=-0.126016`) as named constants with a comment citing their calibration basis. If deferring: leave the docstring mismatch in place, comment `# KNOWN ISSUE, see REORG_CHECKLIST.md "Known bugs"` above it.

### 1.4 Plotting module
- [x] Compare `synthetic_network_analysis/plot_grn.py` vs `synthetic_network_analysis/grn_plot_spread_v2.py` — confirm v2 is the fuller/newer rewrite (adds edge-arc geometry helpers like `_arc3_end_tangent`, `_assign_desired_contact_angles`)
- [x] `git mv synthetic_network_analysis/grn_plot_spread_v2.py package/twinfer/plotting/network_plots.py`; rename any internal `_v2`/`_spread` naming artifacts now that it's the sole canonical version
- [-] `git mv synthetic_network_analysis/plot_grn.py package/_archive/plot_grn_original.py` - #COMMENT : do not need plot_grn anymore. Only v2 is needed/used.
- [x] `git mv fonts/ package/twinfer/plotting/assets/fonts/` (5 `.ttf` files)

### 1.5 Utils module
- [x] Create `package/twinfer/utils/json_utils.py`; consolidate `make_json_safe` + `default()` JSON encoder + `extract_run_id` + `build_simulation_record` — currently duplicated across `infer_network_simulation_boolode_sims.py`, `infer_network_simulation_cyclic.py`, `infer_network_simulation_network_sweep.py`, `infer_network_simulation_real_network.py`, `network_sweep_twinfer_new.py`, plus 15+ notebook copies (heaviest repeat offenders: `figure_5_f_score.ipynb` ×5, `figure_3.ipynb` ×3, `figure_4.ipynb` ×2) #didnot merge the build simulation record since it actually does analysis rather than pure utility.
- [x] Build `package/twinfer/utils/paths.py` — single source of truth for ALL paths in the repo, fixing the "manually keep paths in sync" problem:
  - [x] `get_repo_root()` — derived from the installed package's own file location, not a hardcoded string
  - [x] `get_data_root()` — reads `TWINFER_DATA_ROOT` env var, defaults to `<repo_root>/data`
  - [x] `stage_dir(figure_name, stage, run_tag="latest")` → `data/paper_analysis/<figure_name>/<stage>/<run_tag>/` — used identically by simulate/analyze/plot for a given figure, so the three stages can never point at mismatched directories; reruns get a fresh `run_tag` instead of a hand-edited date stamp
  - [x] `get_external_repo_path(name)` — reads `TWINFER_BEELINE_PATH` / `TWINFER_BOOLODE_PATH` env vars (or a `paths.yaml`)

### 1.6 Packaging
- [x] Finalize `package/pyproject.toml`
- [x] `pip install -e package/` inside the `twinfer-code` conda env — confirm it succeeds
- [x] `python -c "import twinfer"` sanity check

### 1.7 Other same-file double-definition bug
- [l] Note (fix lands in Phase 2 when this file moves): `parameter_scan/analyzing_simulations_parameter_scan/analyze_parameter_scan_correlations.py` defines `calculate_pairwise_gene_gene_correlation_matrix` twice (line ~67 and ~150) — diff them, keep the correct one, comment out the other

### 1.8 Docstrings
- [x] Pass over every public function now under `package/twinfer/`: add a one-line purpose + args/returns docstring where missing —
 target a reader who wants to call the function without reading its body, not a line-by-line narration

### 1.9 Phase 1 checkpoint
- [x] Run a tiny simulate+infer using `simulation_example_input_data/` connectivity matrix; compare output to `simulation_example_output_data/` to confirm the shared.py extraction didn't change behavior
- [x] Ping Claude to review the `package/` layout and confirm the two double-definition resolutions before moving on

---

## Phase 2 — `paper_analysis/`

*(Branch `reorg/paper-analysis` from `reorg/package` once Phase 1 is merged.)*

### 2.1 figure_1/
- [ ] `git mv scripts_simulation_for_figures/figure_1_network.py paper_analysis/figure_1/simulate.py`
- [ ] Update its import to `import twinfer`
- [ ] Write `paper_analysis/figure_1/README.md`

### 2.2 figure_2/ (+ binomial partition)
- [ ] `git mv scripts_simulation_for_figures/figure_2_simulations.py paper_analysis/figure_2/simulate.py`
- [ ] `git mv scripts_analyse_figure_data/figure_2.ipynb paper_analysis/figure_2/analyze.ipynb`
- [ ] `git mv scripts_to_plot_figures/figure_2.ipynb paper_analysis/figure_2/plot.ipynb`
- [ ] `git mv "scripts_analyse_figure_data/figure_2 binomial_partition.ipynb" paper_analysis/figure_2/binomial_partition/analyze.ipynb`
- [ ] `git mv scripts_to_plot_figures/binomial_partition.ipynb paper_analysis/figure_2/binomial_partition/plot.ipynb`
- [ ] `git mv binomial_partitioning/simulating_binomial_partition.py paper_analysis/figure_2/binomial_partition/simulate.py`
- [ ] `git mv binomial_partitioning/run_binomial_partition.sh paper_analysis/figure_2/binomial_partition/run_simulate.sh`
- [ ] `git mv binomial_partitioning/analyze_partitioning.ipynb paper_analysis/figure_2/binomial_partition/` (reconcile with `binomial_partition_analysis.ipynb` — check if genuinely different or a duplicate; archive whichever is superseded)
- [ ] `git mv binomial_partitioning/binomial_partition_analysis.ipynb paper_analysis/_archive/` (if superseded — confirm first)
- [ ] Fix all hardcoded paths in the above (both the stale `grnInference` ones and the different-user `/home/keerthm/...` one) to use `twinfer.utils.paths`
- [ ] Write `paper_analysis/figure_2/README.md`

### 2.3 figure_3/
- [ ] `git mv scripts_simulation_for_figures/figure_3_simulations.py paper_analysis/figure_3/simulate.py`
- [ ] `git mv scripts_analyse_figure_data/figure_3.ipynb paper_analysis/figure_3/analyze.ipynb`
- [ ] `git mv scripts_to_plot_figures/figure_3.ipynb paper_analysis/figure_3/plot.ipynb`
- [ ] Fix hardcoded paths, dedupe the 3x-repeated in-notebook `make_json_safe` in `analyze.ipynb` → import from `twinfer.utils.json_utils`
- [ ] Write `paper_analysis/figure_3/README.md`

### 2.4 figure_4/
- [ ] `git mv scripts_simulation_for_figures/figure_4_simulations.py paper_analysis/figure_4/simulate.py`
- [ ] `git mv scripts_analyse_figure_data/figure_4.ipynb paper_analysis/figure_4/analyze.ipynb`
- [ ] `git mv "scripts_analyse_figure_data/figure_4_f_score copy.ipynb" paper_analysis/_archive/figure_4_f_score_copy.ipynb` (confirm superseded before moving)
- [ ] `git mv scripts_to_plot_figures/Figure_4.ipynb paper_analysis/figure_4/plot.ipynb` (fixes casing inconsistency)
- [ ] Fix hardcoded paths (this trio still references the stale `grnInference` path), dedupe the 2x-repeated in-notebook `make_json_safe`
- [ ] Write `paper_analysis/figure_4/README.md`

### 2.5 figure_5/ (+ real_data_analysis)
- [ ] `git mv scripts_analyse_figure_data/figure_5_f_score.ipynb paper_analysis/figure_5/figure_5_f_score.ipynb`
- [ ] `git mv scripts_to_plot_figures/figure_5.ipynb paper_analysis/figure_5/plot.ipynb`
- [ ] `git mv LARRY_analysis.ipynb paper_analysis/figure_5/real_data_analysis/LARRY_analysis.ipynb` (from repo root)
- [ ] `git mv run_larry_more_TF.py paper_analysis/figure_5/real_data_analysis/run_larry_more_TF.py` (from repo root)
- [ ] `git mv bash_larry_Tf.sh paper_analysis/figure_5/real_data_analysis/run_larry_more_TF.sh` (from repo root; fix stale paths)
- [ ] `git mv real_data_analysis/celltag_analysis.ipynb paper_analysis/figure_5/real_data_analysis/`
- [ ] `git mv real_data_analysis/normalization_comparison.ipynb paper_analysis/figure_5/real_data_analysis/`
- [ ] Fix hardcoded paths throughout
- [ ] Re-run `LARRY_analysis.ipynb`'s twin-pairing/heterogeneity cells after the `pair_id` grouping fix lands in `package/twinfer/inference/correlation.py` (1.3) — prior heterogeneity results here were computed on the buggy `clone_id`-only grouping and silently dropped most day-2/day-4 twin pairs; flag any figure_5 heterogeneity numbers generated before the fix as stale and regenerate them.
- [ ] Write `paper_analysis/figure_5/README.md`, explicitly noting it depends on `paper_analysis/benchmark/` output for F1/AUPRC inputs

### 2.6 benchmark/
- [ ] `git mv synthetic_network_analysis/generating_networks.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/boolode_to_twinfer_format.py paper_analysis/benchmark/conversion/`
- [ ] `git mv synthetic_network_analysis/twinfer_to_boolode_format.py paper_analysis/benchmark/conversion/`
- [ ] `git mv synthetic_network_analysis/boolode_inference_twinfer.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/grn_boost.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/infer_network_simulation_boolode_sims.py paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/infer_network_simulation_cyclic.py paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/infer_network_simulation_network_sweep.py paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/infer_network_simulation_real_network.py paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/network_sweep_twinfer_new.py paper_analysis/benchmark/network_sweep_twinfer.py` (drop "_new" now that it's the only copy)
- [ ] `git mv synthetic_network_analysis/network_sweep_analysis.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/benchmark_network_sweep.ipynb paper_analysis/benchmark/`
- [ ] Compare `evaluate_networksweep_beeline.ipynb` vs `evaluate_networksweep_final_beeline.ipynb` — keep the actually-canonical one in `paper_analysis/benchmark/`, `git mv` the other to `paper_analysis/_archive/`
- [ ] `git mv synthetic_network_analysis/evaluate_real_networks_boolode.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/plot_network_sweep_final_inferred.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/check_twinfer_sim.ipynb paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/verify_networksweep_conversion.py paper_analysis/benchmark/`
- [ ] `git mv synthetic_network_analysis/test_grnboost_vsgenie_3.ipynb paper_analysis/_archive/` (throwaway-iteration naming — confirm superseded first)
- [ ] `git mv synthetic_network_analysis/run_infer_network.sh paper_analysis/benchmark/`
- [ ] Build `paper_analysis/benchmark/metrics.py`, consolidating from the notebooks above: `precision_recall_f1`, `compute_auprc` (+ `_undirected`/`_signed`), `top_k_tie_aware_selection` (+ variants), `build_edge_universe` (+ variants), `dedupe_predictions` (+ variant), `load_ground_truth`
- [ ] Update every notebook in `benchmark/` to import from `metrics.py` instead of redefining
- [ ] Write `paper_analysis/benchmark/README.md`: explain this feeds figure_5's F1/AUPRC scores, document the sibling repos at `../../Beeline` and `../../BoolODE` (user's local edited copies, not part of this repo) and how `TWINFER_BEELINE_PATH`/`TWINFER_BOOLODE_PATH` are used
- [ ] Move `scripts_simulation_for_figures/{network_sweep.py, real_network.py, synthetic_network.py, synthetic_network_high_density.py, cycle.py}` into `paper_analysis/benchmark/` (these feed the benchmark pipeline, not a standalone figure)

### 2.7 extended_figures/
- [ ] `git mv hill_constant_effect/simulating_multiple_hill_constant.py paper_analysis/extended_figures/hill_constant_effect/simulate.py`
- [ ] `git mv hill_constant_effect/run_simulating_multiple_hill_constant.sh paper_analysis/extended_figures/hill_constant_effect/run_simulate.sh`
- [ ] `git mv k_add_effect/multiple_k_add_A_to_B.sh paper_analysis/extended_figures/k_add_effect/`
- [ ] `git mv k_add_effect/multiple_k_add_A_rep_B.sh paper_analysis/extended_figures/k_add_effect/`
- [ ] `git mv k_add_effect/parameters_for_multiple_k_add.ipynb paper_analysis/extended_figures/k_add_effect/`
- [ ] `git mv additional_analysis/saturation_effects/{analyze_pulse.ipynb,noise_analysis.ipynb,saturation_analysis.ipynb,parameters_for_multiple_k_add.ipynb,mutliple_k_add_A_to_B.py,multiple_k_add.sh} paper_analysis/extended_figures/saturation_effects/` (reconcile the duplicate `parameters_for_multiple_k_add.ipynb` that also exists under `k_add_effect/` — check if identical, archive one)
- [ ] Rename the malformed `additional_analysis/saturation_effects/cross_correlati` → a proper `.csv` name (e.g. `cross_correlation_results.csv`) once its actual content/purpose is confirmed; `git mv` into `paper_analysis/extended_figures/saturation_effects/`
- [ ] `git mv additional_analysis/saturation_effects/fixed_z_effect/{fixed_z_simulations.py,analyze_fixed_z.ipynb,run_simulations_for_figures.sh} paper_analysis/extended_figures/saturation_effects/fixed_z_effect/`
- [ ] `git mv drift_multiple_state/{simulate_drift_multiple_states.py,simulate_pulse.py,run_drift_simulation.sh,visualize_drift_simulation.ipynb} paper_analysis/extended_figures/drift_multiple_state/`
- [ ] `git mv linear_simulation/"Linear Simulation and Supplementary Figure.ipynb" paper_analysis/extended_figures/linear_simulation/`
- [ ] Merge `parameter_scan/` + `additional_analysis/parameter_scan/` into `paper_analysis/extended_figures/parameter_scan/`:
  - [ ] `git mv parameter_scan/running_simulations_parameter_scan/* paper_analysis/extended_figures/parameter_scan/simulate/`
  - [ ] `git mv parameter_scan/analyzing_simulations_parameter_scan/* paper_analysis/extended_figures/parameter_scan/analyze/` (fix the double-defined `calculate_pairwise_gene_gene_correlation_matrix` here, per 1.7)
  - [ ] Compare `additional_analysis/parameter_scan/{analyze_param_scan_results.ipynb,parameter_decision_tree.ipynb}` against the above — reconcile overlap, archive whichever is superseded, keep the rest
  - [ ] `git mv scripts_to_plot_figures/parameter_scan.ipynb paper_analysis/extended_figures/parameter_scan/plot.ipynb`
- [ ] `git mv scripts_analyse_figure_data/extended_figures.ipynb paper_analysis/extended_figures/analyze.ipynb`
- [ ] `git mv scripts_to_plot_figures/extended_figures.ipynb paper_analysis/extended_figures/plot.ipynb` (this one currently uses a *third* path base, `/home/keerthm/twinfer/...` and `/home/gzu5140/Font/...` — must be fixed via `twinfer.utils.paths`, not just copy-pasted)
- [ ] `git mv additional_analysis/{creating_benchmark.ipynb,size_of_sample.ipynb,sample_size_correlation_results.csv,sample_size_correlation_results_A_B.csv} paper_analysis/extended_figures/` (or a dedicated subfolder if these turn out to be their own thing — confirm)
- [ ] Write `paper_analysis/extended_figures/README.md` covering every sub-analysis and what it supports

### 2.8 parameter_estimation/
- [ ] `git mv parameter_estimation/ paper_analysis/parameter_estimation/` (notebook + `data_for_estimation/` — the spreadsheets move to `data/` in Phase 3, not here)
- [ ] Write `paper_analysis/parameter_estimation/README.md`

### 2.9 Runner + manifest
- [ ] Write `paper_analysis/figures_manifest.yaml` — one entry per figure/analysis with its stage scripts (no literal paths, those come from `paths.py`)
- [ ] Write `paper_analysis/run_figure.sh` (`./run_figure.sh figure_2 [simulate|analyze|plot|all]`)
- [ ] Write `paper_analysis/reproduce_all.sh`, with a `--validate` mode checking expected outputs exist
- [ ] Fix `scripts_simulation_for_figures/run_simulations_for_figures.sh` bug — it currently launches `synthetic_network_high_density.py` instead of the figure_2/3/4 simulation scripts its name implies; make sure the new runner doesn't inherit this mismatch

### 2.10 Phase 2 checkpoint
- [ ] Run `reproduce_all.sh` (or `run_figure.sh figure_2 all`) end-to-end; confirm output matches the pre-reorg pipeline for at least 2-3 figures
- [ ] Ping Claude to review the full `paper_analysis/` layout and grep for any remaining stale absolute paths

---

## Phase 3 — `tutorials/` + `data/`

*(Branch `reorg/tutorials-data` from `reorg/paper-analysis` once Phase 2 is merged.)*

- [ ] `git mv TwINFER_simulation_and_analysis.ipynb tutorials/01_simulate_and_infer_basics.ipynb`; update to `import twinfer`
- [ ] `git mv simulation_example_input_data/ data/example/input/`
- [ ] `git mv simulation_example_output_data/ data/example/output/`
- [ ] `git mv real_data/ data/real_data/`; confirm `.gitattributes` LFS filter for `*.h5ad` still applies after the move
- [ ] `git mv paper_analysis/parameter_estimation/data_for_estimation/ data/parameter_estimation/`
- [ ] Update `tutorials/01_simulate_and_infer_basics.ipynb` and any script referencing the moved example data to use `twinfer.utils.paths.get_data_root()`
- [ ] (Optional) write `tutorials/02_reproduce_a_figure.ipynb` — short walkthrough bridging to `paper_analysis/run_figure.sh`
- [ ] Rewrite root `README.md`: install instructions, three-folder tour, figure→script→output table (pulled from `figures_manifest.yaml`) — cross-check this table against the external Google Sheet for anything not inferable from filenames alone
- [ ] Update `.gitattributes` if any new large-file paths need LFS tracking

### Phase 3 checkpoint
- [ ] Open `tutorials/01_simulate_and_infer_basics.ipynb` fresh (new kernel), run top to bottom with zero manual path edits
- [ ] Ping Claude for a final full-repo review pass
