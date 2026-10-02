# Third-party code used by this repo (usage trace, 2026-09-30)

Rule: only the fragments the pipelines actually use are vendored; each keeps its LICENSE; local changes are kept as patch files. Nothing is installed from clean_code yet: the pipelines still use the full installs listed below (paths via env.sh).

## Beeline (Murali-group/Beeline), upstream commit 37464085 (merge of PR #139), branch master
- Vendored: `benchmarks/beeline/vendored/{BLRunner.py,BLEvaluator.py,BLPlotter.py,BLRun/,BLEval/,BLPlot/,utils/,LICENSE,README.md}` (copies of the locally modified files).
- Local modifications vs upstream: `benchmarks/beeline/patches/Beeline_local_changes_vs_upstream_37464085.patch` (17 modified tracked files: 14 BLRun/*Runner.py, BLRunner.py, utils/environment.yml, utils/setupAnacondaVENV.sh; +546/-223).
- NOT vendored: `Algorithms/` (algorithm wrappers/sources, incl. SCSGL which `beeline_scoring_extract.py` and `real_networks_summary_table.py` import via BEELINE_SCSGL_DIR / TWINFER_BEELINE_PATH), `docs/`, `inputs/`, conda envs, julia/R libs, singularity images, per-run config dirs.
- Algorithms actually configured in the benchmark configs (config-files/*.yaml): GENIE3, GRNBOOST2, PIDC, PPCOR, PEARSON, SCODE, SCSGL, SINCERITIES, LEAP (SINGE, SCRIBE, GRNVBEM, GRISLI appear only in 2 configs, not run in the benchmarks). Runners for the unused ones are vendored only because BLRun/__init__ imports them all.
- Runs need the full install: `TWINFER_BEELINE_PATH` (default <project root>/code/Beeline) and the separate conda env `BEELINE` (`BEELINE_PYTHON`, default /home/gzu5140/.conda/envs/BEELINE/bin/python). This is the one place `twinfer-code` is not used: the algorithm wrappers need the BEELINE env.

## BoolODE (Murali-group/BoolODE), upstream commit ba8884a, branch master
- Vendored: `benchmarks/boolode/vendored/{BoolODE/,boolode.py,LICENSE,README.md,requirements.txt}` + the user's own scripts in `benchmarks/boolode/twins/`.
- Local modifications: `benchmarks/boolode/patches/BoolODE_local_changes_vs_upstream_ba8884a.patch` (model_generator.py, run_experiment.py, simulator.py, data/dyn-linear.txt, data/mCAD.txt).
- NOT vendored: slingshot-docker, silhouetteanalysis.py, scripts/, config-files/ (unused by the pipelines). `twins/*.sh` call the full install at `TWINFER_BOOLODE_PATH`.

## scclr (cleartools/scclr, Rust + Python), upstream 9528014 "Pin runorm revision"
- USED as an installed package (`import scclr`: scclr.pp.pflog, scclr.tl.pca) by notebooks that are now archived (`_archive/notebooks/real_data_analysis/{normalization_comparison,celltag_analysis}.ipynb`, 2026-10-01). Earlier cleanup notes called it unused; that was wrong.
- Not vendored (Rust build). In env twinfer-code it is an editable install from the OLD tree: /projects/b1042/GoyalLab/Keerthana/TwINFER/code/scclr (see docs/twinfer-code_pip_freeze_2026-09-30.txt). A copy also sits at TwINFER_KA/code/scclr. No local changes (only untracked rustup.sh).

## Other external code copied in
- yscher's TwinScore helpers: `paper_analysis/fatemap_pipeline/external_yscher/` (origin path + md5 header in each file).
## Editable installs in twinfer-code that point OUTSIDE clean_code (check before merging)
- TwINFER 0.1.0 -> /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package (original tree; clean_code is used via PYTHONPATH from env.sh)
- scclr 0.1.0 -> old tree (above)

## BoolODE: unused upstream paths removed (2026-10-01, user)
`benchmarks/boolode/vendored/BoolODE/` is no longer byte-identical to the install in `code/BoolODE/BoolODE`: five upstream paths that referenced undefined names / a typo and are never used by this project now raise `NotImplementedError` (old code commented out in place): dummy genes (`add_dummy`, `addDummyGenes`), deterministic ODE mode (`simulateModel`, `odeint` never imported), non-empty `parameter_set` files (`pvalue` typo in `getParameters`), protein-species output (`proteinlist` in `run_experiment`), slingshot post-processing (`computeSSPT`). Patch against the install: `benchmarks/boolode/patches/BoolODE_removed_unused_paths_2026-10-01.patch`. The stochastic Euler-SDE path with empty parameter sets (used by `benchmarks/boolode/twins/`) is untouched. The original install in `code/BoolODE` was not modified.
