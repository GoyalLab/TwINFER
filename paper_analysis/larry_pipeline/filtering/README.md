# filtering

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| apply_criterion4_rescue_pipeline.py | Runs the current production whitelist (knee v2) through singletCode exactly as the production pipeline does, then applies the criterion-4 cr |
| apply_paper_clone_filters.py | Apply the two documented anti-artifact filters from Weinreb et al. 2020 Methods 3.3 to our final QC'd matrix, removing cells whose clone cal |
| build_final_matrix_umi3.py | THE CURRENT PRODUCTION DEFAULT (writes to qc_filtered/). |
| build_final_matrix_with_criterion4.py | Builds the production final matrix with singletCode's criterion 4 (cross-sample barcode-combination rescue) completed, on top of the existin |
| check_excluded_h5ad_cells.py | Where do the 1,577 h5ad cells excluded by whitelist v2's threshold actually sit relative to that threshold -- right at the boundary (expecte |
| larry_umi_recovery_experiment.py | Experiment: recover low-read-count LARRY barcode observations that the main pipeline's N_READS>=10 filter drops, by Hamming-matching them ag |
| repo_TARGETED-BARCODE-FILTERING+EXPORT_updated_method.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| run_apply_criterion4_rescue.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_apply_paper_clone_filters.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_final_matrix_with_criterion4.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_check_excluded_h5ad_cells.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_recovery_experiment.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
