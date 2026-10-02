# whitelisting

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| build_diptest_otsu_whitelist.py | Cell-calling whitelist via Hartigan's dip test + Otsu's threshold, on the ACTUAL per-cell total_counts (not a binned-histogram approximation |
| build_gmm_whitelist.py | Fully automatic, target-free cell-calling whitelist: per library, fit a 2-component Gaussian mixture to log10(total transcript counts) and k |
| build_spring_whitelist.py | SPRING-style per-library cell-calling whitelist: a knee-point threshold on total transcript counts per library, the same principle data_prep |
| build_spring_whitelist_v2.py | SPRING-style whitelist, tuned to a target total cell count instead of the raw automatic knee. |
| compare_diptest_vs_knee_pipeline.py | Run the dip-test + Otsu whitelist (fully automatic, no external target -- see build_diptest_otsu_whitelist.py) through the SAME full downstr |
| compare_valley_vs_knee_pipeline.py | Compare two per-library cell-calling thresholds -- the scaled-knee threshold currently in production (v2 whitelist) vs. a local-minimum ("va |
| run_build_diptest_otsu_whitelist.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_gmm_whitelist.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_compare_diptest_vs_knee.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_compare_valley_vs_knee.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_spring_whitelist.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_spring_whitelist_v2.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
