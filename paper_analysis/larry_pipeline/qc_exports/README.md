# qc_exports

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| check_unfiltered_scale.sh |  |
| export_all_histograms.py | Same histogram + valley-detection logic as export_histogram_data.py and compare_valley_vs_knee_pipeline.py, but for all 15 libraries (not ju |
| export_histogram_data.py | Export a SPRING-style histogram of total transcripts per barcode, per example library -- log-spaced bins, one panel per library, our actual  |
| export_knee_plot_data.py | Export barcode-rank curve data (rank vs total counts, log-log, downsampled) plus the knee point, for a few representative libraries, to make |
| export_knee_plot_data_v2.py | Update the knee-plot export: show the ACTUAL threshold we use (the v2 whitelist's per-library scaled threshold, knee_thr * k) rather than th |
| run_export_all_histograms.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_export_histogram.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_export_knee_plot_v2.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_knee_plot_export.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
