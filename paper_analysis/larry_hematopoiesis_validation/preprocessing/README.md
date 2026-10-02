# preprocessing

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| build_annotation_filtered_inputs.py | Build twin-input CSVs where a cell's clone_id is combined with its cell-type annotation (`{clone_id}__{Cell type annotation}`), so that assi |
| build_final_matrix_umi3.py | THE CURRENT PRODUCTION DEFAULT (writes to qc_filtered/). |
| build_twinfer_inputs_per_geneset.py | Build one TwINFER canonical-input CSV per gene set in resources/gene_sets.json, following the same approach convert_10x_to_twinfer_input.ipy |
| compare_day2_vs_all_gene_sets.py | One-off: rerun pick_gene_sets.ipynb's exact 9-gene-list recipe restricted to day-2 cells only (treating day 2 as if it were the whole datase |
| compute_random_delta_reference.py | Supplementary computation: random_delta_t1/t2 reference matrices for one gene set, needed for the new TwinScore's "ref_Delta drift contrast" |
| explore_balance_cap.py | Sweep BALANCE_CAP to find a tradeoff point between rare-class recall and majority-class (Undifferentiated etc.) accuracy, instead of committ |
| isg_panel_by_day.py | Direct check: does the day-2-only interferon-stimulated-gene (ISG) signature actually resolve by days 4/6, or is it present throughout and j |
| larry_raw_annotate.py | Attach cell-type annotations to the raw-count LARRY matrices, published where they exist and predicted where they do not, and record which i |
| run_benchmark_notebook.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_final_matrix_umi3.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_celltag3_recon.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitor_methods.py | Run the competitor GRN-inference methods -- GENIE3, GRNBoost2, PIDC, ppcor, rho, and a random baseline -- on one gene set's panel, pooled da |
| run_competitors_all9.sh | rho / ppcor / PIDC / GENIE3 / GRNBoost2 on all 9 gene sets, pooled day2+4, INPUT_DIR=twinfer_input_cp10k |
| run_competitors_allgenes_correlation.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_allgenes_detection.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_allgenes_variability.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_correlation.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_detection.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_variability.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_yscher_single.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_full_comparison.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs20_all9.sh | ALL_PAIRS 5000-shuffle run on the CURRENT gene sets (var reverted to top-HVG terciles, corr |
| run_infer_allpairs20_correlation.sh | ===== SUPERSEDED 2026-09-08 by run_infer_allpairs20_all9.sh ===== |
| run_infer_allpairs20_detection.sh | ===== SUPERSEDED 2026-09-08 by run_infer_allpairs20_all9.sh ===== |
| run_infer_allpairs20_variability.sh | ===== SUPERSEDED 2026-09-08 by run_infer_allpairs20_all9.sh ===== |
| run_infer_allpairs50_correlation.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs50_detection.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs50_single.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs50_single_annotfilter.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs50_variability.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs_correlation.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs_detection.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_allpairs_variability.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_correlation.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_correlation_high.py | Run infer_with_twinfer on one gene set (t1=2, t2=4), save every z-score-bearing structure, and (since this run is expensive) the raw permuta |
| run_infer_correlation_high.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_detection.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_variability.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_yscher_actual.py | Run infer_with_twinfer on one gene set (t1=2, t2=4), save every z-score-bearing structure, and (since this run is expensive) the raw permuta |
| run_infer_yscher_single.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_yscher_single_annotfilter.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_larry_raw_annotate.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_todo4v2_comparison.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| singletcode_cross_sample_rescue.py | Completes singletCode's own unfinished 4th singlet criterion (its TODO: "Integrate multi-sample singlets into above - singlets_step4"): |
