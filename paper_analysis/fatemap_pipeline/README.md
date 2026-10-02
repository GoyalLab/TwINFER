# fatemap_pipeline

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| SUPERSEDED_apply_twinscore_supplement_fatemap.py | Ports larry_hematopoiesis_validation/apply_twinscore_supplement_larry.py (TwinScore_supplement.pdf, 2026-09-17) to FM06/FM08/SpaceBar's sing |
| SUPERSEDED_apply_twinscore_supplement_fatemap_allpairs.py | 2026-09-22 user request: rerun TwinScore_supplement (the formula WITH phi/persistence, distinct from the default package twinScore) under bo |
| SUPERSEDED_apply_twinscore_supplement_spacebar_centroid_ab.py | SpaceBar analog of apply_twinscore_supplement_fatemap_allpairs.py, on the centroid-split A/B construction (run_infer_spacebar_centroid_ab.py |
| SUPERSEDED_run_ab_allpairs_full_fm06.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_ab_allpairs_full_fm08.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_spacebar_centroid_ab_infer_supp.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_supp_ab_fm01.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_supp_ab_fm01_v3.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_supp_allpairs_bootstrap_fm06.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_supp_allpairs_patched.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_twinscore_supplement_all.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_twinscore_supplement_allpairs.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_run_twinscore_supplement_one.sh | SUPERSEDED 2026-09-23 -- DO NOT USE. Runs a superseded twin_supp scorer; use run_supp_gated_bootstrap_*.sh (final gated + bootstrap version) |
| SUPERSEDED_score_negfloor_fm06.py | TwinFER 'neg-only floor' scoring for FM06 A/B-split outputs (2026-09-23). phi is left UNFLOORED for h>0; only genes with h<=0 (in either rep |
| SUPERSEDED_score_phi_variants_fm06.py | Compare phi treatments on FM06 A/B-split outputs (2026-09-23). Caches per-gene bootstrap null SD floors. Variants: unfloored (NaN if h<=0, p |
| apply_singletime_score_spacebar.py | Single-timepoint TwinScore (S/C/U/h; Stages I-III, no direction) on SpaceBar, 2026-09-28. |
| apply_todo4v2_absplit_fm06.py | TODO4v2 (apply_todo4v2_vs_collectri.py's formula, unchanged) on the FM06 A/B split (t1 = A, t2 = B), every ordered pair of a gene set, vs Co |
| apply_todo4v2_vs_collectri.py | Ports larry_hematopoiesis_validation/apply_todo4v2_allpairs_with_competitors.py's TODO4v2 formula and CollecTRI-based validation to FM06/FM0 |
| apply_twinscore_supplement_fatemap_gated_bootstrap.py | TwinScore_supplement, FINAL gated + bootstrapped version (2026-09-23). USE THIS ONE. |
| apply_twinscore_supplement_fatemap_gated_bootstrap_juliaD.py | Final gated + bootstrap TwinScore_supplement with the D term taken from the Julia PIDC (NetworkInference.jl) instead of the Python PIDC. Ide |
| apply_twinscore_supplement_fatemap_gated_bootstrap_parallel.py | TwinScore_supplement, FINAL gated + bootstrapped version, PARALLEL-BOOTSTRAP variant (2026-09-23). |
| apply_twinscore_supplement_hs054_gated_bootstrap.py | TwinScore_supplement, FINAL gated + bootstrapped version, for HS054_Sot48h. |
| apply_twinscore_supplement_sim_gated_bootstrap.py | TwinScore_supplement, gated + bootstrapped version (apply_twinscore_supplement_fatemap_gated_bootstrap.py), applied to SIMULATED twin data ( |
| apply_twinscore_supplement_spacebar_centroid_ab_gated_bootstrap.py | TwinScore_supplement FINAL (gated + bootstrapped phi) on SpaceBar's centroid A/B split, 2026-09-23. |
| build_fm01_integrated.py | FM01 downstream: batch-aware normalization (log1pCP10k, all cells together, batch_key='replicate' for HVG) -> PCA(50) -> scanorama + Harmony |
| build_fm01_qc_matrix.py | FM01 (WM989 + 1uM PLX) QC pipeline: singletCode doublet removal (criteria 1-4, no manual UMI pre-filter, no min_umi_cutoff set) + human MT-  |
| build_fm06_integrated.py | FM06 downstream: batch-aware normalization (log1pCP10k, all cells together, batch_key='replicate' for HVG) -> PCA(50) -> scanorama + Harmony |
| build_fm06_melanoma_ground_truth.py | FM06 melanoma-specific TF-target ground truth (ChIP-Atlas binding + KnockTF knockdown), for comparison with CollecTRI as the positive-edge s |
| build_fm06_qc_matrix.py | FM06 QC pipeline: singletCode doublet removal (criteria 1-4, no manual UMI pre-filter) + human MT- mito filter (<=21%) + gene-count floor/ce |
| build_fm08_integrated.py | FM08 downstream: batch-aware normalization (log1pCP10k, all cells together, batch_key='replicate' for HVG) -> PCA(50) -> scanorama + Harmony |
| build_fm08_qc_matrix.py | FM08 QC pipeline: singletCode doublet removal (criteria 1-4, no manual UMI pre-filter) + human MT- mito filter (<=26%) + gene-count floor/ce |
| build_hpsc_endoderm_chipatlas_candidates.py | Build the PDF's "Candidate pairs" set for hPSC_20260927 (endo_T0/endo_T1, hESC -> definitive-endoderm lineage-tracing): x (a Lambert-catalog |
| build_hpsc_endoderm_chipatlas_candidates_v2.py | Candidate-pair generation for hPSC_20260927, replicating yscher's real code EXACTLY (/gpfs/projects/b1255/yscher/Transcriptomic Distance/exp |
| build_hpsc_endoderm_qc_matrix.py | hPSC_20260927 (endo_T0/endo_T1, two-timepoint hESC->definitive-endoderm lineage-tracing data) QC pipeline: RNA-QC threshold (n_genes>=1000,  |
| build_hpsc_endoderm_qc_matrix_LEGACY_qcfirst.py | hPSC_20260927 (endo_T0/endo_T1, two-timepoint hESC->definitive-endoderm lineage-tracing data) QC pipeline: RNA-QC threshold (n_genes>=1000,  |
| build_hpsc_endoderm_top100_panel.py | Unfiltered 100-gene panel for hPSC_20260927: 40 most-measured TFs (Lambert catalogue, ranked by total detected expression across all QC-filt |
| build_hpsc_endoderm_twinfer_input.py | Builds the canonical TwINFER input (clone_id, cell_id, time_step, {gene}_mRNA) for hPSC_20260927 (endo_T0/endo_T1) from the QC+singletCode-f |
| build_hs054_integrated.py | HS054_Sot48h downstream: log1pCP10k normalization -> PCA(50). Single sample (no replicate/batch split, unlike FM06/FM08), so no batch-aware  |
| build_hs054_qc_matrix.py | HS054_Sot48h (PDAC Sotorasib 48h) QC pipeline: singletCode doublet removal (criteria 1-4, no manual UMI pre-filter) + human MT- mito filter  |
| build_twinscore_ablation_table.py | Rebuild the TwinScore_supplement summary + D/R/Wz ablation table across all 18 FM06+FM08 gene sets directly from the saved *_pair_terms.csv  |
| build_watermelon_qc_matrix.py | Watermelon (T47D naive/lag/late x 2 replicates) QC: singletCode on ALL six samples in one call (treated as one experiment; auto UMI cutoff - |
| build_watermelon_stage_integrated.py | Watermelon (T47D) per-stage integration + UMAP + replicate-separation metrics. |
| check_chipatlas_breast_coverage.py | How many of the Watermelon (T47D breast cancer) panel's TFs have ChIP-Atlas experiments in breast-cell-class lines, and how many of our own  |
| check_chipatlas_melanoma_coverage.py | How many of the FM01/FM06 (WM989 melanoma) panel's TFs have ChIP-Atlas experiments in melanoma-cell-class lines, and how many of our own pan |
| compare_juliaD_vs_python.py | FM06: what changes when PIDC is the Julia implementation (NetworkInference.jl) instead of the Python port. usage: compare_juliaD_vs_python.p |
| compare_julia_pidc_variants.py | FM06: three PIDC implementations, as (a) the D term of TwinFER and (b) the PIDC competitor. python = Python PIDC (helpers/pidc.py of yscher, |
| compare_pidc_julia_python.py | Compare the Julia PIDC (NetworkInference.jl) with the Python PIDC competitor (run_competitors_fatemap.py) on FM06. usage: compare_pidc_julia |
| compare_tf_source_ab.py | Same comparison as compare_twinscore_vs_competitors_ab.py, but ALSO with the universe restricted to pairs whose SOURCE gene (gene_1) is a Co |
| compare_twinscore_paper_vs_competitors.py | AUPRC of TwinScore-paper (TwinScore_LARRY_melanoma.pdf method: Stage I-III, jackknife-50, ranked by /z/) vs the five competitor methods, on  |
| compare_twinscore_vs_competitors_ab.py | AUPRC of TwinScore_supplement (gated bootstrap, A/B split) vs the competitor methods on the SAME cells, per gene set. |
| eval_fm06_vs_melanoma_gt.py | FM06: score the already-computed TwinFER (TwinScore_supplement, gated + bootstrap, A/B split) and competitor networks against the melanoma-s |
| eval_spacebar_centroid_ab_terms.py | Per-term evaluation of TwinScore_supplement (SpaceBar, centroid A/B) vs CollecTRI, next to the 5 competitors, over the FULL ordered-pair uni |
| eval_spacebar_singletime_terms.py | Evaluates the single-timepoint TwinScore (S/C/U/h, Stage I-III, apply_singletime_score_ spacebar.py) against CollecTRI, next to the 5 compet |
| expand_nanog_oct4_targets.py | NANOG and POU5F1(OCT4) are the only 2 TFs in the panel with BOTH ChIP-Atlas binding evidence AND orthogonal Perturb-seq perturbation evidenc |
| fatemap_integration_utils.py | Normalization + PCA + integration + marker-annotation helpers, shared by build_fm06_integrated.py / build_fm08_integrated.py. Mirrors larry_ |
| fatemap_qc_utils.py | Shared helpers for the FM06/FM08 QC pipeline, mirroring the filter/rescue logic in larry_pipeline/filtering/build_final_matrix_umi3.py, adap |
| final_fm06_tables.py | FM06 final tables: TwinFER (gated + bootstrap TwinScore_supplement, A/B split) vs PIDC and the other competitors. |
| julia_pidc_run.sh | Julia PIDC (NetworkInference.jl) for any expression matrix: package PIDC (symmetric) AND the directed variant. |
| null_calibration_fm06.py | Which null sd is right for the spec statistics? FM06, merged sample (A+B, one sample), one gene set. |
| pick_correlation_clean_fm06.py | FM06: rebuild the correlation gene sets with yscher's gene exclusions applied BEFORE the edge selection. |
| pick_correlation_melanoma_tfs_fm06.py | FM06: a correlation_high-style panel that uses ALL the melanoma ground-truth source TFs (MITF, SOX10, JUN, JUND, FOSL1, FOSL2, TEAD1-4). |
| pick_gene_sets_fatemap.py | Adapts larry_hematopoiesis_validation/pick_gene_sets.ipynb's recipe (CollecTRI edges, /rho/ slices, 15-TF/4-target panels -- "THE panel set, |
| pick_gene_sets_hs054.py | Runs pick_gene_sets_fatemap.py's CollecTRI panel-picking recipe on HS054_Sot48h. Single sample (no replicate split), so the "per-replicate m |
| pick_gene_sets_spacebar.py | 2026-09-23: SpaceBar version of pick_gene_sets_fatemap.py's correlation_{high,mid,low} recipe (CollecTRI edges with both ends in the detecti |
| pick_hpsc_endoderm_gene_panel.py | Shrink the raw ChIP-Atlas candidate-pair set (147 TFs x 8,742 targets, 162,251 edges -- computationally infeasible for infer_with_twinfer's  |
| prep_julia_D_fm06.py | Expression table for the Julia-PIDC D term: exactly the cells/genes the scorer's compute_D sees, i.e. replicate-B cells (t2_raw) of the A/B- |
| prep_julia_pidc_fm06.py | Write the expression file for the Julia PIDC (NetworkInference.jl): tab-separated, genes x cells, header = cell ids. Uses the SAME cells the |
| prep_julia_pidc_scaling.py | Expression tables for the Julia-PIDC scaling test (FM06, log1p CP10k of the QC counts, 8,000 randomly chosen QC cells, seed 0). Genes = the  |
| prep_pooled_input_fm06.py | Writes the pooled TwINFER input table twinfer_input_fm06_<gene_set>.csv (all clone-barcoded cells, clone size 2-50, log1p(CP10k) of the pane |
| prep_watermelon_stage_qc.py | Split the combined Watermelon QC'd counts matrix into per-stage qc_filtered folders that look like FM06/FM08's (finalized_data/Watermelon_{s |
| qc_stats_fatemap.py | Per-cell QC distributions (mito %, n_genes) for FM01/FM06/FM08 raw 10x matrices, plus the percentile at which FM06/FM08's chosen cutoffs sit |
| qc_stats_hs054.py | n_genes / mito% distribution for HS054_Sot48h (raw Cell Ranger filtered matrix, no QC filters applied yet) -- to pick sensible mito/gene-cou |
| qc_stats_watermelon.py | Per-sample n_genes / mito% distributions for Watermelon, and how many cells each proposed threshold removes -- to check whether a 2500-gene  |
| run_ab_infer_fm01.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_fm01_integrated_genesets.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_fm01_qc.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_fm06_integrated.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_fm08_integrated.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_hpsc_qc_singlet_first.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_watermelon_qc.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_build_watermelon_stage_integrated.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_ab_fm01_watermelon.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_fatemap.py | Ports preprocessing/run_competitor_methods.py (rho, ppcor, PIDC, GENIE3, GRNBoost2) to FM06/FM08, run in "ALL_GENES" mode (every panel gene  |
| run_competitors_fatemap_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_hpsc_endoderm.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_hpsc_endoderm_top100.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_hs054_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_competitors_spacebar.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_gs_competitors.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_gs_infer.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_gs_infer_gated.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_gs_score.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_fatemap.py | Builds the canonical TwINFER input (clone_id, cell_id, time_step, {gene}_mRNA) from FM06/FM08's un-integrated, batch-aware-normalized log1p( |
| run_infer_fatemap_ab_split.py | Exploratory check (2026-09-22, user request): FM06/FM08 replicate A and B are two wells of the SAME experiment, not a real time-course -- bu |
| run_infer_fatemap_ab_split.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_fatemap_ab_split_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_fatemap_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_fatemap_allpairs.py | 2026-09-22 user request: AUPRC should be computed over ALL gene pairs in a gene set, not just the subset that happened to pass infer_with_tw |
| run_infer_fatemap_allpairs.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_hpsc_endoderm.py | Runs infer_with_twinfer on hPSC_20260927 (endo_T0/endo_T1) as a genuine two-timepoint direction-stage run: t1=0 (endo_T0), t2=1 (endo_T1) -- |
| run_infer_hpsc_endoderm_top100.py | Runs the FULL infer_with_twinfer pipeline (all steps: Step1 existence, Step2 twin heterogeneity/divergence, Stage3 gated regulation, directi |
| run_infer_hpsc_endoderm_top100.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_hs054.py | Builds the canonical TwINFER input (clone_id, cell_id, time_step, {gene}_mRNA) from HS054_Sot48h's un-integrated, log1p(CP10k) expression (H |
| run_infer_hs054_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_spacebar.py | SpaceBar (spatial, lineage-barcoded melanoma tumor sections) TwINFER inference, on the overlap between SpaceBar's fixed 119-gene targeted pa |
| run_infer_spacebar_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_spacebar_allpairs.py | SpaceBar analog of run_infer_fatemap_allpairs.py: relaxed Stage 1 alpha + bypass_stage3_gate so every candidate pair gets a real, non-impute |
| run_infer_spacebar_allpairs.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_spacebar_cap50_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_infer_spacebar_centroid_ab.py | 2026-09-23: real (non-degenerate) A/B split for SpaceBar, built from within-clone SPATIAL STRUCTURE rather than a second replicate (SpaceBar |
| run_juliaD_directed_fm06.sh | Rescoring TwinScore_supplement with the DIRECTED Julia PIDC as the D term. Needs pidc_julia/D_<gs>/outFile_directed.txt (made by |
| run_juliaD_fm06.sh | Per gene set: (1) Julia PIDC on the replicate-B cells (the D input of the scorer), (2) Julia PIDC on the pooled competitor input (same 8,000 |
| run_pdf_final_hpsc_endoderm.py | PDF-exact final results for hPSC_20260927, Stage I survivors only (4,219 pairs): S (same-cell correlation), C (twin/sister correlation), U = |
| run_pidc_julia_fm06.sh | PIPE=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline   # [2026-09-30 replaced by env.sh variable] |
| run_pidc_julia_scaling.sh | Julia PIDC scaling test: nested panels of N = 65, 150, 300, 581 CollecTRI-source genes (>5% detection), 8,000 cells, one task per N. |
| run_pilot_tf150_infer.sh | PILOT (150 of the 581 expressed CollecTRI sources; gene set key tf_pilot150 in gene_sets_fm06.json): TwINFER inference, same settings as the |
| run_pilot_tf150_score.sh | PILOT scoring: (1) directed Julia PIDC on the replicate-B cells (D term), (2) final gated+bootstrap TwinScore_supplement with the PARALLEL b |
| run_qc_stats_fatemap.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_qc_stats_watermelon.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_spacebar_centroid_ab_competitors.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_spacebar_centroid_ab_supp_gated_bootstrap.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_spacebar_singletime_competitors.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_stage1_existence_hpsc_endoderm.py | Stage I existence funnel for hPSC_20260927, using the REAL package function exactly as yscher's own code calls it (confirmed by subagent rea |
| run_stage1_hpsc_endoderm.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_stage234_hpsc_endoderm.py | Stage II-IV (twin S/C/U existence-of-regulation, z_het; cross-time direction z_dagger/ gamma) for hPSC_20260927, restricted to EXACTLY the 4 |
| run_supp_gated_bootstrap_ab_fm01.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_supp_gated_bootstrap_fm06.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_supp_gated_bootstrap_hs054.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_twinscore_paper_all.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_twinscore_spec_fm06.sh | TwinScore pair procedure (spec of 2026-09-25) on FM06, ANALYTIC null only, scored against CollecTRI. |
| run_twinscore_supp_boolode_real.sh | TwinScore_supplement (gated + bootstrapped) on the BoolODE twin sims, for the networks given as arguments, all |
| run_twinscore_v2_merged_fm06.sh | New TwinScore (indicator product x s(z_reg)) on FM06 with A and B merged into one sample. usage: sbatch [--array=0-8] run_twinscore_v2_merge |
| run_wm_ab_infer.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_wm_prep_and_test.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| run_wm_supp_gated_bootstrap.sh | PYTHON=/home/gzu5140/.conda/envs/twinfer-code/bin/python3   # [2026-09-30 replaced by env.sh variable] |
| score_vs_chipatlas_breast.py | Rescore Watermelon TwinFER results (TwinScore-paper and TwinScore_supp gated-bootstrap) plus the same-universe competitors against the ChIP- |
| score_vs_chipatlas_melanoma.py | Rescore FM01 TwinFER results (TwinScore-paper all-z, and TwinScore_supp gated-bootstrap) plus the same-universe competitors against the ChIP |
| spec_pr_tables.py | Precision / recall of TwinScore-recent, TwinScore(phi) and the competitors at the SAME cut, FM06, one gene set. |
| stage1_existence_funnel.py | Cheap closed-form Stage I (existence) screen over the FULL raw ChIP-Atlas candidate-pair set (147 TFs x 8,742 targets, 162,251 edges) for hP |
| test_watermelon_ab_subsample.py | Test of the Watermelon per-stage A/B input with per-side subsampling (cap 100). Exits non-zero on any failure. Checks: (1) counts match the  |
| twinscore_paper.py | TwinScore exactly as specified in TwinScore_LARRY_melanoma.pdf (Sept 2026), the "melanoma" (one-sample) case -- generalized from twinscore_p |
| twinscore_paper_hs054.py | HS054_Sot48h: TwinScore exactly as specified in TwinScore_LARRY_melanoma.pdf (Sept 2026), the "melanoma" (one-sample) case -- Stages I-III,  |
| twinscore_spec_fm06.py | FM06: the TwinScore pair procedure (spec of 2026-09-25, as implemented in yscher's fanout_delta/spec_v3.py), scored against CollecTRI. |
| twinscore_spec_hs054.py | HS054_Sot48h: the most recent TwinScore-recent spec (2026-09-25, yscher's fanout_delta/spec_v3.py, as ported to this repo in twinscore_spec_ |
| twinscore_supp_helpers.py | Ports larry_hematopoiesis_validation/apply_twinscore_supplement_larry.py (TwinScore_supplement.pdf, 2026-09-17) to FM06/FM08/SpaceBar's sing |
| twinscore_v2_merged_fm06.py | Test of a new TwinScore on FM06 with replicates A and B merged into one sample. |
