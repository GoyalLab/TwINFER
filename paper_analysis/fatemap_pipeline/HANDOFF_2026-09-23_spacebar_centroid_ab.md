# HANDOFF 2026-09-23 (evening): SpaceBar centroid A/B split, TwinScore_supplement + competitors

Continues `HANDOFF_2026-09-23_ab_split_phi_investigation.md` (FM06/FM08 real A/B split; SpaceBar had no
usable A/B, only the pooled t1=t2 construction, where rho_cross_xy == rho_cross_yx exactly and gamma/phi are degenerate).
Goal here: build a genuine non-degenerate A/B split for SpaceBar, infer with TwINFER, score with the FINAL
TwinScore_supplement, run the 5 competitors on the same inputs, and see how each score term performs vs CollecTRI.

## STATUS AT HANDOFF
* RUNNING: SLURM array **7195291** (`sb_cab_supp_gb`, tasks 0/1/2 = correlation_high/mid/low), FINAL gated+bootstrap
  supplement, 6 h limit, submitted after the 21:42 fix to `w_rel`. Expected ~4-5 h. **correlation_mid took 5h07m on the
  previous (equivalent-cost) run, so it may hit the 6 h limit** -- if it does, resubmit task 1 alone with a longer limit.
* DONE: inference (job 7173791 step 1) and competitors (job 7173792), all 3 gene sets.
* SUPERSEDED: job 7173791's supplement step (ungated, clipped phi, older w). Its outputs (see "Superseded" below) must not be used.
* NEXT: when `analysis_data/spacebar/data/twinscore_supp_gated_bootstrap/run_params_spacebar_*_absplit_gated_bootstrap.json`
  appear, run `eval_spacebar_centroid_ab_terms.py` (already pointed at that folder; prints w, var_phi_pass, phi_noise_floor,
  n_gated per set; writes `spacebar_centroid_ab_term_performance_gated_bootstrap.csv`). A background waiter was armed
  but times out after 10 min -- just check `squeue`/`ls` manually.

## WHY A SPATIAL SPLIT, AND WHAT WAS TESTED (Sections 2+4; checked on all 5 sections where noted)
SpaceBar clone_id (bc_cluster) is static combinatorial founder tag (bc_001..bc_096), section-local, no replicate A/B. It carries
no generation/branch depth, so twin-vs-cousin cannot be read from the barcode.
* Clonal cells ARE more similar than random pairs in all 5 sections (Cohen d 0.28-0.38) but within-clone similarity is nearly
  flat (CV of pairwise distance ~0.09-0.10 everywhere) -> no clean twin/cousin substructure.
* Spatial distance vs transcriptional distance within clones: rho 0.10 (12 genes) / 0.15 (all 114 genes).
* Distance-matched (KNN) comparison: same-clone pairs sit a constant ~0.5-0.9 below different-clone pairs at every spatial distance
  (72-556) -> real clonal memory on top of generic spatial autocorrelation.
* Farthest-anchor 2-way split: no transcriptional separation (between/within ratio 1.02). Single nearest-vs-farthest PAIR per clone:
  only ~54% of clones ordered correctly (any clone size). Aggregating helps: **median split on each cell's distance to its
  clone's spatial centroid** -> central-vs-peripheral, p=6e-77, 61% of clones in the right direction.
* Degeneracy check on the chosen split: mean |rho_cross_xy - rho_cross_yx| = 0.0235 (max 0.10, 75% of pairs > 0.01, none exactly
  equal). Cross-side same-gene correlation 0.18 vs 0.054 for different genes.
* Caveat to keep in any write-up: this axis is SPATIAL position within one snapshot, not time. Any gamma/direction signal would
  reflect spatial/niche structure, not temporal causality. (Result: direction_term carries no signal, as with FM06.)

## CONSTRUCTION (code: `run_infer_spacebar_centroid_ab.py`, function `build_ab_split_with_full_panel`)
* Cells with a called clone in Section2 + Section4 (identical 114-gene panel; the per-section CSVs ARE the QC'd raw counts).
* Subsample clones > 60 cells to 60 BEFORE the split (per the SpaceBar "subsample, don't exclude" convention; a first version
  dropped 201 clones = 71% of cells -- fixed). Then per clone (n>=3) median-split by distance to spatial centroid:
  time_step 0 = central (A, t1), 1 = peripheral (B, t2). All 3,114 clones kept; 38,242 cells (A 20,004 / B 18,238); max 30/side.
  863 clones are 3-cell (2 central + 1 peripheral). Normalization: log1p(CP10k) over the target gene set.
* Largest clones are 1,000-3,000 cells spanning most of the section; some may be barcode collisions (bc_086, bc_027, bc_001 reused;
  single-barcode clones spanning the tissue). Not investigated further.

## GENE SETS
`pick_gene_sets_spacebar.py` = FM06 `pick_gene_sets_fatemap.py` correlation recipe run natively on the 114-gene panel
(88 genes pass 5% per-section detection floor; 276 CollecTRI edges in panel; top/mid/bottom 50 edges by |Spearman rho| ->
15 TFs x 4 targets). Output `analysis_data/spacebar/data/gene_sets_spacebar.json`: correlation_high 29 genes, mid 36, low 34.
No variability/detection sets (need genome-wide HVG/detection quantiles). **In the centroid scripts `correlation_high` now means
this native 29-gene set, NOT the old 12-gene FM06-intersection set** (still in `run_infer_spacebar.py`).

## SCRIPTS (all in `code/TwINFER/paper_analysis/fatemap_pipeline/`)
* `pick_gene_sets_spacebar.py`; `run_infer_spacebar_centroid_ab.py` (build + `infer_with_twinfer`, t1=0/t2=1, alpha 0.999, rescue,
  bypass_stage3_gate, 500 shuffles; writes input CSV, ranked_edges, z_dagger, z_reg_gated `*_centroid_ab_allpairs*`).
* **FINAL supplement:** `apply_twinscore_supplement_spacebar_centroid_ab_gated_bootstrap.py` -- reuses `run_gene_set` of
  `apply_twinscore_supplement_fatemap_gated_bootstrap.py` (the repo's declared FINAL variant; helpers in `twinscore_supp_helpers.py`)
  unchanged, monkeypatching only (a) the loader -> SpaceBar centroid split, (b) `compute_Wz` -> in-memory split-half Wz from the raw
  114-gene counts. Creates symlink `z_dagger_spacebar_{gs}_absplit_allpairs.json -> ..._centroid_ab_allpairs.json`.
  Launcher `run_spacebar_centroid_ab_supp_gated_bootstrap.sh` (8 cpu, 32 GB, 6 h, 2000 shuffles).
* Competitors: `run_competitors_fatemap.py SPACEBAR --gene-set {gs}_centroid_ab` via `run_spacebar_centroid_ab_competitors.sh`
  -> `analysis_data/spacebar/data/networks/{rho,ppcor,pidc,genie3,grnboost2}_{gs}_centroid_ab_allgenes.csv`.
* Eval: `eval_spacebar_centroid_ab_terms.py` (full ordered-pair universe, CollecTRI directed edges as positives; competitor pairs
  never scored = 0.0; NaN counted in `n_nan` and ranked last, never imputed).
* `apply_twinscore_supplement_spacebar_centroid_ab.py` + `run_spacebar_centroid_ab_infer_supp.sh`: SUPERSEDED (import
  modules that no longer exist; its supplement step used ungated/clipped phi). Kept, not deleted.

## FINAL FORMULA (per repo README in `analysis_data/fm06/data/twinscore_supp_gated_bootstrap/`)
TwinScore = w_rel*s(phi_x) + s(PAIR) + direction_term; PAIR = s(D) + gate_g*s(R) + gate_g*gate_v*s(Wz); phi_x = phi of source gene.
All noise scales are clone-label-permutation bootstraps (analytic null_sd only as NaN fallback). phi gated: gene keeps
rho_dagger_gg/sqrt(h_A*h_B) only if h_A > 2*nf1 and h_B > 2*nf2, else median phi of passing genes; no clip, no NaN.
**w_rel fix (21:42):** 1 - floor/var(phi), BOTH over passing genes only (placeholders excluded). Unit-checked with a stub
(w=0.91 = 1-0.01/0.111); `var_phi_pass` and `phi_noise_floor` are recorded in run_params json -> verify these on real output.

## RESULTS SO FAR
Competitors (VALID; AUPRC fold over random; prevalence 0.142 / 0.118 / 0.131; 812 / 1,260 / 1,122 ordered pairs):
| set | rho | ppcor | pidc | genie3 | grnboost2 |
|---|---|---|---|---|---|
| correlation_high | 1.00 | 1.08 | 1.17 | 1.04 | 1.06 |
| correlation_mid | 1.10 | 1.21 | 1.08 | 1.33 | 0.99 |
| correlation_low | 1.06 | 1.23 | 1.10 | 1.24 | 1.03 |
Everything is within ~1.0-1.6x of random on this small panel -- differences < ~0.1x are probably noise.

SUPERSEDED supplement run (ungated/clipped phi, old w; files `twinscore_supplement_spacebar_*_centroid_ab_allpairs_{pair,gene}_terms.csv`
directly in `analysis_data/spacebar/data/`, and `spacebar_centroid_ab_term_performance.csv`) -- for orientation only, redo with FINAL:
TwinScore 1.37x / 0.99x / 1.02x (high/mid/low); phi 1.61 / 1.09 / 1.13 (phi drove the correlation_high win); D 1.14 / 1.20 / 1.20;
|C| 1.23 / 1.29 / 1.19; Wz and R ~1.0-1.1; direction_term ~1.0 with P@k = 0. Gate weights then: w 0.59/0.71/0.78, gate_v 0.76/0/0.
Note gate_v = 0 in mid/low means Wz was off there. Expect these to change with the gated phi and corrected w.

## OPEN ITEMS
1. Collect 7195291, run eval, compare FINAL terms to the superseded numbers and to competitors; report per-term table + weights.
2. Also score the default package `twinScore` from `ranked_edges_spacebar_*_centroid_ab_allpairs.csv` (806/1,260/1,116 of 812/1,260/1,122
   ordered pairs present -- 6 missing in high, 6 in low; decide how to treat them without imputing).
3. Sensitivity: drop 3-cell clones (863; only 1 cell on side B); try other subsample caps.
4. Check big/multi-barcode clones for barcode collisions (bc_086 etc.).
5. Full 114-gene panel was NOT run (PIDC O(n^3) and permutation cost; 12 genes 20 s, 30 genes 59 s for PIDC; bootstrap ~20-28 s per
   permutation at 29-36 genes). Runtime of the supplement is dominated by single-core clone-permutation bootstraps (~4-5 h/set).
6. Scripts are all in the repo (nothing in /tmp except scratch logs `/tmp/supp_test*.log`).
