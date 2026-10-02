# melanoma_gt_analysis

_Auto-generated 2026-09-30 from file docstrings/headers during the cleanup; edit freely. Files marked UNREVIEWED were rescued and not yet logic-reviewed (see RESCUE_MAP.tsv, REVIEW_LOG.md)._

| File | Summary |
|---|---|
| ab_gamma_merged.py | TwinFER with the direction term but the merged sample everywhere except the cross-sample step. Steps I-IV / R: merged A+B sample (clone = ba |
| build_chipatlas_cutaneous_gt.py | ChIP-Atlas ground truth from cutaneous melanoma cell lines only, for EVERY TF antigen that has data there. |
| build_chipatlas_pancreatic_gt.py | ChIP-Atlas ground truth from pancreatic cancer cell lines, for HS054 -- mirrors build_chipatlas_cutaneous_gt.py (melanoma) exactly, only LIN |
| det.py |  |
| det__chpad_triage.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| features_rand.py | What separates detected/undetected and true/false ChIP edges for TwinFER (spec, merged, no covariate regression) on the random-target panel. |
| hdrs.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| iso.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
| iso_mel.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
| iso_rand.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
| lam.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
| lam_cov.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
| merged_h.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
| score_all_panels_newgt.py | Score TwinFER (spec, merged sample, analytic null, no covariate regression; R for called pairs and ungated R) and the competitors on every p |
| score_mel.py |  |
| score_mel__chpad_triage.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| score_rand.py |  |
| scores_by_cov.py |  |
| scores_by_cov__chpad_triage.py (UNREVIEWED) | [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv] |
| topk_ungated.py | TwinFER without the calling gate: rank ALL source-TF pairs by ungated R (or /z*/, /rho/), top-k with k = number of true ChIP edges. Spec, me |
| zeros.py | sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline") |
