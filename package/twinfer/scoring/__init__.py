"""Scoring helpers shared by benchmarks and paper analyses (moved here 2026-09-30 after logic review; see REVIEW_LOG.md in the repo root of the cleanup).

analytic_zscores     shuffle-free ("analytic") approximations of the infer_with_twinfer z-scores
analytic_core        from-scratch analytic score table for simulated datasets with a known network (any gene count) + the fixed-weight full score
benchmark_truth      ground-truth edge sets / simulation file lookup for synthetic benchmarks
todo4v2              TODO4v2 scoring of twin_score_inputs against ground truth
pidc                 PIDC (partial information decomposition and context) used as the D term of TwinScore_supplement
twinscore_supplement small statistical helpers of TwinScore_supplement (panel z, signal share, CLR calibration, null SDs)
Rule: the package never imports from benchmarks/ or paper_analysis/.
"""
