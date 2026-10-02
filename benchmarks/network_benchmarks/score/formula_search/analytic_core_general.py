"""[2026-09-30 moved to twinfer.scoring.analytic_core after review; the original file is kept in _archive/package/scoring_helpers_original_locations/. This module only re-exports
the package module so that existing imports keep working.]"""
from twinfer.scoring.analytic_core import (  # noqa: F401
    N_RAND, SEED, W_DIR, W_FAN, W_DHET, s, full_table_disjoint, full_table_signed_general, compute_scores, score_topk,
)
