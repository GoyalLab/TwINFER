"""AUPRC of z_rho, z_rho_s and z_rho_delta in the parameter scan (precision-recall curves and areas).

Figure   : parameter_scan (used by heterogeneity_vs_regulation and causal_direction_inference).
Used for : plot_code/plot_auprc.py (z_rho_auprc.pdf, z_rho_s_auprc.pdf, z_rho_delta_auprc.pdf).
Input    : analysis_data/twin_covariance_decomposition_z_scores.csv (made by twin_covariance_decomposition_z_scores.py; one row per
           simulated system: topology A_B / A_to_B / A_and_B / A_rep_B, system single_state / two_state, and the three z-scores at 20 h).
Ranking  : every score is ranked by |z|. Systems without a z-score (a gene without variance) are left out of that score.
Classes (positive vs negative), one curve per topology:
           z_rho        positives = all systems of the topology (A_B: its two-state systems);  negatives = single-state A_B
           z_rho_s      positives = two-state systems of the topology;  negatives = single-state systems of the same topology
           z_rho_delta  positives = all systems of the topology (A_to_B, A_and_B, A_rep_B);  negatives = A_B (single and two-state)
Output   : plot_data/auprc_summary.csv  one row per score and curve: positives, negatives, base_rate, AUPRC, AUPRC_over_base
                                            (AUPRC divided by the base rate, 1.0 = chance)
           plot_data/auprc_curves.csv   score, curve, recall, precision; each curve is thinned to at most --max-points points
           These two files are the data of the plots (final_figure_data/parameter_scan/plot_data).
Usage    : python auprc_of_z_scores.py [--z-scores CSV] [--out-dir DIR (default plot_data)] [--max-points 2000]
Requires : numpy, pandas, scikit-learn.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

TOPOLOGIES = ["A_B", "A_to_B", "A_and_B", "A_rep_B"]
SCORES = ["z_rho", "z_rho_s", "z_rho_delta"]


def classes(score, topology, data):
    """Boolean masks (in_curve, positive) over the rows of `data` for one score and one topology."""
    single = data.system == "single_state"
    mine = data.topology == topology
    if score == "z_rho":
        negative = (data.topology == "A_B") & single
        positive = mine & ~negative
    elif score == "z_rho_s":
        positive, negative = mine & ~single, mine & single
    else:                                                       # z_rho_delta
        negative = data.topology == "A_B"
        positive = mine & ~negative                             # empty for A_B itself: no curve
    return positive | negative, positive


def thin(recall, precision, max_points):
    """At most max_points points of a precision-recall curve, evenly spaced along the curve, always keeping the first and last point."""
    if len(recall) <= max_points:
        return recall, precision
    index = np.unique(np.linspace(0, len(recall) - 1, max_points).astype(int))
    return recall[index], precision[index]


if __name__ == "__main__":
    project = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
    figure_data = os.path.join(project, "final_figure_data", "parameter_scan")
    analysis_dir = os.path.join(figure_data, "analysis_data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-scores", default=os.path.join(analysis_dir, "twin_covariance_decomposition_z_scores.csv"))
    ap.add_argument("--out-dir", default=os.path.join(figure_data, "plot_data"))
    ap.add_argument("--max-points", type=int, default=2000)
    args = ap.parse_args()

    data = pd.read_csv(args.z_scores)
    summary, curves = [], []
    for score in SCORES:
        usable = data[score].notna()
        for curve in TOPOLOGIES:
            keep, positive = classes(score, curve, data)
            keep = keep & usable
            if keep.sum() == 0 or positive[keep].sum() == 0 or (~positive[keep]).sum() == 0:
                continue                                        # the curve has no positives or no negatives for this score
            y, value = positive[keep].to_numpy(), data.loc[keep, score].abs().to_numpy()
            precision, recall, _ = precision_recall_curve(y, value)
            area, base = average_precision_score(y, value), y.mean()
            summary.append(dict(score=score, curve=curve, positives=int(y.sum()), negatives=int((~y).sum()), base_rate=base,
                                AUPRC=area, AUPRC_over_base=area / base))
            recall, precision = thin(recall, precision, args.max_points)
            curves.append(pd.DataFrame(dict(score=score, curve=curve, recall=recall, precision=precision)))
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "auprc_summary.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(args.out_dir, "auprc_curves.csv"), index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
