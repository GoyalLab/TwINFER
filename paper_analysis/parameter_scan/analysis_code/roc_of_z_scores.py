"""ROC curves of z_rho, z_rho_s and z_rho_Delta in the parameter scan: scenario / multiple-state panels and two-state regulation.

Figure   : parameter_scan (used by heterogeneity_vs_regulation and causal_direction_inference).
Used for : plot_code/plot_roc_panels.py (roc_panels.pdf and roc_two_states_regulation.pdf).
Input    : analysis_data/twin_covariance_decomposition_z_scores.csv                (twin pairs)
           analysis_data/twin_covariance_decomposition_z_scores_random_pairs.csv   (control with the twin structure broken; made with
                                                                                   twin_covariance_decomposition_z_scores.py --break-twins)
           Columns used: topology, system (single_state / two_state), z_rho, z_rho_s, z_rho_Delta. Topologies: A_B (X Y, no regulation),
           A_to_B (X -> Y), A_and_B (X <-> Y), A_rep_B (X -| Y).
Panels   : every curve ranks the systems by |z|.
           regulation_vs_scenario_1         score z_rho    positives = single-state systems of A_and_B / A_rep_B / A_to_B (regulation, single state)
                                                           negatives = single-state A_B (scenario 1: no regulation, single state)
           multiple_states_vs_scenario_1    score z_rho    positives = two-state systems of each of the 4 topologies; negatives = single-state A_B
           multiple_states_vs_single_state  score z_rho_s  positives = two-state systems of the topology; negatives = single-state systems of
                                                           the same topology; computed on the twin pairs ("twins") and on the control ("random pairs")
           regulation_two_states            score z_rho_Delta  positives = two-state systems of A_and_B / A_rep_B / A_to_B (regulation, two states)
                                                           negatives = two-state A_B (no regulation, two states)
Output   : plot_data/roc_summary.csv  panel, curve, source, positives, negatives, AUROC
           plot_data/roc_curves.csv   panel, curve, source, fpr, tpr   (each curve thinned to at most --max-points points)
           These two files are the data of the plots (final_figure_data/parameter_scan/plot_data).
Usage    : python roc_of_z_scores.py [--twins CSV] [--random-pairs CSV] [--out-dir DIR (default plot_data)] [--max-points 2000]
Requires : numpy, pandas, scikit-learn.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REGULATED = ["A_and_B", "A_rep_B", "A_to_B"]
ALL_TOPOLOGIES = ["A_and_B", "A_to_B", "A_rep_B", "A_B"]


def curve_definitions():
    """(panel, curve, score, positive mask function, negative mask function) of every ROC curve."""
    single = lambda d: d.system == "single_state"
    two = lambda d: d.system == "two_state"
    definitions = []
    for topology in REGULATED:                                                                   # regulation vs scenario 1
        definitions.append(("regulation_vs_scenario_1", topology, "z_rho", lambda d, t=topology: single(d) & (d.topology == t), lambda d: single(d) & (d.topology == "A_B")))
    for topology in ALL_TOPOLOGIES:                                                              # multiple states vs scenario 1
        definitions.append(("multiple_states_vs_scenario_1", topology, "z_rho", lambda d, t=topology: two(d) & (d.topology == t), lambda d: single(d) & (d.topology == "A_B")))
    for topology in ["A_B", "A_to_B", "A_and_B", "A_rep_B"]:                                     # multiple states vs single state
        definitions.append(("multiple_states_vs_single_state", topology, "z_rho_s", lambda d, t=topology: two(d) & (d.topology == t), lambda d, t=topology: single(d) & (d.topology == t)))
    for topology in REGULATED:                                                                   # regulation among two-state systems
        definitions.append(("regulation_two_states", topology, "z_rho_Delta", lambda d, t=topology: two(d) & (d.topology == t), lambda d: two(d) & (d.topology == "A_B")))
    return definitions


def thin(x, y, max_points):
    """At most max_points points of a curve, evenly spaced along it, always keeping the first and last point."""
    if len(x) <= max_points:
        return x, y
    index = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
    return x[index], y[index]


if __name__ == "__main__":
    project = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
    figure_data = os.path.join(project, "final_figure_data", "parameter_scan")
    analysis_dir = os.path.join(figure_data, "analysis_data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", default=os.path.join(analysis_dir, "twin_covariance_decomposition_z_scores.csv"))
    ap.add_argument("--random-pairs", default=os.path.join(analysis_dir, "twin_covariance_decomposition_z_scores_random_pairs.csv"))
    ap.add_argument("--out-dir", default=os.path.join(figure_data, "plot_data"))
    ap.add_argument("--max-points", type=int, default=2000)
    args = ap.parse_args()

    sources = {"twins": pd.read_csv(args.twins)}                  # z-scores of the twin pairs; the random-pair control is added below if it exists
    if os.path.exists(args.random_pairs):
        sources["random_pairs"] = pd.read_csv(args.random_pairs)
    # [2026-10-24] z_rho_delta, q_rho_delta, rho_delta renamed to z_rho_Delta, q_rho_Delta, rho_Delta: csvs written before the rename are read with the new names
    sources = {name: frame.rename(columns=lambda c: c.replace("rho_delta", "rho_Delta")) for name, frame in sources.items()}
    summary, curves = [], []
    for panel, curve, score, positive_of, negative_of in curve_definitions():
        for source, data in sources.items():
            if source == "random_pairs" and panel != "multiple_states_vs_single_state":
                continue                                                # only this panel has the random-pair control
            usable = data[score].notna()                                # systems without a z-score (gene without variance) are left out
            positive, negative = positive_of(data) & usable, negative_of(data) & usable
            keep = positive | negative
            y, value = positive[keep].to_numpy(), data.loc[keep, score].abs().to_numpy()
            fpr, tpr, _ = roc_curve(y, value)                           # rank by |z|: a larger |z| means 'more positive'
            summary.append(dict(panel=panel, curve=curve, source=source, positives=int(positive.sum()), negatives=int(negative.sum()), AUROC=roc_auc_score(y, value)))
            fpr, tpr = thin(fpr, tpr, args.max_points)
            curves.append(pd.DataFrame(dict(panel=panel, curve=curve, source=source, fpr=fpr, tpr=tpr)))
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "roc_summary.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(args.out_dir, "roc_curves.csv"), index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
