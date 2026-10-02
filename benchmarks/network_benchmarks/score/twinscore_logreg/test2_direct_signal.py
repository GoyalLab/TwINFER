# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Direct test using TwINFER's ACTUAL cross-correlation statistics (not a proxy):
run infer_with_twinfer(verbose=True) on the same HSC and EMT replicates used in
the real benchmark, capture the per-pair "rho_delta_t1=..., rho_delta_t2=...,
d=..., z_d_het=..., z_d_div=..." diagnostic lines, and compare the magnitude of
the directional signal (|d|, |z_d_het|) on TRUE ground-truth edges between the
two networks. If HSC's true edges show systematically weaker |d|/|z_d_het|
than EMT's, that supports "HSC's population has less post-division divergence
signal for TwINFER's causal-direction test to read" as the real explanation
for why PEARSON beats TwINFER on HSC but not EMT.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import contextlib
import io
import re
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from twinfer.inference.infer import infer_with_twinfer

CASES = {
    "HSC": dict(
        twin=f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv',
        topo=f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/HSC.txt',
        n_genes=11, sim_time=6000,
    ),
    "EMT": dict(
        twin=f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv',
        topo=f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/EMT.txt',
        n_genes=17, sim_time=1000,
    ),
}

LINE_RE = re.compile(
    r"gene 1: (gene_\d+), gene 2: (gene_\d+), rho_delta_t1=([-\d.]+), rho_delta_t2=([-\d.]+), "
    r"d=([-\d.]+), z_d_het=([-\d.]+), z_d_div=([-\d.]+)"
)

for label, cfg in CASES.items():
    M = pd.read_csv(cfg["topo"], header=None).to_numpy()
    n = cfg["n_genes"]
    true_edges = {(f"gene_{i+1}", f"gene_{j+1}") for i in range(n) for j in range(n) if M[i, j] != 0}

    base_config = {
        "n_cells": 6000, "simulation_time_before_division": cfg["sim_time"],
        "twin_simulation_time_after_division": 48, "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": cfg["topo"],
        "param_csv": f'{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv',
        "rows_to_use": [[0] * n], "type": label,
    }

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        infer_with_twinfer(
            cfg["twin"], merge_to_multiple_states=False, base_config=base_config,
            t1=1, t2=20, match_sim_details=False, check_for_steady_state=False,
            seed=101010, n_cores=8, z_score_threshold_two_states=12,
            ranked_list=True, verbose=False,
        )
    out = buf.getvalue()

    rows = []
    for g1, g2, rt1, rt2, d, zh, zdv in LINE_RE.findall(out):
        rows.append(dict(g1=g1, g2=g2, d=float(d), z_d_het=float(zh), z_d_div=float(zdv),
                         is_true_edge=((g1, g2) in true_edges or (g2, g1) in true_edges)))
    df = pd.DataFrame(rows)
    print(f"\n===== {label}  (n_pairs={len(df)}, true directed edges={len(true_edges)}) =====")
    for grp_name, grp in df.groupby("is_true_edge"):
        print(f"  is_true_edge={grp_name}: n={len(grp)}  "
              f"mean|d|={grp['d'].abs().mean():.4f}  median|d|={grp['d'].abs().median():.4f}  "
              f"mean|z_d_het|={grp['z_d_het'].abs().mean():.3f}  "
              f"frac |z_d_het|>2.33={ (grp['z_d_het'].abs() > 2.33).mean():.2%}")
