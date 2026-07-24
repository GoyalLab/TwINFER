#!/usr/bin/env python
"""
Convert twin_similarity_sweep.py output (twin_final_states.csv) into the
input format infer_with_twinfer() expects
(TwINFER_function_scripts/infer_with_twinfer.py), for GSD, HSC, mCAD, VSC.

infer_with_twinfer() natively expects data from TwINFER's own stochastic
promoter-state simulator (4 variables/gene: active/inactive promoter,
mRNA, protein), with a full fixed-resolution post-division time grid and
matching simulation-parameter metadata (base_config's param_csv,
rows_to_use, twin_measurement_resolution, etc.) that our simpler BoolODE
2-variable (mRNA/protein) ODE model has no equivalent for. Per instruction,
this converter targets a MINIMAL, honest version of that input:

    - check_for_steady_state=False, match_sim_details=False at call time
      (must be set by the caller -- this script only builds the data
      files, it can't set those kwargs itself). This skips the internal
      consistency asserts that would otherwise require a complete,
      evenly-spaced time grid and a real parameter/connectivity match.
    - time_step uses our own raw, POST-BRANCH global step numbers
      directly (e.g. 400, 500, 600, 700, 799 for GSD/HSC; 250, 300, 400,
      499 for mCAD/VSC) -- no rebasing to "hours since division", no
      resampling to a finer grid. Pre-branch (trunk) rows are excluded:
      twins aren't a distinguishable pair before the branch point, so
      there's nothing for TwINFER's twin-comparison framework to use
      there.
    - replicate=1 <-> our twin_A, replicate=2 <-> our twin_B, and
      cell_id = clone_id + n_cells*(replicate-1), matching the exact
      convention used in convert_network_sweep_to_beeline.py (see its
      module docstring) and therefore in infer_with_twinfer() itself.
    - interaction_matrix.csv: built directly from GroundTruthNetwork.csv
      (the same one convert_twins_to_beeline.py produces/reuses) --
      signed integer adjacency matrix, rows=regulators, cols=targets,
      +1/-1/0, no header -- matching exactly what
      correlation_analysis_helpers.read_input_matrix() parses
      (np.loadtxt(..., dtype=int, delimiter=',')). Row/column order
      matches gene_order.txt written alongside it; pass that same order
      as gene_list_given= to infer_with_twinfer so the matrix and the
      simulation file's gene columns line up.
    - param_csv: infer_with_twinfer unconditionally does
      pd.read_csv(base_config['param_csv'], index_col=0) even when
      check_for_steady_state=False (only the *use* of that data, inside
      get_param_data(), is wrapped in a bare try/except that falls back to
      gene_params=None on any failure). So this needs to be SOME valid,
      loadable CSV, not real parameter values we don't have -- writes a
      minimal placeholder and flags it as such.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

REPLICATES_ROOT = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/boolode_sims_replicates"
GSD_GROUND_TRUTH = "/home/gzu5140/Keerthana_b1042/TwINFER/code/Beeline/inputs/example/GSD/GroundTruthNetwork.csv"
BOOLODE_DATA_DIR = "/home/gzu5140/Keerthana_b1042/TwINFER/code/BoolODE/data"
OUT_ROOT = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/twinfer_format"

NETWORKS = {
    # 'GSD':  dict(rule_file='GSD.txt'),
    # 'HSC':  dict(rule_file='HSC.txt'),
    # 'mCAD': dict(rule_file='mCAD.txt'),
    'VSC':  dict(rule_file='VSC.txt'),
}

N_CELLS = 6000  # twin clones/pairs


def find_available_replicates(network):
    # Layout: REPLICATES_ROOT/<NETWORK>/replicate_<i>/twin_final_states.csv
    reps = []
    for d in sorted(Path(REPLICATES_ROOT, network).glob('replicate_*')):
        idx = int(d.name.split('_')[1])
        if (d / "twin_final_states.csv").exists():
            reps.append(idx)
    return sorted(reps)


def load_states(network, rep_idx):
    path = os.path.join(REPLICATES_ROOT, network, f"replicate_{rep_idx}", "twin_final_states.csv")
    df = pd.read_csv(path)
    gene_cols = [c for c in df.columns if c not in ('pair_id', 'branch_tp', 'source', 'step', 't')]
    return df, gene_cols


def build_ground_truth_network(network):
    """Same logic as convert_twins_to_beeline.py -- kept duplicated (not
    imported) so this script has no import-order dependency on that one."""
    if network == 'GSD':
        return pd.read_csv(GSD_GROUND_TRUTH)
    rule_path = os.path.join(BOOLODE_DATA_DIR, NETWORKS[network]['rule_file'])
    bool_df = pd.read_csv(rule_path, sep='\t')
    genes = set(bool_df['Gene'].values)
    refnet = []
    for g in genes:
        rule = bool_df.loc[bool_df['Gene'] == g, 'Rule'].values[0]
        rhs = rule.replace('(', ' ').replace(')', ' ')
        tokens = rhs.split(' ')
        avoidthese = ['and', 'or', 'not', '']
        regulators = [t for t in tokens if t in genes and t not in avoidthese]
        whereisnot = tokens.index('not') if 'not' in tokens else None
        for r in regulators:
            ty = '+' if (whereisnot is None or tokens.index(r) < whereisnot) else '-'
            refnet.append({'Gene1': r, 'Gene2': g, 'Type': ty})
    return pd.DataFrame(refnet).drop_duplicates()


def write_interaction_matrix(gt_df, gene_order, out_dir):
    idx = {g: i for i, g in enumerate(gene_order)}
    n = len(gene_order)
    mat = np.zeros((n, n), dtype=int)
    for _, row in gt_df.iterrows():
        g1, g2, ty = row['Gene1'], row['Gene2'], row['Type']
        if g1 not in idx or g2 not in idx:
            continue  # shouldn't happen -- ground truth genes should match gene_order exactly
        mat[idx[g1], idx[g2]] = 1 if ty == '+' else -1

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "interaction_matrix.txt", mat, fmt='%d', delimiter=',')
    with open(out_dir / "gene_order.txt", 'w') as f:
        f.write('\n'.join(gene_order) + '\n')


def write_param_placeholder(out_dir):
    """infer_with_twinfer() does pd.read_csv(param_csv, index_col=0)
    unconditionally (get_param_data()'s actual USE of it is what's wrapped
    in try/except, not the load) -- needs to exist and be readable, but
    since check_for_steady_state=False means it's never meaningfully used,
    a minimal placeholder is enough. NOT real parameter values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholder = pd.DataFrame({'placeholder': [0]}, index=[0])
    placeholder.to_csv(out_dir / "param_placeholder.csv")


def build_simulation_file(df, gene_cols, branch_tp):
    """Gene columns are written as '<gene>_mRNA', not bare '<gene>' --
    correlation_analysis_functions.py accesses expression uniformly via
    f"{gene}_mRNA" everywhere in the inference path (matching TwINFER's
    native 4-variables-per-gene simulation format: gene_i_A/I/mRNA/protein,
    of which only _mRNA is used for the correlation-based inference).
    gene_order.txt / gene_list_given still hold the BARE gene names (used
    for matrix row/column labels and printed output) -- only the CSV
    column names need the suffix."""
    a = df[(df.source == 'twin_A') & (df.step >= branch_tp)]
    b = df[(df.source == 'twin_B') & (df.step >= branch_tp)]

    rows = []
    for _, row in a.iterrows():
        rows.append({
            'clone_id': int(row['pair_id']),
            'cell_id': int(row['pair_id']) + N_CELLS * 0,  # replicate=1 -> +0
            'time_step': int(row['step']),
            'replicate': 1,
            **{f"{g}_mRNA": row[g] for g in gene_cols},
        })
    for _, row in b.iterrows():
        rows.append({
            'clone_id': int(row['pair_id']),
            'cell_id': int(row['pair_id']) + N_CELLS * 1,  # replicate=2 -> +6000
            'time_step': int(row['step']),
            'replicate': 2,
            **{f"{g}_mRNA": row[g] for g in gene_cols},
        })

    return pd.DataFrame(rows)


def main():
    for network in NETWORKS:
        reps = find_available_replicates(network)
        print(f"=== {network}: {len(reps)} replicate(s) available: {reps} ===")
        if not reps:
            continue

        net_out_dir = Path(OUT_ROOT, network)
        gt_df = build_ground_truth_network(network)

        # gene_order fixed from the first available replicate's column
        # order; every replicate for a given network shares the same
        # gene set (same underlying model), so this is safe to compute once.
        _, gene_cols = load_states(network, reps[0])
        write_interaction_matrix(gt_df, gene_cols, net_out_dir)
        write_param_placeholder(net_out_dir)

        for rep_idx in reps:
            df, gene_cols_rep = load_states(network, rep_idx)
            assert gene_cols_rep == gene_cols, \
                f"{network} replicate_{rep_idx} has a different gene column order than replicate_{reps[0]}"
            branch_tp = int(df.loc[df.branch_tp != -1, 'branch_tp'].iloc[0])

            sim_df = build_simulation_file(df, gene_cols, branch_tp)
            out_path = net_out_dir / f"replicate_{rep_idx}_simulation.csv"
            sim_df.to_csv(out_path, index=False)

            available_steps = sorted(sim_df['time_step'].unique())
            print(f"  replicate_{rep_idx}: branch_tp={branch_tp}, "
                  f"available time_step values={available_steps}, "
                  f"{len(sim_df)} rows -> {out_path}")
        print()

    print("NOTE: call infer_with_twinfer with check_for_steady_state=False, "
          "match_sim_details=False, and gene_list_given=<gene_order.txt contents, "
          "in order>. base_config needs 'path_to_connectivity_matrix' -> "
          "interaction_matrix.csv, 'param_csv' -> param_placeholder.csv (not real "
          "parameter values -- only loaded, not meaningfully used when "
          "check_for_steady_state=False), 'n_cells' -> 6000, and 'rows_to_use' is "
          "irrelevant since match_sim_details=False skips the code path that reads it.")


if __name__ == '__main__':
    main()
