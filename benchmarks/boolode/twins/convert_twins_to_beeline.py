#!/usr/bin/env python
"""
Convert twin_similarity_sweep.py output (twin_final_states.csv, with its
multi-timepoint trunk/twin_A/twin_B structure) into BEELINE's expected
input format, for GSD, HSC, mCAD, VSC, matching the existing TwINFER
convention seen in
Beeline/inputs/network_sweep_final/grn_n6_e5_pos100_density_rep0/:

    Beeline/inputs/<NETWORK>/
        GroundTruthNetwork.csv              -- one shared file, all simreps
        simrep<i>_twin_paired/
            ExpressionData.csv
            PseudoTime.csv
        simrep<i>_spread/
            ExpressionData.csv
            PseudoTime.csv

one simrep<i> per available replicate (replicate_<i> under
simulation_data/boolode_sims_replicates/), processed independently.

Two sampling schemes per (network, replicate), both drawing from the same
6000 twin pairs, partitioned into three groups of 1500/1500/3000 (fixed
random partition, reproducible via PARTITION_SEED):

  twin_paired:
    - group 1 (1500 pairs): both twins sampled at t_early
    - group 2 (1500 pairs): both twins sampled at t_final
    - group 3 (3000 pairs): one twin at t_early, the other at t_final
    -> 12000 cells total, every pair used exactly once.

  spread:
    - every one of the 12000 twin instances (2 per pair x 6000 pairs)
      independently sampled at ONE uniformly-random timepoint, drawn from
      that instance's own available range: trunk snapshots for steps
      before its branch point (twin == trunk pre-branch, by construction),
      or that specific twin's own snapshots from the branch point onward.
    -> 12000 cells total, spread across the whole simulation window.

t_early/t_final per network are "first saved step strictly after the
branch point" / "final saved step", read directly from the data rather
than hardcoded -- GSD/HSC (simulation_time=8, 800 steps) and mCAD/VSC
(simulation_time=5, 500 steps) have different step grids, so this
resolves to step 500/799 for GSD/HSC and step 300/499 for mCAD/VSC (the
nearest actually-saved step to the exact proportional equivalent, ~312,
which isn't itself a saved snapshot for mCAD/VSC).

Cell naming and PseudoTime.csv format match the existing convention
exactly: columns/rows are named 'cell<i>_t<step>' (i = sequential index
within the file, step = the literal raw step number that cell was sampled
at), and PseudoTime.csv has a single data column 'PseudoTime1' holding
that same literal step number -- not a normalized 0-1 pseudotime, and not
encoding pair/twin identity in the name (matching how the existing
examples don't expose that either).

GroundTruthNetwork.csv: reused as-is for GSD (already present in
Beeline/inputs/example/GSD/), generated fresh for HSC/mCAD/VSC from their
rule files using the exact same sign-extraction algorithm BoolODE's own
utils.generateInputFiles() uses (activator if a regulator token appears
before the rule's first 'not', repressor if after).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import numpy as np
import pandas as pd
from pathlib import Path

# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] REPLICATES_ROOT = f'{TWINFER_PROJECT_ROOT}/code/BoolODE/twins/output/_replicates_view'
REPLICATES_ROOT = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/twins/output/_replicates_view'
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] BOOLODE_DATA_DIR = f'{TWINFER_PROJECT_ROOT}/code/BoolODE/data'
BOOLODE_DATA_DIR = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/model_inputs'
GSD_GROUND_TRUTH = f'{TWINFER_PROJECT_ROOT}/code/Beeline/inputs/example/GSD/GroundTruthNetwork.csv'
OUT_ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/boolode_linear_gene_sim/beeline_format'

NETWORKS = {
    # 'GSD':  dict(rule_file='GSD.txt'),
    # 'HSC':  dict(rule_file='HSC.txt'),
    # 'mCAD': dict(rule_file='mCAD.txt'),
    # 'VSC':  dict(rule_file='VSC.txt'),
    'dyn-linear': dict(rule_file='dyn-linear.txt'),
}

N_GROUP1 = 1500  # both twins @ t_early
N_GROUP2 = 1500  # both twins @ t_final
# remaining pairs (3000 of 6000) -> one twin @ t_early, other @ t_final
PARTITION_SEED = 42
SPREAD_SEED_BASE = 1000  # + replicate index, so each replicate's spread draw differs


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


def get_early_final_steps(df):
    branch_tp = int(df.loc[df.branch_tp != -1, 'branch_tp'].iloc[0])
    twin_steps = sorted(df.loc[df.source == 'twin_A', 'step'].unique())
    t_early = min(s for s in twin_steps if s > branch_tp)
    t_final = max(twin_steps)
    return branch_tp, t_early, t_final


def partition_pairs(pair_ids, seed=PARTITION_SEED):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(pair_ids)
    group1 = shuffled[:N_GROUP1]
    group2 = shuffled[N_GROUP1:N_GROUP1 + N_GROUP2]
    group3 = shuffled[N_GROUP1 + N_GROUP2:]
    return group1, group2, group3


def write_dataset(expr_rows, cell_labels, cell_steps, gene_cols, out_dir):
    """expr_rows: list of 1D arrays (one per cell), aligned with gene_cols.
    Names cells 'cell<i>_t<step>' sequentially, writes ExpressionData.csv
    (genes x cells) and PseudoTime.csv (single 'PseudoTime1' column = the
    literal step number), matching the existing TwINFER/BEELINE convention."""
    cell_names = [f"cell{i}_t{step}" for i, step in enumerate(cell_steps)]
    expr = pd.DataFrame(np.column_stack(expr_rows), index=gene_cols, columns=cell_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    expr.to_csv(out_dir / "ExpressionData.csv")

    pt = pd.DataFrame({'PseudoTime1': cell_steps}, index=cell_names)
    pt.to_csv(out_dir / "PseudoTime.csv")


def build_twin_paired(df, gene_cols, group1, group2, group3, t_early, t_final):
    a = df[df.source == 'twin_A'].set_index(['pair_id', 'step'])
    b = df[df.source == 'twin_B'].set_index(['pair_id', 'step'])

    expr_rows, cell_steps = [], []

    def add(pair_id, twin_df, step):
        row = twin_df.loc[(pair_id, step)]
        expr_rows.append(row[gene_cols].values.astype(float))
        cell_steps.append(int(step))

    for pid in group1:
        add(pid, a, t_early); add(pid, b, t_early)
    for pid in group2:
        add(pid, a, t_final); add(pid, b, t_final)
    for pid in group3:
        add(pid, a, t_early); add(pid, b, t_final)

    return expr_rows, cell_steps


def build_spread(df, gene_cols, branch_tp, pair_ids, seed):
    rng = np.random.RandomState(seed)
    trunk = df[df.source == 'trunk'].set_index(['pair_id', 'step'])
    a = df[df.source == 'twin_A'].set_index(['pair_id', 'step'])
    b = df[df.source == 'twin_B'].set_index(['pair_id', 'step'])

    trunk_steps = sorted(df.loc[df.source == 'trunk', 'step'].unique())
    pre_branch_steps = [s for s in trunk_steps if s < branch_tp]
    twin_steps = sorted(df.loc[df.source == 'twin_A', 'step'].unique())
    pool = pre_branch_steps + twin_steps

    expr_rows, cell_steps = [], []
    for pid in pair_ids:
        for twin_df in (a, b):
            step = rng.choice(pool)
            row = trunk.loc[(pid, step)] if step < branch_tp else twin_df.loc[(pid, step)]
            expr_rows.append(row[gene_cols].values.astype(float))
            cell_steps.append(int(step))

    return expr_rows, cell_steps


def build_ground_truth_network(network):
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


def main():
    for network in NETWORKS:
        reps = find_available_replicates(network)
        print(f"=== {network}: {len(reps)} replicate(s) available: {reps} ===")
        if not reps:
            continue

        net_out_dir = Path(OUT_ROOT, network)
        net_out_dir.mkdir(parents=True, exist_ok=True)
        gt_df = build_ground_truth_network(network)
        gt_df.to_csv(net_out_dir / "GroundTruthNetwork.csv", index=False)

        for rep_idx in reps:
            df, gene_cols = load_states(network, rep_idx)
            branch_tp, t_early, t_final = get_early_final_steps(df)
            pair_ids = sorted(df.pair_id.unique())
            group1, group2, group3 = partition_pairs(pair_ids)

            expr_rows, cell_steps = build_twin_paired(df, gene_cols, group1, group2, group3, t_early, t_final)
            write_dataset(expr_rows, None, cell_steps, gene_cols,
                         net_out_dir / f"simrep{rep_idx}_twin_paired")

            expr_rows, cell_steps = build_spread(df, gene_cols, branch_tp, pair_ids,
                                                  seed=SPREAD_SEED_BASE + rep_idx)
            write_dataset(expr_rows, None, cell_steps, gene_cols,
                         net_out_dir / f"simrep{rep_idx}_spread")

            print(f"  replicate_{rep_idx}: branch_tp={branch_tp} t_early={t_early} t_final={t_final} "
                  f"-> simrep{rep_idx}_twin_paired, simrep{rep_idx}_spread (12000 cells each)")
        print()


if __name__ == '__main__':
    main()
