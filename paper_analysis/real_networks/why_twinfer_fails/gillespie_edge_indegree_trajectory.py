#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Extends 2.7b: does restricting to single-input edges (target has exactly one true regulator) reveal
the expected rise-then-plateau TF-target co-variation build-up, that gets diluted away when pooling all
true edges together (multi-regulator targets mixing in independent/competing dynamics)?

Same S(t) time-resolved cross-gene correlation as before, but true edges are now split into
single-input (target in-degree == 1) vs multi-input (target in-degree > 1), plus the non-edge baseline.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = f'{TWINFER_PROJECT_ROOT}'
OLD = f"{R}/simulation_data/twinfer_format"
RW = f"{R}/input_data/real_world_networks"


def old_genes(net):
    return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt")]


NETS = {
    "mCAD": dict(genes=["Fgf8", "Emx2", "Pax6", "Coup", "Sp8"], matrix=f"{OLD}/mCAD/interaction_matrix.txt",
                 file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv"),
    "GSD": dict(genes=old_genes("GSD"), matrix=f"{OLD}/GSD/interaction_matrix.txt",
                file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv"),
    "HSC": dict(genes=old_genes("HSC"), matrix=f"{OLD}/HSC/interaction_matrix.txt",
                file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv"),
    "VSC": dict(genes=old_genes("VSC"), matrix=f"{OLD}/VSC/interaction_matrix.txt",
                file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_12072026_005406_ncells_6000_VSC_2_0_2befd3f9.csv"),
    "B_cell_activation": dict(genes=["Ikaros", "PU_1", "Flk2", "IL_7R", "GATA_1", "E2A", "EBF", "C_EBPa", "PAX5", "Notch_1"],
                               matrix=f"{RW}/B_cell.txt",
                               file=f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv"),
    "EMT": dict(genes=[f"gene_{i+1}" for i in range(17)], matrix=f"{RW}/EMT.txt",
                file=f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv"),
    "Pluripotent": dict(genes=[f"gene_{i+1}" for i in range(36)], matrix=f"{RW}/Pluripotent.txt",
                        file=f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv"),
}

rows = []
n_single_edges = {}

for net, cfg in NETS.items():
    genes = cfg["genes"]
    native = genes[0].startswith("gene_")
    col = (lambda g: g) if native else (lambda g: f"gene_{genes.index(g)+1}")
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    n = len(genes)
    gene_cols = [f"{col(g)}_protein" for g in genes]

    indeg = (M != 0).sum(axis=0)  # rows=regulator, cols=target
    iu = np.triu_indices(n, 1)
    is_edge_dir = (M != 0) | (M.T != 0)
    single_pair = np.zeros((n, n), dtype=bool)
    multi_pair = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if M[i, j] != 0:  # i -> j
                if indeg[j] == 1:
                    single_pair[i, j] = single_pair[j, i] = True
                else:
                    multi_pair[i, j] = multi_pair[j, i] = True
    single_mask = single_pair[iu]
    multi_mask = multi_pair[iu] & ~single_mask
    nonedge_mask = ~(single_pair | multi_pair)[iu]
    n_single_edges[net] = int(single_mask.sum())
    n_multi_edges = int(multi_mask.sum())
    print(f"{net}: {single_mask.sum()} single-input edge-pairs, {n_multi_edges} multi-input edge-pairs, "
          f"{nonedge_mask.sum()} non-edges (of {len(iu[0])} total pairs)")

    cols = ["time_step", "clone_id", "replicate"] + gene_cols
    df = pd.read_csv(cfg["file"], usecols=cols)
    steps = sorted(df.time_step.unique())

    for t in steps:
        dt = df[df.time_step == t]
        a = dt[dt.replicate == 1].sort_values("clone_id")
        b = dt[dt.replicate == 2].sort_values("clone_id")
        Ra = np.column_stack([rankdata(a[c]) for c in gene_cols])
        Rb = np.column_stack([rankdata(b[c]) for c in gene_cols])
        pool = np.vstack([Ra, Rb])
        S = np.corrcoef(pool.T)
        s_iu = np.abs(S[iu])
        rows.append(dict(net=net, t=t,
                          mean_single=np.nanmean(s_iu[single_mask]) if single_mask.any() else np.nan,
                          mean_multi=np.nanmean(s_iu[multi_mask]) if multi_mask.any() else np.nan,
                          mean_nonedge=np.nanmean(s_iu[nonedge_mask])))

out = pd.DataFrame(rows)
OUTDIR = f"{R}/analysis_data/boolode_sims_real_networks/diagnostics"
out.to_csv(f"{OUTDIR}/gillespie_edge_indegree_trajectory_all_networks.csv", index=False)

print()
print("=== single-input vs multi-input vs non-edge trajectory shape ===")
for net in NETS:
    sub = out[out.net == net].sort_values("t")
    t = sub.t.to_numpy()
    for label, col in [("single", "mean_single"), ("multi", "mean_multi"), ("nonedge", "mean_nonedge")]:
        v = sub[col].to_numpy()
        if np.all(np.isnan(v)):
            print(f"{net:20s} {label:8s}  (no pairs)")
            continue
        peak_t = t[np.nanargmax(v)]
        print(f"{net:20s} {label:8s}  t0={v[0]:.3f} -> peak={np.nanmax(v):.3f} (@t={peak_t}) -> end={v[-1]:.3f}  "
              f"rise={np.nanmax(v)-v[0]:+.3f}")
