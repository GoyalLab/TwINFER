#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Time-resolved S(t) vs C(t) for every single-input edge (from 2.7c) across all 7 real networks --
does the clean divergence seen in the isolated 2-gene A_to_B toy case (S flat, C decaying, S-C growing)
survive in real networks, or does network complexity (multi-regulator targets, feedback, hub structure)
wash it out, consistent with 2.1's finding that corr(S,C) ~ 0.98-1.0 pooled over all edges?
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

all_rows = []
for net, cfg in NETS.items():
    genes = cfg["genes"]
    native = genes[0].startswith("gene_")
    col = (lambda g: g) if native else (lambda g: f"gene_{genes.index(g)+1}")
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    n = len(genes)
    indeg = (M != 0).sum(axis=0)
    single_edges = [(i, j) for i in range(n) for j in range(n) if M[i, j] != 0 and indeg[j] == 1]
    if not single_edges:
        print(f"{net}: no single-input edges, skipping")
        continue
    needed_genes = sorted({i for e in single_edges for i in e})
    gene_cols = {i: f"{col(genes[i])}_protein" for i in needed_genes}
    cols = ["time_step", "clone_id", "replicate"] + list(gene_cols.values())
    df = pd.read_csv(cfg["file"], usecols=cols)
    steps = sorted(df.time_step.unique())

    for (i, j) in single_edges:
        reg, tgt = genes[i], genes[j]
        cr, ct = gene_cols[i], gene_cols[j]
        for t in steps:
            dt = df[df.time_step == t]
            A = dt[dt.replicate == 1].sort_values("clone_id")
            B = dt[dt.replicate == 2].sort_values("clone_id")
            ra_r, ra_t = rankdata(A[cr]), rankdata(A[ct])
            rb_r, rb_t = rankdata(B[cr]), rankdata(B[ct])
            S = np.corrcoef(np.concatenate([ra_r, rb_r]), np.concatenate([ra_t, rb_t]))[0, 1]
            C1 = np.corrcoef(ra_r, rb_t)[0, 1]
            C2 = np.corrcoef(rb_r, ra_t)[0, 1]
            C = (C1 + C2) / 2
            all_rows.append(dict(net=net, reg=reg, tgt=tgt, t=t, S=S, C=C))

out = pd.DataFrame(all_rows)
OUTDIR = f"{R}/analysis_data/boolode_sims_real_networks/diagnostics"
out.to_csv(f"{OUTDIR}/gillespie_single_input_S_vs_C.csv", index=False)

print("=== single-input edge: S(t) vs C(t) shape ===")
for (net, reg, tgt), sub in out.groupby(["net", "reg", "tgt"]):
    sub = sub.sort_values("t")
    S, C, t = sub.S.to_numpy(), sub.C.to_numpy(), sub.t.to_numpy()
    sc_gap = S - C
    print(f"{net:20s} {reg:>10s}->{tgt:<10s}  S: {S[0]:.3f}->{S[-1]:.3f}  C: {C[0]:.3f}->{C[-1]:.3f}  "
          f"S-C: {sc_gap[0]:+.3f}->{sc_gap[-1]:+.3f}  (max S-C={sc_gap.max():+.3f} @t={t[np.argmax(sc_gap)]})")
