#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Extends the mCAD-only same-cell-vs-sister test (handoff 2.7) to all 7 real networks, and adds a new
question: do TRUE regulatory edges show a rise-then-decay cross-gene correlation trajectory over time,
distinguishing them from non-edges' plain decay (both starting near-identical at t1, since twins are ~
identical right after division)?

Part A (per gene, single-gene autocorrelation): same_cell(t2) = corr(twin_A[t1], twin_A[t2]),
sister(t2) = corr(twin_A[t1], twin_B[t2]). Reports mean/max |gap| per network.

Part B (per gene PAIR, cross-gene): S(t) = same-cell cross-gene correlation (pool both twins) at each
raw time_step t (not referenced to t1) -- the ordinary S statistic, time-resolved. Track mean |S(t)| over
time separately for true-edge pairs vs non-edge pairs, per network. Look for a hump (max at some
intermediate t, decaying on both sides) vs monotonic decay from t=0.
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

part_a_rows = []
part_b_rows = []

for net, cfg in NETS.items():
    genes = cfg["genes"]
    native = genes[0].startswith("gene_")
    col = (lambda g: g) if native else (lambda g: f"gene_{genes.index(g)+1}")
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    gene_cols = [f"{col(g)}_mRNA" for g in genes]
    cols = ["time_step", "clone_id", "replicate"] + gene_cols
    df = pd.read_csv(cfg["file"], usecols=cols)
    steps = sorted(df.time_step.unique())
    t1 = steps[0]

    d1 = df[df.time_step == t1]
    A1 = d1[d1.replicate == 1].sort_values("clone_id").set_index("clone_id")
    B1 = d1[d1.replicate == 2].sort_values("clone_id").set_index("clone_id")
    common0 = A1.index.intersection(B1.index)

    # -------- Part A: single-gene same-cell vs sister --------
    gaps = []
    for t2 in steps[1:]:
        dt = df[df.time_step == t2]
        At2 = dt[dt.replicate == 1].sort_values("clone_id").set_index("clone_id")
        Bt2 = dt[dt.replicate == 2].sort_values("clone_id").set_index("clone_id")
        common = common0.intersection(At2.index).intersection(Bt2.index)
        a1 = A1.loc[common]; a2 = At2.loc[common]; b2 = Bt2.loc[common]
        for c in gene_cols:
            if a1[c].std() < 1e-9 or a2[c].std() < 1e-9 or b2[c].std() < 1e-9:
                continue
            same_cell = np.corrcoef(a1[c], a2[c])[0, 1]
            sister = np.corrcoef(a1[c], b2[c])[0, 1]
            gaps.append(abs(same_cell - sister))
    gaps = np.array(gaps)
    part_a_rows.append(dict(net=net, n_gene_t2=len(gaps), mean_abs_gap=gaps.mean(), max_abs_gap=gaps.max()))
    print(f"[A] {net:20s} mean|gap|={gaps.mean():.4f}  max|gap|={gaps.max():.4f}  (n={len(gaps)})")

    # -------- Part B: cross-gene pair trajectory, true edge vs non-edge --------
    n = len(genes)
    iu = np.triu_indices(n, 1)
    is_edge = ((M != 0) | (M.T != 0))[iu]
    for t in steps:
        dt = df[df.time_step == t]
        a = dt[dt.replicate == 1].sort_values("clone_id")
        b = dt[dt.replicate == 2].sort_values("clone_id")
        common = a.clone_id.to_numpy()
        assert np.array_equal(common, b.clone_id.to_numpy())
        # zero-variance genes (e.g. HSC's Fli1/Scl) get rankdata'd to a constant -> corrcoef yields NaN
        # only for entries involving that gene; nanmean below excludes them without dropping the timepoint
        Ra = np.column_stack([rankdata(a[c]) for c in gene_cols])
        Rb = np.column_stack([rankdata(b[c]) for c in gene_cols])
        pool = np.vstack([Ra, Rb])
        S = np.corrcoef(pool.T)
        s_iu = np.abs(S[iu])
        part_b_rows.append(dict(net=net, t=t, mean_abs_S_edge=np.nanmean(s_iu[is_edge]),
                                 mean_abs_S_nonedge=np.nanmean(s_iu[~is_edge])))

outA = pd.DataFrame(part_a_rows)
outB = pd.DataFrame(part_b_rows)
OUTDIR = f"{R}/analysis_data/boolode_sims_real_networks/diagnostics"
outA.to_csv(f"{OUTDIR}/gillespie_same_cell_vs_sister_all_networks_mRNA.csv", index=False)
outB.to_csv(f"{OUTDIR}/gillespie_edge_vs_nonedge_trajectory_all_networks_mRNA.csv", index=False)

print()
print("=== Part A summary ===")
print(outA.to_string(index=False))

print()
print("=== Part B: per-network trajectory shape ===")
for net in NETS:
    sub = outB[outB.net == net].sort_values("t")
    e = sub.mean_abs_S_edge.to_numpy()
    ne = sub.mean_abs_S_nonedge.to_numpy()
    t = sub.t.to_numpy()
    argmax_e = t[np.nanargmax(e)]
    argmax_ne = t[np.nanargmax(ne)]
    rise_e = np.nanmax(e) - e[0]
    rise_ne = np.nanmax(ne) - ne[0]
    print(f"{net:20s} edge: peak@t={argmax_e:>3} (t0={e[0]:.3f}->peak={np.nanmax(e):.3f}->end={e[-1]:.3f}, rise={rise_e:+.3f})  "
          f"non-edge: peak@t={argmax_ne:>3} (t0={ne[0]:.3f}->peak={np.nanmax(ne):.3f}->end={ne[-1]:.3f}, rise={rise_ne:+.3f})")
