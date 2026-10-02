from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, glob, pandas as pd, numpy as np
from scipy.stats import spearmanr
R = f'{TWINFER_PROJECT_ROOT}'
RW = f"{R}/input_data/real_world_networks"
OLD = f"{R}/simulation_data/twinfer_format"

def old_genes(net):
    return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt") if l.strip()]

NETS = {
    "B_cell_activation": dict(
        genes=["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"],
        matrix=f"{RW}/B_cell.txt",
        file=f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv",
        target="Flk2", native=False),
    "EMT": dict(
        genes=[f"gene_{i+1}" for i in range(17)], matrix=f"{RW}/EMT.txt",
        file=f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv",
        target="gene_16", native=True),  # gene_16 = Zeb1
    "Pluripotent": dict(
        genes=[f"gene_{i+1}" for i in range(36)], matrix=f"{RW}/Pluripotent.txt",
        file=f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv",
        target="gene_30", native=True),  # gene_30 = ZNF398
    "GSD": dict(
        genes=old_genes("GSD"), matrix=f"{OLD}/GSD/interaction_matrix.txt",
        file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv",
        target="NR5A1", native=False),
    "HSC": dict(
        genes=old_genes("HSC"), matrix=f"{OLD}/HSC/interaction_matrix.txt",
        file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv",
        target="Pu1", target2="Gata2", native=False),
    "mCAD": dict(
        genes=old_genes("mCAD"), matrix=f"{OLD}/mCAD/interaction_matrix.txt",
        file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv",
        target="Emx2", native=False),
    "VSC": dict(
        genes=old_genes("VSC"), matrix=f"{OLD}/VSC/interaction_matrix.txt",
        file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_12072026_005406_ncells_6000_VSC_2_0_2befd3f9.csv",
        target="Nkx22", native=False),
}
T1, T2 = 1, 48

def check(net, cfg, target):
    genes_disp = cfg["genes"]; gi_disp = {g: i for i, g in enumerate(genes_disp)}
    col = lambda g: f"gene_{gi_disp[g]+1}_protein"
    genes = cfg["genes"]; M = np.loadtxt(cfg["matrix"], delimiter=",")
    gi = {g: i for i, g in enumerate(genes)}
    cols = ["cell_id", "time_step"] + [col(g) for g in genes]
    df = pd.read_csv(cfg["file"], usecols=cols)
    d1 = df[df.time_step == T1].drop_duplicates("cell_id").set_index("cell_id")
    d2 = df[df.time_step == T2].drop_duplicates("cell_id").set_index("cell_id")
    common = d1.index.intersection(d2.index)
    d1, d2 = d1.loc[common], d2.loc[common]
    print(f"\n=== {net}: target={target} (t1={T1}, t2={T2}, n_cells={len(common)}) ===")
    rows = []
    for reg in genes:
        is_edge = M[gi[reg], gi[target]] != 0
        r = spearmanr(d1[col(reg)], d2[col(target)])[0]
        rows.append((reg, is_edge, r))
    for reg, is_edge, r in sorted(rows, key=lambda x: -abs(x[2])):
        flag = "EDGE" if is_edge else "    "
        print(f"  {flag} {reg:10s}->{target:10s}  corr(t1,t2)={r:6.2f}")

for net, cfg in NETS.items():
    check(net, cfg, cfg["target"])
    if "target2" in cfg:
        check(net, cfg, cfg["target2"])
