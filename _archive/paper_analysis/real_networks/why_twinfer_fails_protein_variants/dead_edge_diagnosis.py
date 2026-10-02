from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
RW = f"{R}/input_data/real_world_networks"
OLD = f"{R}/simulation_data/twinfer_format"

def old_genes(net):
    return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt") if l.strip()]

NETS = {
    "mCAD": dict(genes=["Fgf8","Emx2","Pax6","Coup","Sp8"], matrix=f"{OLD}/mCAD/interaction_matrix.txt",
                 file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv"),
    "GSD": dict(genes=old_genes("GSD"), matrix=f"{OLD}/GSD/interaction_matrix.txt",
                file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv"),
    "HSC": dict(genes=old_genes("HSC"), matrix=f"{OLD}/HSC/interaction_matrix.txt",
                file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv"),
    "VSC": dict(genes=old_genes("VSC"), matrix=f"{OLD}/VSC/interaction_matrix.txt",
                file=f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_12072026_005406_ncells_6000_VSC_2_0_2befd3f9.csv"),
    "B_cell_activation": dict(genes=["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"],
                 matrix=f"{RW}/B_cell.txt",
                 file=f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv"),
    "EMT": dict(genes=[f"gene_{i+1}" for i in range(17)], matrix=f"{RW}/EMT.txt",
                file=f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv"),
    "Pluripotent": dict(genes=[f"gene_{i+1}" for i in range(36)], matrix=f"{RW}/Pluripotent.txt",
                file=f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv"),
}
CV_THRESH = 0.15
CORR_THRESH = 0.15

all_rows = []
for net, cfg in NETS.items():
    genes = cfg["genes"]; n = len(genes)
    gi = {g: i for i, g in enumerate(genes)}
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    native = genes[0].startswith("gene_")
    colname = (lambda g: g) if native else (lambda g: f"gene_{gi[g]+1}")
    cols = ["cell_id", "time_step"] + [f"{colname(g)}_protein" for g in genes]
    df = pd.read_csv(cfg["file"], usecols=cols)
    last = df[df.time_step == df.time_step.max()].drop_duplicates("cell_id")
    vals = {g: last[f"{colname(g)}_protein"].to_numpy(dtype=float) for g in genes}
    cv = {g: (np.std(vals[g]) / (np.mean(vals[g]) + 1e-9)) for g in genes}
    in_degree = {g: int((M[:, gi[g]] != 0).sum()) for g in genes}

    reg_idx, tgt_idx = np.where(M != 0)
    for ri, ti in zip(reg_idx, tgt_idx):
        reg, tgt = genes[ri], genes[ti]
        if reg == tgt:
            continue
        corr = spearmanr(vals[reg], vals[tgt])[0]
        reg_dead = cv[reg] < CV_THRESH
        tgt_dead = cv[tgt] < CV_THRESH
        if reg_dead and tgt_dead:
            cls = "both_constant"
        elif reg_dead:
            cls = "regulator_constant"
        elif tgt_dead:
            cls = "target_constant"
        elif abs(corr) >= CORR_THRESH:
            cls = "real_signal"
        else:
            cls = "masked_by_network"
        all_rows.append(dict(net=net, reg=reg, tgt=tgt, corr=corr, reg_cv=cv[reg], tgt_cv=cv[tgt],
                              tgt_in_degree=in_degree[tgt], reg_out_degree=int((M[gi[reg]] != 0).sum()),
                              classification=cls))

res = pd.DataFrame(all_rows)
res.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/dead_edge_diagnosis.csv", index=False)

print(f"Total true directed edges (excl. self-loops): {len(res)}\n")
print("=== Overall classification ===")
print((res.classification.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
print()
print("=== Classification by network ===")
print((pd.crosstab(res.net, res.classification, normalize="index") * 100).round(0))
print()
print("=== Does target in-degree predict 'masked_by_network'? ===")
print(res.groupby(pd.cut(res.tgt_in_degree, [0,1,2,4,8,100]))["classification"].apply(lambda s: (s=="masked_by_network").mean()).round(2))
print()
print("=== Example masked_by_network edges (both genes vary, but corr~0) ===")
print(res[res.classification=="masked_by_network"].sort_values("net").head(15)[["net","reg","tgt","corr","reg_cv","tgt_cv","tgt_in_degree"]].to_string(index=False))
print()
print("=== Example regulator_constant edges ===")
print(res[res.classification=="regulator_constant"].sort_values("net").head(10)[["net","reg","tgt","corr","reg_cv"]].to_string(index=False))
