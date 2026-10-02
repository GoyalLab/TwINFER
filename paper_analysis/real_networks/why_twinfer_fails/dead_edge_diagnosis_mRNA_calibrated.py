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

MASK_PERCENTILE = 75  # a true edge is "masked" only if |corr| <= this percentile of NON-EDGE |corr| in the same net
DISP_THRESH = 2.0      # a gene is "constant" only if index of dispersion (var/mean) <= this -- Poisson noise alone gives 1.0

all_rows = []
for net, cfg in NETS.items():
    genes = cfg["genes"]; n = len(genes)
    gi = {g: i for i, g in enumerate(genes)}
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    native = genes[0].startswith("gene_")
    colname = (lambda g: g) if native else (lambda g: f"gene_{gi[g]+1}")
    cols = ["cell_id", "time_step"] + [f"{colname(g)}_mRNA" for g in genes]
    df = pd.read_csv(cfg["file"], usecols=cols)
    last = df[df.time_step == df.time_step.max()].drop_duplicates("cell_id")
    vals = {g: last[f"{colname(g)}_mRNA"].to_numpy(dtype=float) for g in genes}
    mean_count = {g: np.mean(vals[g]) for g in genes}
    var_count = {g: np.var(vals[g]) for g in genes}
    cv = {g: (np.std(vals[g]) / (mean_count[g] + 1e-9)) for g in genes}
    dispersion = {g: var_count[g] / max(mean_count[g], 1e-6) for g in genes}  # Poisson noise alone -> 1.0
    in_degree = {g: int((M[:, gi[g]] != 0).sum()) for g in genes}

    # full pairwise |corr| matrix (all ordered pairs, excluding self) to build this network's own non-edge null
    all_corr = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            all_corr[(genes[i], genes[j])] = abs(spearmanr(vals[genes[i]], vals[genes[j]])[0])
    is_true = {(genes[i], genes[j]): (M[i, j] != 0) for i in range(n) for j in range(n) if i != j}
    nonedge_vals = [v for k, v in all_corr.items() if not is_true[k] and np.isfinite(v)]
    mask_thresh = np.percentile(nonedge_vals, MASK_PERCENTILE) if len(nonedge_vals) >= 5 else 0.15

    reg_idx, tgt_idx = np.where(M != 0)
    for ri, ti in zip(reg_idx, tgt_idx):
        reg, tgt = genes[ri], genes[ti]
        if reg == tgt:
            continue
        corr = all_corr[(reg, tgt)]
        reg_dead = dispersion[reg] <= DISP_THRESH
        tgt_dead = dispersion[tgt] <= DISP_THRESH
        if reg_dead and tgt_dead:
            cls = "both_constant"
        elif reg_dead:
            cls = "regulator_constant"
        elif tgt_dead:
            cls = "target_constant"
        elif corr > mask_thresh:
            cls = "real_signal"
        else:
            cls = "masked_by_network"
        all_rows.append(dict(net=net, reg=reg, tgt=tgt, corr=corr, reg_dispersion=dispersion[reg],
                              tgt_dispersion=dispersion[tgt], net_mask_thresh=mask_thresh,
                              tgt_in_degree=in_degree[tgt], classification=cls))

res = pd.DataFrame(all_rows)
res.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/dead_edge_diagnosis_mRNA_calibrated.csv', index=False)

print(f"Total true directed edges: {len(res)}\n")
print("=== Overall classification (relative/calibrated thresholds) ===")
print((res.classification.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
print()
print("=== By network ===")
print((pd.crosstab(res.net, res.classification, normalize="index") * 100).round(0))
print()
print("=== Does target in-degree predict 'masked_by_network'? ===")
print(res.groupby(pd.cut(res.tgt_in_degree, [0,1,2,4,8,100]))["classification"].apply(lambda s: (s=="masked_by_network").mean()).round(2))
print()
print("=== per-network mask threshold used (75th pct of non-edge |corr|) ===")
print(res.groupby("net").net_mask_thresh.first().round(4))
print()
print("=== mean |corr| by classification ===")
print(res.groupby("classification")["corr"].mean().round(3))
