from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
OLD = f"{R}/simulation_data/twinfer_format"
RW = f"{R}/input_data/real_world_networks"

def old_genes(net): return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt")]

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

def existence_auprc_x(corr_mat, M, genes):
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    iu = np.triu_indices(n, 1)
    und = ((M != 0) | (M.T != 0))[iu].astype(int)
    if und.sum() == 0 or und.sum() == len(und):
        return np.nan
    vals = np.abs(corr_mat)[iu]
    return average_precision_score(und, np.nan_to_num(vals)) / und.mean()

for net, cfg in NETS.items():
    genes = cfg["genes"]; n = len(genes); native = genes[0].startswith("gene_")
    col = (lambda g: g) if native else (lambda g: f"gene_{genes.index(g)+1}")
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    cols = ["cell_id", "time_step", "clone_id"] + [f"{col(g)}_protein" for g in genes]
    df = pd.read_csv(cfg["file"], usecols=cols)
    last = df[df.time_step == df.time_step.max()]
    clone_sizes = last.clone_id.value_counts()
    pairs = clone_sizes[clone_sizes == 2].index
    sub = last[last.clone_id.isin(pairs)]
    a = sub.groupby("clone_id").nth(0).sort_values("clone_id")
    b = sub.groupby("clone_id").nth(1).sort_values("clone_id")
    assert (a.clone_id.to_numpy() == b.clone_id.to_numpy()).all()

    gene_cols = [f"{col(g)}_protein" for g in genes]
    Ra = np.column_stack([rankdata(a[c]) for c in gene_cols])  # rank-transform (Spearman-safe)
    Rb = np.column_stack([rankdata(b[c]) for c in gene_cols])
    Ra_z = (Ra - Ra.mean(0)) / (Ra.std(0) + 1e-9)
    Rb_z = (Rb - Rb.mean(0)) / (Rb.std(0) + 1e-9)
    avg_z = (Ra_z + Rb_z) / 2.0  # shared "parent state" proxy, per twin pair

    pca = PCA(n_components=1).fit(avg_z)
    factor = pca.transform(avg_z)[:, 0]  # one clonal-identity score per twin pair
    var_explained = pca.explained_variance_ratio_[0]

    # regress the shared factor out of EACH gene's z-scored rank, separately for twin A and B copies
    resid_a = np.zeros_like(Ra_z); resid_b = np.zeros_like(Rb_z)
    for j in range(n):
        sa, ia = np.polyfit(factor, Ra_z[:, j], 1)
        sb, ib = np.polyfit(factor, Rb_z[:, j], 1)
        resid_a[:, j] = Ra_z[:, j] - (sa * factor + ia)
        resid_b[:, j] = Rb_z[:, j] - (sb * factor + ib)

    # existence AUPRC-x: RAW (pooled twin_A+twin_B as "cells") vs RESIDUALIZED
    raw_pool = np.vstack([Ra_z, Rb_z])
    resid_pool = np.vstack([resid_a, resid_b])
    raw_corr = np.corrcoef(raw_pool.T)
    resid_corr = np.corrcoef(resid_pool.T)
    auprc_raw = existence_auprc_x(raw_corr, M, genes)
    auprc_resid = existence_auprc_x(resid_corr, M, genes)

    # twin correlation of residuals, per gene, vs raw twin correlation
    raw_twin_r = [spearmanr(Ra_z[:, j], Rb_z[:, j])[0] for j in range(n)]
    resid_twin_r = [spearmanr(resid_a[:, j], resid_b[:, j])[0] for j in range(n)]

    print(f"\n=== {net}: PC1 of shared parent-state explains {var_explained:.1%} of cross-gene variance ===")
    print(f"  existence AUPRC-x-random: RAW={auprc_raw:.2f}  AFTER REMOVING CLONAL FACTOR={auprc_resid:.2f}  "
          f"(change: {auprc_resid-auprc_raw:+.2f})")
    print(f"  mean twin corr per gene:  RAW={np.mean(raw_twin_r):.3f}  RESIDUAL={np.mean(resid_twin_r):.3f}")
