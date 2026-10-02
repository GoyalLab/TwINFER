from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
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

def existence_auprc_x(vals, M, iu):
    und = ((M != 0) | (M.T != 0))[iu].astype(int)
    if und.sum() in (0, len(und)): return np.nan
    return average_precision_score(und, np.nan_to_num(vals)) / und.mean()

for net, cfg in NETS.items():
    genes = cfg["genes"]; n = len(genes); native = genes[0].startswith("gene_")
    col = (lambda g: g) if native else (lambda g: f"gene_{genes.index(g)+1}")
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    cols = ["cell_id", "time_step", "clone_id"] + [f"{col(g)}_mRNA" for g in genes]
    df = pd.read_csv(cfg["file"], usecols=cols)
    last = df[df.time_step == df.time_step.max()]
    clone_sizes = last.clone_id.value_counts()
    pairs = clone_sizes[clone_sizes == 2].index
    sub = last[last.clone_id.isin(pairs)]
    a = sub.groupby("clone_id").nth(0).sort_values("clone_id")
    b = sub.groupby("clone_id").nth(1).sort_values("clone_id")

    gene_cols = [f"{col(g)}_mRNA" for g in genes]
    keep = [c for c in gene_cols if a[c].std()>1e-9 and b[c].std()>1e-9]
    gene_cols = keep; genes = [genes[gene_cols.index(f"{col(g)}_mRNA")] if False else g for g in genes]  # no-op keep name list length mismatch guard
    n = len(gene_cols)
    Ra = np.column_stack([rankdata(a[c]) for c in gene_cols])
    Rb = np.column_stack([rankdata(b[c]) for c in gene_cols])
    pool = np.vstack([Ra, Rb])  # same-cell matrix S: pool all twin cells as ordinary cells, one correlation matrix
    S = np.corrcoef(pool.T)
    # sister/twin matrix C: gene i in twin A vs gene j in twin B (symmetrized), the actual TwINFER definition
    C_raw = np.corrcoef(np.hstack([Ra, Rb]).T)[:n, n:]  # cross-block: rows=A-genes, cols=B-genes
    C = (C_raw + C_raw.T) / 2.0

    iu = np.triu_indices(n, 1)
    corr_S_C = np.corrcoef(S[iu], C[iu])[0, 1]

    lam = 1.0  # simplest possible zreg proxy: S - C (no shrinkage weighting, isolating the core question)
    zreg_like = S - C

    auprc_S = existence_auprc_x(np.abs(S)[iu], M, iu)
    auprc_zreg = existence_auprc_x(np.abs(zreg_like)[iu], M, iu)
    twin_diag_mean = np.mean([C[i, i] for i in range(n)])

    print(f"{net:20s}  twin(C diag) mean={twin_diag_mean:.3f}  corr(S,C) off-diag={corr_S_C:.3f}  "
          f"AUPRC-x |S|={auprc_S:.2f}  AUPRC-x |S-C|={auprc_zreg:.2f}  (change {auprc_zreg-auprc_S:+.2f})")
