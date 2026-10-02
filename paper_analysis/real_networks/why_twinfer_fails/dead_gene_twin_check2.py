from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
GSD_GENES = [l.strip() for l in open(f"{R}/simulation_data/twinfer_format/GSD/gene_order.txt")]
HSC_GENES = [l.strip() for l in open(f"{R}/simulation_data/twinfer_format/HSC/gene_order.txt")]

DEAD = {
    "GSD": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv",
            ["UGR","DHH","RSPO1","FOXL2","WT1pKTS","NR5A1","SRY"], GSD_GENES),
    "HSC": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv",
            ["Gata1","Fog1","Fli1","Scl","EgrNab"], HSC_GENES),
    "B_cell_activation": (f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv",
            ["IL_7R","E2A","EBF","PAX5"], ["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"]),
    "EMT": (f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv",
            ["gene_3","gene_9","gene_10","gene_13","gene_15","gene_16","gene_17"], [f"gene_{i+1}" for i in range(17)]),
    "Pluripotent": (f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv",
            ["gene_2","gene_3","gene_4","gene_6","gene_7","gene_8","gene_10","gene_11","gene_12","gene_14","gene_17","gene_21","gene_26","gene_30","gene_35","gene_36"],
            [f"gene_{i+1}" for i in range(36)]),
}

def colname(g, genes, native):
    return g if native else f"gene_{genes.index(g)+1}"

for net in DEAD:
    f, dead_genes, genes = DEAD[net]
    native = genes[0].startswith("gene_")
    cols_needed = ["cell_id", "time_step", "clone_id"] + [f"{colname(g, genes, native)}_protein" for g in dead_genes]
    df = pd.read_csv(f, usecols=cols_needed)
    last = df[df.time_step == df.time_step.max()]
    clone_sizes = last.clone_id.value_counts()
    pairs = clone_sizes[clone_sizes == 2].index
    sub = last[last.clone_id.isin(pairs)].sort_values(["clone_id", "cell_id"])
    a = sub.groupby("clone_id").nth(0).sort_values("clone_id"); b = sub.groupby("clone_id").nth(1).sort_values("clone_id")
    assert (a.clone_id.to_numpy() == b.clone_id.to_numpy()).all(), "twin_A/twin_B clone_id misalignment"
    print(f"\n=== {net}: {len(a)} twin pairs ===")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(b))
    for g in dead_genes:
        c = colname(g, genes, native) + "_protein"
        tw = pearsonr(a[c], b[c])[0]
        rn = pearsonr(a[c], b[c].to_numpy()[perm])[0]
        cv = np.std(last[c]) / (np.mean(last[c]) + 1e-9)
        print(f"  {g:10s}: CV={cv:.3f}  twin_r={tw:6.3f}  random_r={rn:6.3f}  gap={tw-rn:6.3f}")
