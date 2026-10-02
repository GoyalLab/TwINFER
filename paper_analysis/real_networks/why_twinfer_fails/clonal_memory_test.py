from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr, spearmanr, rankdata
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
OLD = f"{R}/simulation_data/twinfer_format"
GSD_GENES = [l.strip() for l in open(f"{OLD}/GSD/gene_order.txt")]
HSC_GENES = [l.strip() for l in open(f"{OLD}/HSC/gene_order.txt")]
M = {
    "GSD": np.loadtxt(f"{OLD}/GSD/interaction_matrix.txt", delimiter=","),
    "HSC": np.loadtxt(f"{OLD}/HSC/interaction_matrix.txt", delimiter=","),
    "B_cell_activation": np.loadtxt(f"{R}/input_data/real_world_networks/B_cell.txt", delimiter=","),
    "EMT": np.loadtxt(f"{R}/input_data/real_world_networks/EMT.txt", delimiter=","),
    "Pluripotent": np.loadtxt(f"{R}/input_data/real_world_networks/Pluripotent.txt", delimiter=","),
}
FILES = {
    "GSD": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv", GSD_GENES),
    "HSC": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv", HSC_GENES),
    "B_cell_activation": (f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv",
                          ["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"]),
    "EMT": (f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv",
            [f"gene_{i+1}" for i in range(17)]),
    "Pluripotent": (f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv",
                   [f"gene_{i+1}" for i in range(36)]),
}
DEAD_GENES = {
    "GSD": ["UGR","DHH","RSPO1","FOXL2","WT1pKTS","NR5A1","SRY"],
    "HSC": ["Gata1","Fog1","EgrNab"],  # Fli1/Scl excluded: exactly zero variance
    "B_cell_activation": ["IL_7R","E2A","EBF","PAX5"],
    "EMT": ["gene_3","gene_9","gene_10","gene_13","gene_15","gene_16","gene_17"],
    "Pluripotent": ["gene_2","gene_3","gene_4","gene_6","gene_7","gene_8","gene_10","gene_11","gene_12","gene_14",
                    "gene_17","gene_21","gene_26","gene_30","gene_35","gene_36"],
}

def colname(g, genes, native):
    return g if native else f"gene_{genes.index(g)+1}"

for net, (f, genes) in FILES.items():
    native = genes[0].startswith("gene_")
    n = len(genes); gi = {g: i for i, g in enumerate(genes)}
    cols = ["cell_id", "time_step", "clone_id"] + [f"{colname(g, genes, native)}_protein" for g in genes]
    df = pd.read_csv(f, usecols=cols)
    last = df[df.time_step == df.time_step.max()]
    clone_sizes = last.clone_id.value_counts()
    pairs = clone_sizes[clone_sizes == 2].index
    sub = last[last.clone_id.isin(pairs)]
    a = sub.groupby("clone_id").nth(0).sort_values("clone_id")
    b = sub.groupby("clone_id").nth(1).sort_values("clone_id")
    assert (a.clone_id.to_numpy() == b.clone_id.to_numpy()).all()

    def vec(df_, g): return df_[f"{colname(g, genes, native)}_protein"].to_numpy(dtype=float)

    # ---- Test 1: twin-vs-random gap for EVERY gene (not just flagged-dead) ----
    print(f"\n=== {net}: twin-vs-random gap, ALL genes (dead genes marked *) ===")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(b))
    gaps = {}
    for g in genes:
        va, vb = vec(a, g), vec(b, g)
        if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
            continue
        tw = spearmanr(va, vb)[0]
        rn = spearmanr(va, vb[perm])[0]
        gaps[g] = tw - rn
    dead = set(DEAD_GENES[net])
    dead_gaps = [gaps[g] for g in gaps if g in dead]
    other_gaps = [gaps[g] for g in gaps if g not in dead]
    print(f"  dead genes (n={len(dead_gaps)}):  mean gap = {np.mean(dead_gaps):.3f}, range [{min(dead_gaps):.2f}, {max(dead_gaps):.2f}]")
    print(f"  other genes (n={len(other_gaps)}): mean gap = {np.mean(other_gaps):.3f}, range [{min(other_gaps):.2f}, {max(other_gaps):.2f}]")

    # ---- Test 2: does partialling out the true regulator remove the target's twin correlation? ----
    print(f"  Partial-correlation test (does the true regulator explain the target's heritability?):")
    reg_idx, tgt_idx = np.where(M[net] != 0) if False else np.where(M[net] != 0)
    checked = 0
    for ri, ti in zip(reg_idx, tgt_idx):
        reg, tgt = genes[ri], genes[ti]
        if reg == tgt or tgt not in dead:
            continue
        va_r, vb_r = vec(a, reg), vec(b, reg)
        va_t, vb_t = vec(a, tgt), vec(b, tgt)
        if np.std(va_r) < 1e-9 or np.std(vb_r) < 1e-9 or np.std(va_t) < 1e-9:
            continue
        raw_tw = spearmanr(va_t, vb_t)[0]
        # rank-transform everything first (partial SPEARMAN correlation = partial Pearson on ranks --
        # this captures any monotonic, not just linear, regulator->target relationship)
        ra_r, rb_r = rankdata(va_r), rankdata(vb_r)
        ra_t, rb_t = rankdata(va_t), rankdata(vb_t)
        slope_a, intercept_a = np.polyfit(ra_r, ra_t, 1)
        slope_b, intercept_b = np.polyfit(rb_r, rb_t, 1)
        resid_a = ra_t - (slope_a * ra_r + intercept_a)
        resid_b = rb_t - (slope_b * rb_r + intercept_b)
        partial_tw = pearsonr(resid_a, resid_b)[0]
        print(f"    {reg:10s}->{tgt:10s}: raw twin_r(target)={raw_tw:6.3f}  after removing regulator: {partial_tw:6.3f}  "
              f"(retained {100*partial_tw/raw_tw if raw_tw else 0:.0f}%)")
        checked += 1
        if checked >= 6:
            break
