from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
OLD = f"{R}/simulation_data/twinfer_format"
RW = f"{R}/input_data/real_world_networks"

def old_genes(net): return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt")]

MATRICES = {
    "mCAD": (f"{OLD}/mCAD/interaction_matrix.txt", ["Fgf8","Emx2","Pax6","Coup","Sp8"]),
    "GSD": (f"{OLD}/GSD/interaction_matrix.txt", old_genes("GSD")),
    "HSC": (f"{OLD}/HSC/interaction_matrix.txt", old_genes("HSC")),
    "VSC": (f"{OLD}/VSC/interaction_matrix.txt", old_genes("VSC")),
    "B_cell_activation": (f"{RW}/B_cell.txt", ["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"]),
    "EMT": (f"{RW}/EMT.txt", [f"gene_{i+1}" for i in range(17)]),
    "Pluripotent": (f"{RW}/Pluripotent.txt", [f"gene_{i+1}" for i in range(36)]),
}

diag = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/dead_edge_diagnosis_mRNA_calibrated.csv')

rows = []
for net, (mpath, genes) in MATRICES.items():
    M = np.loadtxt(mpath, delimiter=",")
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    sub = diag[diag.net == net]
    for r in sub.itertuples():
        reg, tgt = r.reg, r.tgt
        ri, ti = gi[reg], gi[tgt]
        regulators_of_reg = set(np.where(M[:, ri] != 0)[0])
        regulators_of_tgt = set(np.where(M[:, ti] != 0)[0])
        shared_regs = len(regulators_of_reg & regulators_of_tgt)
        mutual = M[ti, ri] != 0  # tgt -> reg also exists
        targets_of_reg = set(np.where(M[ri, :] != 0)[0])
        # FFL: reg -> X -> tgt for some third gene X
        ffl = False
        for x in targets_of_reg:
            if x != ti and M[x, ti] != 0:
                ffl = True; break
        tgt_in_degree = int((M[:, ti] != 0).sum())
        reg_out_degree = int((M[ri, :] != 0).sum())
        motif = []
        if shared_regs > 0: motif.append("shared_regulator")
        if mutual: motif.append("mutual_regulation")
        if ffl: motif.append("feed_forward_loop")
        if tgt_in_degree >= 6: motif.append("hub_target")
        if not motif: motif.append("simple_edge")
        rows.append(dict(net=net, reg=reg, tgt=tgt, classification=r.classification, corr=r.corr,
                          shared_regs=shared_regs, mutual=mutual, ffl=ffl, tgt_in_degree=tgt_in_degree,
                          reg_out_degree=reg_out_degree, motif="+".join(motif)))

res = pd.DataFrame(rows)
res.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/motif_analysis_mRNA.csv', index=False)

print("=== fraction 'dead' (any constant/masked classification) by primary motif feature ===")
res["is_dead"] = res.classification != "real_signal"
res["is_masked_specifically"] = res.classification == "masked_by_network"
for feat, label in [("shared_regs", "n_shared_regulators (binned)"), ("ffl", "part of a feed-forward loop"),
                     ("mutual", "mutual (bidirectional) regulation"), ("tgt_in_degree", "target in-degree (binned)")]:
    if feat == "shared_regs":
        g = pd.cut(res[feat], [-1, 0, 1, 2, 100])
    elif feat == "tgt_in_degree":
        g = pd.cut(res[feat], [0, 1, 2, 4, 8, 100])
    else:
        g = res[feat]
    print(f"\n-- {label} --")
    print(res.groupby(g)[["is_dead", "is_masked_specifically"]].agg(["mean", "count"]))

print("\n=== mean |corr| by motif combination ===")
print(res.groupby("motif")["corr"].agg(mean_abs=lambda x: x.abs().mean(), n="count").sort_values("mean_abs"))
