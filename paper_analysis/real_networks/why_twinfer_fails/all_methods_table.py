from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/a2a3302f-b989-404d-a186-76799205cf04/scratchpad")
from paper_analysis.real_networks.why_twinfer_fails.dynamics_fidelity import NETS, ALGOS, load_ranked, load_twinfer_ranked

def auprc_x(pairs_scores, genes, M):
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    iu = np.triu_indices(n, 1)
    und = ((M != 0) | (M.T != 0))[iu].astype(int)
    score_mat = np.zeros((n, n))
    for (a, b), s in pairs_scores.items():
        if a in gi and b in gi:
            score_mat[gi[a], gi[b]] = max(score_mat[gi[a], gi[b]], abs(s))
            score_mat[gi[b], gi[a]] = max(score_mat[gi[b], gi[a]], abs(s))
    vals = score_mat[iu]
    if und.sum() == 0 or und.sum() == len(und):
        return np.nan
    return average_precision_score(und, np.nan_to_num(vals)) / und.mean()

rows = []
for net, cfg in NETS.items():
    genes = cfg["genes"]
    M = np.loadtxt(cfg["matrix"], delimiter=",")

    for algo in ALGOS:
        rd = load_ranked(cfg["beeline_dir"], algo, genes)
        if rd is None:
            rows.append(dict(net=net, method=algo, auprc_x=np.nan, n_genes=len(genes)))
            continue
        pairs_scores = {(r.Gene1, r.Gene2): r.absw for r in rd.itertuples()}
        rows.append(dict(net=net, method=algo, auprc_x=auprc_x(pairs_scores, genes, M), n_genes=len(genes)))

    tw = load_twinfer_ranked(cfg["twinfer_glob"])
    if tw is None:
        rows.append(dict(net=net, method="TwINFER", auprc_x=np.nan, n_genes=len(genes)))
    else:
        df, gmap, native_names = tw
        native = native_names[0].startswith("gene_") is False and native_names[0] not in genes
        # gmap maps gene_i -> native_names[i]; if native_names already are biological names (GSD/HSC/mCAD/VSC), use as-is;
        # if native_names are gene_1..N placeholders (B_cell/EMT/Pluripotent), map gene_1_col values through gmap then to genes-order names
        uses_native_ids = df.gene_1.iloc[0].startswith("gene_") and native_names[0] != df.gene_1.iloc[0]
        if native_names[0] in genes:
            # gene_names already biological
            pairs_scores = {(r.gene_1, r.gene_2): r.twinScore for r in df.itertuples()}
        else:
            # gene_names is the gene_1..N -> biological map itself, in order matching `genes`
            id_to_bio = {f"gene_{i+1}": g for i, g in enumerate(native_names)}
            pairs_scores = {(id_to_bio.get(r.gene_1, r.gene_1), id_to_bio.get(r.gene_2, r.gene_2)): r.twinScore for r in df.itertuples()}
        rows.append(dict(net=net, method="TwINFER", auprc_x=auprc_x(pairs_scores, genes, M), n_genes=len(genes)))

res = pd.DataFrame(rows)
res.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/all_methods_table.csv', index=False)

pivot = res.pivot(index="method", columns="net", values="auprc_x")
order = ["TwINFER"] + ALGOS
pivot = pivot.reindex(order)
net_order = ["mCAD", "GSD", "HSC", "VSC", "B_cell_activation", "EMT", "Pluripotent"]
pivot = pivot[[c for c in net_order if c in pivot.columns]]
print(pivot.round(2).to_string())
print()
print("=== best method per network ===")
print(pivot.idxmax())
print()
print("=== mean AUPRC-x per method (across networks with data) ===")
print(pivot.mean(axis=1).sort_values(ascending=False).round(2))
