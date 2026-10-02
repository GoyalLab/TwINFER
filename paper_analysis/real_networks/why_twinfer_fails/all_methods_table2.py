from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

R = f'{TWINFER_PROJECT_ROOT}'
OLD_FMT = f"{R}/simulation_data/twinfer_format"
RW = f"{R}/input_data/real_world_networks"
ALGOS = ["GENIE3", "GRNBOOST2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]

GENE_LISTS = {
    "mCAD": ["Fgf8", "Emx2", "Pax6", "Coup", "Sp8"],
    "GSD": [l.strip() for l in open(f"{OLD_FMT}/GSD/gene_order.txt") if l.strip()],
    "HSC": [l.strip() for l in open(f"{OLD_FMT}/HSC/gene_order.txt") if l.strip()],
    "VSC": [l.strip() for l in open(f"{OLD_FMT}/VSC/gene_order.txt") if l.strip()],
    "B_cell_activation": ["Ikaros", "PU_1", "Flk2", "IL_7R", "GATA_1", "E2A", "EBF", "C_EBPa", "PAX5", "Notch_1"],
    "EMT": ["Cdh1", "Cldn7", "Foxc2", "Grhl2", "Gsc", "Klf8", "Np63a", "Ovol2", "Snai1", "Snai2", "Tcf3",
            "Tgfbeta", "Twist1", "Twist2", "Vim", "Zeb1", "Zeb2"],
    "Pluripotent": ["ARID3B", "CEBPZ", "ETV4", "FOXH1", "HIC2", "HMGB3", "JARID2", "LIN28B", "MIS18BP1", "MYCN",
                    "NANOG", "POU5F1", "POU5F1B", "PRDM14", "REST", "SALL2", "SALL4", "SMARCC1", "SOX2",
                    "TEAD2", "TERF1", "TGIF1", "WDHD1", "ZBTB12", "ZBTB39", "ZNF281", "ZNF286A", "ZNF286B",
                    "ZNF322", "ZNF398", "ZNF462", "ZNF730", "ZNF90", "ZNF92", "ZSCAN10", "ZSCAN2"],
}
NETS = {
    "mCAD": dict(matrix=f"{OLD_FMT}/mCAD/interaction_matrix.txt",
                 beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/mCAD",
                 twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/mCAD_rep_*_all_results.json"),
    "GSD": dict(matrix=f"{OLD_FMT}/GSD/interaction_matrix.txt",
                beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/GSD",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/GSD_rep_*_all_results.json"),
    "HSC": dict(matrix=f"{OLD_FMT}/HSC/interaction_matrix.txt",
                beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/HSC",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/HSC_rep_*_all_results.json"),
    "VSC": dict(matrix=f"{OLD_FMT}/VSC/interaction_matrix.txt",
                beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/VSC",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/VSC_rep_*_all_results.json"),
    "B_cell_activation": dict(matrix=f"{RW}/B_cell.txt",
                 beeline_dir=f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/B_cell_activation",
                 twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/B_cell_activation_rep_*_all_results.json"),
    "EMT": dict(matrix=f"{RW}/EMT.txt",
                beeline_dir=f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/EMT",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/EMT_rep_*_all_results.json"),
    "Pluripotent": dict(matrix=f"{RW}/Pluripotent.txt",
                beeline_dir=f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/Pluripotent",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/Pluripotent_rep_*_all_results.json"),
}
for n in NETS: NETS[n]["genes"] = GENE_LISTS[n]


def _remap(df, genes):
    native = {f"gene_{i+1}": g for i, g in enumerate(genes)}
    if df.Gene1.iloc[0] in native or (len(df) > 1 and df.Gene1.iloc[1] in native):
        df = df.copy()
        df["Gene1"] = df["Gene1"].map(lambda x: native.get(x, x))
        df["Gene2"] = df["Gene2"].map(lambda x: native.get(x, x))
    return df


def load_ranked(beeline_dir, algo, genes, scheme="twin_paired"):
    reps = sorted(glob.glob(f"{beeline_dir}/simrep*_{scheme}"))
    if not reps:
        return None
    f = f"{reps[0]}/{algo}/rankedEdges.csv"
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, sep="\t")
    df = _remap(df, genes)
    df = df[df.Gene1 != df.Gene2].reset_index(drop=True)
    return df


def load_twinfer_ranked(glob_pat, genes):
    files = sorted(glob.glob(glob_pat))
    if not files:
        return None
    d = json.load(open(files[0]))
    re_ = d["ranked_edges"]
    df = pd.DataFrame(re_["data"], columns=re_["columns"])
    # ranked_edges gene_1..gene_N ALWAYS correspond to genes[i-1] by POSITION (matrix row/col order) --
    # the JSON's own "gene_names" field is always ["gene_1",...] regardless of network, not usable as a map.
    id_to_bio = {f"gene_{i+1}": g for i, g in enumerate(genes)}
    pairs_scores = {(id_to_bio[r.gene_1], id_to_bio[r.gene_2]): r.twinScore for r in df.itertuples()
                    if r.gene_1 in id_to_bio and r.gene_2 in id_to_bio}
    return pairs_scores


def auprc_x(pairs_scores, genes, M):
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    iu = np.triu_indices(n, 1)
    und = ((M != 0) | (M.T != 0))[iu].astype(int)
    score_mat = np.zeros((n, n))
    matched = 0
    for (a, b), s in pairs_scores.items():
        if a in gi and b in gi:
            score_mat[gi[a], gi[b]] = max(score_mat[gi[a], gi[b]], abs(s))
            score_mat[gi[b], gi[a]] = max(score_mat[gi[b], gi[a]], abs(s))
            matched += 1
    vals = score_mat[iu]
    if und.sum() == 0 or und.sum() == len(und):
        return np.nan, matched
    return average_precision_score(und, np.nan_to_num(vals)) / und.mean(), matched


rows = []
for net, cfg in NETS.items():
    genes = cfg["genes"]
    M = np.loadtxt(cfg["matrix"], delimiter=",")

    for algo in ALGOS:
        rd = load_ranked(cfg["beeline_dir"], algo, genes)
        if rd is None:
            rows.append(dict(net=net, method=algo, auprc_x=np.nan, n_matched=0))
            continue
        pairs_scores = {(r.Gene1, r.Gene2): r.EdgeWeight for r in rd.itertuples()}
        a, m = auprc_x(pairs_scores, genes, M)
        rows.append(dict(net=net, method=algo, auprc_x=a, n_matched=m))

    pairs_scores = load_twinfer_ranked(cfg["twinfer_glob"], genes)
    if pairs_scores is None:
        rows.append(dict(net=net, method="TwINFER", auprc_x=np.nan, n_matched=0))
    else:
        a, m = auprc_x(pairs_scores, genes, M)
        rows.append(dict(net=net, method="TwINFER", auprc_x=a, n_matched=m))

res = pd.DataFrame(rows)
res.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/all_methods_table_fixed.csv', index=False)

pivot = res.pivot(index="method", columns="net", values="auprc_x")
order = ["TwINFER"] + ALGOS
pivot = pivot.reindex(order)
net_order = ["mCAD", "GSD", "HSC", "VSC", "B_cell_activation", "EMT", "Pluripotent"]
pivot = pivot[[c for c in net_order if c in pivot.columns]]
print(pivot.round(2).to_string())
print()
print("=== n_matched pairs (sanity check TwINFER isn't 0 anymore) ===")
print(res.pivot(index="method", columns="net", values="n_matched").reindex(order)[[c for c in net_order if c in pivot.columns]])
print()
print("=== best method per network ===")
print(pivot.idxmax())
print()
print("=== mean AUPRC-x per method ===")
print(pivot.mean(axis=1).sort_values(ascending=False).round(2))
