from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
ALGOS = ["GENIE3", "GRNBOOST2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]
BEELINE_DIRS = {
    "mCAD": f"{R}/analysis_data/real_networks/beeline_inference/mCAD",
    "GSD": f"{R}/analysis_data/real_networks/beeline_inference/GSD",
    "HSC": f"{R}/analysis_data/real_networks/beeline_inference/HSC",
    "VSC": f"{R}/analysis_data/real_networks/beeline_inference/VSC",
    "B_cell_activation": f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/B_cell_activation",
    "EMT": f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/EMT",
    "Pluripotent": f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/Pluripotent",
}
TWINFER_GLOBS = {
    "mCAD": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/mCAD_rep_*_all_results.json",
    "GSD": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/GSD_rep_*_all_results.json",
    "HSC": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/HSC_rep_*_all_results.json",
    "VSC": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/VSC_rep_*_all_results.json",
    "B_cell_activation": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/B_cell_activation_rep_*_all_results.json",
    "EMT": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/EMT_rep_*_all_results.json",
    "Pluripotent": f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/Pluripotent_rep_*_all_results.json",
}

def remap(df, genes):
    native = {f"gene_{i+1}": g for i, g in enumerate(genes)}
    if df.Gene1.iloc[0] in native or (len(df) > 1 and df.Gene1.iloc[1] in native):
        df = df.copy()
        df["Gene1"] = df["Gene1"].map(lambda x: native.get(x, x))
        df["Gene2"] = df["Gene2"].map(lambda x: native.get(x, x))
    return df

def load_ranks(beeline_dir, algo, genes, scheme="twin_paired"):
    reps = sorted(glob.glob(f"{beeline_dir}/simrep*_{scheme}"))
    if not reps: return None
    f = f"{reps[0]}/{algo}/rankedEdges.csv"
    df = pd.read_csv(f, sep="\t"); df = remap(df, genes)
    df = df[df.Gene1 != df.Gene2].reset_index(drop=True)
    df["absw"] = df.EdgeWeight.abs()
    df = df.sort_values("absw", ascending=False).reset_index(drop=True)
    df["pctile"] = (df.index + 1) / len(df)  # 0 = top rank, 1 = bottom
    return {(r.Gene1, r.Gene2): r.pctile for r in df.itertuples()}

def load_twinfer_ranks(glob_pat, genes):
    files = sorted(glob.glob(glob_pat))
    if not files: return None
    d = json.load(open(files[0]))
    re_ = d["ranked_edges"]
    df = pd.DataFrame(re_["data"], columns=re_["columns"])
    df = df.rename(columns={"gene_1": "Gene1", "gene_2": "Gene2", "twinScore": "EdgeWeight"})
    df = remap(df, genes)
    df = df[df.Gene1 != df.Gene2].reset_index(drop=True)
    df["absw"] = df.EdgeWeight.abs()
    df = df.sort_values("absw", ascending=False).reset_index(drop=True)
    df["pctile"] = (df.index + 1) / len(df)
    return {(r.Gene1, r.Gene2): r.pctile for r in df.itertuples()}

diag = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/dead_edge_diagnosis.csv')

rows = []
for net in BEELINE_DIRS:
    sub = diag[diag.net == net]
    genes = sorted(set(sub.reg) | set(sub.tgt))
    # need full gene list in matrix order -- reuse from diag (reg/tgt cols already use biological/native names consistent with earlier pipeline)
    for algo in ALGOS:
        ranks = load_ranks(BEELINE_DIRS[net], algo, genes)
        if ranks is None: continue
        for r in sub.itertuples():
            p = ranks.get((r.reg, r.tgt))
            if p is not None:
                rows.append(dict(net=net, algo=algo, reg=r.reg, tgt=r.tgt, classification=r.classification, pctile=p))
    tw = load_twinfer_ranks(TWINFER_GLOBS[net], genes)
    if tw is not None:
        for r in sub.itertuples():
            p = tw.get((r.reg, r.tgt))
            if p is not None:
                rows.append(dict(net=net, algo="TwINFER", reg=r.reg, tgt=r.tgt, classification=r.classification, pctile=p))

res = pd.DataFrame(rows)
res.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/dead_edge_method_comparison.csv', index=False)

print("Mean percentile rank (0=top of ranking, 1=bottom) by classification x algorithm:")
piv = res.pivot_table(index="classification", columns="algo", values="pctile", aggfunc="mean")
print(piv.round(2).to_string())
print("\nOverall mean by classification (pooled across algorithms):")
print(res.groupby("classification").pctile.mean().round(2))
print("\nFraction of each classification that lands in the algorithm's own TOP 20% ranked pairs:")
res["top20"] = res.pctile <= 0.2
print(res.pivot_table(index="classification", columns="algo", values="top20", aggfunc="mean").round(2).to_string())
