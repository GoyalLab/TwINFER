from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, os, glob, json
import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as t

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

# HSC: Fli1 and Scl are exactly-zero-variance in this Gillespie run (established earlier this
# session) -- exclude their pairs from the scored universe for EVERY method uniformly, not just
# TwINFER (which already drops them internally before scoring). IMPORTANT: `genes` must stay the
# FULL original list for M-indexing and gene_N remapping to stay correct -- only U_full/y_full are
# filtered afterward, never the gene list itself.
EXCLUDE_GENES = {"HSC": ["Fli1", "Scl"]}


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


rows = []
for net, cfg in NETS.items():
    genes = cfg["genes"]
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    # full n*(n-1) directed universe, DIRECTED truth (matches TwINFER's own score_one_json convention:
    # only the exact regulator->target direction counts, not OR'd with the reverse)
    excl = set(EXCLUDE_GENES.get(net, []))
    U_full = [(a, b) for a in genes for b in genes if a != b and a not in excl and b not in excl]
    y_full = np.array([1 if M[gi[a], gi[b]] != 0 else 0 for a, b in U_full])

    for algo in ALGOS:
        rd = load_ranked(cfg["beeline_dir"], algo, genes)
        if rd is None:
            rows.append(dict(net=net, method=algo, n_pairs=np.nan, n_true=np.nan))
            continue
        score_map = {(r.Gene1, r.Gene2): abs(r.EdgeWeight) for r in rd.itertuples()}
        sc = np.array([score_map.get(p, np.nan) for p in U_full])
        rep = t.full_report(sc, y_full)
        rows.append(dict(net=net, method=algo, n_pairs=len(U_full), n_true=int(y_full.sum()),
                          base_rate=float(y_full.mean()), auprc_raw=rep["auprc"], auprc_x=rep["auprc_x"],
                          topk_precision=rep["topk_precision"],
                          topk_x=rep["topk_precision"] / y_full.mean() if y_full.mean() > 0 else np.nan))

    files = sorted(glob.glob(cfg["twinfer_glob"]))
    if not files:
        rows.append(dict(net=net, method="TwINFER", n_pairs=np.nan, n_true=np.nan))
        continue
    d = json.load(open(files[0]))
    tsi = d["twin_score_inputs"]
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    id_to_bio = {f"gene_{i+1}": g for i, g in enumerate(genes)}
    U_tw = list(zip(dd.gene_1.map(id_to_bio), dd.gene_2.map(id_to_bio)))
    y_tw = np.array([1 if M[gi[a], gi[b]] != 0 else 0 for a, b in U_tw])
    z_reg_map = d.get("gated_regulation", {}).get("z_reg_gated", {})
    score, n_pass = t.todo4v2_score(dd, list(zip(dd.gene_1, dd.gene_2)), z_reg_map, "nogate")
    rep = t.full_report(score, y_tw)
    rows.append(dict(net=net, method="TwINFER", n_pairs=len(U_tw), n_true=int(y_tw.sum()),
                      base_rate=float(y_tw.mean()), auprc_raw=rep["auprc"], auprc_x=rep["auprc_x"],
                      topk_precision=rep["topk_precision"],
                      topk_x=rep["topk_precision"] / y_tw.mean() if y_tw.mean() > 0 else np.nan,
                      note=f"universe={len(U_tw)}/{n*(n-1)} (TwINFER's own pre-filter)" if len(U_tw) < n * (n - 1) else ""))

res = pd.DataFrame(rows)
res.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/full_metrics_table.csv', index=False)

order = ["TwINFER"] + ALGOS
net_order = ["mCAD", "GSD", "HSC", "VSC", "B_cell_activation", "EMT", "Pluripotent"]
pd.set_option("display.width", 200)
for net in net_order:
    sub = res[res.net == net].set_index("method").reindex(order)
    print(f"\n=== {net} ===")
    print(sub[["n_pairs", "n_true", "base_rate", "auprc_raw", "auprc_x", "topk_precision", "topk_x"]].round(3).to_string())
