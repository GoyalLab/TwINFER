"""2026-09-17: Full table -- unsigned auprc_x, signed auprc_x, unsigned f1_topk, signed f1_topk --
for TODO4v2(nogate), cross-corr-only, and all 7 BEELINE competitors, across the 3 simulated
benchmarks. Per user instruction: methods with no native sign (GENIE3, GRNBOOST2, PIDC, SCODE --
importance/MI scores are never negative) get their sign RECOMPUTED from sign(PEARSON's own
EdgeWeight for that pair) instead of defaulting to all-positive. PEARSON/PPCOR/SCSGL already emit
real signed correlations and keep their own native sign (same convention as the aggregate
beeline_analysis_output.csv's auprc_signed/f1_topk_signed columns). TODO4v2/cross-corr-only both
already use sign(rho_cross_xy) (2026-09-17 change, see signed_scoring_analysis.py).

Recomputation walks the raw BEELINE rankedEdges.csv files directly (the aggregate CSVs only have
the default-positive-sign numbers baked in) -- every replicate subfolder that has both an
unsigned algo's rankedEdges.csv and a sibling PEARSON/rankedEdges.csv in the same
{dataset_id}/{simrepN_scheme}/ directory.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ROOT = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results'
UNSIGNED_ALGOS = ["GENIE3", "GRNBOOST2", "PIDC", "SCODE"]
NATIVE_SIGNED_ALGOS = ["PEARSON", "PPCOR", "SCSGL"]


def load_ranked(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


def true_signed_edges(gt_path, n_genes):
    M = np.loadtxt(gt_path, delimiter=",")
    genes = [f"gene_{i+1}" for i in range(n_genes)]
    return {(genes[i], genes[j], int(np.sign(M[i, j])))
            for i in range(n_genes) for j in range(n_genes) if i != j and M[i, j] != 0}


def topk_f1(y, sc):
    sc = np.asarray(sc, float)
    k = int(y.sum())
    if k == 0 or len(set(sc)) <= 1:
        return float("nan")
    order = np.argsort(-sc, kind="stable")
    boundary = sc[order[k - 1]]
    nonzero = sc[sc > 0]
    floor = nonzero.min() if len(nonzero) else 0.0
    best_val = max(floor, boundary)
    selected = sc >= best_val
    tp = int((y[selected] == 1).sum())
    n_sel = int(selected.sum())
    precision = tp / n_sel if n_sel else 0.0
    recall = tp / k
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def auprc_x_and_f1(y, sc, rand):
    sc = np.asarray(sc, float)
    if len(set(sc)) <= 1 or y.sum() == 0:
        return float("nan"), float("nan")
    prec, rec, _ = precision_recall_curve(y, sc)
    a = auc(rec, prec)
    return (a / rand if rand > 0 else float("nan")), topk_f1(y, sc)


def score_replicate(algo_dir, pearson_dir, gt_path, n_genes, use_pearson_sign):
    algo = load_ranked(f"{algo_dir}/rankedEdges.csv")
    if algo is None or algo.empty:
        return None
    algo_mag = dict(zip(zip(algo.Gene1, algo.Gene2), algo.EdgeWeight.abs()))
    if use_pearson_sign:
        pear = load_ranked(f"{pearson_dir}/rankedEdges.csv")
        if pear is None or pear.empty:
            return None
        sign_map = dict(zip(zip(pear.Gene1, pear.Gene2), np.sign(pear.EdgeWeight).astype(int)))
    else:
        sign_map = {p: 1 for p in algo_mag}  # native-signed algos handled separately; unused here

    genes = [f"gene_{i+1}" for i in range(n_genes)]
    possible = [(a, b) for a in genes for b in genes if a != b]
    true_signed = true_signed_edges(gt_path, n_genes)
    rand_unsigned = sum(1 for a, b in possible if any((a, b, s) in true_signed for s in (1, -1))) / len(possible)
    signed_positions = [(a, b, s) for a, b in possible for s in (1, -1)]
    rand_signed = sum(1 for p in signed_positions if p in true_signed) / len(signed_positions)

    sc_unsigned = np.array([algo_mag.get((a, b), 0.0) for a, b in possible])
    y_unsigned = np.array([1 if any((a, b, s) in true_signed for s in (1, -1)) else 0 for a, b in possible])
    sc_signed = np.array([algo_mag.get((a, b), 0.0) if sign_map.get((a, b), 0) == s else 0.0
                           for a, b, s in signed_positions])
    y_signed = np.array([1 if p in true_signed else 0 for p in signed_positions])

    ax_u, f1_u = auprc_x_and_f1(y_unsigned, sc_unsigned, rand_unsigned)
    ax_s, f1_s = auprc_x_and_f1(y_signed, sc_signed, rand_signed)
    return ax_u, f1_u, ax_s, f1_s


def n_genes_from_gt(path):
    return np.loadtxt(path, delimiter=",").shape[0]


def recompute_pearson_sign_for_benchmark(name, beeline_inference_dir, gt_resolver):
    rows = {algo: [] for algo in UNSIGNED_ALGOS}
    top_dirs = sorted(d for d in glob.glob(f"{beeline_inference_dir}/*") if os.path.isdir(d) and os.path.basename(d) != "logs")
    for top in top_dirs:
        ds = os.path.basename(top)
        gt_path = gt_resolver(ds)
        if gt_path is None or not os.path.exists(gt_path):
            continue
        n_genes = n_genes_from_gt(gt_path)
        for sub in glob.glob(f"{top}/*"):
            pearson_dir = f"{sub}/PEARSON"
            for algo in UNSIGNED_ALGOS:
                algo_dir = f"{sub}/{algo}"
                if not os.path.isdir(algo_dir) or not os.path.isdir(pearson_dir):
                    continue
                r = score_replicate(algo_dir, pearson_dir, gt_path, n_genes, use_pearson_sign=True)
                if r is not None:
                    rows[algo].append(r)
    out = {}
    for algo, vals in rows.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out[algo] = dict(auprc_x=np.nanmean(arr[:, 0]), f1_topk=np.nanmean(arr[:, 1]),
                          auprc_x_signed_pearsonsign=np.nanmean(arr[:, 2]),
                          f1_topk_signed_pearsonsign=np.nanmean(arr[:, 3]), n=len(vals))
    return out


def native_signed_from_aggregate(beeline_csv, gt_resolver, algos=NATIVE_SIGNED_ALGOS):
    b = pd.read_csv(beeline_csv)
    b = b[b.algorithm.isin(algos)].copy()
    gt_cache = {}
    def genes_for(ds):
        if ds not in gt_cache:
            p = gt_resolver(ds)
            gt_cache[ds] = n_genes_from_gt(p) if p and os.path.exists(p) else np.nan
        return gt_cache[ds]
    b["n_genes"] = b.dataset_id.map(genes_for)
    b = b[b.n_genes.notna()]
    total = b.n_genes * (b.n_genes - 1)
    b = b.assign(auprc_x=b.auprc / (b.n_true_edges / total),
                 auprc_x_signed=b.auprc_signed / (b.n_true_edges / (2 * total)))
    g = b.groupby("algorithm").agg(auprc_x=("auprc_x", "mean"), f1_topk=("f1_topk", "mean"),
                                    auprc_x_signed=("auprc_x_signed", "mean"),
                                    f1_topk_signed=("f1_topk_signed", "mean"))
    return {algo: dict(auprc_x=r.auprc_x, f1_topk=r.f1_topk,
                        auprc_x_signed_pearsonsign=r.auprc_x_signed,
                        f1_topk_signed_pearsonsign=r.f1_topk_signed)
            for algo, r in g.iterrows()}


def run(name, beeline_inference_dir, beeline_csv, gt_resolver,
        todo4v2_csv, crosscorr_csv, signed_csv):
    unsigned_algo_rows = recompute_pearson_sign_for_benchmark(name, beeline_inference_dir, gt_resolver)
    native_rows = native_signed_from_aggregate(beeline_csv, gt_resolver) if beeline_csv and os.path.exists(beeline_csv) else {}

    t4 = pd.read_csv(todo4v2_csv)
    cc = pd.read_csv(crosscorr_csv)
    cc_col_x = "crosscorr_auprc_x" if "crosscorr_auprc_x" in cc.columns else "auprc_x"
    cc_col_f1 = "crosscorr_f1" if "crosscorr_f1" in cc.columns else "f1"
    sg = pd.read_csv(signed_csv)

    rows = [
        dict(method="TODO4v2(nogate)", auprc_x=t4.todo4v2_nogate_auprc_x.mean(),
             f1_topk=t4.todo4v2_nogate_topk_f1.mean(),
             auprc_x_signed=sg.todo4v2_signed_auprc_x.mean(), f1_topk_signed=sg.todo4v2_signed_f1.mean()),
        dict(method="cross-corr-only", auprc_x=cc[cc_col_x].mean(), f1_topk=cc[cc_col_f1].mean(),
             auprc_x_signed=sg.crosscorr_signed_auprc_x.mean(), f1_topk_signed=sg.crosscorr_signed_f1.mean()),
    ]
    for algo, r in {**unsigned_algo_rows, **native_rows}.items():
        rows.append(dict(method=algo, auprc_x=r["auprc_x"], f1_topk=r["f1_topk"],
                          auprc_x_signed=r["auprc_x_signed_pearsonsign"],
                          f1_topk_signed=r["f1_topk_signed_pearsonsign"]))

    df = pd.DataFrame(rows).sort_values("auprc_x", ascending=False).reset_index(drop=True)
    print(f"\n=== {name} (signed metrics: PEARSON-sign for GENIE3/GRNBOOST2/PIDC/SCODE, native sign for PEARSON/PPCOR/SCSGL/TODO4v2/cross-corr) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(f"{HERE}/pearson_sign_full_table_{name}.csv", index=False)
    return df


def main():
    run("network_sweep_final",
        f"{ROOT}/analysis_data/network_sweep_final/beeline_inference",
        f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output.csv",
        lambda ds: f"{ROOT}/input_data/network_sweep_final/{ds}.txt",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/todo4v2_network_sweep_final_broader_results.csv",
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/todo4v2_network_sweep_final_broader_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/crosscorr_network_sweep_final_broader_results.csv",
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/crosscorr_network_sweep_final_broader_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/signed_network_sweep_final_results.csv")
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/signed_network_sweep_final_results.csv")

    run("mixed_network_sweep",
        f"{ROOT}/analysis_data/mixed_network_sweep/beeline_inference",
        f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv",
        lambda ds: f"{ROOT}/input_data/mixed_network_sweep/{ds}.txt",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/todo4v2_mixed_network_sweep_results.csv",
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/todo4v2_mixed_network_sweep_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/crosscorr_mixed_network_sweep_results.csv",
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/crosscorr_mixed_network_sweep_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/signed_mixed_network_sweep_results.csv")
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/signed_mixed_network_sweep_results.csv")

    TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
                "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
                "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
    def real_gt_resolver(topdir):
        for suffix in ("_seeded", "_multistate"):
            if topdir.endswith(suffix):
                topdir = topdir[: -len(suffix)]
                break
        return f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(topdir, '')}"

    run("real_networks",
        f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference",
        f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv",
        real_gt_resolver,
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/todo4v2_real_networks_results.csv",
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/todo4v2_real_networks_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/crosscorr_real_networks_results.csv",
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/crosscorr_real_networks_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/signed_real_networks_results.csv")
        f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/signed_real_networks_results.csv")


if __name__ == "__main__":
    main()
