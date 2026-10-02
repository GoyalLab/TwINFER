"""LARRY heatmap data: existdir and full (clone-weighted existdir + permutation z_fanout flat +
0.5*z_d_het) AUPRC + F1(top-k) for all 9 gene sets -- same 'full' formula definition as
network_sweep_final/mixed_network_sweep/real_data's v2 heatmaps (NOT the unweighted/partial-corr
variants explored earlier as side investigations). Reuses compute_zfanout_real from
full_gated_fanout_larry.py (real-data cross-time construction, no disjoint split needed for LARRY
since it already uses the full Cartesian-product cross-time twins).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ROOT = f'{TWINFER_PROJECT_ROOT}'
HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f"{HERE}/resources"
R = f"{RES_HERE}/resources"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
import twinfer.inference.correlation_functions as cf

N_SHUFFLES, N_CORES, SEED = 200, 1, 101010
T1, T2 = 2, 4


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def score_auprc_f1(y, x):
    k = int(y.sum())
    if k == 0 or k == len(y):
        return np.nan, np.nan
    prec, rec, _ = precision_recall_curve(y, x)
    auprc = auc(rec, prec)
    order = np.argsort(-x, kind="stable")
    boundary = x[order[k - 1]]
    selected = x >= boundary
    n_sel = selected.sum()
    tp = (selected & (y == 1)).sum()
    precision_k = tp / n_sel if n_sel else np.nan
    recall_k = tp / k
    f1 = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
    return auprc, f1


def compute_zfanout_real(raw_csv, genes):
    usecols = ["clone_id", "cell_id", "time_step"] + [f"{g}_mRNA" for g in genes]
    df = pd.read_csv(raw_csv, usecols=usecols)
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    across_t1, across_t2 = cf._build_cross_time_twins(t1_raw, t2_raw)
    directed_pairs = [(a, b) for a in genes for b in genes if a != b]
    cross_matrix = cf.get_cross_correlations(across_t1, across_t2, gene_pairs=directed_pairs, unit="clone")
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, details = cf.identify_actual_directed_edges(
            across_t1, across_t2, cross_matrix, gene_pairs=directed_pairs,
            z_score_threshold=np.inf, use_scramble=True, n_shuffles=N_SHUFFLES, n_cores_to_use=N_CORES,
            verbose=False, base_seed=SEED, return_z_scores=True, return_rho_cross_null=True,
            prepare_rho_cross_null=True, unit="clone")
    cross_z = {pair: float(d.get("z_rho_cross", np.nan)) for pair, d in details.items()}
    zfan = {}
    for a, b in directed_pairs:
        cands = []
        for g in genes:
            if g in (a, b):
                continue
            za = max(abs(cross_z.get((g, a), np.nan)), abs(cross_z.get((a, g), np.nan)))
            zb = max(abs(cross_z.get((g, b), np.nan)), abs(cross_z.get((b, g), np.nan)))
            if np.isfinite(za) and np.isfinite(zb):
                cands.append(min(za, zb))
        zfan[(a, b)] = max(cands) if cands else np.nan
    return zfan


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    gene_sets = json.load(open(f"{R}/gene_sets.json"))
    detail = json.load(open(f"{R}/gene_sets_detail.json"))

    GENE_SETS = ["variability_high", "variability_mid", "variability_low",
                 "detection_high", "detection_mid", "detection_low",
                 "correlation_high", "correlation_mid", "correlation_low"]
    rows = []
    for gs in GENE_SETS:
        d_full = pd.read_csv(f"{R}/analytic_infer/{gs}/twin_score_inputs.csv")
        genes = sorted(set(d_full.gene_1) | set(d_full.gene_2))
        crit, lvl = gs.rsplit("_", 1)
        panel = gene_sets[gs]
        tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
        dd = d_full[d_full.gene_1.isin(tfs) & (d_full.gene_1 != d_full.gene_2)].reset_index(drop=True)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if p in CE else 0 for p in U])
        if y.sum() < 3:
            continue

        existdir = s(dd.rho_t1.abs().to_numpy()) + s(dd.rho_t2.abs().to_numpy()) + s(dd.gamma.to_numpy())
        s_zdhet = s(dd.z_d_het.to_numpy())
        zfan_map = compute_zfanout_real(f"{R}/twinfer_input_cp10k/{gs}.csv", genes)
        zfan = np.array([zfan_map.get(p, np.nan) for p in U])
        full = existdir + 1.0 * s(zfan) + 0.5 * s_zdhet

        auprc_ex, f1_ex = score_auprc_f1(y, existdir)
        auprc_full, f1_full = score_auprc_f1(y, full)
        rows.append(dict(gene_set=gs, criterion=crit, level=lvl,
                          auprc_existdir=auprc_ex, f1_existdir=f1_ex,
                          auprc_full=auprc_full, f1_full=f1_full,
                          n_pairs=len(U), n_true=int(y.sum())))
        print(f"{gs}: auprc_full={auprc_full:.4f} f1_full={f1_full:.4f}", flush=True)

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{HERE}/full_formula_v2_larry_twinfer.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/full_formula_v2_larry_twinfer.csv", index=False)
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
