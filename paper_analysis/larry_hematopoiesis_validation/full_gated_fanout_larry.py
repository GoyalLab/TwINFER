"""full_gated on LARRY, using the REAL-DATA cross-time construction (all cells at t1, all cells
at t2, cross-time twins = full Cartesian product within each clone via _build_cross_time_twins
directly -- no 'replicate' column, unlike the simulation-specific select_views) for the genuine
permutation-based z_fanout, gated (penalty only if z_fanout > THR) in place of full's flat
1.0*s(z_fanout) term.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f"{HERE}/resources"
R = f"{RES_HERE}/resources"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
import twinfer.inference.correlation_functions as cf

N_SHUFFLES, N_CORES, SEED, THR = 200, 1, 101010, 2.326
PENALTIES = [-2.0]
T1, T2 = 2, 4


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def score_topk(y, x):
    from sklearn.metrics import auc, precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, x)
    return auc(rec, prec)


def compute_zfanout_real(raw_csv, genes):
    """Real-data cross-time twins: ALL cells at t1, ALL cells at t2, Cartesian product per clone."""
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

    GENE_SETS = ["variability_high","variability_mid","variability_low",
             "detection_high","detection_mid","detection_low",
             "correlation_high","correlation_mid","correlation_low"]
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

        s_zfan_full = s(zfan)
        full_flat = existdir + 1.0 * s_zfan_full + 0.5 * s_zdhet

        print(f"=== {gs}  (n_genes={len(genes)}, n_pairs={len(U)}, n_true={y.sum()}) ===")
        print("  auprc_existdir:", round(score_topk(y, existdir), 4))
        print("  auprc_full_flat_perm:", round(score_topk(y, full_flat), 4))
        for pen in PENALTIES:
            gated = existdir + np.where(zfan > THR, pen, 0.0) + 0.5 * s_zdhet
            print(f"  auprc_full_gated_pen{pen}:", round(score_topk(y, gated), 4))
        print(f"  z_fanout finite frac: {np.isfinite(zfan).mean():.3f}, n>THR: {(zfan>THR).sum()}")


if __name__ == "__main__":
    main()
