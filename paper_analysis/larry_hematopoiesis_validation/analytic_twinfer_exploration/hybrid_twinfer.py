# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Hybrid TwINFER: analytic z's for everything EXCEPT z_het / z_d_het, which come from a cheap
200-shuffle permutation (the two terms twinScore gates on with hard cutoffs). Compare the hybrid
and pure-analytic twinScore rankings to the 5000-shuffle permutation ranking."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer
from twinfer.inference.correlation_functions import calculate_twin_score

H = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
A = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/analytic_infer'
OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/hybrid_infer'
os.makedirs(OUT, exist_ok=True)
NSHUF = int(os.environ.get("NSHUF", "200"))


def het_zscores(gs):
    df = pd.read_csv(f"{H}/resources/twinfer_input/{gs}.csv")
    t0 = time.time()
    r = infer_with_twinfer(
        data=df, is_simulation_data=False, t1=2, t2=4, n_cores=int(os.environ.get("N_CORES", "8")),
        verbose=False, plot=False, return_diagnostics=True, ranked_list=True,
        alpha_gene_gene_corr=0.9999, alpha_stage3=0.9999,
        z_score_threshold_two_states=0.0, z_score_threshold_cross_correlation=0.0,
        n_shuffles_step1=40, n_shuffles_step2=NSHUF, n_shuffles_stage3=NSHUF,
        n_shuffles_direction=40, n_shuffles_fanout=40,
    )
    print(f"  {gs}: {NSHUF}-shuffle infer in {time.time()-t0:.0f}s", flush=True)
    zhet = {tuple(sorted(k)): float(v) for k, v in r["heterogeneity"]["z_het"].items() if v is not None}
    zdhet = {}
    for k, d in r["stage3"].items():
        zdhet[tuple(sorted(k))] = float(d.get("z_d_het", np.nan))
    return zhet, zdhet


def rank_metrics(a, b, label):
    m = a[["gene_1", "gene_2", "twinScore"]].merge(b[["gene_1", "gene_2", "twinScore"]],
                                                   on=["gene_1", "gene_2"], suffixes=("_x", "_y"))
    m["rx"] = m.twinScore_x.rank(ascending=False); m["ry"] = m.twinScore_y.rank(ascending=False)
    row = dict(label=label, n=len(m),
               pearson=np.corrcoef(m.twinScore_x, m.twinScore_y)[0, 1],
               spearman=spearmanr(m.twinScore_x, m.twinScore_y).correlation,
               kendall=kendalltau(m.twinScore_x, m.twinScore_y).correlation,
               exact=(m.rx == m.ry).mean(),
               rankdiff_med=(m.rx - m.ry).abs().median())
    for k in (10, 20, 50):
        sx = set(map(tuple, m.nsmallest(k, "rx")[["gene_1", "gene_2"]].values))
        sy = set(map(tuple, m.nsmallest(k, "ry")[["gene_1", "gene_2"]].values))
        row[f"top{k}"] = len(sx & sy)
    return row


for gs in sys.argv[1:] or ["detection_mid", "correlation_high"]:
    print(f"\n===== {gs} =====")
    tsi = pd.read_csv(f"{A}/{gs}/twin_score_inputs.csv")
    perm = pd.read_csv(f"{H}/resources/infer_results/{gs}_t2_t4_allpairs/ranked_edges.csv")
    analytic = calculate_twin_score({"twin_score_inputs": tsi}).sort_values("twinScore", ascending=False)

    zhet, zdhet = het_zscores(gs)
    hyb = tsi.copy()
    key = list(zip(hyb.gene_1, hyb.gene_2))
    skey = [tuple(sorted(k)) for k in key]
    hyb["z_het"] = [zhet.get(k, hyb.z_het.iloc[i]) for i, k in enumerate(skey)]
    hyb["z_d_het"] = [zdhet.get(k, hyb.z_d_het.iloc[i]) for i, k in enumerate(skey)]
    hyb["z_d"] = hyb["z_d_het"]
    n_hz = sum(k in zhet for k in skey); n_dz = sum(k in zdhet for k in skey)
    print(f"  replaced z_het for {n_hz}/{len(hyb)} rows, z_d_het for {n_dz}/{len(hyb)} rows")
    hybrid = calculate_twin_score({"twin_score_inputs": hyb}).sort_values("twinScore", ascending=False)
    hybrid.to_csv(f"{OUT}/{gs}_ranked_edges.csv", index=False)

    rows = [rank_metrics(analytic, perm, "analytic vs perm5000"),
            rank_metrics(hybrid, perm, f"hybrid({NSHUF}) vs perm5000"),
            rank_metrics(hybrid, analytic, "hybrid vs analytic")]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
