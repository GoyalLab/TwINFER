# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Which true edges does TwINFER (z_het-fixed twinScore) recover, by edge type?
Pool GSD/HSC/VSC/mCAD/EMT. For each true edge: percentile rank of its |twinScore|
among all n(n-1) directed pairs (0 = top), and whether it lands in the top-k.
Grouped by sign, reciprocity, source out-degree, target in-degree, and the pair's
co-expression |rho(t2)|.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, sys
import numpy as np, pandas as pd

H = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/synthetic_network_analysis'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, H)
from benchmarks.network_benchmarks.score import score_real_networks_analytic_zhet_fixed as Z

RN = Z.REAL_NET_ROOT / "twinfer_inference_nofilter"
NETS = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt", "EMT": "EMT.txt"}

rows = []
for net, topo in NETS.items():
    M = np.loadtxt(Z.TOPO_ROOT / topo, delimiter=",", dtype=int)
    n = M.shape[0]
    g = [f"gene_{i+1}" for i in range(n)]
    out_deg = M.astype(bool).sum(1)          # # targets each gene regulates
    in_deg = M.astype(bool).sum(0)           # # regulators each gene has
    poss = [(g[i], g[j]) for i in range(n) for j in range(n) if i != j]
    true = {(g[i], g[j]): M[i, j] for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    recip = {(a, b) for (a, b) in true if (b, a) in true}
    k = len(true)

    # mean |twinScore| over reps
    jfs = sorted(glob.glob(str(RN / f"{net}_rep_*_all_results.json")))
    mats = []
    r2mats = []
    for jf in jfs:
        res = Z.analytic_zhet_fixed(jf)
        if res is None:
            continue
        mag, _ = res
        mats.append(np.array([mag.get(p, np.nan) for p in poss]))
        d = json.load(open(jf)); C = d["correlations"]
        R2 = pd.DataFrame(C["gene_t2"]["data"], index=C["gene_t2"]["index"], columns=C["gene_t2"]["columns"])
        r2mats.append(np.array([abs(R2.loc[a, b]) for a, b in poss]))
    if not mats:
        continue
    score = np.nanmean(mats, axis=0)
    absr2 = np.nanmean(r2mats, axis=0)
    order = np.argsort(-np.where(np.isfinite(score), score, -1e9))
    rankpct = np.empty(len(poss)); rankpct[order] = np.arange(len(poss)) / len(poss)
    topk = {poss[i] for i in order[:k]}

    for idx, (a, b) in enumerate(poss):
        if (a, b) not in true:
            continue
        si, ti = int(a.split("_")[1]) - 1, int(b.split("_")[1]) - 1
        rows.append(dict(
            net=net, edge=f"{a}->{b}", sign="activating" if true[(a, b)] > 0 else "repressing",
            reciprocal=(a, b) in recip,
            src_hub=out_deg[si] >= np.median(out_deg[out_deg > 0]),
            tgt_hub=in_deg[ti] >= np.median(in_deg[in_deg > 0]),
            src_outdeg=int(out_deg[si]), tgt_indeg=int(in_deg[ti]),
            abs_rho_t2=absr2[idx], rank_pct=rankpct[idx], in_topk=(a, b) in topk))

df = pd.DataFrame(rows)
df.to_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/edge_type_bias.csv', index=False)
print(f"{len(df)} true edges across {df.net.nunique()} networks\n")


def grp(col, label=None):
    g = df.groupby(col).agg(n=("edge", "size"),
                            recovered_topk=("in_topk", "mean"),
                            mean_rank_pct=("rank_pct", "mean"))
    print(f"--- by {label or col} ---")
    print(g.round(3).to_string()); print()


grp("sign")
grp("reciprocal", "reciprocal (mutual) edge")
grp("src_hub", "source is a hub (out-deg >= median)")
grp("tgt_hub", "target is a hub (in-deg >= median)")
df["rho_bin"] = pd.qcut(df.abs_rho_t2, 4, labels=["|rho| Q1 (low)", "Q2", "Q3", "Q4 (high)"])
grp("rho_bin", "pair co-expression |rho(t2)| quartile")

# corr: does edge co-expression predict recovery?
print("Spearman(|rho_t2| , rank_pct) =",
      round(df[["abs_rho_t2", "rank_pct"]].corr("spearman").iloc[0, 1], 3),
      "  (negative => higher co-expression -> better/lower rank)")
print("mean |rho_t2|  recovered vs missed:",
      {k: round(v, 3) for k, v in df.groupby("in_topk").abs_rho_t2.mean().items()})
