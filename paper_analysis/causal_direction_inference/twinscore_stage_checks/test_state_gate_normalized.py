"""State gate with normalization instead of the offset kappa.  I = 1[|z_rho_s| > tau].  Branch scores: state: |z_U|; no state: soft = |z_rho| - 0.5|z_rho_s| + |z_U|.
  kappa     : I(|z_U| + kappa) + (1-I) soft, kappa = panel median of (|z_rho| - 0.5|z_rho_s|) over non-state pairs (previous version)
  panel-s   : I s(|z_U|) + (1-I) s(soft), s = (v - mean)/sd over ALL pairs of the panel (one simulation) -- same standardization for both branches
  branch-s  : each branch standardized within its own pairs of the panel
  branch-pct: percentile rank within its own branch
Networks, twin-only N = 1500 / 3000 / 6000, AUPRC x base, mean over groups.  Two-gene: P(A->B > A,B) compares replicates of the same state type (same branch), so any monotone within-branch normalization leaves it unchanged."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int); Z["N"] = Z.design.str.extract(r"(\d+)")[0].astype(int)
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs()
def s(v): v = np.asarray(v, float); sd = v.std(ddof=1) if len(v) > 1 else 0; return (v - v.mean()) / (sd if sd > 0 else 1.0)
def pct(v): v = np.asarray(v, float); return (rankdata(v) - 1) / max(len(v) - 1, 1)
rows = []
for (N, raw, sim), x in Z.groupby(["N", "raw", "sim"]):
    if N not in (1500, 3000, 6000) or x.edge.sum() in (0, len(x)): continue
    b = x.edge.mean(); soft = (x.azr - 0.5 * x.azC + x.azU).to_numpy(); u = x.azU.to_numpy(); r = dict(N=N, group=raw.rsplit("_rep", 1)[0]); r["|z_U|"] = average_precision_score(x.edge, u) / b; r["soft"] = average_precision_score(x.edge, soft) / b
    for tau in (5, 10):
        I = (x.azC > tau).to_numpy(); kap = np.median((x.azr - 0.5 * x.azC).to_numpy()[~I]) if (~I).any() and I.any() else 0.0
        sc = {"kappa": np.where(I, u + kap, soft), "panel-s": np.where(I, s(u), s(soft))}
        bs = np.empty(len(x)); bp = np.empty(len(x))
        for m, v in ((I, u), (~I, soft)):
            if m.sum() > 1: bs[m] = s(v[m]); bp[m] = pct(v[m])
            else: bs[m] = 0.0; bp[m] = 0.5
        sc["branch-s"] = bs; sc["branch-pct"] = bp
        for k, v in sc.items(): r[f"{k} tau={tau}"] = average_precision_score(x.edge, v) / b
    rows.append(r)
G = pd.DataFrame(rows).groupby(["N", "group"]).mean(numeric_only=True).reset_index(); cols = [c for c in G.columns if c not in ("N", "group")]
print("NETWORKS: AUPRC x base, mean over groups"); print(G.groupby("N")[cols].mean().round(3).T.to_string())
G.to_csv(O2 + "test_state_gate_normalized_by_group.csv", index=False)
