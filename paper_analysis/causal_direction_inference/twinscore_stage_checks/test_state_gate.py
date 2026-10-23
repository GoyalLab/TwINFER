"""Indicator-gated score: I = 1[ |z_rho_s| > tau ] (strong shared state).
   score = I * ( |z_U| + kappa )  +  (1 - I) * ( |z_rho| - 0.5 |z_rho_s| + |z_U| )
kappa = median, over the panel's non-state pairs, of ( |z_rho| - 0.5|z_rho_s| ) (label-free offset that puts the two branches on the same scale); kappa = 0 also shown.
Tested on (a) the figure 2 two-gene scenarios (P(A->B replicate > A,B replicate), per state type) and (b) the network groups (AUPRC x base).  z-scores: original (each over its own null SD), twin-only designs."""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; pd.set_option("display.width", 250)
def gated(azr, azC, azU, tau, kappa):
    I = azC > tau; return np.where(I, azU + kappa, azr - 0.5 * azC + azU)
# (a) two-gene
sc = pd.read_csv(O2 + "twin_size_scan_figure2.csv"); sc["azr"] = sc.z_rho.abs(); sc["azC"] = sc.z_C.abs(); sc["azU"] = sc.z_U.abs(); sc["soft"] = sc.azr - 0.5 * sc.azC + sc.azU
P = lambda b, a, c: float(np.mean(b[c].values[:, None] > a[c].values[None, :])); rows = []
for N in (1500, 3000):
    x = sc[sc.N == N].copy()
    for tau in (2, 5, 10): x[f"g{tau}"] = gated(x.azr, x.azC, x.azU, tau, 0.0)
    for st, (a_, b_) in {"single state": ("A_B (single state)", "A_to_B (single state)"), "2 states": ("A_B (two states)", "A_to_B (two states)"), "drift": ("A_B (drift)", "A_to_B (drift)")}.items():
        r = dict(N=N, scenario=st)
        for k in ("azU", "soft", "g2", "g5", "g10"): r[k] = round(P(x[x.scenario == b_], x[x.scenario == a_], k), 2)
        rows.append(r)
print("(a) TWO-GENE: P(A->B replicate > A,B replicate)  [kappa = 0; the gate is decided per replicate]"); print(pd.DataFrame(rows).rename(columns={"azU": "|z_U|", "soft": "soft combined", "g2": "gate tau=2", "g5": "gate tau=5", "g10": "gate tau=10"}).to_string(index=False))
x = sc[sc.N == 3000]; print("\nshare of replicates in the state branch (tau=5):", x.assign(I=x.azC > 5).groupby("scenario").I.mean().round(2).to_dict())
# (b) networks
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Z = Y.merge(d[["raw", "sim", "key", "rel"]], on=["raw", "sim", "key"]).dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["edge"] = Z.rel.str.startswith("edge").astype(int); Z["N"] = Z.design.str.extract(r"(\d+)")[0].astype(int)
Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs(); rows = []
for (N, raw, sim), x in Z.groupby(["N", "raw", "sim"]):
    if N not in (1500, 3000, 6000) or x.edge.sum() in (0, len(x)): continue
    b = x.edge.mean(); base_ = x.azr - 0.5 * x.azC; r = dict(N=N, group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0] if "fam" in x else None)
    r["|z_U|"] = average_precision_score(x.edge, x.azU) / b; r["soft combined"] = average_precision_score(x.edge, x.azr - 0.5 * x.azC + x.azU) / b
    for tau in (2, 5, 10):
        I = x.azC > tau; kap = base_[~I].median() if (~I).any() else 0.0
        r[f"gate tau={tau} (kappa=0)"] = average_precision_score(x.edge, gated(x.azr, x.azC, x.azU, tau, 0.0)) / b; r[f"gate tau={tau} (kappa=median)"] = average_precision_score(x.edge, gated(x.azr, x.azC, x.azU, tau, kap)) / b
        r[f"share in state branch tau={tau}"] = I.mean()
    rows.append(r)
R = pd.DataFrame(rows); G = R.groupby(["N", "group"]).mean(numeric_only=True).reset_index(); cols = [c for c in G.columns if c not in ("N", "group") and not c.startswith("share")]
print("\n(b) NETWORKS: AUPRC x base, mean over groups"); print(G.groupby("N")[cols].mean().round(3).T.to_string()); print("\nshare of pairs in the state branch:"); print(G.groupby("N")[[c for c in G.columns if c.startswith("share")]].mean().round(3).to_string())
