"""Follow-up to scenario_ranking_design.py: wider (beta, gamma) grid for score = |z_rho| - beta|z_C| + gamma|z_U|, top-k precision, and the share of top-k false positives that are fan-out pairs."""
import re
src = open("scenario_ranking_design.py").read(); exec(src.split("print(\"PART 1")[0])
import numpy as np, pandas as pd, itertools
from sklearn.metrics import average_precision_score
def sc(x, b, g): return x.azr - b * x.azC + g * x.azU
grid = [(0, 0)] + [(b, g) for b in (0.5, 1, 1.5, 2, 3) for g in (0, 0.5, 1, 2)] + [(0, 1)]
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    if x.edge.sum() == 0: continue
    k = int(x.edge.sum()); base = x.edge.mean()
    for b, g in grid:
        s = sc(x, b, g).to_numpy(); top = np.argsort(-s, kind="stable")[:k]; fp = x.iloc[top][x.iloc[top].edge == 0]
        rows.append(dict(fam=x.fam.iloc[0], group=raw.rsplit("_rep", 1)[0], b=b, g=g, apx=average_precision_score(x.edge, s) / base, topk=x.iloc[top].edge.sum() / k / base, fp=len(fp), fp_fanout=int(fp.fanout.sum())))
R = pd.DataFrame(rows); G = R.groupby(["fam", "group", "b", "g"]).agg(apx=("apx", "mean"), topk=("topk", "mean"), fp=("fp", "sum"), fp_fanout=("fp_fanout", "sum")).reset_index()
F = G.groupby(["fam", "b", "g"]).agg(apx=("apx", "mean"), topk=("topk", "mean"), fp=("fp", "sum"), fp_fanout=("fp_fanout", "sum")).reset_index(); F["fanout_share_of_FP"] = (F.fp_fanout / F.fp).round(2)
W = F.pivot(index=["b", "g"], columns="fam", values=["apx", "topk", "fanout_share_of_FP"]); W["apx_avg"] = W["apx"].mean(axis=1)
pd.set_option("display.width", 250); print(W.sort_values("apx_avg", ascending=False).round(3).head(14).to_string()); print("\nbaseline |z_rho| (b=0,g=0):"); print(W.loc[(0, 0)].round(3).to_string())
# paired per-group comparison best vs |z_rho|
best = W["apx_avg"].idxmax(); P = G[(G.b == best[0]) & (G.g == best[1])].set_index(["fam", "group"]).apx - G[(G.b == 0) & (G.g == 0)].set_index(["fam", "group"]).apx
print("\nbest", best, "- |z_rho| per group: wins", int((P > 0).sum()), "of", len(P), "| mean diff", round(P.mean(), 3), "| sweep", round(P.loc["sweep"].mean(), 3), "mixed", round(P.loc["mixed"].mean(), 3))
