"""Indicator-style terms: score = |z_rho| - b|z_C| + g|z_U| + u*1[|z_U|>1.96] - c*1[|z_C|>1.96] + w*1[|z_U|>1.96 & |z_C|>1.96]  (w = scenario-3 bonus).  Same pairs/positives/base as scenario_ranking_design.py."""
exec(open("scenario_ranking_design.py").read().split("print(\"PART 1")[0])
import numpy as np, pandas as pd, itertools
from sklearn.metrics import average_precision_score
d["IU"] = (d.azU > T).astype(float); d["IC"] = (d.azC > T).astype(float)
grid = [(b, g, u, c, w) for (b, g) in ((0, 0), (1, 1)) for u in (0, 1, 2, 3) for c in (0, 1, 2, 3) for w in (0, 1, 2)]
rows = []
for (raw, sim), x in d.groupby(["raw", "sim"]):
    if x.edge.sum() == 0: continue
    base = x.edge.mean(); k = int(x.edge.sum()); fam = x.fam.iloc[0]; grp = raw.rsplit("_rep", 1)[0]
    for (b, g, u, c, w) in grid:
        s = (x.azr - b * x.azC + g * x.azU + u * x.IU - c * x.IC + w * x.IU * x.IC).to_numpy(); top = np.argsort(-s, kind="stable")[:k]
        rows.append(dict(fam=fam, group=grp, b=b, g=g, u=u, c=c, w=w, apx=average_precision_score(x.edge, s) / base, topk=x.edge.to_numpy()[top].sum() / k / base))
R = pd.DataFrame(rows); G = R.groupby(["fam", "group", "b", "g", "u", "c", "w"]).mean(numeric_only=True).reset_index()
F = G.groupby(["fam", "b", "g", "u", "c", "w"])[["apx", "topk"]].mean().unstack("fam"); F.columns = [f"{a}_{f}" for a, f in F.columns]; F["apx_avg"] = F[["apx_sweep", "apx_mixed"]].mean(axis=1)
pd.set_option("display.width", 250); print(F.sort_values("apx_avg", ascending=False).round(3).head(15).to_string())
for key, lab in [((0, 0, 0, 0, 0), "|z_rho|"), ((1, 1, 0, 0, 0), "|z_rho|-|z_C|+|z_U| (continuous best)"), ((0, 0, 1, 1, 0), "|z_rho|+1[U]-1[C]"), ((0, 0, 2, 2, 0), "|z_rho|+2[U]-2[C]"), ((0, 0, 0, 0, 2), "|z_rho| + 2 bonus S3")]: print(lab, F.loc[key].round(3).to_dict())
best = F.apx_avg.idxmax(); base = G[(G.b == 0) & (G.g == 0) & (G.u == 0) & (G.c == 0) & (G.w == 0)].set_index(["fam", "group"]).apx; cont = G[(G.b == 1) & (G.g == 1) & (G.u == 0) & (G.c == 0) & (G.w == 0)].set_index(["fam", "group"]).apx
bb = G[(G.b == best[0]) & (G.g == best[1]) & (G.u == best[2]) & (G.c == best[3]) & (G.w == best[4])].set_index(["fam", "group"]).apx
print("\nbest", best, "vs |z_rho|: wins", int((bb > base).sum()), "/", len(bb), "mean diff %.3f" % (bb - base).mean(), "| vs continuous best: wins", int((bb > cont).sum()), "mean diff %.3f" % (bb - cont).mean())
# cross-family: pick on one family, evaluate on the other
for tr, te in (("sweep", "mixed"), ("mixed", "sweep")):
    pick = F[f"apx_{tr}"].idxmax(); print(f"chosen on {tr}: {pick} -> {tr} {F.loc[pick, 'apx_' + tr]:.3f}, {te} {F.loc[pick, 'apx_' + te]:.3f}  (|z_rho| on {te}: {F.loc[(0,0,0,0,0), 'apx_' + te]:.3f}; continuous best on {te}: {F.loc[(1,1,0,0,0), 'apx_' + te]:.3f})")
