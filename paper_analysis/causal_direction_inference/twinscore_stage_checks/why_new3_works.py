"""Why |z_rho| - |z_C| + |z_U| works: z_rho, z_C, z_U by ground-truth pair class (direct edge | indirect (path >= 2) | co-regulated (common ancestor, no path) | unrelated), and AUPRC x base of each term for
'direct edge vs <class>' (positives = direct-edge pairs; negatives = the named class only).  Design X (1500 twins + 3000 singletons, 20 h), sweep + mixed pooled (and by family).  Output: <O>/why_new3_works_*.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
d = d.dropna(subset=["z_rho", "z_C", "z_U"]).copy(); d = d[~((d.fam == "mixed") & (d.sim > 2))]
d["azr"] = d.z_rho.abs(); d["azC"] = d.z_C.abs(); d["azU"] = d.z_U.abs(); d["new3"] = d.azr - d.azC + d.azU; d["soft"] = d.azr - 0.5 * d.azC + d.azU
order = ["direct edge", "indirect", "co-regulated", "unrelated"]; pd.set_option("display.width", 250)
def desc(x):
    g = x.groupby("cls"); t = g.agg(n=("azr", "size"), med_abs_zrho=("azr", "median"), med_abs_zC=("azC", "median"), med_abs_zU=("azU", "median"), pct_zC_sig=("azC", lambda v: 100 * (v > 1.96).mean()), pct_zU_sig=("azU", lambda v: 100 * (v > 1.96).mean()),
                   mean_zrho=("azr", "mean"), mean_zC=("azC", "mean"), mean_zU=("azU", "mean"), mean_new3=("new3", "mean"), med_new3=("new3", "median")).reindex(order)
    return t.round(2)
for lab, x in (("ALL (sweep + mixed)", d), ("sweep", d[d.fam == "sweep"]), ("mixed", d[d.fam == "mixed"])):
    print("\n" + lab); print(desc(x).to_string())
desc(d).to_csv(O2 + "why_new3_works_by_class.csv")
SC = {"|z_rho|": "azr", "|z_C|": "azC", "|z_U|": "azU", "|z_rho|-|z_C|": None, "|z_rho|+|z_U|": None, "new3 b=1": "new3", "soft b=0.5": "soft"}
d["|z_rho|-|z_C|"] = d.azr - d.azC; d["|z_rho|+|z_U|"] = d.azr + d.azU; SC["|z_rho|-|z_C|"] = "|z_rho|-|z_C|"; SC["|z_rho|+|z_U|"] = "|z_rho|+|z_U|"
rows = []
for lab, x in (("ALL", d), ("sweep", d[d.fam == "sweep"]), ("mixed", d[d.fam == "mixed"])):
    for neg in ("indirect", "co-regulated", "unrelated", "all non-edges"):
        y = x[(x.cls == "direct edge") | ((x.cls != "direct edge") if neg == "all non-edges" else (x.cls == neg))]; pos = (y.cls == "direct edge").astype(int); b = pos.mean()
        # per (net, sim) panel so that scores are compared within a panel, then averaged
        aps = {k: [] for k in SC}
        for _, z in y.groupby(["raw", "sim"]):
            p = (z.cls == "direct edge").astype(int)
            if p.sum() in (0, len(z)): continue
            for k, c in SC.items(): aps[k].append(average_precision_score(p, z[c]) / p.mean())
        rows.append(dict(family=lab, direct_vs=neg, n=len(y), base=round(b, 2), **{k: round(float(np.mean(v)), 2) for k, v in aps.items()}))
T = pd.DataFrame(rows); print("\nAUPRC x base for 'direct edge vs <class>' (mean over (network, sim) panels; 1.0 = chance):"); print(T.to_string(index=False)); T.to_csv(O2 + "why_new3_works_direct_vs_class.csv", index=False)
