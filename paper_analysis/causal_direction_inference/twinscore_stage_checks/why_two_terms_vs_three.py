"""Per pair class (direct edge | indirect | co-regulated | unrelated): |z_rho|, |z_C|, |z_U|, |z_rho|-|z_C| (two terms) and |z_rho|-|z_C|+|z_U| (three terms), designs 3,000 twins and 1,500 twins + 3,000 singletons;
how well |z_rho|-|z_C| tracks |z_U| (Spearman, per class); and AUPRC x base of each score for 'direct edge vs <class>' (mean over (network, sim) panels).  Output: <O>/why_two_terms_vs_three_*.csv"""
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
O2 = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
Y = pd.read_csv(O2 + "twin_design_zscores.csv"); Y = Y[Y.design == "3000 twins"].copy(); Y["key"] = [frozenset((int(a), int(b))) for a, b in zip(Y.x, Y.y)]; d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
Y3 = Y.merge(d[["raw", "sim", "key", "cls", "fam"]], on=["raw", "sim", "key"]); X = d[["raw", "sim", "key", "cls", "fam", "z_rho", "z_C", "z_U"]]
order = ["direct edge", "indirect", "co-regulated", "unrelated"]; pd.set_option("display.width", 250); out = {}
for lab, Z in (("3000 twins", Y3), ("1500 twins + 3000 singletons", X)):
    Z = Z.dropna(subset=["z_rho", "z_C", "z_U"]).copy(); Z["azr"] = Z.z_rho.abs(); Z["azC"] = Z.z_C.abs(); Z["azU"] = Z.z_U.abs(); Z["two"] = Z.azr - Z.azC; Z["three"] = Z.two + Z.azU
    t = Z.groupby("cls")[["azr", "azC", "azU", "two", "three"]].median().reindex(order).round(2); t.columns = ["|z_rho|", "|z_C|", "|z_U|", "|zr|-|zC| (2 terms)", "+|zU| (3 terms)"]; t.insert(0, "n", Z.groupby("cls").size().reindex(order))
    sp = Z.groupby("cls").apply(lambda g: spearmanr(g.two, g.azU)[0]).reindex(order).round(2); t["corr(2 terms, |z_U|)"] = sp
    print("\n" + lab + "  (medians; last column = Spearman correlation between |zr|-|zC| and |z_U| within the class)"); print(t.to_string()); out[lab] = t
    SC = {"|z_rho|": "azr", "|z_C|": "azC", "|z_U|": "azU", "2 terms": "two", "3 terms": "three"}; rows = []
    for neg in ("indirect", "co-regulated", "unrelated", "all non-edges"):
        y = Z[(Z.cls == "direct edge") | ((Z.cls != "direct edge") if neg == "all non-edges" else (Z.cls == neg))]; aps = {k: [] for k in SC}
        for _, z in y.groupby(["raw", "sim"]):
            p = (z.cls == "direct edge").astype(int)
            if p.sum() in (0, len(z)): continue
            for k, c in SC.items(): aps[k].append(average_precision_score(p, z[c]) / p.mean())
        rows.append(dict(direct_vs=neg, n=len(y), base=round((y.cls == "direct edge").mean(), 2), **{k: round(float(np.mean(v)), 2) for k, v in aps.items()}))
    print("AUPRC x base, direct edge vs <class> (1.0 = chance):"); print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(O2 + f"why_two_terms_vs_three_auprc_{lab.split()[0]}.csv", index=False); t.to_csv(O2 + f"why_two_terms_vs_three_medians_{lab.split()[0]}.csv")
