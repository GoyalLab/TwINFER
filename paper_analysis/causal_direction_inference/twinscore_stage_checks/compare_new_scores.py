"""All new scores vs old, per group, undirected AUPRC x base (mean over replicates).
New: new3 = |z_rho|-|z_C|+|z_U| on design X (1500 twins + 3000 singletons) and on 3000 twins; zdag = s(max |z_dagger|) (direction-free cross-sample coupling, infer.py partition);
     zdag + 0.5*s(new3) (X).  Old: TODO4v2 (heatmap value = mean over all simulation replicates of each topology instance), TODO4v2 recomputed on the same simulations as the new scores (summary_table_todo4v2.todo4v2_row_for_dataset, auprc_undirected / base),
     BEELINE PEARSON / PPCOR / PIDC / GRNBOOST2 (heatmap).  Bold = beats TODO4v2 (both versions) and all four competitors."""
import glob, json, re, os, numpy as np, pandas as pd
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S
O = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; B = A + "code/Beeline/inputs/"
x = pd.read_csv(O + "compare_old_undirected_by_group.csv", index_col=0)
c = pd.read_csv(O + "todo4v2_undirected_components_replicates.csv").groupby("group")[["new3: |z_rho|-|z_C|+|z_U|", "term: zdag", "s(zdag) + 0.5 s(new3)"]].mean()
y = pd.read_csv(O + "twin_design_compare_by_group.csv"); y3 = y[y.design.str.startswith("Y3")].set_index("group")["new3 apx"]; x_new3 = y[y.design.str.startswith("X")].set_index("group")["new3 apx"]
# TODO4v2 recomputed on all sims, same base as the new scores
base = y[y.design.str.startswith("X")].set_index("group").base
rows = []
for jd, gtroot in ((A + "analysis_data/mixed_network_sweep/twinfer_inference_allpairs", B + "mixed_network_sweep/"), (A + "analysis_data/network_sweep_final/twinfer_inference_allpairs", B + "network_sweep_final/"), (A + "analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs", B + "e13_pos100/")):
    for f in sorted(glob.glob(jd + "/*_all_results.json")):
        m = re.match(r"(.+_rep\d+)_rep(\d+)_all_results\.json", os.path.basename(f))
        if not m: continue
        gp = gtroot + m.group(1) + "/GroundTruthNetwork.csv"
        if not os.path.exists(gp): continue
        r = S.todo4v2_row_for_dataset(f, pd.read_csv(gp))
        if r is not None: rows.append(dict(group=m.group(1).rsplit("_rep", 1)[0], sim=int(m.group(2)), ap=r["auprc_undirected"]))
T4 = pd.DataFrame(rows); T4 = T4[~((T4.group.str.contains("_c\\d_a")) & (T4.sim > 2))]            # mixed: only the original replicates 0-2
t4 = T4.groupby("group").ap.mean() / base
D = pd.DataFrame({"new3 (X)": x_new3, "new3 (3000 twins)": y3, "zdag (cross-sample)": c["term: zdag"], "zdag+0.5 new3": c["s(zdag) + 0.5 s(new3)"], "TODO4v2 (same sims)": t4}).join(x[["old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]], how="inner").dropna(subset=["old TODO4v2"])   # drops the 17-edge set (no old values)
D["best old"] = D[["TODO4v2 (same sims)", "old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]].max(axis=1)
def nm(g):
    m = re.match(r"grn_n(\d+)_e(\d+)_c(\d+)_a(\d+)", g)
    if m: n, e, cy, a = map(int, m.groups()); return f"n{n} e{e} c{cy}" + (" autoreg" if a else "")
    return {"grn_n6_e5_pos100_density": "sweep e5", "grn_n6_e9_pos0_sign_ratio": "sweep e9 all neg", "grn_n6_e9_pos100_center": "sweep e9 all pos", "grn_n6_e9_pos50_sign_ratio": "sweep e9 balanced", "n6_e13_pos100": "sweep e13"}[g]
D["name"] = D.index.map(nm); D = D.sort_index(key=lambda s: s.map(lambda q: (2 if "_c" not in q else 0 if q.startswith("grn_n6") else 1, q)))
new = ["new3 (X)", "new3 (3000 twins)", "zdag (cross-sample)", "zdag+0.5 new3"]
print("| Group | new3 (X) | new3 (3000 twins) | zdag | zdag+0.5 new3 | TODO4v2 (same sims) | TODO4v2 (heatmap) | PEARSON | PPCOR | PIDC | GRNBOOST2 |\n|---|---|---|---|---|---|---|---|---|---|---|")
f = lambda v, b: f"**{v:.2f}**" if b else f"{v:.2f}"
for g, r in D.iterrows(): print(f"| {r['name']} | " + " | ".join(f(r[k], r[k] > r["best old"]) for k in new) + f" | {r['TODO4v2 (same sims)']:.2f} | {r['old TODO4v2']:.2f} | {r['old PEARSON']:.2f} | {r['old PPCOR']:.2f} | {r['old PIDC']:.2f} | {r['old GRNBOOST2']:.2f} |")
mixed = D.index.str.contains("_c\\d_a"); cols = new + ["TODO4v2 (same sims)", "old TODO4v2", "old PEARSON", "old PPCOR", "old PIDC", "old GRNBOOST2"]
print("\nmean over groups:"); print(pd.DataFrame({"mixed (%d)" % mixed.sum(): D[mixed][cols].mean(), "sweep (%d)" % (~mixed).sum(): D[~mixed][cols].mean(), "all": D[cols].mean()}).round(3).to_string())
print("\ngroups beating ALL old scores:", {k: int((D[k] > D["best old"]).sum()) for k in new}, "of", len(D))
print("groups beating TODO4v2 (same sims):", {k: int((D[k] > D["TODO4v2 (same sims)"]).sum()) for k in new})
D.round(3).to_csv(O + "compare_new_scores_by_group.csv")
