#!/usr/bin/env python3
"""Back-to-back bar chart, one row per OFFICIAL Gavish et al. 2023 hallmark (11 rows): Cell cycle (MP1-4), Stress or hypoxia (5-7), Protein regulation (8-11), Mesenchymal (12-16), Interferon and MHC II (17-18),
Senescence (19), Oncogenic = MYC (20), Respiration (21), Secreted (22-23), Cilia (24), and Lineage related (25-37: neural, PDAC classical/alveolar/skin pigmentation, haematopoietic). MP38-41 (Unassigned) are not shown.  Left = number of the top-1000 UNSHARED pairs whose target is in the group, right = the same for the top-1000 SHARED pairs; FM01 (lighter) above FM06 (darker).
ChIP-Atlas candidate edges (FM01 epidermis list), Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded; a target in several groups counts in each.  Under each group name: up to 3 TFs with >= 3 pairs in the group in BOTH
datasets (pairs of the union of the two top-1000 lists) and >= 1.5-fold enrichment in both (enrichment = the TF's fraction of its pairs whose target is in the group / the group's fraction of all pairs), ranked by the smaller enrichment; fold FM01 | FM06.
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_11_hallmarks_butterfly_TF_names.{pdf,png} and fm01_fm06_11_hallmarks_counts_and_TFs.csv.   usage: program_11_hallmarks_butterfly_plot.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000; MINK = 3; MINENR = 1.5
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
X = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/41586_2023_6130_MOESM6_ESM.xlsx", sheet_name="Cancer MPs", header=None); mp = {int(str(X.iloc[0, j]).split()[0][2:]): set(str(g) for g in X.iloc[1:, j].dropna()) for j in range(X.shape[1])}
GROUPS = [("Cell cycle", [1, 2, 3, 4]), ("Stress or hypoxia", [5, 6, 7]), ("Protein regulation", [8, 9, 10, 11]), ("Mesenchymal", [12, 13, 14, 15, 16]), ("Interferon and MHC II", [17, 18]), ("Senescence", [19]), ("Oncogenic (MYC)", [20]), ("Respiration", [21]), ("Secreted", [22, 23]), ("Cilia", [24]),
          ("Lineage related", list(range(25, 38)))]   # the 11 hallmarks of Gavish et al. 2023 (Lineage related = neural 25-29 + other 30-32 + haematopoietic 33-37); MP38-41 'Unassigned' are not a hallmark and are not shown
D, L = {}, {}
for ds in ("FM01", "FM06"):
    P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))].reset_index(drop=True)
    u = P.assign(s=P.z_U.abs()).nlargest(N, "s"); s = P.assign(s=P.h.abs()).nlargest(N, "s"); L[ds] = (u, s); D[ds] = P.loc[sorted(set(u.index) | set(s.index))]
rec, tops = [], {}
for g, ms in GROUPS:
    gs = set().union(*[mp[m] for m in ms]); r = dict(group=g, programs="MP" + ",".join(map(str, ms))); out = {}
    for ds in ("FM01", "FM06"):
        u, s = L[ds]; r[f"{ds}_unshared"] = int(u.gene.isin(gs).sum()); r[f"{ds}_shared"] = int(s.gene.isin(gs).sum())
        d = D[ds]; ing = d.gene.isin(gs); f = ing.mean(); out[ds] = pd.DataFrame({"n_all": d.TF.value_counts(), "k": d[ing].TF.value_counts()}).fillna(0).assign(enr=lambda x, f=f: (x.k / x.n_all) / f if f > 0 else 0.0)
    t = out["FM01"].join(out["FM06"], lsuffix="_01", rsuffix="_06", how="inner"); t = t[(t.k_01 >= MINK) & (t.k_06 >= MINK)].copy(); t["score"] = np.minimum(t.enr_01, t.enr_06); t = t[t.score >= MINENR].sort_values("score", ascending=False).head(3); tops[g] = t
    r["TFs"] = ", ".join(t.index); r["TF_enrichment_FM01_FM06"] = "; ".join(f"{i} {x.enr_01:.1f}|{x.enr_06:.1f}" for i, x in t.iterrows()); rec.append(r)
R = pd.DataFrame(rec); R.to_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_11_hallmarks_counts_and_TFs.csv", index=False)
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
C = {("FM01", "unshared"): "#b3a7ee", ("FM06", "unshared"): "#6a54d3", ("FM01", "shared"): "#8adcbf", ("FM06", "shared"): "#1fb07b"}; GAP = 62; H = 0.34; STEP = 1.55
fig, ax = plt.subplots(figsize=(11.4, 13.6)); fig.patch.set_alpha(0); ax.set_facecolor("none"); ypos = [i * STEP for i in range(len(GROUPS))]
for (g, ms), yy, r in zip(GROUPS, ypos, R.itertuples()):
    for ds, off in (("FM01", -0.19), ("FM06", 0.19)):
        nu, ns = getattr(r, f"{ds}_unshared"), getattr(r, f"{ds}_shared")
        ax.barh(yy + off, nu, left=-GAP - nu, height=H, color=C[(ds, "unshared")], edgecolor="none"); ax.barh(yy + off, ns, left=GAP, height=H, color=C[(ds, "shared")], edgecolor="none")
        ax.text(-GAP - nu - 4, yy + off, str(nu), ha="right", va="center", fontsize=9, color="#333"); ax.text(GAP + ns + 4, yy + off, str(ns), ha="left", va="center", fontsize=9, color="#333")
    ax.text(0, yy, g, ha="center", va="center", fontsize=10.8, color="#111")
    for k, (tf, x) in enumerate(tops[g].iterrows()): ax.text(0, yy + 0.30 + 0.21 * (k + 1), f"{tf}", ha="center", va="center", fontsize=8.2, color="#444")
    if not len(tops[g]): ax.text(0, yy + 0.51, "–", ha="center", va="center", fontsize=8.5, color="#aaa")
YE = ypos[-1] + 1.1
for v in (0, 100, 200):
    for sgn in (1, -1): ax.plot([sgn * (GAP + v)] * 2, [-0.8, YE - 0.4], color="#efefef", lw=0.8, zorder=0); ax.text(sgn * (GAP + v), YE - 0.2, str(v), ha="center", va="top", fontsize=8.5, color="#888")
ax.text(-GAP - 100, -1.3, "Unshared top 1000", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#5a46bd"); ax.text(GAP + 100, -1.3, "Shared top 1000", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#17986b")
ax.set_xlim(-GAP - 235, GAP + 235); ax.set_ylim(YE + 0.9, -2.0); ax.axis("off")
ax.text(0, YE + 0.55, "bars: pairs out of 1000 whose target is in the group.  Under each group: up to 3 TFs, ranked top to bottom (>= 3 pairs in the group in both datasets and >= 1.5-fold enriched in both; folds are in the csv)", ha="center", va="top", fontsize=8.5, color="#777")
fig.legend(handles=[Patch(color=C[("FM01", "unshared")], label="FM01 unshared"), Patch(color=C[("FM06", "unshared")], label="FM06 unshared"), Patch(color=C[("FM01", "shared")], label="FM01 shared"), Patch(color=C[("FM06", "shared")], label="FM06 shared")], loc="lower center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("The 11 hallmarks of Gavish et al. 2023: targets of the top-1000 unshared and shared pairs", x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.992)
fig.text(0.01, 0.973, "ChIP-Atlas edges; lighter bar = FM01, darker = FM06; a target counts in every group it belongs to (11 hallmarks of Gavish et al. 2023; MP38-41 'Unassigned' not shown)", fontsize=8.8, color="#666")
fig.tight_layout(rect=[0, 0.03, 1, 0.965])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_11_hallmarks_butterfly_TF_names"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=105, facecolor="white"); print("saved", out); print(R[["group", "FM01_unshared", "FM01_shared", "FM06_unshared", "FM06_shared", "TFs"]].to_string(index=False))
