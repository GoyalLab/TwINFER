#!/usr/bin/env python3
"""Back-to-back bar chart (as panel b of yscher's figure) for FM01 and FM06 only: one row per Gavish et al. 2023 meta-program group; left = number of the top-1000 UNSHARED pairs whose target is in the group, right = the same for the
top-1000 SHARED pairs; FM01 (lighter) above FM06 (darker).  ChIP-Atlas candidate edges (FM01 epidermis list), Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded; a target in several groups counts in each.
Under each group name: the top 3 TFs most ASSOCIATED with the group = TFs with >= 3 pairs whose target is in the group in BOTH datasets (pairs of the union of the top-1000 unshared and shared lists), ranked by the smaller of the two enrichments, and only TFs enriched >= 1.5-fold in both datasets are shown (so a group can have fewer than three),
enrichment = (the TF's fraction of its list pairs whose target is in the group) / (the group's fraction of all list pairs), shown as FM01 | FM06 fold.  (Ranking by raw pair counts only returns MITF, FOSL1 and TFAP2A for every group, the TFs with the most pairs overall.)
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_program_group_butterfly_TFs.{pdf,png} and fm01_fm06_top3_TFs_per_program_group_enrichment.csv.   usage: program_group_butterfly_TFs_plot.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
X = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/41586_2023_6130_MOESM6_ESM.xlsx", sheet_name="Cancer MPs", header=None)
mp = {int(str(X.iloc[0, j]).split()[0][2:]): set(str(g) for g in X.iloc[1:, j].dropna()) for j in range(X.shape[1])}; cc = set().union(*[mp[m] for m in (1, 2, 3, 4)])
T = pd.read_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_unshared_share.csv")
MINK = 3; MINENR = 1.5; D = {}
for ds in ("FM01", "FM06"):
    P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))].reset_index(drop=True)
    u = set(P.assign(s=P.z_U.abs()).nlargest(N, "s").index); sh = set(P.assign(s=P.h.abs()).nlargest(N, "s").index); D[ds] = P.loc[sorted(u | sh)]
GR = {"Cell cycle": [1, 2, 3, 4], "MYC": [20], "Secreted": [22, 23], "Stress, hypoxia": [5, 6, 7], "Astrocyte, NPC, OPC": [25, 26, 27, 28, 29], "Skin pigmentation": [32], "Mesenchymal": [12, 13, 14, 15, 16]}; tops = {}; rows = []
for g, ms in GR.items():
    gs = set().union(*[mp[m] for m in ms]); out = {}
    for ds in ("FM01", "FM06"):
        d = D[ds]; ing = d.gene.isin(gs); f = ing.mean(); out[ds] = pd.DataFrame({"n_all": d.TF.value_counts(), "k": d[ing].TF.value_counts()}).fillna(0).assign(enr=lambda x, f=f: (x.k / x.n_all) / f)
    t = out["FM01"].join(out["FM06"], lsuffix="_01", rsuffix="_06", how="inner"); t = t[(t.k_01 >= MINK) & (t.k_06 >= MINK)].copy(); t["score"] = np.minimum(t.enr_01, t.enr_06); t = t[t.score >= MINENR].sort_values("score", ascending=False).head(3); tops[g] = t
    for tf, r in t.iterrows(): rows.append(dict(group=g, TF=tf, FM01_group_pairs=int(r.k_01), FM01_TF_pairs=int(r.n_all_01), FM01_enrichment=r.enr_01, FM06_group_pairs=int(r.k_06), FM06_TF_pairs=int(r.n_all_06), FM06_enrichment=r.enr_06))
pd.DataFrame(rows).to_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_top3_TFs_per_program_group_enrichment.csv", index=False)
order = ["Cell cycle", "MYC", "Secreted", "Stress, hypoxia", "Astrocyte, NPC, OPC", "Skin pigmentation", "Mesenchymal"]; yi = {g: i * 1.75 for i, g in enumerate(order)}   # room under every group name for its three TFs
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
C = {("FM01", "unshared"): "#b3a7ee", ("FM06", "unshared"): "#6a54d3", ("FM01", "shared"): "#8adcbf", ("FM06", "shared"): "#1fb07b"}; GAP = 62; H = 0.34; YB = yi["Mesenchymal"] + 1.45
fig, ax = plt.subplots(figsize=(11.2, 10.2)); fig.patch.set_alpha(0); ax.set_facecolor("none")
for g in order:
    for ds, off in (("FM01", -0.19), ("FM06", 0.19)):
        r = T[(T.dataset == ds) & (T.group == g)].iloc[0]; nu, ns = int(r.n_unshared), int(r.n_shared); y = yi[g] + off
        ax.barh(y, nu, left=-GAP - nu, height=H, color=C[(ds, "unshared")], edgecolor="none"); ax.barh(y, ns, left=GAP, height=H, color=C[(ds, "shared")], edgecolor="none")
        ax.text(-GAP - nu - 4, y, str(nu), ha="right", va="center", fontsize=9, color="#333"); ax.text(GAP + ns + 4, y, str(ns), ha="left", va="center", fontsize=9, color="#333")
    ax.text(0, yi[g] + 0.02, g, ha="center", va="center", fontsize=11, color="#111")
for v in (0, 100, 200):
    for sgn in (1, -1):
        ax.plot([sgn * (GAP + v)] * 2, [-0.6, YB], color="#ececec", lw=0.8, zorder=0); ax.text(sgn * (GAP + v), YB + 0.05, str(v), ha="center", va="top", fontsize=8.5, color="#888")
ax.text(-GAP - 100, -0.95, "Unshared top 1000", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#5a46bd"); ax.text(GAP + 100, -0.95, "Shared top 1000", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#17986b")
for g in order:                                           # compact neutral list in the centre column under each group name (not over the bars)
    for k, (tf, r) in enumerate(tops[g].iterrows()): ax.text(0, yi[g] + 0.30 + 0.25 * (k + 1), f"{tf}   {r.enr_01:.1f}\u00d7 | {r.enr_06:.1f}\u00d7", ha="center", va="center", fontsize=8.3, color="#444")
    # (a group can have fewer than three TFs: the rule is stated in the footnote)
ax.set_xlim(-GAP - 235, GAP + 235); ax.set_ylim(YB + 0.4, -1.0); ax.axis("off"); ax.text(0, YB + 0.62, "bars: number of pairs out of 1000 whose target is in the group.  Under each group name: TFs with >= 3 pairs in both datasets and >= 1.5-fold enrichment for the group (top 3; fold FM01 | FM06)", ha="center", va="top", fontsize=9, color="#777")
fig.legend(handles=[Patch(color=C[("FM01", "unshared")], label="FM01 unshared"), Patch(color=C[("FM06", "unshared")], label="FM06 unshared"), Patch(color=C[("FM01", "shared")], label="FM01 shared"), Patch(color=C[("FM06", "shared")], label="FM06 shared")], loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Which meta-program group do the targets of the top-1000 pairs belong to?", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=0.985)
fig.text(0.01, 0.925, "ChIP-Atlas edges; lighter bar = FM01, darker = FM06; a target counts in every group it belongs to (Gavish et al. 2023)", fontsize=8.3, color="#666")
fig.tight_layout(rect=[0, 0.05, 1, 0.91])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_program_group_butterfly_TFs"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=140, facecolor="white"); print("saved", out)
