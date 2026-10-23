#!/usr/bin/env python3
"""Back-to-back bar chart for ALL 41 Gavish et al. 2023 cancer meta-programs (individually, ordered by the seven groups of panel b of yscher's figure followed by the 20 other programs), FM01 and FM06:
left = number of the top-1000 UNSHARED pairs whose target is in the program, right = the same for the top-1000 SHARED pairs; FM01 (lighter) above FM06 (darker).  ChIP-Atlas candidate edges (FM01 epidermis list), Stage I,
no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded; a target in several programs counts in each.  Right-hand column: TFs most associated with the program = TFs with >= 3 pairs whose target is in the
program in BOTH datasets (pairs of the union of the two top-1000 lists) and >= 1.5-fold enrichment in both (enrichment = the TF's fraction of its pairs whose target is in the program / the program's fraction of all pairs), top 3 by the
smaller enrichment.  Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_all41_programs_butterfly.{pdf,png} and fm01_fm06_all41_programs_counts_and_TFs.csv.   usage: program_all41_butterfly_plot.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000; MINK = 3; MINENR = 1.5
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
X = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/41586_2023_6130_MOESM6_ESM.xlsx", sheet_name="Cancer MPs", header=None)
names = {int(str(X.iloc[0, j]).split()[0][2:]): " ".join(str(X.iloc[0, j]).split()) for j in range(X.shape[1])}; mp = {int(str(X.iloc[0, j]).split()[0][2:]): set(str(g) for g in X.iloc[1:, j].dropna()) for j in range(X.shape[1])}
GROUPS = [("Cell cycle", [1, 2, 3, 4]), ("Stress, hypoxia", [5, 6, 7]), ("Mesenchymal", [12, 13, 14, 15, 16]), ("MYC", [20]), ("Secreted", [22, 23]), ("Astrocyte, NPC, OPC", [25, 26, 27, 28, 29]), ("Skin pigmentation", [32])]
used = {m for _, ms in GROUPS for m in ms}; GROUPS.append(("Other programs", [m for m in sorted(mp) if m not in used])); rows_order = [(g, m) for g, ms in GROUPS for m in ms]
D, L = {}, {}
for ds in ("FM01", "FM06"):
    P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))].reset_index(drop=True)
    u = P.assign(s=P.z_U.abs()).nlargest(N, "s"); s = P.assign(s=P.h.abs()).nlargest(N, "s"); L[ds] = (u, s); D[ds] = P.loc[sorted(set(u.index) | set(s.index))]
rec = []
for g, m in rows_order:
    r = dict(group=g, program=names[m]); out = {}
    for ds in ("FM01", "FM06"):
        u, s = L[ds]; r[f"{ds}_unshared"] = int(u.gene.isin(mp[m]).sum()); r[f"{ds}_shared"] = int(s.gene.isin(mp[m]).sum())
        d = D[ds]; ing = d.gene.isin(mp[m]); f = ing.mean(); out[ds] = pd.DataFrame({"n_all": d.TF.value_counts(), "k": d[ing].TF.value_counts()}).fillna(0).assign(enr=lambda x, f=f: (x.k / x.n_all) / f if f > 0 else 0.0)
    t = out["FM01"].join(out["FM06"], lsuffix="_01", rsuffix="_06", how="inner"); t = t[(t.k_01 >= MINK) & (t.k_06 >= MINK)].copy(); t["score"] = np.minimum(t.enr_01, t.enr_06); t = t[t.score >= MINENR].sort_values("score", ascending=False).head(3)
    r["TFs"] = ", ".join(t.index); r["TF_enrichment_FM01_FM06"] = "; ".join(f"{i} {x.enr_01:.1f}|{x.enr_06:.1f}" for i, x in t.iterrows()); rec.append(r)
R = pd.DataFrame(rec); R.to_csv(f"{A}/fm_meta_programs_comparison/fm01_fm06_all41_programs_counts_and_TFs.csv", index=False)
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
C = {("FM01", "unshared"): "#b3a7ee", ("FM06", "unshared"): "#6a54d3", ("FM01", "shared"): "#8adcbf", ("FM06", "shared"): "#1fb07b"}; GAP = 118; H = 0.36; VMAX = 100
ypos, y, lastg = [], 0.0, None
for i, (g, m) in enumerate(rows_order):
    if g != lastg and lastg is not None: y += 0.55
    ypos.append(y); y += 1.0; lastg = g
fig, ax = plt.subplots(figsize=(14.2, 19.5)); fig.patch.set_alpha(0); ax.set_facecolor("none")
for (g, m), yy, r in zip(rows_order, ypos, R.itertuples()):
    for ds, off in (("FM01", -0.2), ("FM06", 0.2)):
        nu, ns = getattr(r, f"{ds}_unshared"), getattr(r, f"{ds}_shared")
        ax.barh(yy + off, nu, left=-GAP - nu, height=H, color=C[(ds, "unshared")], edgecolor="none"); ax.barh(yy + off, ns, left=GAP, height=H, color=C[(ds, "shared")], edgecolor="none")
        ax.text(-GAP - nu - 3, yy + off, str(nu), ha="right", va="center", fontsize=7.6, color="#333"); ax.text(GAP + ns + 3, yy + off, str(ns), ha="left", va="center", fontsize=7.6, color="#333")
    ax.text(0, yy, names[m], ha="center", va="center", fontsize=9.6, color="#111"); ax.text(GAP + VMAX + 42, yy, r.TFs if r.TFs else "–", ha="left", va="center", fontsize=9.2, color="#444" if r.TFs else "#aaa")
for g, ms in GROUPS:
    idx = [i for i, (gg, _) in enumerate(rows_order) if gg == g]; y0, y1 = ypos[idx[0]] - 0.5, ypos[idx[-1]] + 0.5; ax.plot([-GAP - VMAX - 62] * 2, [y0 + 0.08, y1 - 0.08], color="#888", lw=2.2, solid_capstyle="butt")
    ax.text(-GAP - VMAX - 70, (y0 + y1) / 2, g, ha="right", va="center", fontsize=10.5, fontweight="bold", color="#222")
YE = ypos[-1] + 0.9
for v in (0, 50, 100):
    for sgn in (1, -1): ax.plot([sgn * (GAP + v)] * 2, [-0.6, YE - 0.4], color="#efefef", lw=0.8, zorder=0); ax.text(sgn * (GAP + v), YE - 0.2, str(v), ha="center", va="top", fontsize=8.5, color="#888")
ax.text(-GAP - 50, -1.5, "Unshared top 1000", ha="center", va="bottom", fontsize=13, fontweight="bold", color="#5a46bd"); ax.text(GAP + 50, -1.5, "Shared top 1000", ha="center", va="bottom", fontsize=13, fontweight="bold", color="#17986b")
ax.text(GAP + VMAX + 42, -1.5, "TFs most associated\n(>=3 pairs in both datasets,\n>=1.5-fold enriched in both)", ha="left", va="bottom", fontsize=9.2, color="#555")
ax.set_xlim(-GAP - VMAX - 330, GAP + VMAX + 250); ax.set_ylim(YE + 0.6, -3.0); ax.axis("off")
fig.legend(handles=[Patch(color=C[("FM01", "unshared")], label="FM01 unshared"), Patch(color=C[("FM06", "unshared")], label="FM06 unshared"), Patch(color=C[("FM01", "shared")], label="FM01 shared"), Patch(color=C[("FM06", "shared")], label="FM06 shared")], loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("All 41 meta-programs: which programs do the targets of the top-1000 pairs belong to?", x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.992)
fig.text(0.01, 0.972, "ChIP-Atlas edges; number of pairs out of 1000 whose target is in the program (a target counts in every program it belongs to); lighter bar = FM01, darker = FM06 (Gavish et al. 2023)", fontsize=9, color="#666")
fig.tight_layout(rect=[0, 0.025, 1, 0.965])
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_all41_programs_butterfly"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=100, facecolor="white"); print("saved", out)
