#!/usr/bin/env python3
"""Back-to-back bar chart, one row per OFFICIAL Gavish et al. 2023 hallmark (11 rows): Cell cycle (MP1-4), Stress or hypoxia (5-7), Protein regulation (8-11), Mesenchymal (12-16), Interferon and MHC II (17-18),
Senescence (19), Oncogenic = MYC (20), Respiration (21), Secreted (22-23), Cilia (24), and Lineage related (25-37: neural, PDAC classical/alveolar/skin pigmentation, haematopoietic). MP38-41 (Unassigned) are not shown.  Left = number of the top-1000 UNSHARED pairs whose target is in the group, right = the same for the top-1000 SHARED pairs; FM01 (lighter) above FM06 (darker).
ChIP-Atlas candidate edges (FM01 epidermis list), Stage I, no gene filter; AEBP1, CENPA, HMGA1, YBX1 and non-Lambert regulators excluded; a target in several groups counts in each.  Under each group name: up to 3 TFs with >= 3 pairs in the group in BOTH
datasets (pairs of the union of the two top-1000 lists) and >= 1.5-fold enrichment in both (enrichment = the TF's fraction of its pairs whose target is in the group / the group's fraction of all pairs), ranked by the smaller enrichment; fold FM01 | FM06.
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/fm01_fm06_11_hallmarks_vertical_TF_names.{pdf,png}.  Same numbers as program_11_hallmarks_butterfly_plot.py, drawn rotated 90 degrees: hallmarks along x, bars up; within each hallmark FM01 unshared|shared sit side by side, then FM06 unshared|shared.   usage: program_11_hallmarks_vertical_plot.py"""
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
R = pd.DataFrame(rec); DROP = {"Senescence", "Respiration", "Cilia"}; keep = [i for i, (g, _) in enumerate(GROUPS) if g not in DROP]; R = R.iloc[keep].reset_index(drop=True); GROUPS = [GROUPS[i] for i in keep]   # plot only: Senescence and Respiration removed
ORDER = ["Cell cycle", "Stress or hypoxia", "Protein regulation", "Oncogenic (MYC)", "Secreted", "Interferon and MHC II", "Mesenchymal", "Lineage related"]; idx = [[g for g, _ in GROUPS].index(g) for g in ORDER]; R = R.iloc[idx].reset_index(drop=True); GROUPS = [GROUPS[i] for i in idx]   # Cell cycle (unshared-high) first, the others in between, then the shared-high Mesenchymal, Lineage related, Interferon and MHC II
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
DC = {"FM01": "#af8dc3", "FM06": "#7fbf7b"}; plt.rcParams["hatch.linewidth"] = 1.3   # two colours only (dataset); unshared = solid, shared = hatched
W = 0.2; STEP = 1.0
BARS = [("FM01", "unshared", -0.33), ("FM01", "shared", -0.11), ("FM06", "unshared", 0.11), ("FM06", "shared", 0.33)]
fig, ax = plt.subplots(figsize=(14.2, 3.3)); fig.patch.set_alpha(0); ax.set_facecolor("none"); vmax = max(R[[f"{d}_{k}" for d in ("FM01", "FM06") for k in ("unshared", "shared")]].max()); top = (int(vmax / 50) + 1) * 50
for i, (g, ms) in enumerate(GROUPS):
    r = R.iloc[i]
    for ds, k, off in BARS:
        v = int(r[f"{ds}_{k}"]); ax.bar(i * STEP + off, v, width=W, facecolor=DC[ds] if k == "unshared" else "white", edgecolor=DC[ds], linewidth=0, hatch=None if k == "unshared" else "////"); ax.text(i * STEP + off, v + top * 0.012, str(v), ha="center", va="bottom", fontsize=8.4, color="#333")
    ax.text(i * STEP, -top * 0.04, g.replace(" and ", " and\n").replace("Stress or ", "Stress or\n").replace("Protein ", "Protein\n").replace("Lineage ", "Lineage\n").replace("Oncogenic ", "Oncogenic\n"), ha="center", va="top", fontsize=10, color="#111", linespacing=1.05)
#     t = tops[g]
#     for j, tf in enumerate(t.index): ax.text(i * STEP, -top * (0.21 + 0.075 * j), tf, ha="center", va="top", fontsize=8.6, color="#444")
#     if not len(t): ax.text(i * STEP, -top * 0.21, "–", ha="center", va="top", fontsize=9, color="#aaa")
for v in range(0, top + 1, 100): ax.plot([-0.55, (len(GROUPS) - 1) * STEP + 0.55], [v, v], color="#efefef", lw=0.8, zorder=0); ax.text(-0.6, v, str(v), ha="right", va="center", fontsize=8.5, color="#888")

# ax.text(-0.62, -top * 0.21, "TFs", ha="right", va="top", fontsize=8.6, color="#777", style="italic")
ax.text(-0.86, top / 2 - 10, "pairs out of 1000 whose target is in the hallmark", rotation=90, ha="center", va="center", fontsize=9.5, color="#777")
ax.set_xlim(-0.95, (len(GROUPS) - 1) * STEP + 0.6); ax.set_ylim(-top * 0.22, vmax * 1.03); ax.axis("off")
ax.legend(handles=[Patch(facecolor=DC["FM01"], edgecolor="none", label="FM01"), Patch(facecolor=DC["FM06"], edgecolor="none", label="FM06"), Patch(facecolor="#999", edgecolor="none", label="unshared"), Patch(facecolor="white", edgecolor="#777", hatch="////", linewidth=0, label="shared")], loc="upper center", bbox_to_anchor=(0.43, 0.98), ncol=1, frameon=False, fontsize=10.5)
# fig.suptitle("The 11 hallmarks of Gavish et al. 2023: targets of the top-1000 unshared and shared pairs", x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.995)
# fig.text(0.01, 0.865, "Bars = pairs out of 1000 whose target is in the hallmark (ChIP-Atlas edges); per hallmark:  FM01 unshared, FM01 shared, FM06 unshared, FM06 shared; TFs under each hallmark: >= 3 pairs in the hallmark in both datasets and >= 1.5-fold enriched in both, ranked top to bottom (MP38-41 not shown)", fontsize=8.6, color="#666")
fig.subplots_adjust(left=0.005, right=0.998, top=0.995, bottom=0.005)
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm01_fm06_11_hallmarks_vertical_TF_names"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0.08); fig.savefig(out + ".png", dpi=105, facecolor="white", bbox_inches="tight", pad_inches=0.08); print("saved", out)
