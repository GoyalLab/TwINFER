#!/usr/bin/env python3
"""Reproduction of yscher's two-panel figure 'Top 100 TF-target pairs of each list, FM06 and FM01' from our own Stage I tables.
Pairs: Stage I pairs of the ChIP-Atlas candidate edges, all clones, NO gene filter, all regulators kept (analysis_data/<fm>/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv;
FM06 39,320 and FM01 123,544 pairs). Unshared top 100 = top 100 by |z_U|, shared top 100 = top 100 by |z_C| (column h).
a) number of the 100 pairs per TF.  b) number of the 100 pairs whose TARGET is in the program group (Gavish et al. 2023, Supplementary Table 2 = resources/41586_2023_6130_MOESM6_ESM.xlsx, sheet 'Cancer MPs');
   a pair is counted in a row if its target is in at least one program of the row, so a target can count in several rows.
Row groups: Cell cycle = MP1-4; Secreted = MP22-23; MYC = MP20; Stress/hypoxia = MP5-7; Mesenchymal = MP12-16; Skin pigmentation = MP32; Astrocyte/NPC/OPC = MP25-29.
The counts are checked against the numbers read off the original figure (CHECK below). Output: analysis_data/fm_meta_programs_comparison/fm06_fm01_top100_programs_reproduced.{pdf,png} (Arial, fonttype 42, no clip paths, transparent)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
GREEN, PURPLE = "#1fb07b", "#6a54d3"
X = pd.read_excel("/home/gzu5140/TwINFER_KA/resources/41586_2023_6130_MOESM6_ESM.xlsx", sheet_name="Cancer MPs", header=None)
mp = {int(str(X.iloc[0, j]).split()[0][2:]): set(str(g) for g in X.iloc[1:, j].dropna()) for j in range(X.shape[1])}
ROWS = [("Cell cycle", "G2/M, G1/S, HMG-rich, chromatin", [1, 2, 3, 4]), ("Secreted", "secreted I, II", [22, 23]), ("MYC", "", [20]),
        ("Stress, hypoxia", "stress, stress in vitro, hypoxia", [5, 6, 7]), ("Mesenchymal", "EMT-I to IV, MES (glioma)", [12, 13, 14, 15, 16]),
        ("Skin pigmentation", "", [32]), ("Astrocyte, NPC, OPC", "astrocytes, NPC, OPC, oligodendrocyte", [25, 26, 27, 28, 29])]
TFS = ["MITF", "FOSL1", "TFAP2A", "E2F1", "JUN", "SMAD3", "JUNB", "SNAI2", "TFAP2C", "SOX10", "MYC", "ETV1", "FOS"]
# numbers read off the original figure (FM06, FM01) for each row: unshared, shared
CHECK = {"Cell cycle": ((15, 24), (1, 0)), "Secreted": ((9, 9), (7, 4)), "MYC": ((0, 4), (0, 0)), "Stress, hypoxia": ((14, 8), (10, 5)), "Mesenchymal": ((28, 19), (33, 18)),
         "Skin pigmentation": ((14, 2), (16, 11)), "Astrocyte, NPC, OPC": ((9, 4), (6, 15))}
D = {}
for ds in ("fm06", "fm01"):
    P = pd.read_csv(f"{A}/{ds}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]
    D[ds] = dict(n=len(P), unshared=P.reindex(P.z_U.abs().sort_values(ascending=False).index).head(100), shared=P.reindex(P.h.abs().sort_values(ascending=False).index).head(100))
both = {ds: len(set(zip(D[ds]["unshared"].TF, D[ds]["unshared"].gene)) & set(zip(D[ds]["shared"].TF, D[ds]["shared"].gene))) for ds in D}
tfc = {(ds, k): D[ds][k].TF.value_counts() for ds in D for k in ("unshared", "shared")}
prc = {(ds, k, r[0]): int(D[ds][k].gene.map(lambda g: any(g in mp[m] for m in r[2])).sum()) for ds in D for k in ("unshared", "shared") for r in ROWS}
anyc = {(ds, k): int(D[ds][k].gene.map(lambda g: any(g in s for s in mp.values())).sum()) for ds in D for k in ("unshared", "shared")}
bad = [(r[0], k, ds, prc[(ds, k, r[0])], CHECK[r[0]][i][j]) for r in ROWS for i, k in enumerate(("unshared", "shared")) for j, ds in enumerate(("fm06", "fm01")) if prc[(ds, k, r[0])] != CHECK[r[0]][i][j]]
print("CHECK against the original figure: all 28 bars match" if not bad else f"MISMATCH {bad}"); print("pairs in both top-100 lists:", both, "| targets in any program:", anyc)

fig = plt.figure(figsize=(8.6, 14.8)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 8.6); ax.set_ylim(0, 14.8)   # (was 12.4: the bottom rows ran off the page); ax.axis("off")
cx, lx, rx, W = 4.3, 3.45, 5.15, 3.0                                    # centre, left bar edge, right bar edge, longest bar length
vmax = max(max(tfc[k].max() for k in tfc), max(prc.values()))
def bars(y, row, ds_vals, label, sub=None):
    ax.text(cx, y, label, ha="center", va="center", fontsize=10.5, color="#111")
    if sub: ax.text(cx, y - 0.19, sub, ha="center", va="center", fontsize=6.6, color="#777")
    for k, (side, x0) in enumerate((("unshared", lx), ("shared", rx))):
        for j, (ds, col) in enumerate((("fm06", GREEN), ("fm01", PURPLE))):
            v = ds_vals[(ds, side)]; yy = y + 0.1 - 0.2 * j
            if v > 0:
                L = W * v / vmax
                ax.add_patch(plt.Rectangle((x0 - L if side == "unshared" else x0, yy - 0.075), L, 0.15, facecolor=col, edgecolor="none"))
                ax.text(x0 - L - 0.06 if side == "unshared" else x0 + L + 0.06, yy, str(v), ha="right" if side == "unshared" else "left", va="center", fontsize=7, color="#444")
ax.text(0.2, 14.52, "Top 100 TF–target pairs of each list, FM06 and FM01", fontsize=12.5, fontweight="bold", va="center")
ax.text(0.2, 14.26, f"ChIP-Atlas candidate edges, Stage I, no gene filter (FM06 {D['fm06']['n']:,}; FM01 {D['fm01']['n']:,}) · pairs in both top-100 lists: FM06 {both['fm06']}, FM01 {both['fm01']}", fontsize=7.6, color="#555", va="center")
for k, (ds, col, lab) in enumerate((("fm06", GREEN, "FM06 untreated"), ("fm01", PURPLE, "FM01 vemurafenib"))):
    ax.add_patch(plt.Rectangle((0.2 + 2.1 * k, 13.9), 0.17, 0.17, facecolor=col, edgecolor="none")); ax.text(0.45 + 2.1 * k, 13.985, lab, fontsize=8.5, va="center")
ax.text(lx - 1.0, 13.52, "Unshared top 100", fontsize=10.5, fontweight="bold", ha="center"); ax.text(rx + 1.0, 13.52, "Shared top 100", fontsize=10.5, fontweight="bold", ha="center")
ax.text(0.1, 13.2, "a", fontsize=13, fontweight="bold", va="center"); ax.text(0.45, 13.2, "Transcription factor", fontsize=9.5, fontweight="bold", va="center")
y = 12.8
for t in TFS:
    bars(y, None, {(ds, k): int(tfc[(ds, k)].get(t, 0)) for ds in D for k in ("unshared", "shared")}, t); y -= 0.455
ax.text(0.1, y - 0.1, "b", fontsize=13, fontweight="bold", va="center"); ax.text(0.45, y - 0.1, "Meta-program of the target", fontsize=9.5, fontweight="bold", va="center")
ax.text(0.45, y - 0.36, f"targets in any meta-program: unshared FM06 {anyc[('fm06','unshared')]}, FM01 {anyc[('fm01','unshared')]}; shared FM06 {anyc[('fm06','shared')]}, FM01 {anyc[('fm01','shared')]}", fontsize=7.2, color="#555", va="center")
y -= 0.9
for name, sub, _ in ROWS:
    bars(y, None, {(ds, k): prc[(ds, k, name)] for ds in D for k in ("unshared", "shared")}, name, sub or None); y -= 0.66
ax.text(0.45, 0.42, "Each pair is counted under the meta-program(s) of its target (Gavish et al. 2023); a target can sit in more than one program.", fontsize=6.8, color="#777")
ax.text(0.45, 0.24, "Programs outside these rows are not shown. No bar: 0. Reproduced from our Stage I tables; the counts equal the original figure's.", fontsize=6.8, color="#777")
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/fm06_fm01_top100_programs_reproduced"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=110, facecolor="white"); print("saved", out)
