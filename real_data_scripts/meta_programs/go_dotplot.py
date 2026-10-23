#!/usr/bin/env python3
"""GO dot plot, 7 datasets x (unshared | shared): FM01, FM06, Watermelon naive, LARRY d2, CellTag3 d2.5, DTT Telencephalon, Michaels t1.  Input: go_top100/<dataset>_{unshared,shared}.csv (go_top100_all_datasets.py; top-100 pairs, GO BP 2023 Enrichr gmt) and,
for DTT, DTT/results/go_enrichment_top100_genes.csv (Telencephalon, list 'all', ranking z_U = unshared, z_C = shared; local MGI GO, FDR; terms matched to the gmt terms by name).  Dots shown only for q (FDR) < 0.05; size = genes of the list in the term, colour = -log10 q.
Terms: the N_TERMS/2 terms with q < 0.05 in most unshared lists and the N_TERMS/2 in most shared lists (>= 2 lists each), skipping a term whose gene set overlaps an already chosen term by Jaccard >= JMAX in the gmt; rows ordered from unshared-leaning (top) to shared-leaning (bottom).
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/go_top100_dotplot_7datasets.{pdf,png} and go_top100_dotplot_terms.csv   usage: go_dotplot.py"""
import re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; G = f"{A}/fm_meta_programs_comparison/go_top100"; N_TERMS = 24; QMAX = 0.05; JMAX = 0.25
DS = ["FM01", "FM06", "Watermelon_naive", "LARRY_d2", "CellTag3_d2.5", "DTT", "Michaels_t1"]; LAB = {"FM01": "FM01", "FM06": "FM06", "Watermelon_naive": "Watermelon\nnaive", "LARRY_d2": "LARRY\nday 2", "CellTag3_d2.5": "CellTag3\nday 2.5", "DTT": "DTT\nTelencephalon", "Michaels_t1": "Michaels\nt1"}
gmt = {l.split("\t")[0]: set(x for x in l.rstrip("\n").split("\t")[2:] if x) for l in open(f"{A}/gene_set_libraries/GO_Biological_Process_2023.gmt")}
nm = lambda t: re.sub(r" \(GO:\d+\)", "", t).strip(); key = {nm(t).lower(): t for t in gmt}
Z = {}
for ds in DS:
    for k in ("unshared", "shared"):
        if ds != "DTT": d = pd.read_csv(f"{G}/{ds}_{k}.csv")[["term", "k", "q"]]
        else:
            d = pd.read_csv(f"{A}/lineage_tracing_mice_embryonic_organism/DTT/results/go_enrichment_top100_genes.csv"); d = d[(d.celltype == "Telencephalon") & (d.ranking == ("z_U" if k == "unshared" else "z_C")) & (d.list == "all")]
            d = pd.DataFrame({"term": d.term.str.lower().map(key), "k": d.hits, "q": d.fdr}).dropna(subset=["term"])
        Z[(ds, k)] = d.set_index("term")
cols = [(ds, k) for ds in DS for k in ("unshared", "shared")]
sig = pd.DataFrame({c: Z[c].q.lt(QMAX).reindex(list(gmt)).fillna(False) for c in cols}); nsig = sig.sum(axis=1); cand = nsig[nsig >= 2].sort_values(ascending=False)
pm = pd.DataFrame({c: Z[c].q.reindex(list(gmt)) for c in cols}).min(axis=1)
nU = sig[[(d, "unshared") for d in DS]].sum(axis=1); nS = sig[[(d, "shared") for d in DS]].sum(axis=1); chosen = []
def pick(n, other, count):   # most widespread terms of one list type; skipped if overlapping (Jaccard >= JMAX) an already chosen term; must be seen in >= 2 lists of that type
    c = pd.DataFrame({"n": n, "o": other, "p": pm}).query("n >= 2").sort_values(["n", "p"], ascending=[False, True]); got = 0
    for t in c.index:
        if t in chosen or any(len(gmt[t] & gmt[x]) / len(gmt[t] | gmt[x]) >= JMAX for x in chosen): continue
        chosen.append(t); got += 1
        if got == count: break
pick(nU, nS, N_TERMS // 2); pick(nS, nU, N_TERMS // 2)
us = [nU[t] - nS[t] for t in chosen]; order = [chosen[i] for i in np.argsort(-np.array(us), kind="stable")]
rows = [dict(term=nm(t), n_unshared_datasets=int(sig.loc[t, [(d, "unshared") for d in DS]].sum()), n_shared_datasets=int(sig.loc[t, [(d, "shared") for d in DS]].sum())) for t in order]; pd.DataFrame(rows).to_csv(f"{A}/fm_meta_programs_comparison/go_top100_dotplot_terms.csv", index=False)
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf"); plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
fig, ax = plt.subplots(figsize=(11.6, 0.3 * len(order) + 2.2)); fig.patch.set_alpha(0); ax.set_facecolor("none")
xs = {}; x = 0.0
for ds in DS: xs[(ds, "unshared")] = x; xs[(ds, "shared")] = x + 0.62; x += 1.75
kmax = max(Z[c].k.max() for c in cols if len(Z[c])); xx, yy, ss, cc = [], [], [], []
for i, t in enumerate(order):
    for c in cols:
        if t in Z[c].index and Z[c].loc[t, "q"] < QMAX: xx.append(xs[c]); yy.append(i); ss.append(14 + 11 * Z[c].loc[t, "k"]); cc.append(-np.log10(max(Z[c].loc[t, "q"], 1e-12)))
sc = ax.scatter(xx, yy, s=ss, c=cc, cmap="Purples", vmin=1.3, vmax=max(cc + [4]), edgecolor="#555", linewidth=0.4, zorder=3)
for ds in DS:
    ax.text(xs[(ds, "unshared")] + 0.31, -1.35, LAB[ds], ha="center", va="bottom", fontsize=9.5, color="#111", linespacing=1.05)
    ax.text(xs[(ds, "unshared")], -0.6, "U", ha="center", va="center", fontsize=8.5, color="#5a46bd", fontweight="bold"); ax.text(xs[(ds, "shared")], -0.6, "S", ha="center", va="center", fontsize=8.5, color="#17986b", fontweight="bold")
    ax.axvspan(xs[(ds, "unshared")] - 0.4, xs[(ds, "shared")] + 0.4, color="#f5f5f5" if DS.index(ds) % 2 == 0 else "none", zorder=0, lw=0)
ax.set_yticks(range(len(order))); ax.set_yticklabels([nm(t) for t in order], fontsize=9.3); ax.set_xticks([]); ax.set_ylim(len(order) - 0.5, -2.1); ax.set_xlim(-0.45, x - 1.0)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(left=False)
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, aspect=14); cb.set_label("-log10 q", fontsize=9); cb.ax.tick_params(labelsize=8)
for n in (2, 5, 10): ax.scatter([], [], s=14 + 11 * n, c="#bbb", edgecolor="#555", linewidth=0.4, label=f"{n} genes")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.005), ncol=3, frameon=False, fontsize=9, handletextpad=0.3, labelspacing=0.3, columnspacing=1.2)
fig.text(0.005, 0.005, "U = top-100 unshared, S = top-100 shared (GO BP 2023, q < 0.05; 12 most widespread unshared and 12 shared terms, redundant terms merged). DTT uses its own MGI GO analysis (gene pairs, panel background).", fontsize=7.6, color="#777", ha="left", va="bottom")
fig.subplots_adjust(left=0.30, right=0.985, top=0.93, bottom=0.12)
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/go_top100_dotplot_7datasets"; fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0.08); fig.savefig(out + ".png", dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.08)
print(pd.DataFrame(rows).to_string(index=False)); print({c: int(sig[c].sum()) for c in cols})
