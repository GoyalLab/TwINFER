#!/usr/bin/env python3
"""Rank-score plot of the top-N lists of one dataset (line only): x = rank, y = absolute score; left panel unshared (|z_U|), right panel shared (|z_C|, column h of the Stage I table); top 10 pairs labelled 'TF-target'.
Lists are computed here from the Stage I tables (no gene filter; AEBP1, CENPA, HMGA1, YBX1 and regulators outside the Lambert TF list excluded BEFORE ranking, as in build_top100_lists_alltf.py):
  chip  : Stage I pairs of the ChIP-Atlas candidate edges  (analysis_data/<fm>/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv)
  alltf : all TF x target Stage I pairs, BH recomputed over the merged TF-subset parts when the table is stored in parts (analysis_data/<fm>/data/stage1_tf_ge20)
Output (Arial, fonttype 42, no clip paths, transparent): analysis_data/fm_meta_programs_comparison/<DS>_top<N>_rank_score_<mode>.{pdf,png}.   usage: top_n_rank_score_plot.py DATASET chip|alltf [N=1000]"""
import sys, glob
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"
ds = sys.argv[1]; mode = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; d0 = ds.lower()
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
if mode == "chip":
    P = pd.read_csv(f"{A}/{d0}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; subtitle = "ChIP-Atlas candidate edges, Stage I"
else:
    cols = ["TF", "gene", "z_rho", "h", "z_U"]; D = f"{A}/{d0}/data/stage1_tf_ge20"; parts = sorted(glob.glob(f"{D}/part[0-9]/pdf_procedure_all_candidate_pairs.csv"))
    if parts:
        P = pd.concat([pd.read_csv(f, usecols=cols, dtype={"TF": "category", "gene": "category"}) for f in parts], ignore_index=True)
        pv = 2 * norm.sf(np.abs(P.z_rho.to_numpy(np.float64))); o = np.argsort(pv); q = pv[o] * len(pv) / (np.arange(len(pv)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); P["stage1"] = qq <= 0.05
    else: P = pd.read_csv(f"{D}/pdf_procedure_all_candidate_pairs.csv", usecols=cols + ["stage1"], dtype={"TF": "category", "gene": "category"})
    P = P[P.stage1].copy(); subtitle = "all TF × target Stage I pairs"
P["TF"] = P.TF.astype(str); P["gene"] = P.gene.astype(str); P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))]; print(ds, mode, f"Stage I pairs used: {len(P):,}", flush=True)
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "cm"})
COL = {"unshared": "#6a54d3", "shared": "#1fb07b"}; PANELS = [("unshared", "z_U", r"Unshared: $|z_{\hat\rho_\Delta}|$"), ("shared", "h", r"Shared: $|z_{\hat\rho_s}|$")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2)); fig.patch.set_alpha(0)
for ax, (lab, col, title) in zip(axes, PANELS):
    d = P.assign(score=P[col].abs()).nlargest(N, "score").reset_index(drop=True); d["rank"] = np.arange(1, len(d) + 1)
    ax.plot(d["rank"], d.score, "-", color=COL[lab], lw=1.7, zorder=2)
    ymax, ymin = d.score.max(), d.score.min(); span = ymax - ymin
    for i, r in enumerate(d.head(10).itertuples()):
        ax.annotate(f"{r.rank}. {r.TF}–{r.gene}", xy=(r.rank, r.score), xytext=(0.14 * N, ymax - span * (0.02 + 0.062 * i)), fontsize=9, va="center", ha="left", color="#222",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.6, shrinkA=1, shrinkB=1), zorder=4)
    ax.set_xlim(0, N * 1.01); ax.set_ylim(ymin - 0.03 * span, ymax + 0.06 * span); ax.set_xlabel(f"Rank in the top-{N} list", fontsize=11); ax.set_ylabel("Score (absolute z)", fontsize=11)
    ax.set_title(title, fontsize=12.5, loc="left"); ax.set_facecolor("none")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5)
# fig.suptitle(f"{ds}: top {N} TF–target pairs of each list ({subtitle})", x=0.01, ha="left", fontsize=12.5, fontweight="bold")   # removed on request: no figure title
fig.tight_layout()
for a in fig.findobj():
    if hasattr(a, "set_clip_on"): a.set_clip_on(False)
out = f"{A}/fm_meta_programs_comparison/{ds}_top{N}_rank_score_{mode}"
fig.savefig(out + ".pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(out + ".png", dpi=130, facecolor="white"); print("saved", out)
