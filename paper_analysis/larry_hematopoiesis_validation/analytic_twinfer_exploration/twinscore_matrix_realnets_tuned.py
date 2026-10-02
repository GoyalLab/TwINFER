# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Same matrix plot but with the TUNED analytic twinScore
(score_real_networks_analytic_tuned.analytic_tuned_scores):
  E1 - CHG - RD1 - DR - XS - GAM + 0.5*ZDD + dREG,  |z_reg_gated|>1.645 gate else -E1,
recomputed from the JSON correlation matrices (no permutation). Value shown = |twinScore|
(what the ranking uses).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from benchmarks.network_benchmarks.score.score_real_networks_analytic_tuned import analytic_tuned_scores

TOPO = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks'
TWO = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference'
NETS = {"GSD": "GSD.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt", "HSC": "HSC.txt"}

labels, cur = {}, None
for ln in open(f"{TOPO}/network_labels.txt"):
    ln = ln.strip()
    if ln.startswith("#"):
        cur = ln[1:].strip(); labels[cur] = {}
    m = re.match(r"([A-Za-z0-9._/-]+)\s*-\s*g\s*(\d+)", ln)
    if m and cur:
        labels[cur][int(m.group(2))] = m.group(1)


def _topk_prec(sc, T, ng):
    poss = [(f"gene_{i+1}", f"gene_{j+1}") for i in range(ng) for j in range(ng) if i != j]
    true = {p for p in poss if T[int(p[0].split('_')[1])-1, int(p[1].split('_')[1])-1] != 0}
    k = len(true)
    fl = (min(sc.values()) - 1) if sc else -1
    top = sorted(poss, key=lambda p: sc.get(p, fl), reverse=True)[:k]
    return len(set(top) & true) / k


def score_matrix(net, T):
    jfs = sorted(glob.glob(f"{TWO}/{net}_rep_*_all_results.json"))
    ng = json.load(open(jfs[0]))["n_genes"]
    genes = [f"gene_{i+1}" for i in range(ng)]
    mats, precs = [], []
    for jf in jfs:
        sc = analytic_tuned_scores(jf)
        if sc is None:
            continue
        precs.append(_topk_prec(sc, T, ng))          # honest per-rep precision@k
        M = pd.DataFrame(np.nan, index=genes, columns=genes)
        for (a, b), v in sc.items():
            M.loc[a, b] = v
        mats.append(M.to_numpy(float))
    stack = np.array(mats)
    present = np.mean(np.isfinite(stack), axis=0)     # frac of reps the pair cleared the gate
    A = np.nanmean(stack, axis=0)
    A[present < 0.5] = np.nan                         # show only pairs gated-in in >= half the reps
    return A, ng, len(mats), float(np.mean(precs))


fig, axes = plt.subplots(2, 2, figsize=(17, 16))
for ax, (net, topo) in zip(axes.flat, NETS.items()):
    T = np.loadtxt(f"{TOPO}/{topo}", delimiter=",", dtype=int)
    A, ng, n_reps, prec_rep = score_matrix(net, T)
    names = [labels.get(net, {}).get(i + 1, f"g{i+1}") for i in range(ng)]
    np.fill_diagonal(A, np.nan)
    off = A[~np.eye(ng, dtype=bool)]
    lo, hi = np.nanpercentile(off, 5), np.nanpercentile(off, 95)
    cmap = plt.cm.magma.copy(); cmap.set_bad("0.85")
    im = ax.imshow(A, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")

    k = int((T != 0).sum())
    flat = [(A[i, j], i, j) for i in range(ng) for j in range(ng)
            if i != j and np.isfinite(A[i, j])]
    topk = {(i, j) for _, i, j in sorted(flat, reverse=True)[:k]}
    for i in range(ng):
        for j in range(ng):
            if i == j:
                continue
            if T[i, j] != 0:
                c = "#2ca02c" if T[i, j] > 0 else "#e31a1c"
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c, lw=2.8))
            if (i, j) in topk:
                ax.plot(j, i + 0.28, "^", ms=4, mfc="cyan", mec="k", mew=0.4)
            v = A[i, j]
            if np.isfinite(v):
                ax.text(j, i - 0.12, f"{v:.1f}", ha="center", va="center", fontsize=6.2,
                        color="white", path_effects=[pe.withStroke(linewidth=1.3, foreground="black")])
    ax.set_xticks(range(ng)); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(ng)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("target  (gene_2)"); ax.set_ylabel("source  (gene_1)")
    ax.set_title(f"{net}  --  TUNED analytic twinScore, mean of {n_reps} reps  --  "
                 f"{k} true edges boxed  --  per-rep precision@{k} = {prec_rep:.2f}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|twinScore| (tuned)")

fig.suptitle("TUNED analytic twinScore -- rows regulate columns -- green box = activating, "
             "red box = repressing edge -- cyan triangle = in top-k", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/twinscore_matrix_realnets_tuned.png'
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
