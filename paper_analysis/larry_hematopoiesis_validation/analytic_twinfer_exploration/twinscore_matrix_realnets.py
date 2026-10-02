# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""twinScore matrix (gene_1 = source rows x gene_2 = target cols), mean over the
no-filter replicates, for GSD / VSC / mCAD / HSC.  True regulatory edges from the
topology are boxed: green = activation, red = repression.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

RN = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter'
TOPO = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks'
NETS = {"GSD": "GSD.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt", "HSC": "HSC.txt"}

# pretty gene names from network_labels.txt  ("Name - gN = ...")
labels = {}
cur = None
for ln in open(f"{TOPO}/network_labels.txt"):
    ln = ln.strip()
    if ln.startswith("#"):
        cur = ln[1:].strip(); labels[cur] = {}
    m = re.match(r"([A-Za-z0-9._/-]+)\s*-\s*g\s*(\d+)", ln)
    if m and cur:
        labels[cur][int(m.group(2))] = m.group(1)


def score_matrix(net):
    jfs = sorted(glob.glob(f"{RN}/{net}_rep_*_all_results.json"))
    mats = []
    ng = None
    for jf in jfs:
        d = json.load(open(jf))
        ng = d["n_genes"]
        genes = [f"gene_{i+1}" for i in range(ng)]
        red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])
        M = pd.DataFrame(np.nan, index=genes, columns=genes)
        for r in red.itertuples(index=False):
            M.loc[r.gene_1, r.gene_2] = float(r.twinScore)
        mats.append(M.to_numpy(float))
    return np.nanmean(mats, axis=0), ng


fig, axes = plt.subplots(2, 2, figsize=(17, 16))
for ax, (net, topo) in zip(axes.flat, NETS.items()):
    A, ng = score_matrix(net)
    T = np.loadtxt(f"{TOPO}/{topo}", delimiter=",", dtype=int)
    names = [labels.get(net, {}).get(i + 1, f"g{i+1}") for i in range(ng)]

    np.fill_diagonal(A, np.nan)
    off = A[~np.eye(ng, dtype=bool)]
    lo, hi = np.nanpercentile(off, 5), np.nanpercentile(off, 95)
    cmap = plt.cm.magma.copy(); cmap.set_bad("0.85")
    im = ax.imshow(A, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")

    # top-k selected cells (k = # true edges), for a visual read of precision@k
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
                        color="white",
                        path_effects=[pe.withStroke(linewidth=1.3, foreground="black")])
    ax.set_xticks(range(ng)); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(ng)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("target  (gene_2)"); ax.set_ylabel("source  (gene_1)")
    n_reps = len(glob.glob(f"{RN}/{net}_rep_*_all_results.json"))
    tp = len(topk & {(i, j) for i in range(ng) for j in range(ng) if T[i, j] != 0})
    ax.set_title(f"{net}  --  mean twinScore over {n_reps} reps  --  "
                 f"{k} true edges boxed,  top-{k} (cyan dot) hits {tp}/{k}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="twinScore")

fig.suptitle("TwINFER twinScore (no-filter) -- rows regulate columns -- green box = activating, "
             "red box = repressing edge -- cyan dot = in top-k twinScore", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/twinscore_matrix_realnets.png'
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
