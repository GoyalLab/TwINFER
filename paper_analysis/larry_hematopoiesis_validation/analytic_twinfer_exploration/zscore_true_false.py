# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Each TwINFER z-score, true vs. false directed edges, per network.
Permutation z-scores straight from the no-filter ranked_edges (all n(n-1) pairs),
averaged over replicates.  Box + jittered dots.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RN = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter'
TOPO = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks'
NETS = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
        "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt",
        "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt"}
ZCOLS = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het", "z_gamma"]
ZLAB = {"z_abs_rho_t1": r"$z_{|\rho|,t_1}$", "z_abs_rho_t2": r"$z_{|\rho|,t_2}$",
        "z_abs_rho_change": r"$z_{|\Delta\rho|}$", "z_het": r"$z_{\rm het}$",
        "z_div": r"$z_{\rm div}$", "z_d_het": r"$z_{d,\rm het}$", "z_gamma": r"$z_\gamma$"}

data = {}
for net, topo in NETS.items():
    M = np.loadtxt(f"{TOPO}/{topo}", delimiter=",", dtype=int); n = M.shape[0]
    g = [f"gene_{i+1}" for i in range(n)]
    true = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    frames = []
    for jf in sorted(glob.glob(f"{RN}/{net}_rep_*_all_results.json")):
        d = json.load(open(jf))
        red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])
        frames.append(red[["gene_1", "gene_2"] + ZCOLS])
    if not frames:
        continue
    df = pd.concat(frames).groupby(["gene_1", "gene_2"], as_index=False)[ZCOLS].mean()
    df["true"] = [(a, b) in true for a, b in zip(df.gene_1, df.gene_2)]
    data[net] = df
    print(f"{net}: {df.true.sum()} true / {len(df)} pairs", flush=True)

nets = list(data)
fig, axes = plt.subplots(len(ZCOLS), len(nets), figsize=(3.1 * len(nets), 2.5 * len(ZCOLS)),
                         squeeze=False)
rng = np.random.default_rng(0)
for r, zc in enumerate(ZCOLS):
    # shared y per row
    allv = np.concatenate([data[nt][zc].replace([np.inf, -np.inf], np.nan).dropna().values for nt in nets])
    lo, hi = np.percentile(allv, 1), np.percentile(allv, 99)
    pad = (hi - lo) * 0.1
    for c, nt in enumerate(nets):
        ax = axes[r, c]
        df = data[nt]
        for k, (lab, col) in enumerate([("false", "#9aa0a6"), ("true", "#2ca02c")]):
            v = df.loc[df.true == (lab == "true"), zc].replace([np.inf, -np.inf], np.nan).dropna().values
            if not len(v):
                continue
            ax.boxplot(v, positions=[k], widths=0.55, showfliers=False, patch_artist=True,
                       medianprops=dict(color=col, lw=2),
                       boxprops=dict(facecolor=col, alpha=0.2, edgecolor=col),
                       whiskerprops=dict(color=col), capprops=dict(color=col))
            vv = v if len(v) <= 200 else rng.choice(v, 200, replace=False)
            ax.scatter(k + rng.uniform(-0.16, 0.16, len(vv)), vv, s=7, color=col, alpha=0.5,
                       edgecolor="none")
        ax.axhline(0, color="0.8", lw=0.7)
        for t in (2.576, -2.576):
            ax.axhline(t, ls=":", lw=0.8, color="0.6")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non-edge", "true"], fontsize=8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.spines[["top", "right"]].set_visible(False)
        if c == 0:
            ax.set_ylabel(ZLAB[zc], fontsize=13)
        if r == 0:
            ax.set_title(nt.replace("_activation", "").replace("_cycle", ""), fontsize=10)
fig.suptitle("TwINFER permutation z-scores — true directed edges vs. non-edges "
             "(no-filter, mean over replicates)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/zscore_true_vs_false.png'
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)

# quick numeric: median true - median false, and AUROC of each z for edge/non-edge
from sklearn.metrics import roc_auc_score
print(f"\n{'network':16s} " + " ".join(f"{ZLAB[z].strip('$'):>10s}" for z in ZCOLS))
for nt in nets:
    df = data[nt]
    aucs = []
    for zc in ZCOLS:
        v = df[zc].replace([np.inf, -np.inf], np.nan)
        m = v.notna()
        s = v[m]
        if zc == "z_gamma":
            s = s.abs()   # gamma discriminative power in magnitude
        try:
            a = roc_auc_score(df.true[m], s)
        except Exception:
            a = np.nan
        aucs.append(a)
    print(f"{nt:16s} " + " ".join(f"{a:10.2f}" for a in aucs))
