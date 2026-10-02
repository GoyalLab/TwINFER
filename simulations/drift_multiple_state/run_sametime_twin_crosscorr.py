"""
IDEA TEST: same-timepoint twin cross-correlation.

Step 4 correlates  gene_X in sibling-1 @ t1  vs  gene_Y in sibling-2 @ t2
(a time-lagged, cross-twin comparison -> direction).

Here we do the SAME construction but with BOTH siblings read at the SAME time t:

    rho_same(X, Y | t) = weighted Spearman( X in twin-1 @ t , Y in twin-2 @ t )

evaluated for (g1,g2), (g2,g1), (g1,g1), (g2,g2) at t = 10 h and t = 20 h, for
all 6 scenarios x 20 replicates.  z-score vs the same permutation null as Step 4
(shuffle which twin-2 pairs with which twin-1, excluding the source clone).

Interpretation:
  * g1-g1 / g2-g2  -> ordinary twin heritability of that gene at time t.
  * g1-g2 cross    -> does the shared clonal state make gene_1 in one sibling
                      predict gene_2 in the other sibling, with NO time lag and
                      NO direction?  If yes in the *unregulated* drift scenario,
                      that shared-state term is what inflates Step 4 at t1=10 h.

Output (analysis_data/drift_inference/sametime_twin/):
  sametime_twin_crosscorr.csv           long table (scenario, rep, t, src, dst, rho, z, ...)
  sametime_twin_crosscorr.png           boxplots, 6 scenarios x 4 pair-panels x {t10,t20}
  sametime_twin_threshold_crossing.png  |z|>2.33 count heatmap
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os
import time

import numpy as np
import pandas as pd
import numba
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from twinfer.inference.correlation_functions import (
    get_cross_correlations,
    identify_actual_directed_edges,
    _build_cross_time_twins,
)
# reuse the exact file table / loader from the 6-scenario driver
from simulations.drift_multiple_state.run_6scenario_zscores import collect_tasks, _load

OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/sametime_twin'
TIMES = [10, 20]
PAIRS = [("gene_1", "gene_2"), ("gene_2", "gene_1"),
         ("gene_1", "gene_1"), ("gene_2", "gene_2")]
SEED = 101010
N_SHUFFLES = 2000
THRESH = 2.33

SCEN = [
    ("no_regulation",     "no reg",        "unreg"),
    ("A_to_B",            "A->B",          "reg"),
    ("multistate_A_B",    "2-state A,B",   "unreg"),
    ("multistate_A_to_B", "2-state A->B",  "reg"),
    ("drift_A_B",         "drift A,B",     "unreg"),
    ("drift_A_to_B",      "drift A->B",    "reg"),
]
COL = {"reg": "#0072B2", "unreg": "#D55E00"}


def sametime_pair(sim, t, n_cores=1):
    """rho + z for every PAIR at a single timepoint t, twin-1 vs twin-2."""
    reps = sorted(sim["replicate"].unique())
    left = sim[(sim["time_step"] == t) & (sim["replicate"] == reps[0])].copy()
    right = sim[(sim["time_step"] == t) & (sim["replicate"] == reps[1])].copy()
    a1, a2 = _build_cross_time_twins(left, right)
    dmat = get_cross_correlations(a1, a2, gene_pairs=PAIRS, unit="clone")
    _, _, xnull = identify_actual_directed_edges(
        a1, a2, dmat, gene_pairs=PAIRS, z_score_threshold=2.5,
        use_scramble=True, n_shuffles=N_SHUFFLES, n_cores_to_use=n_cores,
        verbose=False, return_z_scores=True, return_rho_cross_null=True,
        prepare_rho_cross_null=True, unit="clone", base_seed=SEED,
    )
    rows = []
    for (x, y) in PAIRS:
        d = xnull[(x, y)]
        rows.append({"src": x, "dst": y, "t": t, "n_twins": len(a1),
                     "rho_same": d["observed_rho_cross"], "z": d["z_rho_cross"],
                     "null_mean": d["null_mean"], "null_std": d["null_std"]})
    return rows


def run_one(scenario, rep_id, path, n_cores=1):
    numba.set_num_threads(1)
    out = []
    try:
        with threadpool_limits(limits=1):
            sim = _load(path)
            for t in TIMES:
                for r in sametime_pair(sim, t, n_cores=n_cores):
                    r.update({"scenario": scenario, "rep_id": rep_id})
                    out.append(r)
    except Exception as e:  # noqa: BLE001
        out.append({"scenario": scenario, "rep_id": rep_id, "error": f"{type(e).__name__}: {e}"})
    return out


def make_plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = df[df.get("error").isna()] if "error" in df else df
    pair_titles = [("gene_1", "gene_2", r"$g_1^{(1)}$ vs $g_2^{(2)}$  (cross)"),
                   ("gene_2", "gene_1", r"$g_2^{(1)}$ vs $g_1^{(2)}$  (cross)"),
                   ("gene_1", "gene_1", r"$g_1^{(1)}$ vs $g_1^{(2)}$  (same gene)"),
                   ("gene_2", "gene_2", r"$g_2^{(1)}$ vs $g_2^{(2)}$  (same gene)")]

    for metric, tag, ylab in [("z", "z", "z-score"), ("rho_same", "rho", r"$\rho_{\rm same}$")]:
        fig, axes = plt.subplots(len(TIMES), 4, figsize=(20, 4.6 * len(TIMES)), squeeze=False)
        allv = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = allv.min(), allv.max()
        m = (hi - lo) * 0.08 or 0.1
        for ri, t in enumerate(TIMES):
            for ci, (x, y, ttl) in enumerate(pair_titles):
                ax = axes[ri][ci]
                for k, (scen, lbl, fam) in enumerate(SCEN):
                    v = df[(df.scenario == scen) & (df.t == t) & (df.src == x) & (df.dst == y)][metric]
                    v = v.replace([np.inf, -np.inf], np.nan).dropna().values
                    if not len(v):
                        continue
                    ax.boxplot(v, positions=[k], widths=0.6, patch_artist=True, showfliers=True,
                               medianprops=dict(color=COL[fam], lw=2.2),
                               boxprops=dict(facecolor=COL[fam], alpha=0.16, edgecolor=COL[fam], lw=1.3),
                               whiskerprops=dict(color=COL[fam], lw=1.3),
                               capprops=dict(color=COL[fam], lw=1.3))
                if metric == "z":
                    ax.axhline(THRESH, ls="--", lw=1, color="0.5")
                    ax.axhline(-THRESH, ls="--", lw=1, color="0.5")
                ax.axhline(0, color="0.75", lw=0.8)
                ax.set_title(f"{ttl}   ·   t = {t} h", fontsize=10)
                ax.set_xticks(range(len(SCEN)))
                ax.set_xticklabels([l for _, l, _ in SCEN], fontsize=7, rotation=30, ha="right")
                ax.set_xlim(-0.6, len(SCEN) - 0.4)
                ax.set_ylim(lo - m, hi + m)
                ax.spines[["top", "right"]].set_visible(False)
                if ci == 0:
                    ax.set_ylabel(ylab)
        fig.legend(handles=[Patch(facecolor=COL["reg"], label="regulated (A->B)"),
                            Patch(facecolor=COL["unreg"], label="unregulated (A, B)")],
                   loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Same-timepoint twin cross-correlation  ·  twin-1 vs twin-2 at the same t  ·  "
                     f"6 scenarios x 20 reps  ·  {ylab}", fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        p = os.path.join(OUT_DIR, f"sametime_twin_{tag}.png")
        fig.savefig(p, dpi=170, bbox_inches="tight")
        print("saved", p)

    # threshold-crossing heatmap (|z| > THRESH), rows = scenario x t, cols = pair
    rows = []
    for scen, lbl, _ in SCEN:
        for t in TIMES:
            rec = {"scenario": f"{lbl}  t{t}"}
            for x, y, ttl in pair_titles:
                v = df[(df.scenario == scen) & (df.t == t) & (df.src == x) & (df.dst == y)]["z"].abs().dropna()
                rec[f"{x[-1]}->{y[-1]}"] = int((v > THRESH).sum())
            rows.append(rec)
    H = pd.DataFrame(rows).set_index("scenario")
    fig, ax = plt.subplots(figsize=(1.5 + 1.3 * H.shape[1], 0.5 * H.shape[0] + 1.5))
    im = ax.imshow(H.values / 20.0, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, fontsize=9)
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=9)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.values[i, j]}/20", ha="center", va="center",
                    fontsize=8, color="white" if H.values[i, j] / 20 > 0.55 else "0.15")
    ax.set_title(f"same-time twin cross-corr: replicates with |z| > {THRESH}", fontsize=11, pad=8)
    fig.colorbar(im, ax=ax, label="fraction", fraction=0.025, pad=0.02)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "sametime_twin_threshold_crossing.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)
    print("\n|z| > %.2f  (count/20):" % THRESH)
    print(H.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    csv = os.path.join(OUT_DIR, "sametime_twin_crosscorr.csv")

    if not args.plot_only:
        tasks = collect_tasks()
        print(f"{len(tasks)} tasks x {len(TIMES)} timepoints  ({args.jobs} parallel)", flush=True)
        t0 = time.time()
        results = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(run_one)(s, r, p, args.n_cores) for s, r, p in tasks)
        rows = [rec for sub in results for rec in sub]
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"wrote {csv}  ({time.time()-t0:.0f}s, {len(rows)} rows)", flush=True)

    make_plots(pd.read_csv(csv))
