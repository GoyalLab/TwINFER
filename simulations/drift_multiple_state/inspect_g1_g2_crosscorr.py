"""
Inspect the Step-4 cross-time relationship  gene_1(t1) -> gene_2(t2)  (and the
reverse) for a single drift-simulation replicate, exactly as
run_6scenario_zscores.py builds it:

  * infer.py clone partition (seed 101010): the second half of the shuffled
    clones supplies the cross-time twins  (rep-1 sibling at t1, rep-2 sibling
    at t2).
  * rho_cross = weighted Spearman(gene_x_mRNA @ t1, gene_y_mRNA @ t2), clone unit.
  * z = (rho_cross - mean null) / std null,  null = weighted cross permutation
    excluding the source clone (identify_actual_directed_edges).

Usage:
  python inspect_g1_g2_crosscorr.py FILE.csv  T1 T2  [label]

Writes  <label>_g1g2_cross_t{T1}_{T2}.png  and prints rho + z for both directions.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from twinfer.inference.correlation_functions import (
    assign_twin_id,
    get_cross_correlations,
    identify_actual_directed_edges,
    _build_cross_time_twins,
)

SEED = 101010
N_SHUFFLES = 5000
PAIRS = [("gene_1", "gene_2"), ("gene_2", "gene_1"),
         ("gene_1", "gene_1"), ("gene_2", "gene_2")]
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/g1g2_cross_inspect'


def partition_ac(sim, t1, t2, seed=SEED):
    """Second-half (cross-time) clones of infer.py's split: rep-1 @ t1, rep-2 @ t2."""
    rng = np.random.default_rng(seed)
    clone_ids = sim["clone_id"].drop_duplicates().to_numpy()
    sh = rng.permutation(clone_ids)
    n = len(sh) // 4
    ac = sh[2 * n:]
    reps = sorted(sim["replicate"].unique())
    ac_left = sim[sim["clone_id"].isin(ac) & (sim["time_step"] == t1)
                  & (sim["replicate"] == reps[0])].copy()
    ac_right = sim[sim["clone_id"].isin(ac) & (sim["time_step"] == t2)
                   & (sim["replicate"] == reps[1])].copy()
    return ac_left.reset_index(drop=True), ac_right.reset_index(drop=True)


def main():
    path, t1, t2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(os.path.basename(path))[0]
    os.makedirs(OUT_DIR, exist_ok=True)

    sim = pd.read_csv(path)
    ac_left, ac_right = partition_ac(sim, t1, t2)
    at1, at2 = _build_cross_time_twins(ac_left, ac_right)
    n_twins = len(at1)

    dmat = get_cross_correlations(at1, at2, gene_pairs=PAIRS, unit="clone")
    _, zc, xnull = identify_actual_directed_edges(
        at1, at2, dmat, gene_pairs=PAIRS, z_score_threshold=2.5,
        use_scramble=True, n_shuffles=N_SHUFFLES, n_cores_to_use=4,
        verbose=False, return_z_scores=True, return_rho_cross_null=True,
        prepare_rho_cross_null=True, unit="clone", base_seed=SEED,
    )

    print(f"\n===== {label}   t1={t1}h  t2={t2}h   ({n_twins} cross-time twins) =====")
    rows = []
    for a, b in [("gene_1", "gene_2"), ("gene_2", "gene_1"),
                 ("gene_1", "gene_1"), ("gene_2", "gene_2")]:
        d = xnull[(a, b)]
        print(f"  {a}(t1) -> {b}(t2):  rho_cross = {d['observed_rho_cross']:+.4f}   "
              f"z = {d['z_rho_cross']:+.3f}   "
              f"(null mean {d['null_mean']:+.4f}, sd {d['null_std']:.4f}, n={d['n_shuffles']})")
        rows.append({"label": label, "t1": t1, "t2": t2, "src": a, "dst": b,
                     "rho_cross": d["observed_rho_cross"], "z": d["z_rho_cross"],
                     "null_mean": d["null_mean"], "null_std": d["null_std"],
                     "n_twins": n_twins})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{label}_g1g2_cross_t{t1}_{t2}.csv"), index=False)

    # ---- scatter: gene_1 mRNA @ t1  vs  gene_2 mRNA @ t2  (ranks, as Spearman sees it) ----
    x = at1["gene_1_mRNA"].to_numpy(float)
    y = at2["gene_2_mRNA"].to_numpy(float)
    xr, yr = rankdata(x), rankdata(y)
    d12 = xnull[("gene_1", "gene_2")]
    d21 = xnull[("gene_2", "gene_1")]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].scatter(x, y, s=10, alpha=0.35, color="#0072B2")
    ax[0].set_xlabel("gene_1 mRNA  @ t1"); ax[0].set_ylabel("gene_2 mRNA  @ t2")
    ax[0].set_title(f"raw counts (n={n_twins})")

    ax[1].scatter(xr, yr, s=10, alpha=0.35, color="#0072B2")
    ax[1].set_xlabel("rank  gene_1 mRNA @ t1"); ax[1].set_ylabel("rank  gene_2 mRNA @ t2")
    ax[1].set_title(f"ranks  |  rho_cross(g1->g2) = {d12['observed_rho_cross']:+.3f}   "
                    f"z = {d12['z_rho_cross']:+.2f}")
    # least-squares guide line on ranks
    b1, b0 = np.polyfit(xr, yr, 1)
    xs = np.array([xr.min(), xr.max()])
    ax[1].plot(xs, b0 + b1 * xs, color="#D55E00", lw=2)

    ax[2].hist(d12["null_values"], bins=60, alpha=0.7, color="0.6", label="null g1->g2")
    ax[2].axvline(d12["observed_rho_cross"], color="#0072B2", lw=2,
                  label=f"obs g1->g2  z={d12['z_rho_cross']:+.2f}")
    ax[2].axvline(d21["observed_rho_cross"], color="#D55E00", lw=2, ls="--",
                  label=f"obs g2->g1  z={d21['z_rho_cross']:+.2f}")
    ax[2].set_xlabel("rho_cross"); ax[2].set_title("permutation null vs observed")
    ax[2].legend(fontsize=8)

    fig.suptitle(f"{label}   ·   gene_1(t1={t1}h) -> gene_2(t2={t2}h)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT_DIR, f"{label}_g1g2_cross_t{t1}_{t2}.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    main()
