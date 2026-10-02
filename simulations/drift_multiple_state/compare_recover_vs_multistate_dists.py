"""
Compare gene-expression distributions:  low/mid DRIFT (recover, 300 h) vs the
static 2-state low/mid merge (multistate_lowbase = low_k_on + non-drift baseline).

Both are 50/50 mixtures of a low-k_on sub-population and a baseline-k_on
sub-population.  Question: are the two 'baseline' halves (recovered vs
never-suppressed) actually the same distribution?

Per network (A_B, A_to_B) and per species (g1/g2 x mRNA/protein) overlay:
  recover  - recovered half (state == up)      @ t = T_DRIFT
  recover  - suppressed half (state == down)   @ t = T_DRIFT
  static   - baseline half                     @ t = T_STAT
  static   - low_k_on half                     @ t = T_STAT

Output: analysis_data/drift_inference/lowmid_300h/recover_vs_multistate_distributions.png / .csv
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIFT = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants/lowmid_300h'
YSCHER = "/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000"
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/lowmid_300h'
T_DRIFT = 300
T_STAT = 20
SPECIES = ["gene_1_mRNA", "gene_2_mRNA", "gene_1_protein", "gene_2_protein"]

NETS = [("A, B (unregulated)", "A_B_no_reg_2_states_recover", "A_B_low_k_on", "A_B"),
        ("A->B (regulated)",   "A_to_B_2_states_recover",     "A_to_B_low_k_on", "A_to_B")]


def _first(pat):
    fs = sorted(glob.glob(pat))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[0]


def load_groups(drift_tag, low_dir, base_dir):
    dr = pd.read_csv(_first(f"{DRIFT}/df_rows_0_0_1_*_{drift_tag}_*.csv"))
    dr = dr[dr.time_step == T_DRIFT]
    lo = pd.read_csv(_first(f"{YSCHER}/{low_dir}/df_*.csv"))
    lo = lo[lo.time_step == T_STAT]
    ba = pd.read_csv(_first(f"{YSCHER}/{base_dir}/df_rows_0_1_*_ncells_6000_{base_dir}_rep_*.csv"))
    ba = ba[ba.time_step == T_STAT]
    return {
        "recover: recovered": dr[dr.state == "up"],
        "recover: suppressed": dr[dr.state == "down"],
        "static: baseline":    ba,
        "static: low_k_on":    lo,
    }


COLORS = {"recover: recovered": "#0072B2", "recover: suppressed": "#56B4E9",
          "static: baseline": "#D55E00", "static: low_k_on": "#E69F00"}
STYLE = {"recover: recovered": "-", "recover: suppressed": "-",
         "static: baseline": "--", "static: low_k_on": "--"}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stat_rows = []
    fig, axes = plt.subplots(len(NETS), len(SPECIES), figsize=(5 * len(SPECIES), 4 * len(NETS)))

    for i, (net_label, dtag, low_dir, base_dir) in enumerate(NETS):
        grp = load_groups(dtag, low_dir, base_dir)
        for j, sp in enumerate(SPECIES):
            ax = axes[i][j]
            allv = np.concatenate([g[sp].to_numpy(float) for g in grp.values()])
            lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
            is_mrna = "mRNA" in sp
            bins = np.arange(-0.5, hi + 1.5, 1) if is_mrna else np.linspace(lo, hi, 60)
            for name, g in grp.items():
                v = g[sp].to_numpy(float)
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                        color=COLORS[name], ls=STYLE[name], label=name)
                stat_rows.append({"network": net_label, "species": sp, "group": name,
                                  "mean": v.mean(), "std": v.std(),
                                  "cv": v.std() / v.mean() if v.mean() else np.nan,
                                  "median": np.median(v), "frac_zero": (v == 0).mean()})
            ax.set_title(f"{net_label}\n{sp}", fontsize=9)
            ax.set_xlabel(sp)
            if j == 0:
                ax.set_ylabel("density")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Gene-expression distributions: recover drift (t={T_DRIFT} h)  vs  "
                 f"static 2-state low/mid (t={T_STAT} h)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "recover_vs_multistate_distributions.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    print("saved", p)

    sdf = pd.DataFrame(stat_rows)
    sdf.to_csv(os.path.join(OUT, "recover_vs_multistate_distributions.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(sdf.round(3).to_string(index=False))
