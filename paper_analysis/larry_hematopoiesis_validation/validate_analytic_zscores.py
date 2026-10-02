"""Validate analytic_zscores.py against the 5000-shuffle z-scores on the 6 drift/regulation
scenarios (K_frozen excluded).

The analytic method (see analytic_zscores.py) is:
    z = (rho_obs - centre) / sd          # signed statistics
    z = (|rho_obs| - sd*sqrt(2/pi)) / (sd*sqrt(1-2/pi))    # |.| statistics
with
    sd     = the permutation-null SD, CALIBRATED ONCE from the dataset's clone structure
             (measured here from the median of each scenario's own reported null_std -- the
             sims store it in every replicate JSON; on real data you run ~1 short shuffle job).
    centre = 0 for the "destroy X-Y association" nulls (step1, z_div, z_d_div, z_rho_cross);
             E[rho_Delta of random pairs] for z_het / z_d_het -- estimated in practice from
             ~30-60 cheap re-pairings; here we use het_null_mean (the 5000-draw value) to stand
             in for a well-estimated centre, and also report the naive single-draw version.

Each scenario replicate JSON stores, per gene_1-gene_2 pair: rho_obs, the 5000-shuffle null mean
& SD, and the shuffle z. Writes validate_analytic_zscores.{csv,png} and prints the agreement.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from twinfer.scoring.analytic_zscores import S2PI, VHALF, z_signed

ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario'
HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = ["no_regulation", "A_to_B", "multistate_A_to_B", "multistate_A_B", "kramp_A_to_B", "kramp_A_B"]
STATS = ["step1", "z_div", "z_het", "z_het_naive", "step3_z_d", "step3_z_d_div",
         "step4_z_rho_cross", "z_gamma"]


def calibrate_sds():
    """Median reported 5000-shuffle null SD per (scenario, null type). This is step 2 of the
    analytic method -- a one-time structural measurement, not per-pair."""
    acc = {}
    for jf in glob.glob(f"{ROOT}/t1_1_t2_20/*_rep_*.json") + glob.glob(f"{ROOT}/t1_10_t2_20/*_rep_*.json"):
        scen = os.path.basename(jf).rsplit("_rep_", 1)[0]
        if scen not in SCEN:
            continue
        d = json.load(open(jf))
        for key, jk in [("step1_t1", "step1_null_std_t1"), ("step1_t2", "step1_null_std_t2"),
                        ("div_t1", "div_null_std_t1"), ("div_t2", "div_null_std_t2"),
                        ("het_t1", "het_null_std_t1"), ("het_t2", "het_null_std_t2"),
                        ("d", "step3_null_std")]:
            acc.setdefault((scen, key), []).append(d[jk])
        # cross-corr null SD: from step4 z and rho (|rho|/|z| ~ SD if centre ~ 0)
        for k in ("1to2", "2to1"):
            z, r = d[f"step4_z_{k}"], d[f"step4_rho_cross_{k}"]
            if z:
                acc.setdefault((scen, "cross"), []).append(abs(r / z))
    return {k: float(np.median(v)) for k, v in acc.items()}


def collect(SD):
    rows = []
    for sub in ("t1_1_t2_20", "t1_10_t2_20"):
        for jf in sorted(glob.glob(f"{ROOT}/{sub}/*_rep_*.json")):
            scen = os.path.basename(jf).rsplit("_rep_", 1)[0]
            if scen not in SCEN:
                continue
            d = json.load(open(jf))
            s = lambda k: SD[(scen, k)]
            for tp in ("t1", "t2"):
                rows += [
                    dict(scen=scen, sub=sub, tp=tp, stat="step1", z_shuf=d[f"step1_z_{tp}"],
                         z_an=z_signed(d[f"rho_{tp}"], s(f"step1_{tp}"))),
                    dict(scen=scen, sub=sub, tp=tp, stat="z_div", z_shuf=d.get(f"z_div_{tp}"),
                         z_an=z_signed(d[f"rho_delta_{tp}"], s(f"div_{tp}"))),
                    dict(scen=scen, sub=sub, tp=tp, stat="z_het", z_shuf=d[f"step2_z_het_{tp}"],
                         z_an=z_signed(d[f"rho_delta_{tp}"], s(f"het_{tp}"),
                                       centre=d[f"het_null_mean_{tp}"])),      # well-estimated centre
                    dict(scen=scen, sub=sub, tp=tp, stat="z_het_naive", z_shuf=d[f"step2_z_het_{tp}"],
                         z_an=z_signed(d[f"rho_delta_{tp}"], s(f"het_{tp}"),
                                       centre=d[f"rho_delta_random_{tp}"])),   # naive single draw
                ]
            rows += [
                dict(scen=scen, sub=sub, tp="-", stat="step3_z_d", z_shuf=d["step3_z_d"],
                     z_an=z_signed(d["step3_d"], s("d"),
                                   centre=d["het_null_mean_t2"] - d["het_null_mean_t1"])),
                dict(scen=scen, sub=sub, tp="-", stat="step3_z_d_div", z_shuf=d["step3_z_d_div"],
                     z_an=z_signed(d["step3_d"], s("d"))),
                dict(scen=scen, sub=sub, tp="-", stat="step4_z_rho_cross", z_shuf=d["step4_z_1to2"],
                     z_an=z_signed(d["step4_rho_cross_1to2"], s("cross"))),
                dict(scen=scen, sub=sub, tp="-", stat="z_gamma", z_shuf=d["z_gamma_1to2"],
                     z_an=(abs(d["step4_rho_cross_1to2"]) - abs(d["step4_rho_cross_2to1"]))
                          / (s("cross") * np.sqrt(2 * VHALF))),
            ]
    df = pd.DataFrame(rows).dropna(subset=["z_shuf", "z_an"])
    return df[np.isfinite(df.z_shuf) & np.isfinite(df.z_an) & (df.z_shuf.abs() < 1e4)].copy()


def main():
    SD = calibrate_sds()
    df = collect(SD)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{HERE}/validate_analytic_zscores.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/validate_analytic_zscores.csv", index=False)

    print(f"{'statistic':18s} {'n':>4s} {'corr':>7s} {'med|dz|':>8s} {'RMS':>6s} {'max|dz|':>8s} {'slope':>6s}")
    for st in STATS:
        g = df[df.stat == st]
        if len(g) < 3:
            continue
        e = g.z_an - g.z_shuf
        sl = np.linalg.lstsq(np.vstack([g.z_an, np.ones(len(g))]).T, g.z_shuf.to_numpy(),
                             rcond=None)[0][0]
        print(f"{st:18s} {len(g):4d} {np.corrcoef(g.z_an, g.z_shuf)[0, 1]:7.4f} "
              f"{e.abs().median():8.3f} {np.sqrt((e ** 2).mean()):6.3f} {e.abs().max():8.2f} {sl:6.3f}")
    core = df[df.stat != "z_het_naive"]
    print(f"\nOVERALL (excl. z_het_naive)  corr {np.corrcoef(core.z_an, core.z_shuf)[0, 1]:.4f}   "
          f"median|dz| {(core.z_an - core.z_shuf).abs().median():.3f}   n={len(core)}")

    _make_figs(df)
    print("\nwrote validate_analytic_zscores.png / .pdf  and  .csv")


# ---------------------------------------------------------------------------
# figure, styled after drift_multiple_state/plot_6scenario.py:
#   one panel per z-statistic; x-axis = the 6 scenarios; at each scenario a pair
#   of boxplots over the replicates -- 5000-shuffle z (grey) vs analytic z
#   (orange) -- with the +/-2.33 decision lines. Combined into a multi-page PDF.
# ---------------------------------------------------------------------------
SCEN_LBL = {
    "no_regulation":     "A, B",
    "A_to_B":            "A→B",
    "multistate_A_B":    "A, B\nmultistate",
    "multistate_A_to_B": "A→B\nmultistate",
    "kramp_A_B":         "A, B\ndrift",
    "kramp_A_to_B":      "A→B\ndrift",
}
C_SHUF, C_AN = "#555555", "#D55E00"
THRESH = 2.33

PANELS = [
    ("step1",             "t1", r"Step 1 z  ($t_1$)"),
    ("step1",             "t2", r"Step 1 z  ($t_2$)"),
    ("step3_z_d",         "-",  r"Step 3 z  $z_d$"),
    ("z_het",             "t1", r"Step 2 z$_{\rm het}$  ($t_1$)"),
    ("z_het",             "t2", r"Step 2 z$_{\rm het}$  ($t_2$)"),
    ("step3_z_d_div",     "-",  r"Step 3 z  $z_{d,\rm div}$"),
    ("z_div",             "t1", r"z$_{\rm regulation}$  ($t_1$)"),
    ("z_div",             "t2", r"z$_{\rm regulation}$  ($t_2$)"),
    ("step4_z_rho_cross", "-",  r"Step 4 z  $\rho_{\rm cross}$"),
    ("z_gamma",           "-",  r"Step 4 z$_\gamma$"),
    ("z_het_naive",       "t1", r"z$_{\rm het}$ naive centre  ($t_1$)"),
    ("z_het_naive",       "t2", r"z$_{\rm het}$ naive centre  ($t_2$)"),
]


def _panel(ax, df, stat, tp, title, ylim):
    g = df[(df.stat == stat) & (df.tp == tp)]
    for k, sc in enumerate(SCEN):
        gg = g[g.scen == sc]
        for dx, col, key in ((-0.19, C_SHUF, "z_shuf"), (0.19, C_AN, "z_an")):
            v = gg[key].replace([np.inf, -np.inf], np.nan).dropna().values
            if not len(v):
                continue
            ax.boxplot(v, positions=[k + dx], widths=0.33, patch_artist=True, showfliers=True,
                       medianprops=dict(color=col, lw=2.0),
                       boxprops=dict(facecolor=col, alpha=0.18, edgecolor=col, lw=1.2),
                       whiskerprops=dict(color=col, lw=1.1),
                       capprops=dict(color=col, lw=1.1),
                       flierprops=dict(marker="o", ms=3, mfc=col, mec=col, alpha=0.5), zorder=2)
    for y in (THRESH, -THRESH):
        ax.axhline(y, ls="--", lw=1, color="0.5")
    ax.axhline(0, color="0.8", lw=0.8, zorder=1)
    md = (g.z_an - g.z_shuf).abs().median() if len(g) else float("nan")
    ax.set_title(f"{title}\nmedian |$\\Delta z$| = {md:.2f}", fontsize=9.5)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([SCEN_LBL[s] for s in SCEN], fontsize=7.5)
    ax.set_xlim(-0.7, len(SCEN) - 0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)


def _make_figs(df):
    ncol = 3
    nrow = int(np.ceil(len(PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow), squeeze=False)
    # shared y across the "real" panels (exclude the naive-centre diagnostic pair)
    core = df[df.stat != "z_het_naive"]
    v = pd.concat([core.z_shuf, core.z_an]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    pad = (hi - lo) * 0.08
    ylim_core = (lo - pad, hi + pad)
    for ax, (stat, tp, title) in zip(axes.flat, PANELS):
        yl = None if stat == "z_het_naive" else ylim_core
        _panel(ax, df, stat, tp, title, yl)
    for ax in axes.flat[len(PANELS):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(facecolor=C_SHUF, alpha=0.4, label="z  (5000 shuffles)"),
                        Patch(facecolor=C_AN, alpha=0.4, label="z  (analytic)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.008))
    fig.suptitle("Analytic z vs 5000-shuffle z  ·  gene$_1$–gene$_2$  ·  "
                 "6 scenarios × replicates  ·  $t_1\\in\\{1,10\\}$ h, $t_2$=20 h",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(wspace=0.24, hspace=0.75)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] fig.savefig(f"{HERE}/validate_analytic_zscores.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/validate_analytic_zscores.png", dpi=200, bbox_inches="tight")
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] with PdfPages(f"{HERE}/validate_analytic_zscores.pdf") as pdf:
    with PdfPages(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/validate_analytic_zscores.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
