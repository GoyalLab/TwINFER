#!/usr/bin/env python
"""
Does the multistate / seeded population actually contain MULTIPLE STATES at the
end of the before-division run (t=2000 for multistate, 800/1500 for seeded), or
is the wide variance in the trajectory plots just a broad unimodal transient?

For each rep-0 simulation_before_division_df we take every cell's expression
vector at the LAST time step, then:
  - log1p + per-gene z-score, PCA to whiten
  - GaussianMixture k=1..6 (n_init=5); pick k by BIC, but ACCEPT k>1 only if
    the best silhouette (k>=2) >= 0.15 -- otherwise it's one broad blob, report k=1
  - Hartigan dip test on PC1 (p<0.05 => significantly non-unimodal)
  - report per-cluster occupancy (fraction of the 6000 cells)

The ORIGINAL monostable sims are included as the negative control -- they
should come back k=1, dip p>>0.05.

Output: analysis_data/paper_analysis/real_networks/multistate_state_check.csv
        analysis_data/paper_analysis/real_networks/multistate_state_check.pdf
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, os
import numpy as np
import pandas as pd
import pyarrow.csv as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

ROOT = f'{TWINFER_PROJECT_ROOT}'
REAL = f"{ROOT}/simulation_data/real_data"
PAPER = f"{ROOT}/analysis_data/paper_analysis"
OUT = f"{PAPER}/real_networks"

CASES = [
    ("GSD_original",   f"{REAL}/simulation_before_division_df*_ncells_6000_GSD_0_0_*.csv"),
    ("GSD_multistate", f"{PAPER}/GSD/simulate/multistate_2000steps/simulation_before_division_df*_GSD_multistate_rep_0_*.csv"),
    ("GSD_seeded",     f"{PAPER}/GSD/simulate/multistate_2000steps_seeded/simulation_before_division_df*_GSD_seeded_rep_0_*.csv"),
    ("HSC_original",   f"{REAL}/simulation_before_division_df*_ncells_6000_HSC_balanced_0_0_*.csv"),
    ("HSC_multistate", f"{PAPER}/HSC/simulate/multistate_2000steps/simulation_before_division_df*_HSC_multistate_rep_0_*.csv"),
    ("HSC_seeded",     f"{PAPER}/HSC/simulate/multistate_2000steps_seeded/simulation_before_division_df*_HSC_seeded_rep_0_*.csv"),
    ("EMT_original",   f"{PAPER}/EMT/simulate/20260825_224653/simulation_before_division_df*_EMT_rep_0_*.csv"),
    ("EMT_multistate", f"{PAPER}/EMT/simulate/multistate_2000steps/simulation_before_division_df*_EMT_multistate_rep_0_*.csv"),
    ("EMT_seeded",     f"{PAPER}/EMT/simulate/multistate_2000steps_seeded/simulation_before_division_df*_EMT_seeded_rep_0_*.csv"),
    ("VSC_original",   f"{REAL}/simulation_before_division_df*_ncells_6000_VSC_2_0_*.csv"),
    ("VSC_seeded",     f"{PAPER}/VSC/simulate/multistate_6000steps_seeded/simulation_before_division_df*_VSC_seeded_rep_0_*.csv"),
    ("mCAD_original",  f"{REAL}/simulation_before_division_df*_ncells_6000_mCAD_1_0_*.csv"),
    ("mCAD_seeded",    f"{PAPER}/mCAD/simulate/multistate_2000steps_seeded/simulation_before_division_df*_mCAD_seeded_rep_0_*.csv"),
]

K_MAX, N_INIT, SIL_MIN = 6, 5, 0.15


def dip_test(x, n_boot=500, rng=None):
    """Hartigan's dip statistic + bootstrap p-value vs the uniform null."""
    rng = rng or np.random.default_rng(0)
    x = np.sort(np.asarray(x, float))
    n = len(x)

    def dip_stat(xs):
        xs = np.sort(xs)
        m = len(xs)
        cdf = np.arange(1, m + 1) / m
        # greatest convex minorant / least concave majorant gap approximation
        lo = np.maximum.accumulate(cdf - np.arange(m) / m)
        hi = (cdf - (np.arange(m) + 1) / m)
        hi = hi[::-1]
        hi = np.minimum.accumulate(hi)[::-1]
        return float(np.max(np.abs(np.concatenate([lo, hi])))) / 2

    d = dip_stat((x - x.min()) / (np.ptp(x) + 1e-12))
    boot = np.array([dip_stat(rng.uniform(0, 1, n)) for _ in range(n_boot)])
    return d, float((boot >= d).mean())


def analyse(name, pattern, rng):
    f = sorted(glob.glob(pattern))
    if not f:
        return None
    f = f[0]
    header = pd.read_csv(f, nrows=0).columns.tolist()
    genes = sorted({c.rsplit("_", 1)[0] for c in header if c.endswith("_mRNA")},
                   key=lambda g: int(g.split("_")[1]))
    mrna_cols = [f"{g}_mRNA" for g in genes]

    # pass 1: time_step column only -> tmax  (streamed, tiny memory)
    import pyarrow.compute as pc
    tmax = -1
    reader = pv.open_csv(f, convert_options=pv.ConvertOptions(include_columns=["time_step"]))
    for batch in reader:
        tmax = max(tmax, int(pc.max(batch.column("time_step")).as_py()))
    # pass 2: keep only the last-time-step rows
    keep = []
    reader = pv.open_csv(f, convert_options=pv.ConvertOptions(include_columns=["time_step"] + mrna_cols))
    for batch in reader:
        b = batch.to_pandas()
        sel = b[b["time_step"] == tmax]
        if len(sel):
            keep.append(sel[mrna_cols].to_numpy(float))
    X = np.vstack(keep)
    fin = X
    Xz = np.log1p(X)
    Xz = (Xz - Xz.mean(0)) / (Xz.std(0) + 1e-9)
    pcs = PCA(n_components=min(8, Xz.shape[1])).fit_transform(Xz)

    bics, sils = [], []
    labels_by_k = {}
    for k in range(1, K_MAX + 1):
        gm = GaussianMixture(k, covariance_type="full", n_init=N_INIT, random_state=0).fit(pcs)
        bics.append(gm.bic(pcs))
        lab = gm.predict(pcs)
        labels_by_k[k] = lab
        sils.append(silhouette_score(pcs, lab) if k > 1 and len(np.unique(lab)) > 1 else np.nan)
    k_bic = int(np.argmin(bics) + 1)
    best_sil = np.nanmax(sils) if np.any(~np.isnan(sils)) else np.nan
    k_final = k_bic if (k_bic > 1 and best_sil >= SIL_MIN) else 1
    lab = labels_by_k[k_final]
    occ = np.bincount(lab, minlength=k_final) / len(lab)

    d, dip_p = dip_test(pcs[:, 0], rng=rng)
    return dict(name=name, tmax=tmax, n_cells=len(fin), n_genes=len(genes),
               k_bic=k_bic, best_silhouette=round(float(best_sil), 3) if not np.isnan(best_sil) else None,
               k_final=k_final, occupancy=";".join(f"{o:.2f}" for o in sorted(occ, reverse=True)),
               dip_p=round(dip_p, 3), pcs=pcs, labels=lab)


def main():
    rng = np.random.default_rng(0)
    rows, plots = [], []
    for name, pat in CASES:
        print(f"[{name}] reading...", flush=True)
        r = analyse(name, pat, rng)
        if r is None:
            print(f"  MISSING: {pat}")
            continue
        print(f"  t={r['tmax']}  k_bic={r['k_bic']}  best_sil={r['best_silhouette']}  "
              f"=> k_final={r['k_final']}  occ={r['occupancy']}  dip_p={r['dip_p']}", flush=True)
        plots.append((name, r.pop("pcs"), r.pop("labels"), r["k_final"], r["dip_p"], r["best_silhouette"]))
        rows.append(r)

    pd.DataFrame(rows).to_csv(f"{OUT}/multistate_state_check.csv", index=False)
    print(f"\nwrote {OUT}/multistate_state_check.csv")

    with PdfPages(f"{OUT}/multistate_state_check.pdf") as pdf:
        import math
        ncol = 4
        nrow = math.ceil(len(plots) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
        for ax, (name, pcs, lab, kf, dp, bs) in zip(axes.ravel(), plots):
            ax.scatter(pcs[:, 0], pcs[:, 1], c=lab, s=3, cmap="tab10", alpha=0.4, lw=0)
            ax.set_title(f"{name}\nk={kf}  sil={bs}  dip_p={dp}", fontsize=8)
            ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(plots):]:
            ax.set_visible(False)
        fig.suptitle("Cell population at the last before-division time step  "
                     "(k>1 accepted only if silhouette >= 0.15)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT}/multistate_state_check.pdf")


if __name__ == "__main__":
    main()
