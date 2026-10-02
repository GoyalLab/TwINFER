#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Twin-pair vs random-pair Euclidean distance, post-division, for BOTH the
original ("initial") real-network sims and the seeded (balanced-basin-
occupancy) sims.

Post-division files (df_*.csv, NOT simulation_before_division_df_*) have
cell_id, time_step, <gene>_{A,I,mRNA,protein}, replicate (1/2), clone_id
(mother-cell / twin-pair key). Twin pair = same clone_id, different
replicate. Random pair = replicate-1 cell paired with a DIFFERENT clone's
replicate-2 cell (derangement of clone_id, no self-pairing), same
methodology TwINFER's own twin-vs-random null uses conceptually, but here
computed directly as a raw Euclidean distance rather than a rank
correlation, per the user's request.

Feature space: protein counts across all genes, z-scored PER GENE PER TIME
POINT (using that time point's cross-cell mean/std, pooling both
replicates) before computing Euclidean distance -- otherwise the distance
would just reflect whichever gene has the largest raw count scale.

One PDF, one page per network (grouped: initial sims first, then seeded),
mean +/- SEM band across the twin/random pair population at each post-
division time step.

Output: analysis_data/paper_analysis/real_networks/twin_vs_random_distance.pdf
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = f'{TWINFER_PROJECT_ROOT}'
PAPER = f"{ROOT}/analysis_data/paper_analysis"
REAL_DATA = f"{ROOT}/simulation_data/real_data"
OUT_PDF = f"{PAPER}/real_networks/twin_vs_random_distance.pdf"

INITIAL_PATTERN = {
    "Circadian_cycle":    f"{PAPER}/Circadian_cycle/simulate/*/df_*_Circadian_rep_0_*.csv",
    "mCAD":               f"{REAL_DATA}/df_*_ncells_6000_mCAD_1_0_*.csv",
    "VSC":                f"{REAL_DATA}/df_*_ncells_6000_VSC_2_0_*.csv",
    "B_cell_activation":  f"{PAPER}/B_cell_activation/simulate/*/df_*_B_cell_activation_rep_0_*.csv",
    "HSC_balanced":       f"{REAL_DATA}/df_*_ncells_6000_HSC_balanced_0_0_*.csv",
    "EMT":                f"{PAPER}/EMT/simulate/*/df_*_EMT_rep_0_*.csv",
    "GSD":                f"{REAL_DATA}/df_*_ncells_6000_GSD_0_0_*.csv",
    "Pluripotent":        f"{PAPER}/Pluripotent/simulate/20260825_224511/df_*_Pluripotent_rep_0_*.csv",
}

SEEDED_PATTERN = {
    "VSC (seeded)":  f"{PAPER}/VSC/simulate/multistate_6000steps_seeded/df_*_VSC_seeded_rep_0_*.csv",
    "mCAD (seeded)": f"{PAPER}/mCAD/simulate/multistate_2000steps_seeded/df_*_mCAD_seeded_rep_0_*.csv",
    "GSD (seeded)":  f"{PAPER}/GSD/simulate/multistate_2000steps_seeded/df_*_GSD_seeded_rep_0_*.csv",
    "HSC (seeded)":  f"{PAPER}/HSC/simulate/multistate_2000steps_seeded/df_*_HSC_seeded_rep_0_*.csv",
    "EMT (seeded)":  f"{PAPER}/EMT/simulate/multistate_2000steps_seeded/df_*_EMT_seeded_rep_0_*.csv",
}

RNG_SEED = 0


def pick(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def _derangement_pairing(rng, n):
    """Random permutation with no fixed points (no cell paired with itself)."""
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    if len(fixed) > 0:
        shift = np.roll(fixed, 1) if len(fixed) > 1 else (fixed + 1) % n
        perm[fixed] = shift
    return perm


def compute_distances(path):
    df = pd.read_csv(path)
    prot_cols = sorted(
        [c for c in df.columns if c.endswith("_protein")],
        key=lambda c: int(c.split("_")[1]),
    )
    times = sorted(df["time_step"].unique())
    rng = np.random.default_rng(RNG_SEED)

    # ---- fixed reference scale: t=0 (division moment) cross-cell std per gene.
    # At t=0 twins are identical (no within-clone noise yet), so ALL of the t=0
    # population variance is between-clone -- a clean, noise-free normalization
    # constant, unlike re-deriving the std at every later time point (which
    # mixes in the growing within-clone noise and masks the true divergence
    # rate -- see the ratio-panel discussion this was added to address).
    d0 = df[df["time_step"] == times[0]]
    sd_ref = d0[prot_cols].to_numpy(dtype=float).std(axis=0)
    sd_ref = np.where(sd_ref < 1e-9, 1.0, sd_ref)

    twin_mean, twin_sem, rand_mean, rand_sem = [], [], [], []
    twin_mean_fx, twin_sem_fx, rand_mean_fx, rand_sem_fx = [], [], [], []
    for t in times:
        d = df[df["time_step"] == t]
        r1 = d[d["replicate"] == 1].set_index("clone_id").sort_index()
        r2 = d[d["replicate"] == 2].set_index("clone_id").sort_index()
        common = r1.index.intersection(r2.index)
        r1 = r1.loc[common, prot_cols].to_numpy(dtype=float)
        r2 = r2.loc[common, prot_cols].to_numpy(dtype=float)
        n = len(common)
        perm = _derangement_pairing(rng, n)

        # -- (1) per-time-point z-score (original; moving denominator) --------
        pooled = np.vstack([r1, r2])
        mu = pooled.mean(axis=0)
        sd = pooled.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        r1z = (r1 - mu) / sd
        r2z = (r2 - mu) / sd
        twin_d = np.linalg.norm(r1z - r2z, axis=1)
        rand_d = np.linalg.norm(r1z - r2z[perm], axis=1)
        twin_mean.append(twin_d.mean()); twin_sem.append(twin_d.std() / np.sqrt(n))
        rand_mean.append(rand_d.mean()); rand_sem.append(rand_d.std() / np.sqrt(n))

        # -- (2) FIXED t=0-scale z-score (mean cancels in a difference, so only
        # the scale matters -- no need to also fix a reference mean) ---------
        r1zf = r1 / sd_ref
        r2zf = r2 / sd_ref
        twin_dfx = np.linalg.norm(r1zf - r2zf, axis=1)
        rand_dfx = np.linalg.norm(r1zf - r2zf[perm], axis=1)
        twin_mean_fx.append(twin_dfx.mean()); twin_sem_fx.append(twin_dfx.std() / np.sqrt(n))
        rand_mean_fx.append(rand_dfx.mean()); rand_sem_fx.append(rand_dfx.std() / np.sqrt(n))

    return (np.array(times), np.array(twin_mean), np.array(twin_sem),
            np.array(rand_mean), np.array(rand_sem),
            np.array(twin_mean_fx), np.array(twin_sem_fx),
            np.array(rand_mean_fx), np.array(rand_sem_fx),
            len(prot_cols), len(common))


def plot_network(pdf, name, times, tm, ts, rm, rs, tmf, tsf, rmf, rsf, n_genes, n_pairs, src):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 4.5))

    ax1.plot(times, tm, color="#4C72B0", lw=1.6, label="twin pairs")
    ax1.fill_between(times, tm - ts, tm + ts, color="#4C72B0", alpha=0.25, lw=0)
    ax1.plot(times, rm, color="#C44E52", lw=1.6, label="random pairs")
    ax1.fill_between(times, rm - rs, rm + rs, color="#C44E52", alpha=0.25, lw=0)
    ax1.set_xlabel("time since division")
    ax1.set_ylabel("Euclidean distance (z-scored protein space)")
    ax1.set_title("raw distance")
    ax1.legend(frameon=False)
    ax1.margins(x=0)

    # normalized panel: random/twin ratio. Skip t=0, where twin distance is
    # ~0 by construction (division just happened) and the ratio blows up --
    # not informative, just an artifact of the starting condition.
    keep = tm > 1e-6
    ratio = np.divide(rm, tm, out=np.full_like(rm, np.nan), where=keep)
    # propagate SEM of twin/random means into the ratio's SEM (delta method,
    # independent-error approximation)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_sem = ratio * np.sqrt((rs / np.where(keep, rm, 1)) ** 2 + (ts / np.where(keep, tm, 1)) ** 2)

    ax2.plot(times[keep], ratio[keep], color="#55A868", lw=1.6)
    ax2.fill_between(times[keep], np.clip((ratio - ratio_sem)[keep], 1e-3, None), (ratio + ratio_sem)[keep],
                      color="#55A868", alpha=0.25, lw=0)
    ax2.axhline(1.0, color="grey", ls="--", lw=1)
    ax2.set_yscale("log")
    ax2.set_xlabel("time since division")
    ax2.set_ylabel("random / twin distance ratio (log scale)")
    ax2.set_title("normalized separation (t=0 omitted: twin dist. ~0 there;\n"
                   "early spike = twins still near-identical just after division, not noise)")
    ax2.margins(x=0)

    # panel 3: same raw-distance view as panel 1, but z-scored using the FIXED
    # t=0 (division-moment) std per gene instead of the moving per-time-point
    # std. At t=0 twins are identical, so that reference is pure between-clone
    # variance with zero noise contamination -- this isolates the genuine
    # divergence rate instead of the rate-independent ratio panel 2 collapses to.
    ax3.plot(times, tmf, color="#4C72B0", lw=1.6, label="twin pairs")
    ax3.fill_between(times, tmf - tsf, tmf + tsf, color="#4C72B0", alpha=0.25, lw=0)
    ax3.plot(times, rmf, color="#C44E52", lw=1.6, label="random pairs")
    ax3.fill_between(times, rmf - rsf, rmf + rsf, color="#C44E52", alpha=0.25, lw=0)
    ax3.set_xlabel("time since division")
    ax3.set_ylabel("Euclidean distance (fixed t=0-scale z-score)")
    ax3.set_title("fixed-scale distance (isolates true divergence rate;\n"
                   "random pairs ~flat here by construction -- they're the t=0 scale itself)")
    ax3.legend(frameon=False)
    ax3.margins(x=0)

    fig.suptitle(f"{name}  --  twin vs random pair distance  "
                 f"({n_genes} genes, n={n_pairs} pairs/time point, mean +/- SEM)", fontsize=11)
    fig.text(0.01, 0.01, src, fontsize=5, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for group_name, patterns in (("INITIAL SIMS", INITIAL_PATTERN), ("SEEDED SIMS", SEEDED_PATTERN)):
            for net, pattern in patterns.items():
                path = pick(pattern)
                src = os.path.relpath(path, ROOT)
                print(f"[{group_name}] {net}: {os.path.basename(path)}", flush=True)
                (times, tm, ts, rm, rs, tmf, tsf, rmf, rsf,
                 n_genes, n_pairs) = compute_distances(path)
                print(f"    genes={n_genes}  pairs={n_pairs}  "
                      f"t={times.min()}..{times.max()}  "
                      f"twin(final)={tm[-1]:.3f}  random(final)={rm[-1]:.3f}  "
                      f"fixed-scale twin(final)={tmf[-1]:.3f}  random(final)={rmf[-1]:.3f}", flush=True)
                plot_network(pdf, net, times, tm, ts, rm, rs, tmf, tsf, rmf, rsf, n_genes, n_pairs, src)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
