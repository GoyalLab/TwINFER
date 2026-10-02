# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Per-gene population distributions (final timestep, 3 reps pooled) for the
seeded mCAD and VSC sims. Log10(1+count) x-axis so the low mode (~0) and the
high mode (~1e4-1e5) are both resolved. Mean-field stable-FP levels marked."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code")
from simulations.network_analysis.predict_multistability import MeanField, load_matrix, load_params, NETWORKS as PMN

PA = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis'
OUT = f"{PA}/real_networks/seeded_population_distributions.pdf"

RUNS = {
    "mCAD": (f"{PA}/mCAD/simulate/multistate_2000steps_seeded", 799, 2.0, 1.0),
    "VSC":  (f"{PA}/VSC/simulate/multistate_6000steps_seeded", 1499, 2.0, 1.0),
}
P = load_params()


def load_final(folder, tmax):
    frames = []
    for f in sorted(glob.glob(f"{folder}/simulation_before_division_df_*.csv")):
        keep = [ch[ch["time_step"] == tmax] for ch in pd.read_csv(f, chunksize=1_000_000)]
        frames.append(pd.concat(keep))
    return pd.concat(frames, ignore_index=True)


def mf_fps(net, n_hill, kadd_scale):
    M = load_matrix(PMN[net][0])
    mf = MeanField(M, P, kadd_scale=kadd_scale, n_hill=n_hill)
    fps = [f for f in mf.fixed_points(n_starts=200) if f["stable"]]
    mrna = np.array([P["kpm"] * f["a"] / P["kdm"] for f in fps])
    prot = np.array([f["p"] for f in fps])
    a = np.array([f["a"] for f in fps])
    return mrna, prot, a


def page(pdf, net, df, species, mf_vals, afrac):
    genes = sorted({c.rsplit("_", 1)[0] for c in df.columns if c.endswith(f"_{species}")},
                   key=lambda g: int(g.split("_")[1]))
    n = len(genes)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False)
    af = axes.ravel()
    cmap = plt.get_cmap("tab10")
    k = mf_vals.shape[0]
    bins = np.linspace(0, np.log10(1 + df[[f"{g}_{species}" for g in genes]].to_numpy().max()) + 0.2, 70)
    for gi, g in enumerate(genes):
        ax = af[gi]
        lx = np.log10(1 + df[f"{g}_{species}"].to_numpy(float))
        ax.hist(lx, bins=bins, color="0.5", edgecolor="none")
        for si in range(k):
            style = "-" if afrac[si, gi] > 0.5 else ":"
            ax.axvline(np.log10(1 + mf_vals[si, gi]), color=cmap(si % 10), lw=1.8, ls=style,
                       label=(f"FP{si}" if gi == 0 else None))
        hi_frac = (df[f"{g}_{species}"].to_numpy(float) > 0.3 * mf_vals[:, gi].max()).mean() if mf_vals[:, gi].max() > 0 else 0
        ax.set_title(f"{g}   ({hi_frac*100:.0f}% high)", fontsize=9)
        ax.set_xlabel(f"log10(1 + {species})")
        ax.set_yticks([])
    for j in range(n, len(af)):
        af[j].set_visible(False)
    if k <= 6:
        fig.legend(loc="upper right", fontsize=8, frameon=False, title="mean-field FP\n(solid=gene ON)")
    fig.suptitle(f"{net} seeded  |  per-gene population distribution ({species}, final step, 3 reps pooled, "
                 f"{len(df)} cells)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(OUT) as pdf:
    for net, (folder, tmax, nh, ks) in RUNS.items():
        print(f"[{net}] loading ...", flush=True)
        df = load_final(folder, tmax)
        mrna_fp, prot_fp, a_fp = mf_fps(net, nh, ks)
        print(f"[{net}] {len(df)} cells, {prot_fp.shape[0]} mean-field stable FPs", flush=True)
        page(pdf, net, df, "protein", prot_fp, a_fp)
        page(pdf, net, df, "mRNA", mrna_fp, a_fp)
print(f"wrote {OUT}")
