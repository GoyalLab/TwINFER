#!/usr/bin/env python
"""
Cluster cells by their *final-timepoint* expression state, then redraw the
mean +/- std gene-expression trajectories (mRNA and protein) with one curve per
cluster.

Per network (rep 0 simulation_before_division_df), three PDF pages:
  1. clustering diagnostics  (BIC & silhouette vs k, PCA scatter, cluster sizes)
  2. mRNA trajectories, one mean +/- std band per final-timepoint cluster
  3. protein trajectories, same

Clustering
----------
Features  : per-cell mean of each gene's mRNA over the last WINDOW_FRAC of
            time steps (denoises the Poisson-noisy instantaneous value),
            then log1p and z-scored per gene.
Model     : GaussianMixture(full covariance), k = 1..K_MAX, n_init=5.
n_clusters: argmin BIC.  Silhouette (k>=2) and a PCA scatter are reported
            alongside so the choice is auditable; if BIC picks k>1 but the
            best silhouette < SIL_WEAK the page is annotated "weak separation".

Two streaming passes over each CSV (files run to 36M rows); per-network results
are cached to <cache>/<network>.npz so re-plotting is cheap.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.csv as pv
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

REAL_DATA = f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data'
PAPER = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis'
OUT_DIR = os.path.join(PAPER, "real_networks")
OUT_PDF = os.path.join(OUT_DIR, "real_networks_gene_expression_by_final_cluster.pdf")
CACHE_DIR = os.path.join(OUT_DIR, "final_cluster_cache_v2")

NETWORKS = {
    "Circadian_cycle": f"{PAPER}/Circadian_cycle/simulate/*/simulation_before_division_df*_Circadian_rep_0_*.csv",
    "mCAD": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_mCAD_1_0_*.csv",
    "VSC": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_VSC_2_0_*.csv",
    "B_cell_activation": f"{PAPER}/B_cell_activation/simulate/*/simulation_before_division_df*_B_cell_activation_rep_0_*.csv",
    "HSC_balanced": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_HSC_balanced_0_0_*.csv",
    "EMT": f"{PAPER}/EMT/simulate/*/simulation_before_division_df*_EMT_rep_0_*.csv",
    "GSD": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_GSD_0_0_*.csv",
    "Pluripotent": f"{PAPER}/Pluripotent/simulate/*/simulation_before_division_df*_Pluripotent_rep_0_*.csv",
}

WINDOW_FRAC = 0.05        # fraction of trailing time steps used for the cluster features
K_MAX = 6                 # max clusters to try
N_INIT = 5
SIL_MIN = 0.15            # need best silhouette >= this to accept k>1, else k=1
SIL_WEAK = 0.25           # 0.15-0.25 -> accepted but flagged as modest separation
BAND_ALPHA = 0.18
BLOCK_SIZE = 64 << 20     # pyarrow csv read block


def pick_file(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def gene_list(path):
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    genes = sorted(
        {c.rsplit("_", 1)[0] for c in header if c.endswith("_mRNA")},
        key=lambda g: int(g.split("_")[1]),
    )
    return genes


def stream_batches(path, columns, col_types):
    co = pv.ConvertOptions(include_columns=columns, column_types=col_types)
    reader = pv.open_csv(
        path, read_options=pv.ReadOptions(block_size=BLOCK_SIZE), convert_options=co
    )
    for batch in reader:
        yield {name: batch.column(name).to_numpy(zero_copy_only=False)
               for name in columns}


def compute_network(path, genes):
    n_genes = len(genes)
    mrna_cols = [f"{g}_mRNA" for g in genes]
    prot_cols = [f"{g}_protein" for g in genes]

    # ---- pass 0: time_step range + n cells (cheap single columns) -------------
    t0 = pv.read_csv(
        path,
        read_options=pv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=pv.ConvertOptions(
            include_columns=["cell_id", "time_step"],
            column_types={"cell_id": "int32", "time_step": "int32"},
        ),
    )
    ts_all = t0.column("time_step").to_numpy()
    cid_all = t0.column("cell_id").to_numpy()
    tmax = int(ts_all.max())
    ncells = int(cid_all.max()) + 1
    T = tmax + 1
    window = max(1, int(round(WINDOW_FRAC * T)))
    t_lo = T - window
    del t0, ts_all, cid_all

    # ---- pass 1: per-cell trailing-window mean of each gene's mRNA -----------
    feat_sum = np.zeros((ncells, n_genes))
    feat_cnt = np.zeros(ncells)
    cols1 = ["cell_id", "time_step"] + mrna_cols
    types1 = {"cell_id": "int32", "time_step": "int32", **{c: "float64" for c in mrna_cols}}
    for b in stream_batches(path, cols1, types1):
        m = b["time_step"] >= t_lo
        if not m.any():
            continue
        cid = b["cell_id"][m]
        vals = np.stack([b[c][m] for c in mrna_cols], axis=1)
        np.add.at(feat_sum, cid, vals)
        np.add.at(feat_cnt, cid, 1.0)
    feats = feat_sum / feat_cnt[:, None]                     # (ncells, n_genes)

    # ---- choose k on log1p + z-scored features -----------------------------
    # Selection rule: silhouette (separation) is primary; BIC is a sanity gate.
    #   k* = argmax_{k>=2} silhouette(k)
    #        IF   max silhouette >= SIL_MIN  AND  BIC(k*) < BIC(k=1)
    #        ELSE k* = 1   (population is a single state in this feature space)
    # This returns k=1 honestly when there is no real sub-structure, instead of
    # letting BIC run to K_MAX fitting noise.
    X = np.log1p(np.clip(feats, 0, None))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ks = list(range(1, K_MAX + 1))
    bics, sils, models = [], [], {}
    sil_idx = np.random.RandomState(0).choice(
        len(X), min(4000, len(X)), replace=False
    )
    for k in ks:
        gm = GaussianMixture(
            n_components=k, covariance_type="full", n_init=N_INIT,
            reg_covar=1e-4, random_state=0,
        ).fit(X)
        lab = gm.predict(X)
        bics.append(gm.bic(X))
        models[k] = (gm, lab)
        if k >= 2 and len(np.unique(lab[sil_idx])) > 1:
            sils.append(silhouette_score(X[sil_idx], lab[sil_idx]))
        else:
            sils.append(np.nan)

    sils_arr = np.array(sils, dtype=float)
    k_sil = ks[int(np.nanargmax(sils_arr))] if np.isfinite(sils_arr).any() else 1
    best_sil = np.nanmax(sils_arr) if np.isfinite(sils_arr).any() else np.nan
    if (
        k_sil >= 2
        and np.isfinite(best_sil)
        and best_sil >= SIL_MIN
        and bics[k_sil - 1] < bics[0]
    ):
        k_star = k_sil
    else:
        k_star = 1
    gm, labels = models[k_star]
    # relabel clusters by size (0 = largest) for stable colours
    order = np.argsort(-np.bincount(labels, minlength=k_star))
    remap = np.zeros(k_star, dtype=int)
    remap[order] = np.arange(k_star)
    labels = remap[labels]
    sil_star = sils[k_star - 1] if k_star >= 2 else np.nan

    pca = PCA(n_components=2).fit(X)
    pcs = pca.transform(X)

    # ---- pass 2: per-(cluster, time_step) stats for mRNA and protein --------
    K = k_star
    shp = (K, T, n_genes)
    n_ct = np.zeros((K, T))
    sums = {"mRNA": np.zeros(shp), "protein": np.zeros(shp)}
    sqs = {"mRNA": np.zeros(shp), "protein": np.zeros(shp)}
    cols2 = ["cell_id", "time_step"] + mrna_cols + prot_cols
    types2 = {"cell_id": "int32", "time_step": "int32",
              **{c: "float64" for c in mrna_cols + prot_cols}}
    for b in stream_batches(path, cols2, types2):
        lab = labels[b["cell_id"]]
        ts = b["time_step"]
        flat = lab * T + ts
        np.add.at(n_ct.reshape(-1), flat, 1.0)
        for sp, cnames in (("mRNA", mrna_cols), ("protein", prot_cols)):
            vals = np.stack([b[c] for c in cnames], axis=1)
            np.add.at(sums[sp].reshape(K * T, n_genes), flat, vals)
            np.add.at(sqs[sp].reshape(K * T, n_genes), flat, vals * vals)

    n = n_ct[:, :, None]
    stats = {}
    for sp in ("mRNA", "protein"):
        mean = np.where(n > 0, sums[sp] / np.clip(n, 1, None), np.nan)
        var = sqs[sp] / np.clip(n, 1, None) - np.nan_to_num(mean) ** 2
        var = var * np.where(n > 1, n / np.clip(n - 1, 1, None), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        stats[sp] = (mean, std)                              # each (K, T, n_genes)

    return dict(
        genes=np.array(genes), time=np.arange(T), window=window,
        feats=feats, X=X, labels=labels, k_star=k_star,
        ks=np.array(ks), bics=np.array(bics), sils=np.array(sils, dtype=float),
        sil_star=np.float64(sil_star), pcs=pcs,
        pca_evr=pca.explained_variance_ratio_,
        cluster_sizes=np.bincount(labels, minlength=k_star),
        mean_mRNA=stats["mRNA"][0], std_mRNA=stats["mRNA"][1],
        mean_protein=stats["protein"][0], std_protein=stats["protein"][1],
    )


def load_or_compute(network, path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{network}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    genes = gene_list(path)
    print(f"[{network}] computing ({len(genes)} genes) ...", flush=True)
    res = compute_network(path, genes)
    np.savez_compressed(cache, **res)
    return res


# ----------------------------------------------------------------------------- plots
def page_diagnostics(pdf, network, d, src):
    ks, bics, sils = d["ks"], d["bics"], d["sils"]
    k_star = int(d["k_star"])
    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ks, bics, "o-")
    ax.axvline(k_star, color="crimson", ls="--", lw=1)
    ax.set_xlabel("k"); ax.set_ylabel("GMM BIC"); ax.set_title("BIC (min = chosen k)")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ks, sils, "o-", color="teal")
    ax.axhline(SIL_WEAK, color="grey", ls=":", lw=1)
    ax.axvline(k_star, color="crimson", ls="--", lw=1)
    ax.set_xlabel("k"); ax.set_ylabel("mean silhouette"); ax.set_title("silhouette (k>=2)")

    ax = fig.add_subplot(gs[0, 2])
    pcs, labels = d["pcs"], d["labels"]
    cmap = plt.get_cmap("tab10")
    for c in range(k_star):
        m = labels == c
        ax.scatter(pcs[m, 0], pcs[m, 1], s=4, alpha=0.4, color=cmap(c % 10),
                   label=f"c{c}: n={int(d['cluster_sizes'][c])}")
    evr = d["pca_evr"]
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    ax.set_title("cells in PCA space (final-window mRNA)")
    ax.legend(fontsize=7, frameon=False, markerscale=2)

    if k_star == 1:
        note = "  [single state - no separated sub-populations]"
    elif np.isnan(d["sil_star"]) or float(d["sil_star"]) < SIL_WEAK:
        note = "  [modest separation]"
    else:
        note = ""
    fig.suptitle(
        f"{network} - final-timepoint clustering:  k = {k_star}"
        f"  (silhouette={float(d['sil_star']):.2f}, window=last {int(d['window'])} steps){note}",
        fontsize=12,
    )
    fig.text(0.01, 0.01, src, fontsize=5, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    pdf.savefig(fig); plt.close(fig)


def page_trajectories(pdf, network, d, species, src):
    genes = [str(g) for g in d["genes"]]
    time = d["time"]
    mean = d[f"mean_{species}"]      # (K, T, n_genes)
    std = d[f"std_{species}"]
    K, _, n = mean.shape
    sizes = d["cluster_sizes"]
    total = int(sizes.sum())
    cmap = plt.get_cmap("tab10")

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    lo = np.nanmin(np.clip(mean - std, 0, None))
    hi = np.nanmax(mean + std)
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )
    af = axes.ravel()
    for gi in range(n):
        ax = af[gi]
        for c in range(K):
            m, s = mean[c, :, gi], std[c, :, gi]
            col = cmap(c % 10)
            ax.plot(time, m, color=col, lw=1.2)
            ax.fill_between(time, np.clip(m - s, 0, None), m + s,
                            color=col, alpha=BAND_ALPHA, lw=0)
        ax.set_title(genes[gi], fontsize=9)
        ax.set_ylim(ylim); ax.margins(x=0)
    for j in range(n, len(af)):
        af[j].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("time step")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{species} count")

    handles = [plt.Line2D([], [], color=cmap(c % 10), lw=2,
                          label=f"cluster {c}: n={int(sizes[c])} ({100*sizes[c]/total:.0f}%)")
               for c in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=min(K, 5),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{network} - {species} by final-timepoint cluster  "
        f"(k={K}, rep 0, mean +/- std within cluster; shared y-scale)",
        fontsize=11,
    )
    fig.text(0.01, 0.005, src, fontsize=5, color="grey")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _worker(item):
    """Compute + cache one network (runs in its own process)."""
    network, pattern = item
    # keep each worker single-threaded for BLAS/arrow so N workers != oversubscribe
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    try:
        import pyarrow
        pyarrow.set_cpu_count(2)
    except Exception:
        pass
    path = pick_file(pattern)
    d = load_or_compute(network, path)
    return network, int(d["k_star"]), d["cluster_sizes"].tolist(), float(d["sil_star"])


def main():
    import multiprocessing as mp

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    todo = [(n, p) for n, p in NETWORKS.items()
            if not os.path.exists(os.path.join(CACHE_DIR, f"{n}.npz"))]
    if todo:
        nproc = min(len(todo), int(os.environ.get("N_WORKERS", "8")))
        print(f"computing {len(todo)} networks with {nproc} parallel workers", flush=True)
        with mp.get_context("spawn").Pool(nproc) as pool:
            for network, k, sizes, sil in pool.imap_unordered(_worker, todo):
                print(f"[{network}] k={k} sizes={sizes} sil={sil:.3f}", flush=True)

    # ---- single-process PDF assembly (deterministic network order) ----------
    with PdfPages(OUT_PDF) as pdf:
        for network, pattern in NETWORKS.items():
            path = pick_file(pattern)
            src = os.path.relpath(path, f'{TWINFER_PROJECT_ROOT}')
            d = load_or_compute(network, path)
            page_diagnostics(pdf, network, d, src)
            page_trajectories(pdf, network, d, "mRNA", src)
            page_trajectories(pdf, network, d, "protein", src)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
