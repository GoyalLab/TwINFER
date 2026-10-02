# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Read-only comparison of BoolODE (ODE) twin/branch simulation data vs
TwINFER's own Gillespie SSA twin/branch simulation data, for the HSC and
mCAD gene regulatory networks.

Outputs:
  - text summary tables printed to stdout (redirected to a .txt report)
  - a handful of PNG plots (trajectory means, similarity histograms)

Does NOT modify any file under simulation_data/.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = f'{TWINFER_PROJECT_ROOT}'
BOOLODE_DIR = f"{REPO}/simulation_data/boolode_sims_replicates"
REAL_DIR = f"{REPO}/simulation_data/real_data"
GENE_ORDER_DIR = f"{REPO}/simulation_data/twinfer_format"
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/gillespie_comparison/out'
os.makedirs(OUT_DIR, exist_ok=True)

NETWORKS = ["HSC", "mCAD"]
N_BOOLODE_REPLICATES = 4          # replicate_0..3
N_GILLESPIE_BATCHES = 4           # first 4 unique batches per network
N_CELLS_TRUNK_SAMPLE = 60         # cells sampled from the huge before-division files
GILLESPIE_TYPE_TAG = {"HSC": "HSC_balanced", "mCAD": "mCAD"}


def gene_order(network):
    with open(f"{GENE_ORDER_DIR}/{network}/gene_order.txt") as fh:
        return [l.strip() for l in fh if l.strip()]


# ---------------------------------------------------------------------------
# BoolODE side
# ---------------------------------------------------------------------------

def load_boolode_final_states(network, replicate):
    path = f"{BOOLODE_DIR}/{network}/replicate_{replicate}/twin_final_states.csv"
    return pd.read_csv(path)


def load_boolode_similarity(network, replicate):
    path = f"{BOOLODE_DIR}/{network}/replicate_{replicate}/twin_similarity_results.csv"
    df = pd.read_csv(path)
    return df[df.comparison == "twin"].copy()


def boolode_trajectory_summary(network, genes, n_rep=N_BOOLODE_REPLICATES):
    """Mean +/- std trajectory per gene per step, pooled over n_rep replicates,
    split into trunk phase and branch (twin) phase."""
    trunk_frames, branch_frames = [], []
    for r in range(n_rep):
        df = load_boolode_final_states(network, r)
        df["replicate_file"] = r
        trunk_frames.append(df[df.source == "trunk"])
        branch_frames.append(df[df.source.isin(["twin_A", "twin_B"])])
    trunk = pd.concat(trunk_frames, ignore_index=True)
    branch = pd.concat(branch_frames, ignore_index=True)

    trunk_summary = trunk.groupby("step")[genes].agg(["mean", "std"])
    branch_summary = branch.groupby("step")[genes].agg(["mean", "std"])
    return trunk, branch, trunk_summary, branch_summary


# ---------------------------------------------------------------------------
# Gillespie side
# ---------------------------------------------------------------------------

def find_gillespie_batches(network, n_batches=N_GILLESPIE_BATCHES):
    """Return list of (batch_label, df_rows_path, before_division_path) for the
    first n_batches distinct batch indices, picking the first hash occurrence
    for each ("type_batchidx") when duplicated."""
    tag = GILLESPIE_TYPE_TAG[network]
    all_df_rows = sorted(glob.glob(f"{REAL_DIR}/df_rows_*_{tag}_*.csv"))
    # exclude any that are actually GSD/VSC matches by re-checking the tag exactly
    pattern = re.compile(rf"_{re.escape(tag)}_(\d+)_(\d+)_[0-9a-f]{{8}}\.csv$")
    batches = {}
    for p in all_df_rows:
        m = pattern.search(os.path.basename(p))
        if not m:
            continue
        key = f"{tag}_{m.group(1)}_{m.group(2)}"
        batches.setdefault(key, []).append(p)

    selected = []
    for key in sorted(batches.keys(), key=lambda s: int(s.split("_")[-1]))[:n_batches]:
        df_rows_path = sorted(batches[key])[0]  # first hash alphabetically
        before_path = df_rows_path.replace(
            "df_rows_", "simulation_before_division_df_rows_", 1
        )
        if not os.path.exists(before_path):
            # locate by same suffix hash
            hash_suffix = os.path.basename(df_rows_path)
            cand = glob.glob(
                f"{REAL_DIR}/simulation_before_division_df_rows_*{key}_*{hash_suffix.split('_')[-1]}"
            )
            before_path = cand[0] if cand else None
        selected.append((key, df_rows_path, before_path))
    return selected


def gillespie_trunk_trajectory(before_path, n_genes, n_cells=N_CELLS_TRUNK_SAMPLE):
    """Sample the first n_cells cells from the (huge, cell-major-ordered)
    before-division file without loading the whole thing."""
    mrna_cols = [f"gene_{i}_mRNA" for i in range(1, n_genes + 1)]
    prot_cols = [f"gene_{i}_protein" for i in range(1, n_genes + 1)]
    usecols = ["cell_id", "time_step"] + mrna_cols + prot_cols
    # steps per cell in the trunk phase is fixed (6000 for the files inspected);
    # read a generous fixed number of rows and then trim to the requested cells.
    nrows_guess = n_cells * 6000
    df = pd.read_csv(before_path, usecols=usecols, nrows=nrows_guess)
    keep_ids = sorted(df.cell_id.unique())[:n_cells]
    df = df[df.cell_id.isin(keep_ids)]
    return df, mrna_cols, prot_cols


def gillespie_post_division(df_rows_path, n_genes):
    mrna_cols = [f"gene_{i}_mRNA" for i in range(1, n_genes + 1)]
    prot_cols = [f"gene_{i}_protein" for i in range(1, n_genes + 1)]
    usecols = ["time_step", "replicate", "clone_id"] + mrna_cols + prot_cols
    df = pd.read_csv(df_rows_path, usecols=usecols)
    return df, mrna_cols, prot_cols


def gillespie_twin_similarity(df_rows_path, n_genes, value="mRNA", late_steps=None):
    """Per clone_id, correlate/distance replicate-1 vs replicate-2 gene vectors
    (log1p transformed) at one or more late time_steps. Returns a DataFrame with
    columns clone_id, time_step, pearson, euclidean."""
    cols = [f"gene_{i}_{value}" for i in range(1, n_genes + 1)]
    usecols = ["time_step", "replicate", "clone_id"] + cols
    df = pd.read_csv(df_rows_path, usecols=usecols)
    max_step = df.time_step.max()
    if late_steps is None:
        late_steps = [max_step]

    df = df[df.time_step.isin(late_steps)]
    df_log = df.copy()
    df_log[cols] = np.log1p(df_log[cols])

    r1 = df_log[df_log.replicate == 1].set_index(["clone_id", "time_step"])[cols]
    r2 = df_log[df_log.replicate == 2].set_index(["clone_id", "time_step"])[cols]
    common_idx = r1.index.intersection(r2.index)
    r1 = r1.loc[common_idx]
    r2 = r2.loc[common_idx]

    results = []
    a = r1.values
    b = r2.values
    for i, idx in enumerate(common_idx):
        va, vb = a[i], b[i]
        if np.std(va) == 0 or np.std(vb) == 0:
            r = np.nan
        else:
            r = np.corrcoef(va, vb)[0, 1]
        eu = np.linalg.norm(va - vb)
        results.append((idx[0], idx[1], r, eu))
    out = pd.DataFrame(results, columns=["clone_id", "time_step", "pearson", "euclidean"])
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def describe(series, label):
    s = series.dropna()
    return (
        f"{label}: n={len(s)}, mean={s.mean():.4f}, median={s.median():.4f}, "
        f"std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}"
    )


def main():
    report_lines = []

    for network in NETWORKS:
        report_lines.append(f"\n{'='*70}\nNETWORK: {network}\n{'='*70}")
        genes = gene_order(network)
        n_genes = len(genes)
        report_lines.append(f"genes ({n_genes}): {genes}")

        # ---------------- BoolODE ----------------
        trunk, branch, trunk_summary, branch_summary = boolode_trajectory_summary(
            network, genes
        )
        report_lines.append(
            f"\n--- BoolODE (ODE) : pooled over {N_BOOLODE_REPLICATES} replicates ---"
        )
        report_lines.append(
            f"trunk rows: {len(trunk)}, branch rows: {len(branch)}, "
            f"pair_ids per replicate: {trunk.groupby('replicate_file').pair_id.nunique().to_dict()}"
        )
        first_step = trunk_summary.index.min()
        last_trunk_step = trunk_summary.index.max()
        first_branch_step = branch_summary.index.min()
        last_branch_step = branch_summary.index.max()
        report_lines.append(
            f"trunk steps: {first_step}..{last_trunk_step} ; branch steps: "
            f"{first_branch_step}..{last_branch_step}"
        )
        for g in genes:
            m0 = trunk_summary.loc[first_step, (g, "mean")]
            m1 = trunk_summary.loc[last_trunk_step, (g, "mean")]
            sd1 = trunk_summary.loc[last_trunk_step, (g, "std")]
            mb0 = branch_summary.loc[first_branch_step, (g, "mean")]
            mb1 = branch_summary.loc[last_branch_step, (g, "mean")]
            sdb1 = branch_summary.loc[last_branch_step, (g, "std")]
            report_lines.append(
                f"  {g:8s} trunk: {m0:7.3f} -> {m1:7.3f} (end std {sd1:6.3f}) | "
                f"branch: {mb0:7.3f} -> {mb1:7.3f} (end std {sdb1:6.3f})"
            )

        # BoolODE twin similarity
        sim_frames = [load_boolode_similarity(network, r) for r in range(N_BOOLODE_REPLICATES)]
        boolode_sim = pd.concat(sim_frames, ignore_index=True)
        report_lines.append(f"\nBoolODE twin similarity (pooled, n_pairs={len(boolode_sim)}):")
        report_lines.append("  " + describe(boolode_sim.pearson, "pearson"))
        report_lines.append("  " + describe(boolode_sim.euclidean, "euclidean"))

        # plot boolode trajectories
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for g in genes:
            axes[0].plot(trunk_summary.index, trunk_summary[(g, "mean")], label=g)
            axes[1].plot(branch_summary.index, branch_summary[(g, "mean")], label=g)
        axes[0].set_title(f"BoolODE {network} trunk mean trajectory")
        axes[1].set_title(f"BoolODE {network} branch(twin) mean trajectory")
        for ax in axes:
            ax.set_xlabel("step")
            ax.set_ylabel("expression (ODE units)")
        axes[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/boolode_{network}_trajectories.png", dpi=140)
        plt.close(fig)

        # ---------------- Gillespie ----------------
        report_lines.append(
            f"\n--- Gillespie (SSA) : {N_GILLESPIE_BATCHES} batches ---"
        )
        batches = find_gillespie_batches(network)
        report_lines.append(f"batches used: {[b[0] for b in batches]}")

        # trunk (pre-division) dynamics from a subsample of cells, first batch only
        # (huge files -- only sample the first batch's file to keep runtime sane)
        key0, df_rows0, before0 = batches[0]
        if before0 and os.path.exists(before0):
            trunk_g, mrna_cols, prot_cols = gillespie_trunk_trajectory(before0, n_genes)
            trunk_g_log = trunk_g.copy()
            trunk_g_log[mrna_cols] = np.log1p(trunk_g_log[mrna_cols])
            trunk_g_log[prot_cols] = np.log1p(trunk_g_log[prot_cols])
            tg_summary_mrna = trunk_g_log.groupby("time_step")[mrna_cols].agg(["mean", "std"])
            tg_summary_prot = trunk_g_log.groupby("time_step")[prot_cols].agg(["mean", "std"])
            report_lines.append(
                f"Gillespie trunk phase (batch {key0}, {trunk_g.cell_id.nunique()} cells sampled, "
                f"time_step 0..{trunk_g.time_step.max()}), log1p(mRNA)/log1p(protein):"
            )
            for i, g in enumerate(genes, start=1):
                mc, pc = f"gene_{i}_mRNA", f"gene_{i}_protein"
                m0 = tg_summary_mrna.loc[0, (mc, "mean")]
                m1 = tg_summary_mrna.loc[trunk_g.time_step.max(), (mc, "mean")]
                sd1 = tg_summary_mrna.loc[trunk_g.time_step.max(), (mc, "std")]
                p0 = tg_summary_prot.loc[0, (pc, "mean")]
                p1 = tg_summary_prot.loc[trunk_g.time_step.max(), (pc, "mean")]
                psd1 = tg_summary_prot.loc[trunk_g.time_step.max(), (pc, "std")]
                report_lines.append(
                    f"  {g:8s}(gene_{i:<2d}) log1p(mRNA): {m0:6.3f} -> {m1:6.3f} (end std {sd1:5.3f}) | "
                    f"log1p(protein): {p0:6.3f} -> {p1:6.3f} (end std {psd1:5.3f})"
                )

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for i, g in enumerate(genes, start=1):
                axes[0].plot(tg_summary_mrna.index, tg_summary_mrna[(f"gene_{i}_mRNA", "mean")], label=g)
                axes[1].plot(tg_summary_prot.index, tg_summary_prot[(f"gene_{i}_protein", "mean")], label=g)
            axes[0].set_title(f"Gillespie {network} trunk mean log1p(mRNA) [{key0}]")
            axes[1].set_title(f"Gillespie {network} trunk mean log1p(protein) [{key0}]")
            for ax in axes:
                ax.set_xlabel("time_step")
                ax.set_ylabel("log1p(expression)")
            axes[1].legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(f"{OUT_DIR}/gillespie_{network}_trunk_trajectories.png", dpi=140)
            plt.close(fig)
        else:
            report_lines.append("  (before-division file not found for first batch; skipped trunk dynamics)")

        # post-division (twin) dynamics + similarity, all selected batches
        post_frames = []
        sim_frames_g = []
        for key, df_rows_path, _ in batches:
            post, mrna_cols, prot_cols = gillespie_post_division(df_rows_path, n_genes)
            post["batch"] = key
            post_frames.append(post)
            sim = gillespie_twin_similarity(df_rows_path, n_genes, value="mRNA")
            sim["batch"] = key
            sim_frames_g.append(sim)
        post_all = pd.concat(post_frames, ignore_index=True)
        gillespie_sim = pd.concat(sim_frames_g, ignore_index=True)

        post_log = post_all.copy()
        post_log[mrna_cols] = np.log1p(post_log[mrna_cols])
        post_summary = post_log.groupby("time_step")[mrna_cols].agg(["mean", "std"])
        report_lines.append(
            f"\nGillespie post-division (twin) phase, {len(batches)} batches pooled "
            f"({post_all.clone_id.nunique()} clone_ids/batch x 2 replicates), log1p(mRNA):"
        )
        for i, g in enumerate(genes, start=1):
            mc = f"gene_{i}_mRNA"
            m0 = post_summary.loc[0, (mc, "mean")]
            m1 = post_summary.loc[post_summary.index.max(), (mc, "mean")]
            sd1 = post_summary.loc[post_summary.index.max(), (mc, "std")]
            report_lines.append(f"  {g:8s} log1p(mRNA): {m0:6.3f} -> {m1:6.3f} (end std {sd1:5.3f})")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for i, g in enumerate(genes, start=1):
            ax.plot(post_summary.index, post_summary[(f"gene_{i}_mRNA", "mean")], label=g)
        ax.set_title(f"Gillespie {network} post-division mean log1p(mRNA)\n({len(batches)} batches pooled)")
        ax.set_xlabel("time_step (post-division)")
        ax.set_ylabel("log1p(mRNA)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/gillespie_{network}_post_division_trajectories.png", dpi=140)
        plt.close(fig)

        report_lines.append(
            f"\nGillespie twin similarity at final time_step, log1p(mRNA), "
            f"pooled over {len(batches)} batches (n_pairs={len(gillespie_sim)}):"
        )
        report_lines.append("  " + describe(gillespie_sim.pearson, "pearson"))
        report_lines.append("  " + describe(gillespie_sim.euclidean, "euclidean"))

        # side-by-side histogram
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].hist(boolode_sim.pearson.dropna(), bins=40, alpha=0.6, label="BoolODE", density=True)
        axes[0].hist(gillespie_sim.pearson.dropna(), bins=40, alpha=0.6, label="Gillespie", density=True)
        axes[0].set_title(f"{network}: twin pearson correlation")
        axes[0].set_xlabel("pearson r")
        axes[0].legend()
        axes[1].hist(boolode_sim.euclidean.dropna(), bins=40, alpha=0.6, label="BoolODE", density=True)
        axes[1].hist(gillespie_sim.euclidean.dropna(), bins=40, alpha=0.6, label="Gillespie (log1p mRNA)", density=True)
        axes[1].set_title(f"{network}: twin euclidean distance\n(NOT comparable in magnitude across methods)")
        axes[1].set_xlabel("euclidean distance")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/similarity_hist_{network}.png", dpi=140)
        plt.close(fig)

        # save the raw similarity tables for reference
        boolode_sim.to_csv(f"{OUT_DIR}/boolode_{network}_twin_similarity_pooled.csv", index=False)
        gillespie_sim.to_csv(f"{OUT_DIR}/gillespie_{network}_twin_similarity_computed.csv", index=False)

    report_text = "\n".join(str(l) for l in report_lines)
    with open(f"{OUT_DIR}/report.txt", "w") as fh:
        fh.write(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
