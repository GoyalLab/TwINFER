#!/usr/bin/env python3
"""Single-timepoint TwinScore (S/C/U/h; Stages I-III, no direction) on SpaceBar, 2026-09-28.

Implements TwinScore_LARRY_melanoma.pdf Sec 1.2's within-sample statistics -- the same ones the
paper itself uses for its one-sample melanoma (FM06) run (Table 2; "Melanoma has one sample: no
direction, the sign of U is the sign of the correlation"). Unlike the centroid A/B construction
(run_infer_spacebar_centroid_ab.py), this needs no artificial split: a "twin pair" is just two
cells sharing a clone in one section, which SpaceBar has directly. Only the Direction stage
(needs a t1/t2 pair per clone) is out of reach here and is not attempted.

2026-09-29 fidelity fixes (paper Sec 1.1-1.2, checked against the first pooled version):
  1. Normalization denominator is the FULL measured gene panel (~114 genes), not the target gene
     set -- "log(1 + 10^4 x_gi / sum_g x_gi)" sums over all genes g, not just the ones being
     scored (the target-set-only sum was inherited from run_infer_spacebar.py's convention and
     is circular: a cell's depth would depend on how much of THIS gene set it expresses).
  2. ONE sample, not pooled -- "Stages I-III use the sample with more within-sample twin pairs."
     Section2 and Section4 are each built independently (same cap/gene set) and the one with
     more twin pairs is kept; the other is dropped entirely, not pooled.
  3. Explicit centering "over twin cells" -- after twin pairs are selected, each gene's
     normal-score values are re-centered on the pair-weighted mean of the CELLS ACTUALLY USED,
     not just assumed ~0 from the global rank transform (twin cells are a subset -- clones of
     size 1 and the odd leftover cell of an odd clone are excluded).

Data (post-fix): the single section with more twin pairs, MAX_CLONE_SIZE=30 subsample of
oversized clones (same cap/seed as run_infer_spacebar.py), clones with >=2 cells (not >=3 -- no
split, so 2 is enough for one twin pair), log1p(CP10k) over the target gene set (full-panel
denominator). Also writes twinfer_input_spacebar_{gs}_singletime.csv so run_competitors_fatemap.py
can be run on the IDENTICAL cells for a fair comparison.

Statistics (paper Sec 1.2, single-sample columns only; jackknife over 50 clone groups for sd):
  normal-score transform per gene: rank -> inverse-normal quantile (ties averaged), across all
    cells of the chosen sample, then re-centered on the twin-cell subset (see fix 3 above)
  twin pairs: per clone of size n, floor(n/2) random disjoint pairs, weight 1/n_pairs each
    (clone total weight 1, matching "each clone has total weight 1... split evenly over its pairs")
  mx_g  = sum_p w_p (xa_g^2 + xb_g^2)/2                      == diag of the S numerator
  S(x,y)= sum_p w_p (xa_x ya_y + xb_x yb_y)/2 / sqrt(mx_x mx_y)   (same-cell corr; S(x,x)==1 always)
  C(x,y)= sum_p w_p (xa_x yb_y + xb_x ya_y)/2 / sqrt(mx_x mx_y)   (twin corr)
  U(x,y)= S - C                                              (unshared corr)
  h_x   = C(x,x) / sd(C(x,x))                                (Stage II; sign(S(x,x))=+1 always)
  z(x,y)= U(x,y) / sd(U(x,y))                                (Stage III; called pairs ranked by |z|)
sd(theta) = sqrt(49/50 * sum_g (theta_(-g) - mean_g theta_(-g))^2), delete-one-group jackknife,
50 random-but-balanced groups of clones (each leave-out recomputed by subtracting that group's
precomputed partial sums, fully vectorized -- no refitting).
No candidate-pair ChIP filter here (not wired up for SpaceBar): scored over every ordered pair of
the gene set, same as eval_spacebar_centroid_ab_terms.py's competitor/term evaluation.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline.run_infer_spacebar import subsample_oversized_clones, MAX_CLONE_SIZE, SEED, RAW_DIR, SECTIONS
from paper_analysis.fatemap_pipeline.run_infer_spacebar_centroid_ab import GENE_SETS, _full_gene_panel

D = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
N_JACK = 50


def log(m):
    print(m, flush=True)


def _load_one_section(section, genes, genes_full):
    """Fix 1: normalization denominator is the FULL measured panel, not the target gene set --
    loads genes_full to compute each cell's total, but returns only the target gene columns."""
    cols = ["cell_id", "bc_cluster"] + genes_full
    path = f"{RAW_DIR}/Section{section}_cell_by_gene_clustered.csv"
    df = pd.read_csv(path, usecols=cols)
    df = df[df["bc_cluster"].notna()].copy()
    df["clone_id"] = f"S{section}_" + df["bc_cluster"].astype(int).astype(str)
    df["cell_id"] = f"S{section}_" + df["cell_id"].astype(str)
    total = df[genes_full].sum(axis=1).to_numpy().astype(float)  # fix 1: full panel, not target set
    total[total == 0] = 1.0
    mat = np.log1p(df[genes].to_numpy(dtype=float) / total[:, None] * 1e4)
    return df["clone_id"].to_numpy(), df["cell_id"].to_numpy(), mat


def build_singletime(genes, max_clone_size=MAX_CLONE_SIZE, seed=SEED):
    """Fix 2: build EACH section independently (same cap/gene set), count within-sample twin
    pairs each would form, and keep only the section with more -- "Stages I-III use the sample
    with more within-sample twin pairs." The other section is dropped, not pooled."""
    genes_full = _full_gene_panel()
    best = None
    for section in SECTIONS:
        clone_id, cell_id, mat = _load_one_section(section, genes, genes_full)
        df = pd.DataFrame({"clone_id": clone_id, "cell_id": cell_id})
        df["_row"] = np.arange(len(df))
        keep_idx = subsample_oversized_clones(df, max_clone_size, seed=seed)["_row"].to_numpy()
        clone_id, cell_id, mat = clone_id[keep_idx], cell_id[keep_idx], mat[keep_idx]
        counts = pd.Series(clone_id).value_counts()
        keep_clones = set(counts[counts >= 2].index)
        keep = np.array([c in keep_clones for c in clone_id])
        clone_id, cell_id, mat = clone_id[keep], cell_id[keep], mat[keep]
        n_pairs = int((counts[counts >= 2] // 2).sum())
        log(f"[singletime] Section{section}: {len(clone_id):,} cells, {len(keep_clones):,} clones "
            f"(n>=2), ~{n_pairs:,} twin pairs")
        if best is None or n_pairs > best[0]:
            best = (n_pairs, section, clone_id, cell_id, mat)
    n_pairs, section, clone_id, cell_id, mat = best
    log(f"[singletime] using Section{section} (more twin pairs, ~{n_pairs:,})")
    return clone_id, cell_id, mat, section


def normal_scores(mat):
    """Per-gene: rank across all cells of the chosen sample -> inverse-normal quantile, ties
    averaged. Not yet centered on twin cells specifically -- see _center_on_twin_cells (fix 3)."""
    n = mat.shape[0]
    out = np.empty_like(mat)
    for j in range(mat.shape[1]):
        r = stats.rankdata(mat[:, j], method="average")
        out[:, j] = stats.norm.ppf((r - 0.5) / n)
    return out


def _center_on_twin_cells(Xa, Xb, w):
    """Fix 3: 'genes centred over twin cells' -- subtract, per gene, the pair-weighted mean of
    the cells actually used as twins (sum_p w_p (xa+xb)/2 / sum_p w_p), not the global rank-
    transform mean. Twin cells are a subset (size-1 clones and the odd leftover cell of an
    odd-sized clone are excluded), so this isn't exactly 0 even though normal_scores() is
    ~centered over the full sample."""
    mean_g = (w[:, None] * (Xa + Xb) / 2.0).sum(axis=0) / w.sum()
    return Xa - mean_g, Xb - mean_g


def make_twin_pairs(clone_id, seed=SEED):
    """floor(n/2) random disjoint pairs per clone (n>=2), weight 1/n_pairs each -> clone total
    weight 1. Returns row indices ia, ib (each length n_pairs), weight array w, and the clone
    label per pair (for jackknife grouping)."""
    rng = np.random.default_rng(seed)
    ia, ib, w, clone_of_pair = [], [], [], []
    for cid, idx in pd.Series(np.arange(len(clone_id)), index=clone_id).groupby(level=0):
        idx = idx.to_numpy()
        n = len(idx)
        n_pairs = n // 2
        if n_pairs == 0:
            continue
        perm = rng.permutation(idx)[:2 * n_pairs]
        a, b = perm[0::2], perm[1::2]
        ia.append(a)
        ib.append(b)
        w.append(np.full(n_pairs, 1.0 / n_pairs))
        clone_of_pair.append(np.full(n_pairs, cid, dtype=object))
    return (np.concatenate(ia), np.concatenate(ib), np.concatenate(w),
            np.concatenate(clone_of_pair))


def _num_matrices(Xa, Xb, w):
    """numS[g1,g2] = sum_p w_p (Xa_g1 Xa_g2 + Xb_g1 Xb_g2)/2 ; numC likewise with Xa/Xb swapped
    on one side. Vectorized via BLAS matmuls (pairs x genes)."""
    wXa = w[:, None] * Xa
    wXb = w[:, None] * Xb
    numS = 0.5 * (Xa.T @ wXa + Xb.T @ wXb)
    numC = 0.5 * (Xa.T @ wXb + Xb.T @ wXa)
    return numS, numC


def _S_C_U(numS, numC):
    mx = np.diag(numS).copy()
    mx = np.clip(mx, 1e-12, None)
    denom = np.sqrt(np.outer(mx, mx))
    S = numS / denom
    C = numC / denom
    return S, C, S - C


def compute_scores(Xa, Xb, w, clone_of_pair, genes, seed=SEED, n_jack=N_JACK):
    n_genes = len(genes)
    Xa, Xb = _center_on_twin_cells(Xa, Xb, w)  # fix 3: center on the twin-cell subset once,
    # then jackknife the (already-centered) S/C/U statistics below -- consistent with mx already
    # being defined from group-subtracted numerator sums.
    numS, numC = _num_matrices(Xa, Xb, w)
    S, C, U = _S_C_U(numS, numC)

    # jackknife: partition clones into n_jack balanced groups, precompute per-group partial sums
    clones = np.unique(clone_of_pair)
    rng = np.random.default_rng(seed + 1)
    shuffled = rng.permutation(clones)
    group_of_clone = {c: i % n_jack for i, c in enumerate(shuffled)}
    pair_group = np.array([group_of_clone[c] for c in clone_of_pair])

    numS_grp = np.zeros((n_jack, n_genes, n_genes))
    numC_grp = np.zeros((n_jack, n_genes, n_genes))
    for g in range(n_jack):
        m = pair_group == g
        if not m.any():
            continue
        s, c = _num_matrices(Xa[m], Xb[m], w[m])
        numS_grp[g], numC_grp[g] = s, c

    numS_loo = numS[None, :, :] - numS_grp   # (n_jack, g, g)
    numC_loo = numC[None, :, :] - numC_grp
    mx_loo = np.clip(np.diagonal(numS_loo, axis1=1, axis2=2), 1e-12, None)   # (n_jack, g)
    denom_loo = np.sqrt(mx_loo[:, :, None] * mx_loo[:, None, :])
    S_loo = numS_loo / denom_loo
    C_loo = numC_loo / denom_loo
    U_loo = S_loo - C_loo

    def jack_sd(theta_loo):  # theta_loo: (n_jack, ...)
        mean = theta_loo.mean(axis=0)
        return np.sqrt((n_jack - 1) / n_jack * np.sum((theta_loo - mean) ** 2, axis=0))

    sd_U = jack_sd(U_loo)
    sd_C_diag = jack_sd(np.diagonal(C_loo, axis1=1, axis2=2))  # (n_genes,)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = U / sd_U
        h = np.diag(C) / sd_C_diag

    return dict(S=S, C=C, U=U, z=z, h=h, sd_U=sd_U, sd_C_diag=sd_C_diag,
                n_jack_nonempty=int((numS_grp.sum(axis=(1, 2)) != 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", choices=[k for k in GENE_SETS if k.startswith("correlation_")],
                    default="correlation_high")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    gs = args.gene_set
    genes = sorted(GENE_SETS[gs])

    clone_id, cell_id, mat, section = build_singletime(genes, seed=args.seed)
    log(f"[{gs}] Section{section}: {mat.shape[0]:,} cells x {len(genes)} genes "
        f"(clone n>=2, cap {MAX_CLONE_SIZE})")

    out_dir = D
    os.makedirs(out_dir, exist_ok=True)
    input_csv = os.path.join(out_dir, f"twinfer_input_spacebar_{gs}_singletime.csv")
    pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes]) \
        .assign(clone_id=clone_id, cell_id=cell_id, time_step=0) \
        [["clone_id", "cell_id", "time_step"] + [f"{g}_mRNA" for g in genes]] \
        .to_csv(input_csv, index=False)
    log(f"wrote {input_csv}")

    Xn = normal_scores(mat)
    ia, ib, w, clone_of_pair = make_twin_pairs(clone_id, seed=args.seed)
    n_clones_with_pairs = len(np.unique(clone_of_pair))
    log(f"[{gs}] {len(ia):,} twin pairs from {n_clones_with_pairs:,} clones "
        f"(total weight {w.sum():.1f})")

    res = compute_scores(Xn[ia], Xn[ib], w, clone_of_pair, genes, seed=args.seed)
    log(f"[{gs}] jackknife groups with data: {res['n_jack_nonempty']}/{N_JACK}")

    h = dict(zip(genes, res["h"]))
    gene_rows = [{"gene": g, "h": h[g], "inherited": bool(h[g] > 2.405)} for g in genes]
    gene_df = pd.DataFrame(gene_rows).sort_values("h", ascending=False)
    gene_out = os.path.join(out_dir, f"singletime_score_spacebar_{gs}_gene_terms.csv")
    gene_df.to_csv(gene_out, index=False)
    log(f"wrote {gene_out} ({int(gene_df.inherited.sum())}/{len(genes)} genes inherited, h>2.405)")

    GI = {g: i for i, g in enumerate(genes)}
    rows = []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            i, j = GI[a], GI[b]
            rows.append(dict(gene_1=a, gene_2=b, S=res["S"][i, j], C=res["C"][i, j],
                              U=res["U"][i, j], z=res["z"][i, j], sd_U=res["sd_U"][i, j]))
    pair_df = pd.DataFrame(rows).sort_values("z", key=lambda s: s.abs(), ascending=False)
    pair_out = os.path.join(out_dir, f"singletime_score_spacebar_{gs}_pair_terms.csv")
    pair_df.to_csv(pair_out, index=False)
    log(f"wrote {pair_out} ({len(pair_df)} ordered pairs, |z| symmetric per unordered pair)")

    params = dict(dataset="SPACEBAR", gene_set=gs, section=int(section), n_genes=len(genes),
                  n_cells=int(mat.shape[0]),
                  n_twin_pairs=int(len(ia)), n_clones_with_pairs=int(n_clones_with_pairs),
                  max_clone_size=MAX_CLONE_SIZE, n_jack=N_JACK, seed=args.seed,
                  n_genes_inherited=int(gene_df.inherited.sum()),
                  fidelity_fixes=["full_panel_normalization", "single_sample_only",
                                   "centered_on_twin_cells"],
                  script=os.path.basename(__file__))
    with open(os.path.join(out_dir, f"run_params_spacebar_{gs}_singletime.json"), "w") as fh:
        json.dump(params, fh, indent=2, default=str)
    log(f"diagnostics: {params}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
