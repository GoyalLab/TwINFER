"""
Score the 8 original real-network simulations with the ANALYTIC, yscher-tuned
TwinScore (the one built for the LARRY panels in
paper_analysis/larry_hematopoiesis_validation/run_analytic_twinfer_tuned.py):
no permutation shuffle -- every term is a panel-standardized function s(.) of a
correlation quantity, plus the |z_reg_gated| > 1.645 gate.

    E1  = s(|rho_t1|)
    CHG = s(|rho_t2| - |rho_t1|)          # magnitude change t1->t2  (larry: |rho_t2 - rho_t1|)
    RD1 = s(|rho_delta_twin(t1)|)
    DR  = s(|rho_delta_rand(t2) - rho_delta_rand(t1)|)
    XS  = s(min(|rho_cross_xy|, |rho_cross_yx|))
    GAM = s(|rho_cross_xy| - |rho_cross_yx|)
    ZDD = s(rho_cross_xy - rho_cross_yx)
    REG[g] = zscore over genes of  mean_w (rho_cross[g,w] - rho_cross[w,g])
    TS  = E1 - CHG - RD1 - DR - XS - GAM + 0.5*ZDD + (REG[x] - REG[y])
    gate: z_het      = (rho_delta_twin(t1) - rho_delta_rand(t1)) / sd_reg
          lambda     = min(1, |z_het| / 2.33)
          z_reg_gated = (rho_t1 - lambda * 0.5*(rho_cross_xy + rho_cross_yx)) / sd_reg
          |z_reg_gated| <= 1.645  ->  score = -E1
    sd_reg = SD of the off-diagonal random_delta_t1 entries (a structural,
             measure-once estimate of the twin-delta null SD for this replicate).

Correlations are taken verbatim from the existing full-inference JSONs in
analysis_data/paper_analysis/real_networks/twinfer_inference/ -- i.e. the twin
correlation matrices TwINFER already computed from each simulated replicate
(clone-weighted, unit='clone'; the LARRY tuned variant used unweighted twin
correlations -- that is the one deviation, noted because re-deriving unweighted
correlations needs TwINFER's internal twin re-simulation).

Scored over the FULL n*(n-1) directed-pair universe, same conventions as
score_original_benchmark_twinscore.py (floor for any unscored pair; tie-aware
top-k F1). BEELINE baselines are reused from
real_networks_twinscore_benchmark_scores.csv.

Output: real_networks_analytic_tuned_scores.csv
        real_networks_analytic_tuned_heatmap.pdf   (AUPRC / F1 / each vs random)
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, precision_recall_curve

from twinfer.utils.paths import get_data_root

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
TOPO_ROOT = PROJECT_ROOT / "input_data" / "real_world_networks"
REAL_NET_ROOT = DATA_ROOT / "paper_analysis" / "real_networks"
TWINFER_OUT = REAL_NET_ROOT / "twinfer_inference"
BEELINE_SCORES_CSV = REAL_NET_ROOT / "real_networks_twinscore_benchmark_scores.csv"

NETWORKS = {  # network -> (topology file, JSON filename token)
    "GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
    "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt",
    "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
}
BEELINE = ["PIDC", "GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON"]
TW_ANALYTIC = "TwINFER (analytic, yscher-tuned)"
GATE_Z = 1.645

S2PI = np.sqrt(2.0 / np.pi)


# --------------------------------------------------------------------- helpers
def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def true_edges_from_topo(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    g = [f"gene_{i + 1}" for i in range(n)]
    true = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    possible = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j}
    return true, possible


def auprc_from_scored_pairs(pair_scores, true_edges, possible_edges):
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    y_true = np.fromiter((1 if p in true_edges else 0 for p in possible_edges), dtype=int)
    y_score = np.fromiter((pair_scores.get(p, floor) for p in possible_edges), dtype=float)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return float(auc(rec, prec))


def f1_topk(pair_scores, true_edges, possible_edges):
    k = len(true_edges)
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    scored = sorted(possible_edges, key=lambda p: pair_scores.get(p, floor), reverse=True)
    if k <= 0 or not scored:
        return 0.0
    k = min(k, len(scored))
    boundary = pair_scores.get(scored[k - 1], floor)
    selected = {p for p in scored if pair_scores.get(p, floor) >= boundary}
    tp = len(selected & true_edges)
    if not selected or not true_edges:
        return 0.0
    prec, rec = tp / len(selected), tp / len(true_edges)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _mat(block):
    if block is None:
        return None
    return pd.DataFrame(block["data"], index=block["index"], columns=block["columns"])


def analytic_tuned_scores(json_path):
    """{(gene_1, gene_2): |TS|} for every directed pair, from the JSON correlations.
    Returns None if the replicate is missing any correlation matrix (e.g. a
    timepoint that never reached steady state)."""
    d = json.load(open(json_path))
    C = d["correlations"]
    need = ["gene_t1", "gene_t2", "twin_delta_t1", "random_delta_t1", "random_delta_t2", "direction"]
    if any(C.get(k) is None for k in need):
        return None
    R1, R2 = _mat(C["gene_t1"]), _mat(C["gene_t2"])
    D1 = _mat(C["twin_delta_t1"])
    RR1, RR2 = _mat(C["random_delta_t1"]), _mat(C["random_delta_t2"])
    XC = _mat(C["direction"])                      # XC.loc[a, b] = rho_cross(a -> b)
    genes = list(R1.columns)

    off = ~np.eye(len(genes), dtype=bool)
    sd_reg = float(np.nanstd(RR1.to_numpy()[off], ddof=1)) or 1e-6

    reg_raw = np.array([np.nanmean((XC.loc[g] - XC[g]).drop(g).to_numpy()) for g in genes])
    REG = dict(zip(genes, (reg_raw - reg_raw.mean()) / max(reg_raw.std(ddof=1), 1e-12)))

    a_idx, b_idx = [], []
    rt1, rt2, rd1, drr, rxy, ryx, het = [], [], [], [], [], [], []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            a_idx.append(a); b_idx.append(b)
            rt1.append(R1.loc[a, b]); rt2.append(R2.loc[a, b]); rd1.append(D1.loc[a, b])
            drr.append(RR2.loc[a, b] - RR1.loc[a, b])
            rxy.append(XC.loc[a, b]); ryx.append(XC.loc[b, a])
            het.append((D1.loc[a, b] - RR1.loc[a, b]) / sd_reg)
    rt1 = np.array(rt1); rt2 = np.array(rt2); rd1 = np.array(rd1); drr = np.array(drr)
    rxy = np.array(rxy); ryx = np.array(ryx); het = np.array(het)

    E1 = s(np.abs(rt1))
    TS = (E1
          - s(np.abs(rt2) - np.abs(rt1))
          - s(np.abs(rd1))
          - s(np.abs(drr))
          - s(np.minimum(np.abs(rxy), np.abs(ryx)))
          - s(np.abs(rxy) - np.abs(ryx))
          + 0.5 * s(rxy - ryx)
          + np.array([REG[a] - REG[b] for a, b in zip(a_idx, b_idx)]))
    lam = np.minimum(1.0, np.abs(het) / 2.33)
    z_reg_gated = (rt1 - lam * 0.5 * (rxy + ryx)) / sd_reg
    gate = np.abs(z_reg_gated) > GATE_Z
    # gate-fail -> drop the pair (floored to the bottom of the ranking). NOT -E1: the
    # downstream |twinScore| flips a very-negative -E1 back to the top (see
    # run_analytic_twinfer_tuned.py, superseded 2026-09-09).
    return {(a, b): abs(float(v)) for a, b, v, g in zip(a_idx, b_idx, TS, gate)
            if g and np.isfinite(v)}


# --------------------------------------------------------------------- scoring
def score_analytic():
    rows = []
    for net, topo in NETWORKS.items():
        true_edges, possible_edges = true_edges_from_topo(TOPO_ROOT / topo)
        base = len(true_edges) / len(possible_edges)
        jfs = sorted(glob.glob(str(TWINFER_OUT / f"{net}_rep_*_all_results.json")))
        n_used = 0
        for jf in jfs:
            sc = analytic_tuned_scores(jf)
            if sc is None:
                continue
            n_used += 1
            a = auprc_from_scored_pairs(sc, true_edges, possible_edges)
            f = f1_topk(sc, true_edges, possible_edges)
            rows.append(dict(dataset=net, algorithm=TW_ANALYTIC, rep=Path(jf).stem,
                             auprc=a, f1=f, base=base))
        print(f"[{net}] {n_used}/{len(jfs)} reps used, {len(true_edges)}/{len(possible_edges)} "
              f"edges (base {base:.3f})", flush=True)
    return pd.DataFrame(rows)


def beeline_rows():
    """Per-network mean AUPRC/F1 for the 7 BEELINE methods + TwINFER (twinScore)
    from the existing benchmark, so the heatmap has the same reference baselines."""
    b = pd.read_csv(BEELINE_SCORES_CSV)
    b = b[b.algorithm.isin(BEELINE + ["TwINFER (twinScore)"])]
    g = b.groupby(["network", "algorithm"], as_index=False)[["auprc", "f1"]].mean()
    g = g.rename(columns={"network": "dataset"})
    bases = {net: (lambda t: len(t[0]) / len(t[1]))(true_edges_from_topo(TOPO_ROOT / topo))
             for net, topo in NETWORKS.items()}
    g["base"] = g.dataset.map(bases)
    g.algorithm = g.algorithm.replace({"TwINFER (twinScore)": "TwINFER (shipped twinScore)"})
    return g


# --------------------------------------------------------------------- heatmap
CMAP = LinearSegmentedColormap.from_list(
    "score", ["#f4fbf0", "#a8dba0", "#4fb99f", "#2874a6", "#241468"])
COLUMN_LABELS = {"B_cell_activation": "B_cell", "Circadian_cycle": "Circadian"}


def make_panel(ax, piv, columns, title, vmin, vmax, fmt):
    piv = piv.reindex(columns=columns)
    piv["Avg"] = piv.mean(axis=1, skipna=True)
    piv = piv.sort_values("Avg", ascending=False)
    piv["Rank"] = range(1, len(piv) + 1)
    n_rows, n_cols = len(piv), len(columns)
    mat = np.hstack([piv[columns + ["Avg"]].values.astype(float),
                     np.full((n_rows, 1), np.nan)])
    masked = np.ma.masked_invalid(mat)
    cmap = CMAP.copy(); cmap.set_bad(color="white")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    thr = vmin + 0.62 * (vmax - vmin)
    for i in range(n_rows):
        for j in range(n_cols + 1):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="white" if v > thr else "#1a1a1a",
                    fontweight="bold" if j == n_cols else "normal")
        ax.text(n_cols + 1, i, f"#{piv['Rank'].iloc[i]}", ha="center", va="center",
                fontsize=8, color="#555")
    ax.set_xticks(range(n_cols + 2))
    ax.set_xticklabels([COLUMN_LABELS.get(c, c) for c in columns] + ["Avg", "Rank"],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(piv.index.tolist(), fontsize=9)
    for i, alg in enumerate(piv.index):
        if alg.startswith("TwINFER"):
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.axvline(n_cols - 0.5, color="#888", lw=1.2)
    ax.axvline(n_cols + 0.5, color="#888", lw=1.2)
    ax.set_xlim(-0.5, n_cols + 1.5)
    ax.set_xticks(np.arange(-0.5, n_cols + 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", bottom=False, left=False, top=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    return im


METRICS = [
    ("auprc", "AUPRC", 0.0, 1.0, "{:.2f}"),
    ("f1", "F1 (top-k, tie-aware)", 0.0, 1.0, "{:.2f}"),
    ("ratio_auprc", "AUPRC / random baseline", 0.0, 4.0, "{:.1f}x"),
    ("ratio_f1", "F1 (top-k) / random baseline", 0.0, 4.0, "{:.1f}x"),
]


def main():
    an = score_analytic()
    an_mean = an.groupby(["dataset", "algorithm"], as_index=False).agg(
        auprc=("auprc", "mean"), f1=("f1", "mean"), base=("base", "first"))
    combined = pd.concat([an_mean, beeline_rows()], ignore_index=True)
    combined["ratio_auprc"] = combined.auprc / combined.base
    combined["ratio_f1"] = combined.f1 / combined.base

    out_csv = REAL_NET_ROOT / "real_networks_analytic_tuned_scores.csv"
    an.to_csv(REAL_NET_ROOT / "real_networks_analytic_tuned_scores_perrep.csv", index=False)
    combined.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n=== mean AUPRC ===")
    print(combined.pivot(index="algorithm", columns="dataset", values="auprc").round(3).to_string())

    columns = list(NETWORKS)
    pdf_path = REAL_NET_ROOT / "real_networks_analytic_tuned_heatmap.pdf"
    with PdfPages(pdf_path) as pdf:
        for metric, mlabel, vmin, vmax, fmt in METRICS:
            piv = combined.pivot(index="algorithm", columns="dataset", values=metric)
            fig, ax = plt.subplots(figsize=(10.5, 5.5))
            im = make_panel(ax, piv, columns,
                            f"{mlabel}  --  original real networks  --  "
                            f"analytic yscher-tuned TwinScore vs BEELINE",
                            vmin, vmax, fmt)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("ratio to random" if metric.startswith("ratio") else "score", fontsize=9)
            if metric.startswith("ratio"):
                cbar.ax.axhline(1.0, color="red", lw=1.2)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
