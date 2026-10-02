"""
Rate TwINFER on the ORIGINAL 8 real-network simulations (GSD/HSC/VSC/mCAD/
EMT/Pluripotent/B_cell_activation/Circadian_cycle -- the pre-multistate-work
runs in twinfer_inference/), scored by its own proposed final output: the
TwinScore ranking (ranked_edges' twinScore column), matched against every
BEELINE algorithm's rankedEdges.csv on the SAME simulated replicate (matched
by the 8-hex-char simulator hash embedded in both TwINFER's and BEELINE's
filenames).

Unlike direction.unfiltered_matrix (used in score_multistate_benchmark.py),
ranked_edges/twinScore only covers TwINFER's own tested candidate panel (a
subset of all n*(n-1) pairs) -- exactly as the paper's method is meant to be
read out. Untested pairs get the worst (floor) score in the full-universe
AUPRC, same fair treatment as every other method; no method is scored on an
easier subset than any other.

Output: real_networks_twinscore_benchmark_scores.csv (per rep/scheme/algorithm)
        real_networks_twinscore_benchmark.pdf (one page per network)
"""
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import auc, precision_recall_curve

from twinfer.utils.paths import get_data_root

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
TOPO_ROOT = PROJECT_ROOT / "input_data" / "real_world_networks"
TWINFER_OUT = DATA_ROOT / "paper_analysis" / "real_networks" / "twinfer_inference"
OUT_DIR = DATA_ROOT / "paper_analysis" / "real_networks"

LEGACY_BEELINE_ROOT = PROJECT_ROOT / "benchmarking_analysis" / "twinfer_real" / "outputs"
NEW_BEELINE_ROOT = DATA_ROOT / "paper_analysis" / "real_networks" / "beeline_inference"

# network -> (topology file, json filename glob token, BEELINE output root, BEELINE
# dataset dir name, match_mode). Legacy (GSD/HSC/VSC/mCAD) BEELINE run dirs are named
# simrep<hash>_<scheme> (matched via the simulator hash shared by both filenames); the
# "new" track's (EMT/Pluripotent/B_cell_activation/Circadian_cycle) are named
# simrep<rep_index>_<scheme> instead (see twinfer_to_boolode_format.py's
# sim_filename_re, which captures the replicate NUMBER, not a hash).
NETWORKS = {
    "GSD":  dict(topo="GSD.txt", json_token="GSD", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="GSD", match_mode="hash"),
    "HSC":  dict(topo="HSC.txt", json_token="HSC", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="HSC", match_mode="hash"),
    "VSC":  dict(topo="VSC.txt", json_token="VSC", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="VSC", match_mode="hash"),
    "mCAD": dict(topo="mCAD.txt", json_token="mCAD", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="mCAD", match_mode="hash"),
    "EMT":  dict(topo="EMT.txt", json_token="EMT", beeline_root=NEW_BEELINE_ROOT, beeline_id="EMT", match_mode="index"),
    "Pluripotent": dict(topo="Pluripotent.txt", json_token="Pluripotent", beeline_root=NEW_BEELINE_ROOT, beeline_id="Pluripotent", match_mode="index"),
    "B_cell_activation": dict(topo="B_cell.txt", json_token="B_cell_activation", beeline_root=NEW_BEELINE_ROOT, beeline_id="B_cell_activation", match_mode="index"),
    "Circadian_cycle": dict(topo="circadian.txt", json_token="Circadian_cycle", beeline_root=NEW_BEELINE_ROOT, beeline_id="Circadian_cycle", match_mode="index"),
}
ALGOS = ["PIDC", "GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON"]
SCHEMES = ["twin_paired", "spread"]
HASH_RE = re.compile(r"_([0-9a-fA-F]{8})_all_results\.json$")
INDEX_RE = re.compile(r"_rep_(\d+)_[0-9a-fA-F]{8}_all_results\.json$")


def true_edges_from_topo(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    genes = [f"gene_{i + 1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    possible = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j}
    true_sign = {(genes[i], genes[j]): ("+" if M[i, j] > 0 else "-")
                 for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    return true, possible, true_sign


def auprc_from_scored_pairs(pair_scores: dict, true_edges: set, possible_edges: set) -> float:
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    y_true = np.fromiter((1 if p in true_edges else 0 for p in possible_edges), dtype=int)
    y_score = np.fromiter((pair_scores.get(p, floor) for p in possible_edges), dtype=float)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return float(auc(rec, prec))


def _sign_gated_scores(mag: dict, sign: dict, true_sign: dict, possible_edges: set) -> dict:
    """Same score dict as `mag`, except a true edge whose predicted sign is wrong --
    or that was never scored at all -- is dropped (so it falls back to the caller's
    floor treatment), i.e. counted as a MISSED required positive, exactly like an
    unscored pair is in the unsigned metrics. Negatives (non-edges) are untouched.
    This makes the signed metric a strict superset of failure modes of the unsigned
    one (same positive set, same floor-for-missing convention), rather than silently
    shrinking the recall target to only the scored+correctly-signed edges, which is
    what naively checking `sign.get(p) == true_sign[p]` on the LABEL does (an
    unscored true edge has sign.get(p) is None, which never matches -- that would
    exempt it from the recall requirement entirely instead of counting it as a miss)."""
    gated = {}
    for p in possible_edges:
        if p in true_sign:
            if sign.get(p) == true_sign[p]:
                gated[p] = mag[p]
        elif p in mag:
            gated[p] = mag[p]
    return gated


def auprc_signed(mag: dict, sign: dict, true_sign: dict, possible_edges: set) -> float:
    """Same PR-curve AUPRC and same positive set (every true edge) as the unsigned
    metric, but a true edge only keeps its ranked score if the predicted sign matches
    ground truth -- a wrong-sign call, or no call at all, is treated as a miss (floor
    score), just like an unscored pair always is."""
    gated = _sign_gated_scores(mag, sign, true_sign, possible_edges)
    return auprc_from_scored_pairs(gated, set(true_sign.keys()), possible_edges)


def _topk_selection(pair_scores: dict, possible_edges: set, k: int) -> set:
    """Tie-aware top-k: rank all possible_edges by score (unscored pairs = worst),
    then include every pair tied with the score at the k-th boundary -- so the
    selected set can be larger than k when there's a tie there, same tie-aware
    convention as the original summary table's f1_topk."""
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    scored = sorted(possible_edges, key=lambda p: pair_scores.get(p, floor), reverse=True)
    if k <= 0 or not scored:
        return set()
    k = min(k, len(scored))
    boundary_score = pair_scores.get(scored[k - 1], floor)
    return {p for p in scored if pair_scores.get(p, floor) >= boundary_score}


def f1_topk(pair_scores: dict, true_edges: set, possible_edges: set) -> float:
    k = len(true_edges)
    selected = _topk_selection(pair_scores, possible_edges, k)
    tp = len(selected & true_edges)
    if not selected or not true_edges:
        return 0.0
    precision = tp / len(selected)
    recall = tp / len(true_edges)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def f1_topk_signed(mag: dict, sign: dict, true_sign: dict, possible_edges: set) -> float:
    """Same top-k selection logic as f1_topk, but ranked on the sign-gated scores (see
    _sign_gated_scores): a true edge with the wrong predicted sign, or no scored
    prediction at all, is pushed to the floor -- so it can't occupy a top-k slot on
    the strength of a sign it got wrong, and correctly can't count as a hit either."""
    gated = _sign_gated_scores(mag, sign, true_sign, possible_edges)
    return f1_topk(gated, set(true_sign.keys()), possible_edges)


def load_beeline_scores(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, sep="\t")
    return {(r.Gene1, r.Gene2): abs(float(r.EdgeWeight)) for r in df.itertuples() if pd.notna(r.EdgeWeight)}


def load_twinscore(json_path: Path) -> dict:
    d = json.load(open(json_path))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    return {(r.gene_1, r.gene_2): abs(float(r.twinScore)) for r in df.itertuples() if pd.notna(r.twinScore)}


def load_twinscore_signed(json_path: Path):
    """magnitude for ranking = |twinScore| (unchanged); predicted sign = sign of the
    plain co-expression correlation rho_t2 for that edge, falling back to rho_t1 and
    then the directed cross-correlation rho_cross_xy only when both are unavailable
    -- sign from the correlation, not from |twinScore| itself, which has no sign."""
    d = json.load(open(json_path))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    mag, sign = {}, {}
    for r in df.itertuples():
        if pd.isna(r.twinScore):
            continue
        key = (r.gene_1, r.gene_2)
        mag[key] = abs(float(r.twinScore))
        s = getattr(r, "rho_t2", np.nan)
        if pd.isna(s):
            s = getattr(r, "rho_t1", np.nan)
        if pd.isna(s):
            s = getattr(r, "rho_cross_xy", np.nan)
        sign[key] = "+" if (pd.notna(s) and s >= 0) else "-"
    return mag, sign


def main():
    rows = []
    for net, cfg in NETWORKS.items():
        true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / cfg["topo"])
        json_files = sorted(glob.glob(str(TWINFER_OUT / f"{cfg['json_token']}_rep_*_all_results.json")))
        print(f"[{net}] {len(true_edges)} true / {len(possible_edges)} possible edges, "
              f"{len(json_files)} TwINFER replicate(s)", flush=True)

        n_scored = 0
        for jf in json_files:
            if cfg["match_mode"] == "hash":
                m = HASH_RE.search(jf)
                if not m:
                    print(f"  [warn] couldn't parse hash from {jf}")
                    continue
                run_key = m.group(1)
            else:
                m = INDEX_RE.search(jf)
                if not m:
                    print(f"  [warn] couldn't parse rep index from {jf}")
                    continue
                run_key = m.group(1)

            tw_scores = load_twinscore(Path(jf))
            tw_auprc = auprc_from_scored_pairs(tw_scores, true_edges, possible_edges)
            tw_f1 = f1_topk(tw_scores, true_edges, possible_edges)
            tw_coverage = len(tw_scores) / len(possible_edges)

            tw_mag, tw_sign = load_twinscore_signed(Path(jf))
            tw_auprc_signed = auprc_signed(tw_mag, tw_sign, true_sign, possible_edges)
            tw_f1_signed = f1_topk_signed(tw_mag, tw_sign, true_sign, possible_edges)

            found_any_beeline = False
            for scheme in SCHEMES:
                run_dir = cfg["beeline_root"] / cfg["beeline_id"] / f"simrep{run_key}_{scheme}"
                if not run_dir.is_dir():
                    continue
                found_any_beeline = True
                rows.append(dict(network=net, sim_rep_key=run_key, scheme=scheme,
                                 algorithm="TwINFER (twinScore)", auprc=tw_auprc, f1=tw_f1, coverage=tw_coverage))
                rows.append(dict(network=net, sim_rep_key=run_key, scheme=scheme,
                                 algorithm="TwINFER (signed)", auprc=tw_auprc_signed, f1=tw_f1_signed,
                                 coverage=tw_coverage))
                for algo in ALGOS:
                    ranked_path = run_dir / algo / "rankedEdges.csv"
                    if not ranked_path.exists():
                        continue
                    scores = load_beeline_scores(ranked_path)
                    a = auprc_from_scored_pairs(scores, true_edges, possible_edges)
                    f = f1_topk(scores, true_edges, possible_edges)
                    rows.append(dict(network=net, sim_rep_key=run_key, scheme=scheme,
                                     algorithm=algo, auprc=a, f1=f, coverage=len(scores) / len(possible_edges)))
            if found_any_beeline:
                n_scored += 1
        print(f"  -> {n_scored}/{len(json_files)} replicates had a matching BEELINE run", flush=True)

    scores_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "real_networks_twinscore_benchmark_scores.csv"
    scores_df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    # ---------------------------------------------------------------- summary
    method_order = ALGOS + ["TwINFER (twinScore)", "TwINFER (signed)"]
    piv = scores_df.groupby(["network", "algorithm"])["auprc"].mean().unstack()
    piv = piv.reindex(columns=method_order)
    print("\n=== mean AUPRC (all reps x schemes pooled) ===")
    print(piv.round(3).to_string())
    piv_f1 = scores_df.groupby(["network", "algorithm"])["f1"].mean().unstack()
    piv_f1 = piv_f1.reindex(columns=method_order)
    print("\n=== mean F1 top-k, tie-aware (all reps x schemes pooled) ===")
    print(piv_f1.round(3).to_string())
    cov = scores_df[scores_df.algorithm == "TwINFER (twinScore)"].groupby("network")["coverage"].mean()
    print("\n=== TwINFER candidate-panel coverage (fraction of all directed pairs it scores) ===")
    print(cov.round(3).to_string())

    # ---------------------------------------------------------------- plot
    pdf_path = OUT_DIR / "real_networks_twinscore_benchmark.pdf"
    with PdfPages(pdf_path) as pdf:
        for net in NETWORKS:
            sub = scores_df[scores_df["network"] == net]
            if sub.empty:
                continue
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
            x = np.arange(len(method_order))
            width = 0.35
            colors = ["#4C72B0" if m in ALGOS else ("#C44E52" if m == "TwINFER (twinScore)" else "#DD8452")
                     for m in method_order]
            for metric, axis, ylabel in [("auprc", ax, "AUPRC (full directed-pair universe)"),
                                          ("f1", ax2, "F1 top-k (tie-aware)")]:
                for off, scheme in zip([-width / 2, width / 2], SCHEMES):
                    means, stds = [], []
                    for m in method_order:
                        vals = sub[(sub["algorithm"] == m) & (sub["scheme"] == scheme)][metric].dropna()
                        means.append(vals.mean() if len(vals) else np.nan)
                        stds.append(vals.std() if len(vals) else 0)
                    axis.bar(x + off, means, width, yerr=stds, capsize=3,
                            color=colors, alpha=(0.55 if scheme == "spread" else 0.95),
                            label=scheme, edgecolor="black", linewidth=0.5)
                axis.set_ylabel(ylabel)
                axis.axhline(0, color="grey", lw=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(method_order, rotation=30, ha="right")
            n_reps = sub["sim_rep_key"].nunique()
            cov_pct = cov.get(net, float("nan")) * 100
            ax.set_title(f"{net} -- original sims, TwinScore ranking ({n_reps} reps, mean +/- std)\n"
                         f"TwINFER candidate-panel coverage: {cov_pct:.0f}% of all directed pairs")
            ax.legend(title="sampling scheme", frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
