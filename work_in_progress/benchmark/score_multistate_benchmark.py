"""
Score the multistate/seeded real-network benchmark: 7 BEELINE algorithms
(PIDC, GENIE3, GRNBOOST2, PPCOR, SCODE, SCSGL, PEARSON) from
beeline_inference/<dataset>/simrep<r>_{spread,twin_paired}/<algo>/rankedEdges.csv,
plus TwINFER scored FAIRLY via its full unfiltered directed cross-correlation
matrix (direction.unfiltered_matrix -- every directed pair, no candidate-panel
restriction; see ranked_edges.py's own docstring, which recommends this mode
for scoring against a full ground truth) from
twinfer_inference_multistate/<dataset>_rep_<r>_<hash>_all_results.json.

AUPRC over the FULL n*(n-1) directed-pair universe for every method (methods/
pairs with no score get the minimum observed score, i.e. worst rank -- not
dropped), so no method benefits from only being scored on an easy subset.
TwINFER's score does not depend on the BEELINE sampling scheme (spread vs
twin_paired -- both are just different re-samplings of the same underlying
simulation TwINFER was run on directly), so its AUPRC is identical in both.

Output: real_networks_multistate_benchmark_scores.csv (per rep/scheme/algorithm)
        real_networks_multistate_benchmark.pdf (one page per dataset)
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
from sklearn.metrics import auc, precision_recall_curve

from twinfer.utils.paths import get_data_root

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
TOPO_ROOT = PROJECT_ROOT / "input_data" / "real_world_networks"
BEELINE_OUT = DATA_ROOT / "paper_analysis" / "real_networks" / "beeline_inference"
TWINFER_OUT = DATA_ROOT / "paper_analysis" / "real_networks" / "twinfer_inference_multistate"
OUT_DIR = DATA_ROOT / "paper_analysis" / "real_networks"

DATASETS = ["GSD_multistate", "GSD_seeded", "HSC_multistate", "HSC_seeded",
            "EMT_multistate", "EMT_seeded", "VSC_seeded", "mCAD_seeded"]
BASE_TOPO = {"GSD": "GSD.txt", "HSC": "HSC.txt", "EMT": "EMT.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt"}
ALGOS = ["PIDC", "GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON"]
SCHEMES = ["twin_paired", "spread"]
N_REPS = 3


def true_edges_from_topo(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    genes = [f"gene_{i + 1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    possible = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j}
    return true, possible


def auprc_from_scored_pairs(pair_scores: dict, true_edges: set, possible_edges: set) -> float:
    """Full-universe AUPRC; pairs absent from pair_scores get the worst (floor) score."""
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    y_true = np.fromiter((1 if p in true_edges else 0 for p in possible_edges), dtype=int)
    y_score = np.fromiter((pair_scores.get(p, floor) for p in possible_edges), dtype=float)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return float(auc(rec, prec))


def load_beeline_scores(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, sep="\t")
    return {(r.Gene1, r.Gene2): abs(float(r.EdgeWeight)) for r in df.itertuples()
            if pd.notna(r.EdgeWeight)}


def load_twinfer_scores(json_path: Path) -> dict:
    d = json.load(open(json_path))
    um = d["direction"]["unfiltered_matrix"]
    mat = pd.DataFrame(um["data"], index=um["index"], columns=um["columns"])
    scores = {}
    for g1 in mat.index:
        for g2 in mat.columns:
            if g1 == g2:
                continue
            v = mat.loc[g1, g2]
            if pd.notna(v):
                scores[(g1, g2)] = abs(float(v))
    return scores


def main():
    rows = []
    for dataset_id in DATASETS:
        base = dataset_id.split("_")[0]
        true_edges, possible_edges = true_edges_from_topo(TOPO_ROOT / BASE_TOPO[base])
        print(f"[{dataset_id}] {len(true_edges)} true / {len(possible_edges)} possible directed edges", flush=True)

        for rep in range(N_REPS):
            # ---- TwINFER (scheme-independent) ----
            matches = sorted(glob.glob(str(TWINFER_OUT / f"{dataset_id}_rep_{rep}_*_all_results.json")))
            twinfer_auprc = np.nan
            if matches:
                tw_scores = load_twinfer_scores(Path(matches[0]))
                twinfer_auprc = auprc_from_scored_pairs(tw_scores, true_edges, possible_edges)
            else:
                print(f"  [warn] no TwINFER json for {dataset_id} rep {rep}")

            for scheme in SCHEMES:
                rows.append(dict(dataset_id=dataset_id, rep=rep, scheme=scheme,
                                 algorithm="TwINFER", auprc=twinfer_auprc))

                run_dir = BEELINE_OUT / dataset_id / f"simrep{rep}_{scheme}"
                for algo in ALGOS:
                    ranked_path = run_dir / algo / "rankedEdges.csv"
                    if not ranked_path.exists():
                        print(f"  [warn] missing {ranked_path}")
                        continue
                    scores = load_beeline_scores(ranked_path)
                    a = auprc_from_scored_pairs(scores, true_edges, possible_edges)
                    rows.append(dict(dataset_id=dataset_id, rep=rep, scheme=scheme, algorithm=algo, auprc=a))

    scores_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "real_networks_multistate_benchmark_scores.csv"
    scores_df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    # ---------------------------------------------------------------- plot
    pdf_path = OUT_DIR / "real_networks_multistate_benchmark.pdf"
    method_order = ALGOS + ["TwINFER"]
    with PdfPages(pdf_path) as pdf:
        for dataset_id in DATASETS:
            sub = scores_df[scores_df["dataset_id"] == dataset_id]
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(method_order))
            width = 0.35
            for off, scheme in zip([-width / 2, width / 2], SCHEMES):
                means, stds = [], []
                for m in method_order:
                    vals = sub[(sub["algorithm"] == m) & (sub["scheme"] == scheme)]["auprc"].dropna()
                    means.append(vals.mean() if len(vals) else np.nan)
                    stds.append(vals.std() if len(vals) else 0)
                colors = ["#4C72B0" if m != "TwINFER" else "#C44E52" for m in method_order]
                ax.bar(x + off, means, width, yerr=stds, capsize=3,
                      color=colors, alpha=(0.55 if scheme == "spread" else 0.95),
                      label=scheme, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(method_order, rotation=30, ha="right")
            ax.set_ylabel("AUPRC (full directed-pair universe)")
            ax.set_title(f"{dataset_id} -- benchmark comparison (3 reps, mean +/- std)\n"
                         f"TwINFER scored via full unfiltered cross-correlation matrix (fair, unrestricted)")
            ax.legend(title="sampling scheme", frameon=False)
            ax.axhline(0, color="grey", lw=0.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
