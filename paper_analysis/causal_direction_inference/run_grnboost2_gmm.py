"""
GRNBoost2 baseline comparison for the 5-gene cascade (Part D), GMM-threshold
only -- no shuffle-null generator. Runs in the `grnboost310` conda env (has
arboreto/sklearn; NOT twinfer-code).

Per replicate: run grnboost2 on the t1 expression matrix, fit a 2-component
Gaussian mixture to the resulting importance-score distribution, threshold
at the two components' intersection (falls back to no edges predicted if
the two components don't cross), then score precision/recall/F1 against the
ground-truth connectivity matrix -- same scoring logic as
score_cascade_results.py (Part C), applied to GRNBoost2's predictions
instead of TwINFER's.

Output: one grnboost2_rep_{rep_id}.json per task, under --output-dir. Skips
(does not recompute) any task whose output file already exists.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from arboreto.algo import grnboost2
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import norm
from scipy.optimize import brentq

SEED = 101010


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_grnboost2_replicate(path_to_simulation_file, gt_matrix_path, t1=1):
    gt_matrix = np.loadtxt(gt_matrix_path, delimiter=",")
    n_genes = gt_matrix.shape[0]
    genes = [f"gene_{i+1}_mRNA" for i in range(n_genes)]
    gt_df = pd.DataFrame(gt_matrix, index=genes, columns=genes)
    gt_binary = (gt_df.values != 0).astype(int)
    np.fill_diagonal(gt_binary, 0)
    mask = ~np.eye(len(genes), dtype=bool).flatten()
    y_true = gt_binary.flatten()[mask]

    df = pd.read_csv(path_to_simulation_file)
    df_t1 = df[df["time_step"] == t1][genes]
    X = df_t1.to_numpy(dtype=np.float64)

    network = grnboost2(
        expression_data=X,
        gene_names=genes,
        tf_names=genes,
        seed=SEED,
        verbose=False,
    )

    pred_df = (
        network
        .pivot(index="TF", columns="target", values="importance")
        .reindex(index=genes, columns=genes, fill_value=0.0)
    )
    np.fill_diagonal(pred_df.values, 0.0)
    y_score = pred_df.values.flatten()[mask]

    # ---- GMM threshold (2-component intersection) ----
    x = network["importance"].values.reshape(-1, 1)
    importance_threshold = np.inf
    threshold_type = "GMM_no_intersection"
    if len(x) >= 2 and len(np.unique(x)) >= 2:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gmm.fit(x)
        weights = gmm.weights_
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        order = np.argsort(means)
        w1, w2 = weights[order]
        m1, m2 = means[order]
        s1, s2 = stds[order]
        f = lambda v: w1 * norm.pdf(v, m1, s1) - w2 * norm.pdf(v, m2, s2)
        try:
            importance_threshold = brentq(f, m1, m2)
            threshold_type = "GMM_intersection"
        except ValueError:
            pass

    y_pred_gmm = (y_score >= importance_threshold).astype(int)
    if y_pred_gmm.sum() > 0:
        precision = float(precision_score(y_true, y_pred_gmm, zero_division=0))
        recall = float(recall_score(y_true, y_pred_gmm))
        f1 = float(f1_score(y_true, y_pred_gmm))
    else:
        precision = recall = f1 = 0.0

    # Naive coin-flip baseline for context -- unrelated to the (skipped)
    # shuffle-null generator, just an independent random 0/1 predictor.
    rng = np.random.default_rng(SEED)
    y_pred_random = rng.binomial(1, 0.5, size=len(y_true))
    precision_random = float(precision_score(y_true, y_pred_random, zero_division=0))
    recall_random = float(recall_score(y_true, y_pred_random, zero_division=0))
    f1_random = float(f1_score(y_true, y_pred_random, zero_division=0))

    return {
        "network": network.to_dict(orient="records"),
        "gmm_threshold": float(importance_threshold),
        "gmm_mode": threshold_type,
        "n_edges_predicted": int(y_pred_gmm.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_random": precision_random,
        "recall_random": recall_random,
        "f1_random": f1_random,
    }


def collect_tasks(sim_folder, name_contains):
    tasks = []
    for f in sorted(glob.glob(os.path.join(sim_folder, "df_*.csv"))):
        if name_contains not in os.path.basename(f):
            continue
        m = re.search(r"rep_(\d+)", os.path.basename(f))
        rep_id = m.group(1) if m else "0"
        tasks.append((f, rep_id))
    return tasks


def run_one(path, rep_id, output_dir, gt_matrix_path):
    out_path = os.path.join(output_dir, f"grnboost2_rep_{rep_id}.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    record = run_grnboost2_replicate(path, gt_matrix_path)
    record["rep_id"] = rep_id
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-folder", type=str,
                         default="/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_3_simulations")
    parser.add_argument("--file-prefix", type=str, default="five_gene_cascade")
    parser.add_argument(
        "--gt-matrix",
        type=str,
        default=f'{TWINFER_PROJECT_ROOT}/code/TwINFER/simulation_example_input_data/connectivity_matrix_5_gene_linear_cascade.txt',
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    if not args.list_tasks and not args.output_dir:
        parser.error("--output-dir is required unless --list-tasks is given")

    tasks = collect_tasks(args.sim_folder, args.file_prefix)

    if args.list_tasks:
        print("\n".join(rep_id for _, rep_id in tasks))
        raise SystemExit(0)

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t[1] in wanted]

    print(f"Running {len(tasks)} GRNBoost2 task(s), output -> {args.output_dir}", flush=True)
    for i, (path, rep_id) in enumerate(tasks):
        t0 = time.time()
        out_path, status = run_one(path, rep_id, args.output_dir, args.gt_matrix)
        print(f"[{i+1}/{len(tasks)}] grnboost2 rep {rep_id}: {status} ({time.time()-t0:.1f}s) -> {out_path}", flush=True)
