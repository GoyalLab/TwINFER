"""Plot Figure 4 fan-out and gated-regulation Z-scores for all gene pairs.

The script uses 20 Fan_out and 20 Mutual_regulation simulations and ignores
Feed_forward simulations. For every unordered pair (x, y) in Fan_out data it
plots both

    z_fanout(x, y) = max over g != x,y of
                     min(|z_cross(g -> x)|, |z_cross(g -> y)|)

and z_reg_gated(x, y). For every unordered pair in Mutual_regulation data it
plots z_reg_gated(x, y). Twin and clone weighting are shown side by side in one
box-plot figure. Each box label names the genes used by the statistic.

This file is safe to run with ``%run`` in a Jupyter notebook.
"""

from __future__ import annotations
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import argparse
import contextlib
import io
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path(
    f'{TWINFER_PROJECT_ROOT}/simulation_data/figure_4'
)
DEFAULT_PACKAGE_ROOT = Path(
    f'{TWINFER_PROJECT_ROOT}/code/TwINFER/package'
)
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "fanout_mutual_z_boxplot"


@dataclass
class SimulationViews:
    t1_twins_raw: pd.DataFrame
    across_t1_raw: pd.DataFrame
    across_t2_raw: pd.DataFrame
    rho_t1: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Plot z_fanout and z_reg_gated for every gene pair in Figure 4 "
            "Fan_out and Mutual_regulation simulations."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-per-motif", type=int, default=20)
    parser.add_argument("--t1", type=float, default=1)
    parser.add_argument("--t2", type=float, default=20)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101010)
    parser.add_argument(
        "--mode",
        choices=("twin", "clone", "both"),
        default="both",
    )

    # Jupyter supplies -f/--f for its kernel file. Long-option abbreviations
    # are disabled above, and the notebook-only argument is ignored here.
    if "IPython" in sys.modules:
        args, _ = parser.parse_known_args()
        return args
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(path))
    ]


def find_motif_files(
    data_dir: Path,
    motif_text: str,
    n_files: int,
) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Figure 4 directory not found: {data_dir}")
    if n_files < 1:
        raise ValueError("--n-per-motif must be at least 1")

    motif_text = motif_text.lower()
    files = [
        path
        for path in data_dir.rglob("*.csv")
        if path.name.lower().startswith("df_rows")
        and motif_text in path.name.lower()
        and "feed_forward" not in path.name.lower()
        and "simulation_before_division" not in path.name.lower()
        and "fanout_mutual_z_boxplot" not in path.parts
    ]
    files = sorted(files, key=natural_sort_key)
    if len(files) < n_files:
        raise RuntimeError(
            f"Requested {n_files} {motif_text} simulations, but found only "
            f"{len(files)} under {data_dir}."
        )
    return files[:n_files]


def gene_names(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns
    genes = [column[:-5] for column in columns if column.endswith("_mRNA")]
    if len(genes) < 2:
        raise ValueError(f"{path.name} has fewer than two *_mRNA columns")
    return genes


def read_simulation(path: Path, expected_genes: list[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = ["clone_id", "cell_id", "time_step", "replicate"]
    required += [f"{gene}_mRNA" for gene in expected_genes]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    actual_genes = [
        column[:-5] for column in data.columns if column.endswith("_mRNA")
    ]
    if actual_genes != expected_genes:
        raise ValueError(
            f"Gene columns differ from the first file. Expected "
            f"{expected_genes}; found {actual_genes}."
        )
    return data


def select_views(
    data: pd.DataFrame,
    t1: float,
    t2: float,
    seed: int,
) -> SimulationViews:
    available_times = set(data["time_step"].drop_duplicates().tolist())
    if t1 not in available_times or t2 not in available_times:
        raise ValueError(
            f"Requested t1={t1:g}, t2={t2:g}; available times are "
            f"{sorted(available_times)}"
        )

    replicates = data["replicate"].drop_duplicates().sort_values().to_numpy()
    if len(replicates) < 2:
        raise ValueError("Simulation input requires at least two replicate labels")

    rng = np.random.default_rng(seed)
    clone_ids = data["clone_id"].drop_duplicates().to_numpy()
    shuffled = rng.permutation(clone_ids)
    n_t1_only = n_t2_only = len(shuffled) // 4
    t1_only = shuffled[:n_t1_only]
    t2_only = shuffled[n_t1_only:n_t1_only + n_t2_only]
    across_time = shuffled[n_t1_only + n_t2_only:]

    t1_twins_raw = data[
        data["clone_id"].isin(t1_only)
        & (data["time_step"] == t1)
    ].copy()
    t2_twins_raw = data[
        data["clone_id"].isin(t2_only)
        & (data["time_step"] == t2)
    ].copy()
    across_t1 = data[
        data["clone_id"].isin(across_time)
        & (data["time_step"] == t1)
        & (data["replicate"] == replicates[0])
    ].copy()
    across_t2 = data[
        data["clone_id"].isin(across_time)
        & (data["time_step"] == t2)
        & (data["replicate"] == replicates[1])
    ].copy()

    if t1_twins_raw.empty or t2_twins_raw.empty:
        raise ValueError("No within-time twins remain after clone subsetting")
    if across_t1.empty or across_t2.empty:
        raise ValueError("No cross-time twins remain after clone subsetting")

    return SimulationViews(
        t1_twins_raw=t1_twins_raw.reset_index(drop=True),
        across_t1_raw=across_t1.reset_index(drop=True),
        across_t2_raw=across_t2.reset_index(drop=True),
        rho_t1=pd.concat([t1_twins_raw, across_t1], ignore_index=True),
    )


def standardized_z(observed: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) < 2:
        return np.nan
    null_std = float(np.std(null, ddof=1))
    if not np.isfinite(observed) or not np.isfinite(null_std) or null_std <= 0:
        return np.nan
    return float((observed - np.mean(null)) / null_std)


def lookup_pair(mapping: dict, gene_a: str, gene_b: str):
    for key in (
        (gene_a, gene_b),
        (gene_b, gene_a),
        tuple(sorted((gene_a, gene_b))),
    ):
        if key in mapping:
            return mapping[key]
    raise KeyError(f"No value found for gene pair {gene_a}, {gene_b}")


def calculate_all_z_reg_gated(
    views: SimulationViews,
    genes: list[str],
    unit: str,
    n_shuffles: int,
    seed: int,
    n_cores: int,
    cf,
) -> dict[tuple[str, str], dict[str, float]]:
    pairs = list(combinations(genes, 2))
    t1_twins = cf.assign_twin_id(views.t1_twins_raw).reset_index(drop=True)
    twin_delta, _ = cf.calculate_twin_random_correlations(
        views.rho_t1,
        t1_twins,
        genes,
        random_state=seed + 1,
        unit=unit,
    )
    het_null = cf.generate_random_shuffle(
        t1_twins,
        gene_list=genes,
        gene_pairs=pairs,
        n_shuffles=n_shuffles,
        unit=unit,
        raw_cells=views.rho_t1,
        random_state=seed + 2,
        n_cores_to_use=n_cores,
    )

    scores = {}
    for pair_index, (gene_a, gene_b) in enumerate(pairs):
        observed = float(twin_delta.loc[gene_a, gene_b])
        z_het = standardized_z(
            observed,
            lookup_pair(het_null, gene_a, gene_b),
        )
        gated = cf.calculate_gated_regulation_statistic(
            t1_twins,
            gene_a,
            gene_b,
            z_het=z_het,
            n_shuffles=n_shuffles,
            random_state=seed + 100 + pair_index,
            unit=unit,
        )
        value = gated.get("z_reg_gated", np.nan)
        scores[(gene_a, gene_b)] = {
            "z_reg_gated": float(value) if value is not None else np.nan,
            "z_het": z_het,
        }
    return scores


def calculate_all_cross_z(
    views: SimulationViews,
    genes: list[str],
    unit: str,
    n_shuffles: int,
    seed: int,
    n_cores: int,
    cf,
) -> dict[tuple[str, str], float]:
    across_t1, across_t2 = cf._build_cross_time_twins(
        views.across_t1_raw,
        views.across_t2_raw,
    )
    directed_pairs = [
        (source, target)
        for source in genes
        for target in genes
        if source != target
    ]
    cross_matrix = cf.get_cross_correlations(
        across_t1,
        across_t2,
        gene_pairs=directed_pairs,
        unit=unit,
    )

    # Calculate every directional cross-time Z-score without making pipeline
    # edge calls. The infinite threshold prevents significance decisions while
    # the raw Z-score details are still returned for z_fanout.
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, cross_details = cf.identify_actual_directed_edges(
            across_t1,
            across_t2,
            cross_matrix,
            gene_pairs=directed_pairs,
            z_score_threshold=np.inf,
            use_scramble=True,
            n_shuffles=n_shuffles,
            n_cores_to_use=n_cores,
            verbose=False,
            base_seed=seed + 3,
            return_z_scores=True,
            return_rho_cross_null=True,
            prepare_rho_cross_null=True,
            unit=unit,
        )

    return {
        pair: float(details.get("z_rho_cross", np.nan))
        for pair, details in cross_details.items()
    }


def calculate_z_fanout(
    gene_x: str,
    gene_y: str,
    genes: list[str],
    cross_z: dict[tuple[str, str], float],
) -> dict[str, object]:
    candidates = []
    for regulator in genes:
        if regulator in {gene_x, gene_y}:
            continue
        z_to_x = cross_z.get((regulator, gene_x), np.nan)
        z_to_y = cross_z.get((regulator, gene_y), np.nan)
        if not np.isfinite(z_to_x) or not np.isfinite(z_to_y):
            continue
        candidates.append(
            {
                "regulator": regulator,
                "score": min(abs(z_to_x), abs(z_to_y)),
                "z_to_x": z_to_x,
                "z_to_y": z_to_y,
            }
        )

    if not candidates:
        return {
            "z_fanout": np.nan,
            "best_regulator": None,
            "z_best_regulator_to_x": np.nan,
            "z_best_regulator_to_y": np.nan,
        }
    best = max(candidates, key=lambda item: item["score"])
    return {
        "z_fanout": float(best["score"]),
        "best_regulator": best["regulator"],
        "z_best_regulator_to_x": float(best["z_to_x"]),
        "z_best_regulator_to_y": float(best["z_to_y"]),
    }


def evaluate_file(
    path: Path,
    motif: str,
    genes: list[str],
    unit: str,
    args: argparse.Namespace,
    run_seed: int,
    cf,
) -> list[dict[str, object]]:
    data = read_simulation(path, genes)
    views = select_views(data, args.t1, args.t2, run_seed)
    gated_scores = calculate_all_z_reg_gated(
        views,
        genes,
        unit,
        args.n_shuffles,
        run_seed,
        args.n_cores,
        cf,
    )
    cross_z = (
        calculate_all_cross_z(
            views,
            genes,
            unit,
            args.n_shuffles,
            run_seed,
            args.n_cores,
            cf,
        )
        if motif == "Fan-out"
        else None
    )

    rows = []
    for gene_x, gene_y in combinations(genes, 2):
        pair_details = gated_scores[(gene_x, gene_y)]
        common = {
            "file": str(path),
            "simulation": path.stem,
            "motif": motif,
            "weighting": unit,
            "gene_x": gene_x,
            "gene_y": gene_y,
            "genes_scored": f"{gene_x}, {gene_y}",
            "t1": args.t1,
            "t2": args.t2,
            "z_het_scored_pair": pair_details["z_het"],
        }

        if motif == "Fan-out":
            fanout = calculate_z_fanout(gene_x, gene_y, genes, cross_z)
            possible_regulators = [
                gene for gene in genes if gene not in {gene_x, gene_y}
            ]
            regulator_label = (
                possible_regulators[0]
                if len(possible_regulators) == 1
                else "best third gene"
            )
            rows.append(
                {
                    **common,
                    "statistic": "z_fanout",
                    "plot_label": (
                        f"Fan-out\nz_fanout({gene_x}, {gene_y} | "
                        f"{regulator_label})"
                    ),
                    "z_score": fanout["z_fanout"],
                    "best_regulator": fanout["best_regulator"],
                    "z_best_regulator_to_x": fanout[
                        "z_best_regulator_to_x"
                    ],
                    "z_best_regulator_to_y": fanout[
                        "z_best_regulator_to_y"
                    ],
                }
            )

        rows.append(
            {
                **common,
                "statistic": "z_reg_gated",
                "plot_label": (
                    f"{motif}\nz_reg,gated({gene_x}, {gene_y})"
                ),
                "z_score": pair_details["z_reg_gated"],
                "best_regulator": None,
                "z_best_regulator_to_x": np.nan,
                "z_best_regulator_to_y": np.nan,
            }
        )
    return rows


def plot_boxplot(results: pd.DataFrame, output_path: Path) -> None:
    plot_data = results.copy()
    plot_data["z_score"] = pd.to_numeric(plot_data["z_score"], errors="coerce")
    plot_data = plot_data[np.isfinite(plot_data["z_score"])]
    if plot_data.empty:
        raise RuntimeError("No finite Z-scores were calculated")

    order = list(dict.fromkeys(results["plot_label"].tolist()))
    hue_order = [
        mode for mode in ("twin", "clone")
        if mode in set(plot_data["weighting"])
    ]
    palette = {"twin": "#E1812C", "clone": "#3274A1"}
    width = max(16, 2.3 * len(order))
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(width, 8.5))
    sns.boxplot(
        data=plot_data,
        x="plot_label",
        y="z_score",
        hue="weighting",
        order=order,
        hue_order=hue_order,
        palette=palette,
        width=0.62,
        whis=1.5,
        linewidth=1.5,
        boxprops={"alpha": 0.30},
        medianprops={"linewidth": 2.3},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        ax=axis,
    )
    axis.axhline(0, color="#888888", linewidth=1, alpha=0.65)
    for critical in (-2.33, 2.33):
        axis.axhline(
            critical,
            color="#777777",
            linewidth=1.2,
            linestyle=(0, (5, 5)),
        )
    axis.set_xlabel("")
    axis.set_ylabel("Z-score")
    axis.set_title(
        "Figure 4: all-pair fan-out and heterogeneity-gated regulation",
        fontweight="bold",
    )
    axis.legend(title="Weighting", frameon=False)
    axis.tick_params(
        axis="x",
        labelsize=10,
        rotation=20 if len(order) > 6 else 0,
    )
    sns.despine(ax=axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def main() -> pd.DataFrame:
    args = parse_args()
    if args.t1 == args.t2:
        raise ValueError("t1 and t2 must be different")

    package_root = args.package_root.expanduser().resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"TwINFER package directory not found: {package_root}")
    if str(package_root) not in sys.path:
        pass
        # [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
        # sys.path.insert(0, str(package_root))
    import twinfer.inference.correlation_functions as cf

    data_dir = args.data_dir.expanduser().resolve()
    fanout_files = find_motif_files(data_dir, "fan_out", args.n_per_motif)
    mutual_files = find_motif_files(
        data_dir,
        "mutual_regulation",
        args.n_per_motif,
    )
    fanout_genes = gene_names(fanout_files[0])
    mutual_genes = gene_names(mutual_files[0])
    units = [args.mode] if args.mode != "both" else ["twin", "clone"]

    print(
        f"Using {len(fanout_files)} Fan_out and {len(mutual_files)} "
        "Mutual_regulation simulations; Feed_forward ignored."
    )
    print(f"Fan-out pairs: {list(combinations(fanout_genes, 2))}")
    print(f"Mutual-regulation pairs: {list(combinations(mutual_genes, 2))}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "fanout_mutual_all_pairs_z_scores.csv"
    plot_path = output_dir / "fanout_mutual_all_pairs_z_boxplot.png"
    failure_path = output_dir / "fanout_mutual_all_pairs_failures.csv"

    tasks = [
        ("Fan-out", path, fanout_genes) for path in fanout_files
    ] + [
        ("Mutual regulation", path, mutual_genes) for path in mutual_files
    ]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total = len(tasks) * len(units)
    task_number = 0

    for unit in units:
        for file_index, (motif, path, genes) in enumerate(tasks):
            task_number += 1
            print(f"[{task_number}/{total}] {unit}: {path.name}")
            run_seed = args.seed + 1000 * file_index
            try:
                rows.extend(
                    evaluate_file(
                        path,
                        motif,
                        genes,
                        unit,
                        args,
                        run_seed,
                        cf,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "file": str(path),
                        "motif": motif,
                        "weighting": unit,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  Failed: {type(exc).__name__}: {exc}")

            if rows:
                pd.DataFrame(rows).to_csv(result_path, index=False)
            if failures:
                pd.DataFrame(failures).to_csv(failure_path, index=False)

    if not rows:
        raise RuntimeError("All fan-out and mutual-regulation calculations failed")

    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    plot_boxplot(results, plot_path)
    print(f"Saved: {plot_path}")
    print(f"Successful score rows: {len(results)}; failed runs: {len(failures)}")
    return results


if __name__ == "__main__":
    fanout_mutual_z_scores = main()
