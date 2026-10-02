"""
Plot the inferred cyclic-6-node network for each replicate side by side
with the ground-truth network, using plot_grn from grn_plot_spread_v2.py.

Self-regulatory edges are completely excluded:
  - They are not plotted.
  - They are not counted as TP.
  - They are not counted as FP.
  - They are not counted as FN.

Saves:
  - replicates_network_plots.png
  - replicates_network_plots.pdf
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# 1. Paths and imports
# ============================================================================

# HERE = Path(__file__).resolve().parent   [2026-09-30 replaced: this script used to sit in the data dir simulation_data/cyclic_6_nodes and read/write next to itself; that dir is now referenced explicitly]
HERE = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/cyclic_6_nodes')
CODE_REPO = HERE.parent.parent / "code" / "TwINFER"

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.append(str(CODE_REPO / "synthetic_network_analysis"))

from benchmarks.network_benchmarks.analysis_plots.grn_plot_spread_v2 import plot_grn as plot_grn_spread  # noqa: E402


# ============================================================================
# 2. Configuration
# ============================================================================

N_GENES = 6

GENES = [f"gene_{i + 1}" for i in range(N_GENES)]
SHORT_LABELS = [f"g{i + 1}" for i in range(N_GENES)]

GT_PATH = (
    HERE.parent.parent
    / "input_data"
#     / "cycle_6_node.txt"   [2026-09-30 replaced per user: the original name no longer exists; cycle_6_node.txt -> input_data/cycle_g6.txt; simulation_data/cycle_data -> simulation_data/cyclic_6_nodes]
    / "cycle_g6.txt"
)

# Same six-node hexagon layout used elsewhere in the repository.
FIXED_POS = {
    0: np.array([0.28, 0.78]),
    1: np.array([0.72, 0.78]),
    2: np.array([0.86, 0.50]),
    3: np.array([0.72, 0.22]),
    4: np.array([0.28, 0.22]),
    5: np.array([0.14, 0.50]),
}

PLOT_KWARGS = dict(
    show_colorbar=True,
    font_size=11,
    terminal_spread=1.1,
    terminal_angle_tolerance=1.10,
    reciprocal_separation=0.70,
    max_shift_fraction=1,
    vmin=-1,
    vmax=1,
)

# Any absolute value greater than this is treated as an inferred edge.
# This avoids counting tiny floating-point values as edges.
EDGE_THRESHOLD = 1e-8


# ============================================================================
# 3. Matrix utilities
# ============================================================================

def validate_matrix(matrix, name):
    """Validate that a matrix has the expected square shape."""
    matrix = np.asarray(matrix, dtype=float)

    expected_shape = (N_GENES, N_GENES)

    if matrix.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {matrix.shape}; "
            f"expected {expected_shape}."
        )

    return matrix


def remove_self_regulation(matrix):
    """
    Remove all self-regulatory edges.

    The returned matrix is a copy, so the original matrix is not modified.
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


def prepare_matrix(matrix, name):
    """
    Validate a matrix and remove all diagonal/self-regulatory entries.
    """
    matrix = validate_matrix(matrix, name=name)
    matrix = remove_self_regulation(matrix)
    return matrix


# ============================================================================
# 4. Data loading
# ============================================================================

def load_ground_truth():
    """Load the ground-truth network with self-regulation removed."""
    matrix = np.loadtxt(GT_PATH, delimiter=",")

    return prepare_matrix(
        matrix,
        name="Ground-truth matrix",
    )


def load_inferred(replicate_dir):
    """
    Load one replicate's inferred direction matrix.

    Self-regulatory entries are set to zero before plotting or evaluation.
    """
    matrix_path = replicate_dir / "correlation_matrices.pkl"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing inferred matrix file: {matrix_path}"
        )

    with open(matrix_path, "rb") as file:
        data = pickle.load(file)

    if "direction_matrix" not in data:
        raise KeyError(
            f"'direction_matrix' not found in {matrix_path}"
        )

    direction_matrix = data["direction_matrix"]

    if isinstance(direction_matrix, pd.DataFrame):
        direction_matrix = direction_matrix.reindex(
            index=GENES,
            columns=GENES,
            fill_value=0,
        )
        matrix = direction_matrix.to_numpy(dtype=float)
    else:
        matrix = np.asarray(direction_matrix, dtype=float)

    return prepare_matrix(
        matrix,
        name=f"Inferred matrix for {replicate_dir.name}",
    )


# ============================================================================
# 5. Edge evaluation
# ============================================================================

def calculate_edge_metrics(
    ground_truth,
    inferred,
    threshold=EDGE_THRESHOLD,
):
    """
    Calculate directed-edge TP, FP, and FN.

    Self-regulatory edges are excluded from every metric.

    Edge presence is defined by absolute matrix magnitude:
        abs(value) > threshold

    This evaluates whether a directed interaction exists. It does not require
    the inferred interaction sign to match the ground-truth sign.
    """
    ground_truth = validate_matrix(
        ground_truth,
        name="Ground-truth matrix",
    )
    inferred = validate_matrix(
        inferred,
        name="Inferred matrix",
    )

    ground_truth_edges = np.abs(ground_truth) > threshold
    inferred_edges = np.abs(inferred) > threshold

    # False on the diagonal and True everywhere else.
    off_diagonal_mask = ~np.eye(N_GENES, dtype=bool)

    # Explicitly exclude self-regulatory edges from both networks.
    ground_truth_edges &= off_diagonal_mask
    inferred_edges &= off_diagonal_mask

    true_positive = int(
        np.count_nonzero(
            ground_truth_edges & inferred_edges
        )
    )

    false_positive = int(
        np.count_nonzero(
            ~ground_truth_edges
            & inferred_edges
            & off_diagonal_mask
        )
    )

    false_negative = int(
        np.count_nonzero(
            ground_truth_edges
            & ~inferred_edges
            & off_diagonal_mask
        )
    )

    return {
        "true_positive_edges": true_positive,
        "false_positive_edges": false_positive,
        "false_negative_edges": false_negative,
    }


# ============================================================================
# 6. Plotting helpers
# ============================================================================

def format_network_axis(ax):
    """Apply identical formatting to every network panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_failed_panel(ax, replicate_index, error):
    """Display an error message inside a failed replicate panel."""
    ax.clear()
    ax.axis("off")

    ax.text(
        0.5,
        0.5,
        (
            f"Plot failed for replicate {replicate_index}\n"
            f"{type(error).__name__}: {error}"
        ),
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
        wrap=True,
    )


# ============================================================================
# 7. Main
# ============================================================================

def main():
    replicate_dirs = sorted(
        [
            path
            for path in HERE.glob("replicate_*")
            if path.is_dir()
        ],
        key=lambda path: int(path.name.split("_")[1]),
    )

    if not replicate_dirs:
        raise FileNotFoundError(
            f"No replicate directories were found in {HERE}. "
            "Expected directories named replicate_1, replicate_2, etc."
        )

    ground_truth_matrix = load_ground_truth()

    n_panels = 1 + len(replicate_dirs)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    panel_size = 4.5

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            panel_size * n_cols,
            panel_size * n_rows,
        ),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    # ------------------------------------------------------------------------
    # Panel 0: ground truth
    # ------------------------------------------------------------------------

    ground_truth_ax = axes_flat[0]

    plot_grn_spread(
        ground_truth_matrix,
        node_labels=SHORT_LABELS,
        title="Ground truth\n(cyclic_6_nodes)",
        ax=ground_truth_ax,
        fixed_pos=FIXED_POS,
        **PLOT_KWARGS,
    )

    format_network_axis(ground_truth_ax)

    # ------------------------------------------------------------------------
    # Panels 1 through N: inferred replicate networks
    # ------------------------------------------------------------------------

    for panel_index, replicate_dir in enumerate(
        replicate_dirs,
        start=1,
    ):
        ax = axes_flat[panel_index]
        replicate_index = int(
            replicate_dir.name.split("_")[1]
        )

        try:
            inferred_matrix = load_inferred(replicate_dir)

            metrics = calculate_edge_metrics(
                ground_truth=ground_truth_matrix,
                inferred=inferred_matrix,
            )

            true_positive = metrics["true_positive_edges"]
            false_positive = metrics["false_positive_edges"]
            false_negative = metrics["false_negative_edges"]

            title = (
                f"Replicate {replicate_index}\n"
                f"TP={true_positive} "
                f"FP={false_positive} "
                f"FN={false_negative}"
            )

            plot_grn_spread(
                inferred_matrix,
                node_labels=SHORT_LABELS,
                title=title,
                ax=ax,
                fixed_pos=FIXED_POS,
                **PLOT_KWARGS,
            )

            format_network_axis(ax)

            print(
                f"Replicate {replicate_index}: "
                f"TP={true_positive}, "
                f"FP={false_positive}, "
                f"FN={false_negative}"
            )

        except Exception as error:
            plot_failed_panel(
                ax=ax,
                replicate_index=replicate_index,
                error=error,
            )

            print(
                f"Plot failed for replicate "
                f"{replicate_index}: {error}"
            )

    # Hide unused subplot axes.
    for unused_index in range(
        n_panels,
        len(axes_flat),
    ):
        axes_flat[unused_index].axis("off")

    fig.suptitle(
        (
            "TwINFER inference on cyclic_6_nodes: "
            f"{len(replicate_dirs)} replicates vs ground truth"
        ),
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.93,
        wspace=0.25,
        hspace=0.35,
    )

    out_png = HERE / "replicates_network_plots.png"
    out_pdf = HERE / "replicates_network_plots.pdf"

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()