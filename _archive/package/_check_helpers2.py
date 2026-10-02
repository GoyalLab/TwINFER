import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SNA_DIR = Path("/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/synthetic_network_analysis")
if str(_SNA_DIR) not in sys.path:
    sys.path.insert(0, str(_SNA_DIR))

import grn_plot_spread_v2
importlib.reload(grn_plot_spread_v2)
from grn_plot_spread_v2 import plot_grn as plot_grn_spread


def _short_labels(genes):
    """Map long gene names like 'gene_1' -> short labels 'g1' to avoid text
    overlap in the plot (falls back to the original name if it doesn't end
    in an underscore + number)."""
    short = []
    for g in genes:
        g = str(g)
        if "_" in g:
            prefix, _, suffix = g.rpartition("_")
            short.append("g" + suffix if suffix.isdigit() else g)
        else:
            short.append(g)
    return short


def _dataframe_from_matrix_dict(d):
    """Rebuild a square pandas DataFrame from a nested dict, e.g. when
    correlation_matrices has been round-tripped through JSON."""
    df = pd.DataFrame(d)
    genes = df.index.tolist()
    return df.reindex(index=genes, columns=genes, fill_value=0)


def _circular_layout(genes):
    """Simple circular layout used as a fallback when the notebook does not
    provide its own fixed_pos."""
    n = len(genes)
    pos = {}
    for i in range(n):
        angle = 2 * np.pi * i / n + np.pi / 2
        pos[i] = np.array([0.5 + 0.38 * np.cos(angle), 0.5 + 0.38 * np.sin(angle)])
    return pos


def load_inferred_network(source):
    """Load an inferred network (direction matrix) from a DataFrame/dict
    already in memory (e.g. this notebook's own `correlation_matrices`), or
    from a path to a .pkl/.json file produced by TwINFER."""
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, dict):
        if "direction_matrix" in source:
            return _dataframe_from_matrix_dict(source["direction_matrix"])
        return _dataframe_from_matrix_dict(source)
    path = Path(str(source))
    if not path.exists():
        raise FileNotFoundError(f"Inferred network path not found: {path}")
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            obj = json.load(f)
    else:
        raise ValueError(f"Unsupported inferred-network file type: {path.suffix}")
    if isinstance(obj, dict) and "direction_matrix" in obj:
        return _dataframe_from_matrix_dict(obj["direction_matrix"])
    return _dataframe_from_matrix_dict(obj)


def load_ground_truth_network(source, genes=None):
    """Load a ground-truth adjacency matrix from a DataFrame/ndarray already
    in memory, or from a path to a .txt/.csv file. Accepts either a plain
    headerless numeric matrix (e.g. cycle_6_node.txt) or a matrix with a
    header row/index column."""
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, np.ndarray):
        labels = genes if genes is not None else list(range(source.shape[0]))
        return pd.DataFrame(source, index=labels, columns=labels)
    path = Path(str(source))
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth path not found: {path}")
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    df = df.dropna(axis=1, how="all")
    n = df.shape[0]
    labels = genes if (genes is not None and len(genes) == n) else list(range(n))
    df.index = labels
    df.columns = labels
    return df


def visualize_grn(inferred, ground_truth=None, figsize=None, **kwargs):
    """Plot the inferred network, optionally side by side with a
    ground-truth network for comparison.

    `inferred` / `ground_truth` may be DataFrames, dicts, ndarrays, or file
    paths (.pkl/.json for inferred, .txt/.csv for ground truth).
    """
    matrix = load_inferred_network(inferred)
    genes = list(matrix.index)
    labels = _short_labels(genes)
    fixed_pos = _circular_layout(genes)
    plot_kwargs = dict(show_colorbar=True, vmin=-1, vmax=1, font_size=11)
    plot_kwargs.update(kwargs)

    if ground_truth is not None:
        gt_matrix = load_ground_truth_network(ground_truth, genes=genes)
        gt_matrix = gt_matrix.reindex(index=genes, columns=genes, fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5.5))
        plot_grn_spread(matrix, node_labels=labels, title="Inferred network",
                         ax=axes[0], fixed_pos=fixed_pos, **plot_kwargs)
        plot_grn_spread(gt_matrix, node_labels=labels, title="Ground truth",
                         ax=axes[1], fixed_pos=fixed_pos, **plot_kwargs)
        for ax in axes:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (6, 5.5))
        plot_grn_spread(matrix, node_labels=labels, title="Inferred network",
                         ax=ax, fixed_pos=fixed_pos, **plot_kwargs)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

    plt.show()
    return fig
