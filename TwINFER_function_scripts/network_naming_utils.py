"""
Utilities for converting real-gene-named regulatory networks into the gene_1, gene_2, ...
numeric format TwINFER's Gillespie simulator requires, and back again after simulation.

Expected notation per network block in a "network_labels.txt"-style file:

    #NetworkName
    RealName - g<i> = <signed, comma-separated list of targets, e.g. "-g3, +g5">

Convention: "RealName - g<i> = -g3" means g<i> REPRESSES g3 -- i.e. the left-hand gene is
the regulator, the right-hand list is its signed targets. This matches TwINFER's
Gillespie simulator connectivity_matrix convention exactly: connectivity_matrix[i, j] is
the effect of gene i (row, regulator) on gene j (column, target).
"""
import re
import numpy as np


def parse_network_labels(path):
    """
    Parse a network_labels.txt-style file into per-network gene name legends and
    reconstructed signed connectivity matrices.

    Returns
    -------
    dict: {network_name: {"gene_names": [name_for_g1, name_for_g2, ...],
                           "matrix": np.ndarray (signed connectivity matrix, gene_1..gene_N order)}}
    """
    networks = {}
    current_name = None
    current_lines = []

    def flush():
        if current_name is None or not current_lines:
            return
        idx_to_name = {}
        parsed_edges = []  # (regulator_idx, [(sign, target_idx), ...])

        for line_no, line in current_lines:
            m = re.match(r'^\s*(.+?)\s*-\s*g(\d+)\s*=\s*(.*)$', line)
            if not m:
                print(f"[{current_name}] line {line_no}: could not parse, skipping: {line!r}")
                continue
            name, g_idx, rhs = m.group(1).strip(), int(m.group(2)), m.group(3)
            idx_to_name[g_idx] = name

            targets = []
            for item in rhs.split(','):
                item = item.strip()
                if not item:
                    continue
                m2 = re.match(r'^([+-])?\s*g(\d+)$', item)
                if not m2:
                    print(f"[{current_name}] line {line_no}: could not parse target {item!r}, skipping")
                    continue
                sign_str, t_idx = m2.group(1), int(m2.group(2))
                if sign_str is None:
                    print(f"[{current_name}] line {line_no}: {name} -> g{t_idx} has NO explicit "
                          f"sign (ambiguous) -- skipping this edge. Fix the source file if it "
                          f"should be included.")
                    continue
                sign = 1 if sign_str == '+' else -1
                targets.append((sign, t_idx))
            parsed_edges.append((g_idx, targets))

        n_genes = max(idx_to_name.keys())
        if sorted(idx_to_name.keys()) != list(range(1, n_genes + 1)):
            print(f"[{current_name}] WARNING: gene indices are not contiguous 1..{n_genes}: "
                  f"{sorted(idx_to_name.keys())}")

        gene_names = [idx_to_name.get(i, f"UNKNOWN_g{i}") for i in range(1, n_genes + 1)]
        matrix = np.zeros((n_genes, n_genes), dtype=int)
        for reg_idx, targets in parsed_edges:
            for sign, t_idx in targets:
                matrix[reg_idx - 1, t_idx - 1] = sign

        networks[current_name] = {"gene_names": gene_names, "matrix": matrix}

    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip('\n')
            if line.startswith('#'):
                flush()
                current_name = line[1:].strip()
                current_lines = []
            elif line.strip() == '':
                continue
            else:
                current_lines.append((line_no, line))
    flush()
    return networks


def gene_name_map(gene_names):
    """{'gene_1': gene_names[0], 'gene_2': gene_names[1], ...} for renaming simulator output."""
    return {f"gene_{i+1}": name for i, name in enumerate(gene_names)}


def rename_simulation_columns(df, gene_names, inplace=False):
    """
    Rename a Gillespie-simulated DataFrame's gene_{i}_mRNA / gene_{i}_protein columns to
    real gene names.

    gene_names may be either:
      - a list/tuple, where gene_names[i-1] is the real name for gene_i (positional), or
      - a dict shaped like gene_name_map()'s output, e.g. {"gene_1": "Pax6", "gene_2": "Coup", ...}

    Passing a dict to the list-shaped code path would silently produce wrong results (iterating
    a dict yields its keys, e.g. the literal string "gene_1", not the intended gene name) --
    this dispatches on type explicitly instead of assuming.
    """
    if isinstance(gene_names, dict):
        ordered_names = [gene_names[f"gene_{i}"] for i in range(1, len(gene_names) + 1)]
    else:
        ordered_names = list(gene_names)

    rename_map = {}
    for i, name in enumerate(ordered_names, start=1):
        for suffix in ("mRNA", "protein", "A", "I"):
            rename_map[f"gene_{i}_{suffix}"] = f"{name}_{suffix}"
    return df.rename(columns=rename_map, inplace=inplace)
