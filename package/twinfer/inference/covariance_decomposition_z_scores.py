"""Covariance decomposition z-scores of twin pairs: z_rho, z_rho_s and z_rho_Delta.

Used by every figure that scores a pair of genes with twin data (heterogeneity_vs_regulation, parameter_scan, ...).

Input    : for two genes X and Y, the expression in the two sisters of N twin pairs, as two arrays of shape (N, 2)
           (column 0 = sister 1, column 1 = sister 2). `sister_arrays` builds them from a table with the columns
           clone_id, replicate (1, 2), time_step and one column per gene.
Output   : `twin_stats(x, y)` returns a dictionary with
             z_rho        z-score of the expression correlation rho
             z_rho_s      z-score of the twin-shared correlation rho_s
             z_rho_Delta  z-score of the twin-difference correlation (rho-hat_Delta)
           and the quantities they are built from.

Definitions (ranks are taken over all 2N cells of a gene, ties get the average rank, ranks are centred)
  S            Spearman correlation of X and Y over all cells (both sisters)
  C            cross-sister part: correlation of X in sister 1 with Y in sister 2, and of X in sister 2 with Y in sister 1
  U            S - C, the part not shared between sisters
  h_x, h_y     sister correlation of each gene (how similar the two sisters are); negative values are set to 0 in the null variances
  sigma_rho    null standard deviation of S,  sqrt((0.5 + 0.5 h_x h_y) / N)
  z_rho        S / sigma_rho
  z_rho_s      sign(S) * C / sqrt((1 + S^2 + h_x h_y) / (2N))
  r_delta      correlation of the sister differences of X and Y;   z_rho_Delta = r_delta * sqrt(N)
  A_s, A_u     C / sigma_rho and U / sigma_rho (z_rho = A_s + A_u);   c = sqrt((1 - h_x)(1 - h_y)) / (sqrt(N) sigma_rho)  (A_u = c z_rho_Delta)
N is always the number of twin pairs. The formulas are analytic: no shuffling is needed.

`infer_with_twinfer(table, genes, time_points)` scores the gene pairs of a table. It sets up three scenarios (chosen automatically from the
table, or with `scenario=`):
  "single_time_point"    twin pairs (columns clone_id, replicate 1/2, time_step) at ONE time -> z_rho, z_rho_s, z_rho_Delta per gene pair
  "multiple_time_points" the same table with several time points: for every pair of time points (t_a < t_b) the clones are split at random
                         (fixed seed, `split`, default 1:1:2) into twin pairs at t_a, twin pairs at t_b and a cross group (one sister at
                         t_a, the other sister at t_b). At each of t_a and t_b the covariance z-scores use ALL cells measured at that
                         time, with the clone-size statistics below: the twin pairs of that group (clones of 2) and the cross-group
                         sister measured then (singletons) for z_rho, the twin pairs for z_rho_s and z_rho_Delta. The cross group gives
                         the directed cross-time correlations rho_dag_xy = corr(X at t_a, Y at t_b) and rho_dag_yx (Spearman) and
                         z_dag = rho_dag * sqrt(N_cross - 1). A larger z_dag_xy than z_dag_yx points to X -> Y.
  "clones"               real data: one row per cell with clone_id and the genes (no replicate column), clones of any size, singletons
                         allowed, one time point -> the TwinScore statistics below (all gene pairs, or regulators x targets).
(The cross-time direction for real data with several time points is not implemented.)
A gene without variance (at a time point, or in the cross group) gives NaN in every statistic of its pairs.
`save_to=<path>` saves the result table and the settings of the call (`save_result`).
Every pair also gets an inference scenario (column `inference_scenario`, see `classify_inference_scenario`):
  1  nothing                         z_rho not significant
  2  regulation                      z_rho significant, z_rho_s not significant
  3  regulation + heterogeneity      z_rho significant, z_rho_s significant and z_rho_Delta significant
  4  heterogeneity (shared state)    z_rho significant, z_rho_s significant, z_rho_Delta not significant
Significant means |z| above 1.96 (z_rho), 2.326 (z_rho_s) and 1.96 (z_rho_Delta). For many pairs at once (the "clones" scenario, real data) z_rho is
instead tested with Benjamini-Hochberg over all pairs scored (significant when q_rho <= 0.05), and z_rho_Delta with Benjamini-Hochberg over the
pairs whose z_rho is significant (q_rho_Delta <= 0.05), as Stages I and III of the TwinScore procedure (`correction`, `fdr`; z_rho_s keeps its threshold). The twin-pair scenarios score one pre-specified pair per simulated system and use no correction by default.

Clones of any size (real data): `clone_twin_stats(x, y, clone)` and `clone_twin_stats_matrix(X, Y, clone)` take one value per cell and a
clone label per cell (singletons allowed) instead of (N, 2) arrays and follow the TwinScore procedure of twinscore_pdf_candidates.py:
  ranks         clone-weighted mid-ranks over ALL cells (cell weight 1/n_c, so a singleton has weight 1)
  twin terms    S, C, U, h, rho_Delta from ALL within-clone cell pairs of clones with n_c >= 2, each clone's weight 1 split over its
                P_c = n_c (n_c - 1) / 2 pairs; ranks centred over the twin cells
  Stage I       rho = clone-weighted Spearman correlation over all cells (singletons included);
                sigma_rho = sqrt(sum_c [1/n_c + (1 - 1/n_c) h_x h_y]) / N_1                       (N_1 = number of clones, singletons too)
  shared        z_rho_s = sign(S) C / sigma_C,  sigma_C^2 = sum_{n_c>=2} [1 + S^2 + (n_c-2)(h_x+h_y) + (n_c^2-3n_c+3) h_x h_y] / (2 P_c) / N_t^2
  unshared      z_rho_Delta = rho_Delta / sigma_D,  sigma_D^2 = sum_{n_c>=2} 1/(n_c - 1) / N_t^2                                (N_t = clones with n_c >= 2)
  h_x, h_y are floored at 0. For clones that all have two cells and no singletons these are exactly the formulas of `twin_stats`.
  The values are used as given: normalise (e.g. CP10k) before calling.

Requires : numpy, pandas, scipy.
Example  : from twinfer.inference.covariance_decomposition_z_scores import infer_with_twinfer
           result = infer_with_twinfer(table, ["gene_1_mRNA", "gene_2_mRNA"], time_points=[20])        # one time point
           result = infer_with_twinfer(table, ["gene_1_mRNA", "gene_2_mRNA"], time_points=[1, 20])     # two time points, with cross-correlation
           result = infer_with_twinfer(cells, panel, regulators=tfs, targets=genes)                    # real data: one row per cell, clone_id
"""
# [2026-10-24 renamed: z_rho_delta, q_rho_delta, rho_delta and q_values_of_z_rho_delta to z_rho_Delta, q_rho_Delta, rho_Delta, q_values_of_z_rho_Delta]
import json
import os
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

STAT_COLUMNS = ["z_rho", "z_rho_s", "z_rho_Delta", "h_x", "h_y", "S", "C", "U", "sigma_rho", "A_s", "A_u", "c", "r_delta"]


def average_rank(values):
    """Ranks 1..n of a 1-d array; ties get the average rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - counts + (counts + 1) / 2)[inverse]


def twin_stats(x, y):
    """Statistics of two genes in N twin pairs. x, y: arrays of shape (N, 2). Returns a dictionary with the keys of STAT_COLUMNS."""
    n = len(x)
    rx = average_rank(x.ravel()).reshape(n, 2)
    ry = average_rank(y.ravel()).reshape(n, 2)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    mean_product = lambda a, b: ((a * b).sum(1) / 2).mean()          # mean over the two sisters, then over the twin pairs
    norm = np.sqrt(mean_product(rx, rx) * mean_product(ry, ry))
    S = ((rx[:, 0] * ry[:, 0] + rx[:, 1] * ry[:, 1]) / 2).mean() / norm
    C = ((rx[:, 0] * ry[:, 1] + rx[:, 1] * ry[:, 0]) / 2).mean() / norm
    h_x = (rx[:, 0] * rx[:, 1]).mean() / mean_product(rx, rx)
    h_y = (ry[:, 0] * ry[:, 1]).mean() / mean_product(ry, ry)
    hx_pos, hy_pos = max(0., h_x), max(0., h_y)
    sigma_rho = np.sqrt((0.5 + 0.5 * hx_pos * hy_pos) / n)
    sigma_delta = 1 / np.sqrt(n)
    dx, dy = rx[:, 0] - rx[:, 1], ry[:, 0] - ry[:, 1]
    r_delta = (dx * dy).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    return dict(
        z_rho=S / sigma_rho,
        z_rho_s=np.sign(S) * C / np.sqrt((1 + S * S + hx_pos * hy_pos) / (2 * n)),
        z_rho_Delta=r_delta / sigma_delta,
        h_x=h_x, h_y=h_y, S=S, C=C, U=S - C, sigma_rho=sigma_rho,
        A_s=C / sigma_rho, A_u=(S - C) / sigma_rho,
        c=np.sqrt(max((1 - h_x) * (1 - h_y), 0.)) * sigma_delta / sigma_rho,
        r_delta=r_delta)


def sister_arrays(table, genes, time_step):
    """Arrange a long table into per-gene sister arrays.

    table: DataFrame with clone_id, replicate (1 and 2), time_step and one column per gene.
    Returns (clone ids sorted, {gene: array of shape (N, 2)}) at `time_step`; raises ValueError unless every clone has both sisters.
    """
    rows = table[table["time_step"] == time_step].sort_values(["clone_id", "replicate"])
    n = rows["clone_id"].nunique()
    if len(rows) != 2 * n or rows["replicate"].nunique() != 2:
        raise ValueError(f"expected exactly two sisters (replicate 1 and 2) per clone at time step {time_step}")
    return rows["clone_id"].values[::2], {gene: rows[gene].values.reshape(n, 2) for gene in genes}


def twin_stats_or_nan(x, y):
    """Like twin_stats, but a gene without variance (no correlation defined) gives NaN for every statistic."""
    if x.std() == 0 or y.std() == 0:
        return {name: np.nan for name in STAT_COLUMNS}
    with np.errstate(all="ignore"):
        return twin_stats(x, y)


def _split_clones(clones, split, rng):
    """Random split of the clone ids into (twins at t_a, twins at t_b, cross group) in the proportions `split`."""
    order = rng.permutation(np.asarray(clones))
    fractions = np.asarray(split, dtype=float) / sum(split)
    n_a, n_b = int(len(order) * fractions[0]), int(len(order) * fractions[1])
    return order[:n_a], order[n_a:n_a + n_b], order[n_a + n_b:]


SCENARIOS = ("single_time_point", "multiple_time_points", "clones")
INFERENCE_SCENARIOS = {1: "nothing", 2: "regulation", 3: "regulation + heterogeneity", 4: "heterogeneity"}


def benjamini_hochberg(p):
    """Benjamini-Hochberg q-values of an array of p-values; NaN stays NaN and is not counted as a test."""
    p = np.atleast_1d(np.asarray(p, float))
    q = np.full(len(p), np.nan)
    defined = np.flatnonzero(~np.isnan(p))
    order = defined[np.argsort(p[defined])]
    ranked = len(order) * p[order] / np.arange(1, len(order) + 1)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q


def q_values_of_z_rho(z_rho):
    """BH q-values of z_rho: two-sided normal p-values ``2 * P(Z > |z_rho|)``, corrected over all pairs given (NaN excluded)."""
    return benjamini_hochberg(2 * norm.sf(np.abs(np.atleast_1d(np.asarray(z_rho, float)))))


def q_values_of_z_rho_Delta(z_rho_Delta, z_rho, fdr=0.05):
    """BH q-values of z_rho_Delta (two-sided normal p-values), corrected over the pairs whose z_rho passes BH (q_rho <= fdr);
    NaN for the other pairs (as Stage III of the TwinScore procedure: BH over the Stage I pairs)."""
    z_rho_Delta = np.atleast_1d(np.asarray(z_rho_Delta, float))
    passing = q_values_of_z_rho(z_rho) <= fdr
    q = np.full(len(z_rho_Delta), np.nan)
    q[passing] = benjamini_hochberg(2 * norm.sf(np.abs(z_rho_Delta[passing])))
    return q


def classify_inference_scenario(z_rho, z_rho_s, z_rho_Delta, thresholds=(1.96, 2.326, 1.96), correction="none", fdr=0.05):
    """Inference scenario (1-4) of each pair from its three z-scores (arrays or scalars); NaN in a needed z-score gives <NA>.

    Parameters
    ----------
    z_rho, z_rho_s, z_rho_Delta : float or array
        The three z-scores of each pair.
    thresholds : tuple of float
        (z_rho, z_rho_s, z_rho_Delta): ``|z|`` above the value counts as significant::

            z_rho not significant                                   -> 1 nothing
            z_rho significant, z_rho_s not significant              -> 2 regulation
            z_rho, z_rho_s and z_rho_Delta significant              -> 3 regulation + heterogeneity
            z_rho and z_rho_s significant, z_rho_Delta not          -> 4 heterogeneity

    correction : {"none", "bh"}
        - "none": every threshold as above (one pre-specified pair).
        - "bh": z_rho is significant when its Benjamini-Hochberg q-value over ALL pairs given is <= fdr; z_rho_Delta is
          significant when its BH q-value over the pairs whose z_rho is significant is <= fdr (thresholds[0] and
          thresholds[2] are not used); z_rho_s keeps its threshold. Use for many pairs at once (real data).
    fdr : float
        q-value cut-off of the "bh" correction.
    """
    z_rho, z_rho_s, z_rho_Delta = (np.atleast_1d(np.asarray(z, float)) for z in (z_rho, z_rho_s, z_rho_Delta))
    # [2026-10-23 replaced: z_rho can be tested with BH]
    # sig_rho, sig_s, sig_delta = np.abs(z_rho) > thresholds[0], np.abs(z_rho_s) > thresholds[1], np.abs(z_rho_delta) > thresholds[2]
    if correction not in ("none", "bh"):
        raise ValueError('correction must be "none" or "bh"')
    # [2026-10-23 replaced: z_rho_delta also tested with BH]
    # sig_rho = (q_values_of_z_rho(z_rho) <= fdr) if correction == "bh" else (np.abs(z_rho) > thresholds[0])
    # sig_s, sig_delta = np.abs(z_rho_s) > thresholds[1], np.abs(z_rho_delta) > thresholds[2]
    sig_s = np.abs(z_rho_s) > thresholds[1]
    if correction == "bh":
        sig_rho = q_values_of_z_rho(z_rho) <= fdr
        sig_delta = q_values_of_z_rho_Delta(z_rho_Delta, z_rho, fdr) <= fdr          # NaN (z_rho not significant) -> False, not used
    else:
        sig_rho, sig_delta = np.abs(z_rho) > thresholds[0], np.abs(z_rho_Delta) > thresholds[2]
    scenario = np.where(~sig_rho, 1, np.where(~sig_s, 2, np.where(sig_delta, 3, 4)))
    undefined = np.isnan(z_rho) | (sig_rho & np.isnan(z_rho_s)) | (sig_rho & sig_s & np.isnan(z_rho_Delta))
    result = pd.array(scenario, dtype="Int64")
    result[undefined] = pd.NA
    return result


# [2026-10-23 replaced: correction (BH on z_rho) added]
# def infer_with_twinfer(table, genes, time_points=None, scenario="auto", seed=101010, split=(1, 1, 2),
#                        regulators=None, targets=None, chunk_x=100, chunk_y=300, z_thresholds=(1.96, 2.326, 1.96), n_permutations=1000):
# [2026-10-23 replaced: cross_correlation_pairs added]
#                        correction="auto", fdr=0.05):
# [2026-10-24 replaced: save_to added]
#                        correction="auto", fdr=0.05, cross_correlation_pairs="all"):
def save_result(result, path, settings=None):
    """Write a result table to `path` (format from the extension: .csv, .csv.gz, .tsv, .tsv.gz or .parquet; folders are created).
    With `settings` (a dict), also write them to <path without extension>_settings.json. Returns the paths written."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    name = os.path.basename(path)
    if name.endswith(".parquet"):
        result.to_parquet(path, index=False)
    elif name.endswith((".tsv", ".tsv.gz")):
        result.to_csv(path, sep="\t", index=False)
    elif name.endswith((".csv", ".csv.gz")):
        result.to_csv(path, index=False)
    else:
        raise ValueError("save_to must end in .csv, .csv.gz, .tsv, .tsv.gz or .parquet")
    written = [path]
    if settings is not None:
        stem = name
        for extension in (".csv.gz", ".tsv.gz", ".parquet", ".csv", ".tsv"):
            if stem.endswith(extension):
                stem = stem[: -len(extension)]
                break
        settings_path = os.path.join(folder, stem + "_settings.json")
        with open(settings_path, "w") as handle:
            json.dump(settings, handle, indent=2, default=str)
        written.append(settings_path)
    return written


def infer_with_twinfer(table, genes, time_points=None, scenario="auto", seed=101010, split=(1, 1, 2),
                       regulators=None, targets=None, chunk_x=100, chunk_y=300, z_thresholds=(1.96, 2.326, 1.96), n_permutations=1000,
                       correction="auto", fdr=0.05, cross_correlation_pairs="all", save_to=None):
    """Score gene pairs with the covariance decomposition (and the cross-time correlation when there are several time points).

    Parameters
    ----------
    table : pandas.DataFrame
        Twin-pair scenarios: clone_id, replicate (1 and 2 = the two sisters), time_step and one column per gene.
        "clones" scenario (real data): one row per cell with clone_id, the genes (values used as given, e.g. CP10k) and
        optionally time_step.
    genes : list of str
        Gene columns. Pairs (X, Y) with X before Y in the list are scored; in the "clones" scenario regulators x targets
        are scored instead when given.
    time_points : list, optional
        Time steps to use. Twin-pair scenarios: one value -> "single_time_point", several -> "multiple_time_points".
        "clones": optional, selects the single time step of the cells.
    scenario : str
        "auto" (default: "clones" when the table has no replicate column, otherwise by the number of time points) or one of SCENARIOS.
    seed, split : int, tuple
        Random split of the clones for "multiple_time_points": seed and proportions of (twins at t_a, twins at t_b, cross group).
    regulators, targets, chunk_x, chunk_y : list of str, list of str, int, int
        "clones" only: gene lists of the pairs to score and the block sizes of the computation.
    z_thresholds : tuple of float
        (z_rho, z_rho_s, z_rho_Delta) significance thresholds of `classify_inference_scenario` (adds the column inference_scenario).
    n_permutations : int
        "multiple_time_points": permutations of the cross group for the p-value of delta (``p = (1 + #{|null| >= |delta|}) / (n + 1)``).
    correction, fdr : {"auto", "bh", "none"}, float
        Multiple testing of z_rho in the inference scenario: "bh" = Benjamini-Hochberg over all pairs of the result, z_rho
        significant when q_rho <= fdr, and z_rho_Delta BH over the pairs with significant z_rho, significant when q_rho_Delta <= fdr
        (adds the columns q_rho, q_rho_Delta; with several time points with the suffix _ta / _tb); "none" = the ``|z|`` thresholds.
        "auto" (default): "bh" for the "clones" scenario (real data, many pairs), "none" for the twin-pair scenarios (one pre-specified
        pair per simulated system). z_rho_s always uses its threshold.
    cross_correlation_pairs : {"all", "significant_rho", "regulation"}
        "multiple_time_points": which gene pairs get the cross-time correlation (z_dag_xy, z_dag_yx, delta, p_delta):
        "all" (default) every pair; "significant_rho" pairs whose z_rho is significant at t_b; "regulation" pairs in inference scenario
        2 or 3 at t_b. The other pairs get NaN in these columns (and skip the permutations).
    save_to : str, optional
        Path to save the result table (.csv, .csv.gz, .tsv, .tsv.gz or .parquet). The settings of the call (genes, time points,
        scenario, split, seed, thresholds, correction, permutations, cross-correlation pairs, date) are written next to it as
        <name>_settings.json. See `save_result`.

    Returns
    -------
    pandas.DataFrame
        One row per gene pair (and time pair). Columns: gene_x, gene_y, t_a, [t_b,] the N of each group and the
        statistics. One time point: STAT_COLUMNS. Several time points: for each of t_a and t_b (suffix _ta / _tb) the clone-size statistics
        (rho, S, C, U, rho_Delta, h_x, h_y, sigma_rho, sigma_C, sigma_D, z_rho, z_rho_s, z_rho_Delta) of ALL cells measured at that time:
        z_rho uses the twin pairs of that group (N_a / N_b, each a clone of 2) and the cross-group sister measured then (N_cross singletons),
        N_clones_ta / N_clones_tb in total; z_rho_s and z_rho_Delta use the twin pairs. Then rho_dag_xy, rho_dag_yx, z_dag_xy, z_dag_yx,
        delta (= rho_dag_xy - rho_dag_yx) and p_delta (its permutation p-value). Clones: rho, S, C, U, rho_Delta, h_x, h_y, sigma_rho,
        sigma_C, sigma_D, z_rho, z_rho_s, z_rho_Delta. Every scenario adds inference_scenario (q_rho, q_rho_Delta with correction="bh").
        A gene without variance gives NaN in the statistics of its pairs (and in the cross-time columns, including p_delta).
    """
    if save_to is not None:
        if not str(save_to).endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet")):
            raise ValueError("save_to must end in .csv, .csv.gz, .tsv, .tsv.gz or .parquet")
        result = infer_with_twinfer(table, genes, time_points, scenario, seed, split, regulators, targets, chunk_x, chunk_y, z_thresholds,
                                    n_permutations, correction, fdr, cross_correlation_pairs)
        settings = dict(genes=list(genes), time_points=None if time_points is None else list(time_points), scenario=scenario, seed=seed,
                        split=list(split), regulators=None if regulators is None else list(regulators),
                        targets=None if targets is None else list(targets), z_thresholds=list(z_thresholds), n_permutations=n_permutations,
                        correction=correction, fdr=fdr, cross_correlation_pairs=cross_correlation_pairs, n_rows=len(result),
                        saved=datetime.now().isoformat(timespec="seconds"))
        save_result(result, save_to, settings)
        return result
    if scenario == "auto":
        scenario = "clones" if "replicate" not in table.columns else ("single_time_point" if len(time_points) == 1 else "multiple_time_points")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if correction == "auto":
        correction = "bh" if scenario == "clones" else "none"
    if correction not in ("none", "bh"):
        raise ValueError('correction must be "auto", "none" or "bh"')
    if cross_correlation_pairs not in ("all", "significant_rho", "regulation"):
        raise ValueError('cross_correlation_pairs must be "all", "significant_rho" or "regulation"')
    if not table.columns.is_unique:
        raise ValueError("the table has duplicate column names: " + ", ".join(sorted(set(table.columns[table.columns.duplicated()]))))
    for label, names in (("genes", genes), ("regulators", regulators), ("targets", targets)):
        if names is not None and len(set(names)) != len(list(names)):
            raise ValueError(f"duplicate names in {label}: " + ", ".join(sorted({n for n in names if list(names).count(n) > 1})))
    if scenario == "clones":
        # [2026-10-23 replaced: correction (BH on z_rho) added]
        # return _infer_clones(table, genes, time_points, regulators, targets, chunk_x, chunk_y, z_thresholds)
        return _infer_clones(table, genes, time_points, regulators, targets, chunk_x, chunk_y, z_thresholds, correction, fdr)
    if time_points is None:
        raise ValueError("time_points is required for twin-pair tables")
    time_points = sorted(time_points)
    rows = []
    if len(time_points) == 1:
        clones, data = sister_arrays(table, genes, time_points[0])
        for gx, gy in combinations(genes, 2):
            rows.append(dict(gene_x=gx, gene_y=gy, t_a=time_points[0], N=len(clones), **twin_stats_or_nan(data[gx], data[gy])))
        frame = pd.DataFrame(rows)
        # [2026-10-23 replaced: correction (BH on z_rho) added]
        # frame["inference_scenario"] = classify_inference_scenario(frame.z_rho, frame.z_rho_s, frame.z_rho_delta, z_thresholds)
        # return frame
        if correction == "bh":
            frame["q_rho"] = q_values_of_z_rho(frame.z_rho)
            frame["q_rho_Delta"] = q_values_of_z_rho_Delta(frame.z_rho_Delta, frame.z_rho, fdr)
        frame["inference_scenario"] = classify_inference_scenario(frame.z_rho, frame.z_rho_s, frame.z_rho_Delta, z_thresholds, correction, fdr)
        return frame

    # [2026-10-23 replaced: cross-time correlations can be limited to chosen pairs (cross_correlation_pairs); two passes]
    # for t_a, t_b in combinations(time_points, 2):
    #     rows_a = table[table["time_step"] == t_a].set_index(["clone_id", "replicate"])
    #     rows_b = table[table["time_step"] == t_b].set_index(["clone_id", "replicate"])
    #     both = sorted(set(rows_a.index.get_level_values(0)) & set(rows_b.index.get_level_values(0)))
    #     rng = np.random.RandomState(seed)
    #     twins_a, twins_b, cross = _split_clones(both, split, rng)
    #     sister_a = rng.randint(1, 3, len(cross))                    # which sister is measured at t_a; the other one at t_b
    #     perm_rng = np.random.RandomState(seed + 1)
    #     for gx, gy in combinations(genes, 2):
    #         _, data_a = sister_arrays(table[table["clone_id"].isin(twins_a)], [gx, gy], t_a)
    #         _, data_b = sister_arrays(table[table["clone_id"].isin(twins_b)], [gx, gy], t_b)
    #         stats_a, stats_b = twin_stats_or_nan(data_a[gx], data_a[gy]), twin_stats_or_nan(data_b[gx], data_b[gy])
    #         at_a = lambda gene: rows_a[gene].reindex(list(zip(cross, sister_a))).values
    #         at_b = lambda gene: rows_b[gene].reindex(list(zip(cross, 3 - sister_a))).values
    #         ranks = lambda v: (lambda r: r - r.mean())(average_rank(v))
    #         xa, ya, xb, yb = ranks(at_a(gx)), ranks(at_a(gy)), ranks(at_b(gx)), ranks(at_b(gy))      # centred ranks, cross group
    #         norm_xy, norm_yx = np.sqrt((xa ** 2).sum() * (yb ** 2).sum()), np.sqrt((ya ** 2).sum() * (xb ** 2).sum())
    #         rho_xy, rho_yx = (xa * yb).sum() / norm_xy, (ya * xb).sum() / norm_yx
    #         delta = rho_xy - rho_yx
    #         null = np.empty(n_permutations)                          # permute which t_b measurement pairs with which t_a measurement
    #         for k in range(n_permutations):
    #             perm = perm_rng.permutation(len(cross))
    #             null[k] = (xa * yb[perm]).sum() / norm_xy - (ya * xb[perm]).sum() / norm_yx
    #         p_delta = (1 + np.sum(np.abs(null) >= abs(delta))) / (n_permutations + 1)
    #         row = dict(gene_x=gx, gene_y=gy, t_a=t_a, t_b=t_b, N_a=len(twins_a), N_b=len(twins_b), N_cross=len(cross))
    #         row.update({f"{name}_ta": value for name, value in stats_a.items()})
    #         row.update({f"{name}_tb": value for name, value in stats_b.items()})
    #         row.update(rho_dag_xy=rho_xy, rho_dag_yx=rho_yx, z_dag_xy=rho_xy * np.sqrt(len(cross) - 1), z_dag_yx=rho_yx * np.sqrt(len(cross) - 1),
    #                    delta=delta, p_delta=p_delta)
    #         rows.append(row)
    # frame = pd.DataFrame(rows)
    # for suffix in ("ta", "tb"):
    #     # [2026-10-23 replaced: correction (BH on z_rho) added]
    #     # frame[f"inference_scenario_{suffix}"] = classify_inference_scenario(frame[f"z_rho_{suffix}"], frame[f"z_rho_s_{suffix}"], frame[f"z_rho_delta_{suffix}"], z_thresholds)
    #     if correction == "bh":
    #         frame[f"q_rho_{suffix}"] = q_values_of_z_rho(frame[f"z_rho_{suffix}"])
    #         frame[f"q_rho_delta_{suffix}"] = q_values_of_z_rho_delta(frame[f"z_rho_delta_{suffix}"], frame[f"z_rho_{suffix}"], fdr)
    #     frame[f"inference_scenario_{suffix}"] = classify_inference_scenario(frame[f"z_rho_{suffix}"], frame[f"z_rho_s_{suffix}"], frame[f"z_rho_delta_{suffix}"],
    #                                                                         z_thresholds, correction, fdr)
    # return frame
    # [2026-10-24 replaced: the statistics at t_a and t_b now use ALL cells measured at that time (the twin pairs of that group plus the
    #  cross-group sister measured then, as singletons), with the clone-size statistics: z_rho over all of them, z_rho_s and z_rho_Delta
    #  from the twin pairs. Previously only the twin pairs of each group were used (twin_stats).]
    # contexts, rows = {}, []
    # for t_a, t_b in combinations(time_points, 2):                   # pass 1: twin statistics of every pair at t_a and t_b
    #     rows_a = table[table["time_step"] == t_a].set_index(["clone_id", "replicate"])
    #     rows_b = table[table["time_step"] == t_b].set_index(["clone_id", "replicate"])
    #     both = sorted(set(rows_a.index.get_level_values(0)) & set(rows_b.index.get_level_values(0)))
    #     rng = np.random.RandomState(seed)
    #     twins_a, twins_b, cross = _split_clones(both, split, rng)
    #     sister_a = rng.randint(1, 3, len(cross))                    # which sister is measured at t_a; the other one at t_b
    #     contexts[(t_a, t_b)] = (rows_a, rows_b, cross, sister_a)
    #     for gx, gy in combinations(genes, 2):
    #         _, data_a = sister_arrays(table[table["clone_id"].isin(twins_a)], [gx, gy], t_a)
    #         _, data_b = sister_arrays(table[table["clone_id"].isin(twins_b)], [gx, gy], t_b)
    #         stats_a, stats_b = twin_stats_or_nan(data_a[gx], data_a[gy]), twin_stats_or_nan(data_b[gx], data_b[gy])
    #         row = dict(gene_x=gx, gene_y=gy, t_a=t_a, t_b=t_b, N_a=len(twins_a), N_b=len(twins_b), N_cross=len(cross))
    #         row.update({f"{name}_ta": value for name, value in stats_a.items()})
    #         row.update({f"{name}_tb": value for name, value in stats_b.items()})
    #         rows.append(row)
    # frame = pd.DataFrame(rows)
    contexts, rows = {}, []
    pair_index = list(combinations(range(len(genes)), 2))
    for t_a, t_b in combinations(time_points, 2):                   # pass 1: statistics of every pair at t_a and t_b
        rows_a = table[table["time_step"] == t_a].set_index(["clone_id", "replicate"])
        rows_b = table[table["time_step"] == t_b].set_index(["clone_id", "replicate"])
        both = sorted(set(rows_a.index.get_level_values(0)) & set(rows_b.index.get_level_values(0)))
        rng = np.random.RandomState(seed)
        twins_a, twins_b, cross = _split_clones(both, split, rng)
        sister_a = rng.randint(1, 3, len(cross))                    # which sister is measured at t_a; the other one at t_b
        contexts[(t_a, t_b)] = (rows_a, rows_b, cross, sister_a)
        stats = {}
        for suffix, rows_t, twins, cross_sister in (("ta", rows_a, twins_a, sister_a), ("tb", rows_b, twins_b, 3 - sister_a)):
            # all cells measured at this time: both sisters of its twin pairs (clones of 2) + one sister of each cross pair (singletons)
            measured = list(zip(np.repeat(twins, 2), np.tile([1, 2], len(twins)))) + list(zip(cross, cross_sister))
            cells = rows_t.reindex(measured)
            values = cells[list(genes)].to_numpy(float)
            with np.errstate(all="ignore"):
                stats[suffix] = clone_twin_stats_matrix(values, values, cells.index.get_level_values(0).to_numpy())
            constant = values.std(axis=0) == 0                       # a gene without variance: no correlation defined
            for key in stats[suffix]:
                stats[suffix][key] = np.where(constant[:, None] | constant[None, :], np.nan, stats[suffix][key])
        for i, j in pair_index:
            row = dict(gene_x=genes[i], gene_y=genes[j], t_a=t_a, t_b=t_b, N_a=len(twins_a), N_b=len(twins_b), N_cross=len(cross),
                       N_clones_ta=len(twins_a) + len(cross), N_clones_tb=len(twins_b) + len(cross))
            for suffix in ("ta", "tb"):
                row.update({f"{name}_{suffix}": float(matrix[i, j]) for name, matrix in stats[suffix].items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    stat_columns = list(frame.columns)
    for suffix in ("ta", "tb"):
        if correction == "bh":
            frame[f"q_rho_{suffix}"] = q_values_of_z_rho(frame[f"z_rho_{suffix}"])
            frame[f"q_rho_Delta_{suffix}"] = q_values_of_z_rho_Delta(frame[f"z_rho_Delta_{suffix}"], frame[f"z_rho_{suffix}"], fdr)
        frame[f"inference_scenario_{suffix}"] = classify_inference_scenario(frame[f"z_rho_{suffix}"], frame[f"z_rho_s_{suffix}"], frame[f"z_rho_Delta_{suffix}"],
                                                                            z_thresholds, correction, fdr)
    if cross_correlation_pairs == "all":
        chosen = np.ones(len(frame), dtype=bool)
    elif cross_correlation_pairs == "significant_rho":
        chosen = (frame["q_rho_tb"] <= fdr).to_numpy() if correction == "bh" else (np.abs(frame["z_rho_tb"]) > z_thresholds[0]).to_numpy()
    else:                                                            # "regulation": scenario 2 or 3 at t_b
        chosen = frame["inference_scenario_tb"].isin([2, 3]).fillna(False).to_numpy(dtype=bool)
    cross_columns = ["rho_dag_xy", "rho_dag_yx", "z_dag_xy", "z_dag_yx", "delta", "p_delta"]
    for column in cross_columns:
        frame[column] = np.nan
    ranks = lambda v: (lambda r: r - r.mean())(average_rank(v))
    for (t_a, t_b), (rows_a, rows_b, cross, sister_a) in contexts.items():   # pass 2: cross-time correlation of the chosen pairs
        perm_rng = np.random.RandomState(seed + 1)
        at_a = lambda gene: rows_a[gene].reindex(list(zip(cross, sister_a))).values
        at_b = lambda gene: rows_b[gene].reindex(list(zip(cross, 3 - sister_a))).values
        for i in np.flatnonzero(chosen & (frame["t_a"] == t_a).to_numpy() & (frame["t_b"] == t_b).to_numpy()):
            gx, gy = frame.at[i, "gene_x"], frame.at[i, "gene_y"]
            xa, ya, xb, yb = ranks(at_a(gx)), ranks(at_a(gy)), ranks(at_b(gx)), ranks(at_b(gy))      # centred ranks, cross group
            norm_xy, norm_yx = np.sqrt((xa ** 2).sum() * (yb ** 2).sum()), np.sqrt((ya ** 2).sum() * (xb ** 2).sum())
            # [2026-10-24 added: a gene without variance in the cross group has no cross-time correlation; before, its p_delta came out as
            #  1 / (n_permutations + 1) because every comparison with NaN is False. The permutations are still drawn so that the random
            #  stream, and with it the p_delta of every other pair, is unchanged.]
            if norm_xy == 0 or norm_yx == 0:
                for k in range(n_permutations):
                    perm_rng.permutation(len(cross))
                continue
            rho_xy, rho_yx = (xa * yb).sum() / norm_xy, (ya * xb).sum() / norm_yx
            delta = rho_xy - rho_yx
            null = np.empty(n_permutations)                          # permute which t_b measurement pairs with which t_a measurement
            for k in range(n_permutations):
                perm = perm_rng.permutation(len(cross))
                null[k] = (xa * yb[perm]).sum() / norm_xy - (ya * xb[perm]).sum() / norm_yx
            p_delta = (1 + np.sum(np.abs(null) >= abs(delta))) / (n_permutations + 1)
            frame.loc[i, cross_columns] = [rho_xy, rho_yx, rho_xy * np.sqrt(len(cross) - 1), rho_yx * np.sqrt(len(cross) - 1), delta, p_delta]
    return frame[stat_columns + cross_columns + [c for c in frame.columns if c not in stat_columns + cross_columns]]


# ---------------------------------------------------------------------------- clones of any size, singletons included
def weighted_ranks(values, weights):
    """Weighted mid-ranks of a 1-d array: for each value the weight of all smaller values plus half the weight of its ties."""
    _, inverse = np.unique(values, return_inverse=True)
    tie_weight = np.bincount(inverse, weights=weights)
    return (np.cumsum(tie_weight) - tie_weight + 0.5 * tie_weight)[inverse]


class CloneStructure:
    """Clone sizes, weights, within-clone cell pairs and variance constants for one set of cells.

    clone: one clone label per cell (any hashable; every cell counts, singletons included).
    Attributes: w (cell weights 1/n_c), N1 (clones), Nt (clones with n_c >= 2), ia/ib/wp (the within-clone cell pairs and their weights),
    A_rho, B_rho, a0, a1, a2, sigma_D (the constants of the variances, see the module header).
    """

    def __init__(self, clone):
        clone = np.asarray(clone)
        labels, inverse = np.unique(clone, return_inverse=True)
        size = np.bincount(inverse).astype(float)
        self.n_cells = len(clone)
        self.w = 1.0 / size[inverse]
        self.N1 = len(labels)
        twin = size >= 2
        self.Nt = int(twin.sum())
        if self.Nt == 0:
            raise ValueError("no clone has two or more cells")
        n_twin = size[twin]
        pairs_per_clone = n_twin * (n_twin - 1) / 2
        order = np.argsort(inverse, kind="stable")
        starts = np.concatenate([[0], np.cumsum(size.astype(int))[:-1]])
        ia, ib, wp = [], [], []
        for start, n in zip(starts[twin], size[twin].astype(int)):
            i, j = np.triu_indices(n, 1)
            ia.append(order[start + i]); ib.append(order[start + j]); wp.append(np.full(len(i), 1.0 / (n * (n - 1) / 2)))
        self.ia, self.ib, self.wp = np.concatenate(ia), np.concatenate(ib), np.concatenate(wp)
        self.A_rho, self.B_rho = float((1.0 / size).sum()), float((1.0 - 1.0 / size).sum())
        self.a0 = float((1 / (2 * pairs_per_clone)).sum())
        self.a1 = float(((n_twin - 2) / (2 * pairs_per_clone)).sum())
        self.a2 = float(((n_twin ** 2 - 3 * n_twin + 3) / (2 * pairs_per_clone)).sum())
        self.sigma_D = np.sqrt((1.0 / (n_twin - 1)).sum()) / self.Nt


def _gene_block(structure, values):
    """Ranks and per-gene terms (cells x genes input): centred twin ranks, M, h (floored at 0), centred cell ranks and their weighted variance."""
    s = structure
    R = np.stack([weighted_ranks(values[:, j], s.w) for j in range(values.shape[1])], 1)
    mu = (s.wp[:, None] * 0.5 * (R[s.ia] + R[s.ib])).sum(0) / s.Nt
    ra, rb = R[s.ia] - mu, R[s.ib] - mu
    M = (s.wp[:, None] * 0.5 * (ra ** 2 + rb ** 2)).sum(0)
    h = np.maximum(0.0, (s.wp[:, None] * ra * rb).sum(0) / M)
    Rc = R - (s.w[:, None] * R).sum(0) / s.w.sum()
    return dict(ra=ra, rb=rb, M=M, h=h, Rc=Rc, v=(s.w[:, None] * Rc ** 2).sum(0), d=ra - rb, dd=(s.wp[:, None] * (ra - rb) ** 2).sum(0))


def _pair_matrices(s, bx, by):
    """Statistics of every pair between two gene blocks (see _gene_block); dict of (nx, ny) arrays."""
    wp, w = s.wp[:, None], s.w[:, None]
    rho = ((bx["Rc"] * w).T @ by["Rc"]) / np.sqrt(np.outer(bx["v"], by["v"]))
    norm = np.sqrt(np.outer(bx["M"], by["M"]))
    S = 0.5 * ((bx["ra"] * wp).T @ by["ra"] + (bx["rb"] * wp).T @ by["rb"]) / norm
    C = 0.5 * ((bx["ra"] * wp).T @ by["rb"] + (bx["rb"] * wp).T @ by["ra"]) / norm
    rho_Delta = ((bx["d"] * wp).T @ by["d"]) / np.sqrt(np.outer(bx["dd"], by["dd"]))
    hx, hy = bx["h"][:, None], by["h"][None, :]
    sigma_rho = np.sqrt(s.A_rho + s.B_rho * hx * hy) / s.N1
    sigma_C = np.sqrt((1 + S ** 2) * s.a0 + (hx + hy) * s.a1 + hx * hy * s.a2) / s.Nt
    return dict(rho=rho, S=S, C=C, U=S - C, rho_Delta=rho_Delta, h_x=np.broadcast_to(hx, S.shape), h_y=np.broadcast_to(hy, S.shape),
                sigma_rho=sigma_rho, sigma_C=sigma_C, sigma_D=np.full(S.shape, s.sigma_D),
                z_rho=rho / sigma_rho, z_rho_s=np.sign(S) * C / sigma_C, z_rho_Delta=rho_Delta / s.sigma_D)


def clone_twin_stats_matrix(X, Y, clone, chunk_x=None, chunk_y=None, keys=None):
    """Statistics of every pair (column of X, column of Y) for cells with clone labels.

    Parameters
    ----------
    X, Y, clone : array, array, array
        X: cells x nx array, Y: cells x ny array (values used as given, e.g. CP10k), clone: one label per cell.
    chunk_x, chunk_y : int, optional
        Process the pairs in blocks of that many regulator / target columns (needed when nx * ny is large: the cost in
        memory is about (number of within-clone cell pairs) x chunk columns); None = everything at once.
    keys : list of str, optional
        Which statistics to return (default all).

    Returns
    -------
    dict
        (nx, ny) arrays: rho, S, C, U, rho_Delta, h_x, h_y, sigma_rho, sigma_C, sigma_D, z_rho, z_rho_s, z_rho_Delta.
    """
    s = CloneStructure(clone)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    step_x, step_y = chunk_x or X.shape[1], chunk_y or Y.shape[1]
    out = {}
    for a in range(0, X.shape[1], step_x):
        bx = _gene_block(s, X[:, a:a + step_x])
        for b in range(0, Y.shape[1], step_y):
            res = _pair_matrices(s, bx, _gene_block(s, Y[:, b:b + step_y]))
            for key in (keys or res):
                out.setdefault(key, np.empty((X.shape[1], Y.shape[1])))[a:a + step_x, b:b + step_y] = res[key]
    return out


def clone_twin_stats(x, y, clone):
    """Statistics of one gene pair: x, y one value per cell, clone one label per cell. Returns a dict of floats (see clone_twin_stats_matrix)."""
    out = clone_twin_stats_matrix(np.asarray(x, float)[:, None], np.asarray(y, float)[:, None], clone)
    return {key: float(np.asarray(value).ravel()[0]) for key, value in out.items()}


# [2026-10-23 replaced: correction (BH on z_rho) added]
# def _infer_clones(table, genes, time_points, regulators, targets, chunk_x, chunk_y, z_thresholds):
def _infer_clones(table, genes, time_points, regulators, targets, chunk_x, chunk_y, z_thresholds, correction="bh", fdr=0.05):
    """Real-data scenario: clone-size aware statistics for all gene pairs (or regulators x targets) at one time point."""
    if time_points is not None and len(time_points) > 1:
        raise NotImplementedError("several time points are not implemented for the clones scenario (cross-time direction needs the real-data spec)")
    cells = table
    if time_points is not None:
        cells = table[table["time_step"] == time_points[0]]
    clone = cells["clone_id"].to_numpy()
    if regulators is None:
        regulators, targets = list(genes), list(genes)
        pairs_wanted = "upper"
    else:
        targets = list(targets if targets is not None else genes)
        pairs_wanted = "all"
    result = clone_twin_stats_matrix(cells[list(regulators)].to_numpy(float), cells[list(targets)].to_numpy(float), clone, chunk_x=chunk_x, chunk_y=chunk_y)
    n_clones, n_twin = CloneStructure(clone).N1, CloneStructure(clone).Nt
    rows_i, cols_j = np.triu_indices(len(regulators), 1) if pairs_wanted == "upper" else np.indices((len(regulators), len(targets))).reshape(2, -1)
    keep = np.asarray(regulators)[rows_i] != np.asarray(targets)[cols_j]            # a regulator that is also a target is not paired with itself
    rows_i, cols_j = rows_i[keep], cols_j[keep]
    frame = pd.DataFrame({"gene_x": np.asarray(regulators)[rows_i], "gene_y": np.asarray(targets)[cols_j]})
    frame["t_a"] = time_points[0] if time_points is not None else np.nan
    frame["N_clones"], frame["N_twin_clones"] = n_clones, n_twin
    for key, matrix in result.items():
        frame[key] = matrix[rows_i, cols_j]
    # [2026-10-23 replaced: correction (BH on z_rho) added]
    # frame["inference_scenario"] = classify_inference_scenario(frame.z_rho, frame.z_rho_s, frame.z_rho_delta, z_thresholds)
    if correction == "bh":
        frame["q_rho"] = q_values_of_z_rho(frame.z_rho)
        frame["q_rho_Delta"] = q_values_of_z_rho_Delta(frame.z_rho_Delta, frame.z_rho, fdr)
    frame["inference_scenario"] = classify_inference_scenario(frame.z_rho, frame.z_rho_s, frame.z_rho_Delta, z_thresholds, correction, fdr)
    return frame
