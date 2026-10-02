# [2026-09-30 removed a stray "so" that preceded the docstring (the original file did not parse: SyntaxError)]


"""Adds ratio-to-random columns (auprc_x, f1_topk_x, precision_topk_x = value / random_baseline)
to summary_metrics_table_larry.csv, per (twin_def, gene_set, variant). Random baseline:
  directed_unsigned: n_true_directed / n_pairs_directed
  signed_directed:    n_true_signed  / (2 * n_pairs_directed)   (2 signed positions per pair)
  undirected:         n_true_undirected / n_pairs_undirected
The 'ALL' row's baseline is the size-weighted mean of its 9 gene sets' baselines (matching how
the underlying auprc/f1_topk 'ALL' rows are themselves a mean across gene sets).
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

from paper_analysis.larry_hematopoiesis_validation import summary_table_todo4v2_larry as L
from paper_analysis.larry_hematopoiesis_validation import apply_todo4v2_allpairs_with_competitors as A

from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: project root for the clean_data/ repointing, see REPOINT_LOG.tsv]
HERE = L.HERE


def random_baselines(twin_def, gs, CE, true_signed):
    cfg = L.TWIN_DEFS[twin_def]
    dpath = f"{L.R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
    if not os.path.exists(dpath):
        return None
    dd = pd.read_csv(dpath)
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    genes = sorted(set(dd.gene_1) | set(dd.gene_2))
    U_full = [(a, b) for a in genes for b in genes if a != b]
    true_directed = CE & set(U_full)
    true_undirected = {frozenset(p) for p in true_directed}
    U_full_und = [frozenset(p) for p in combinations(genes, 2)]

    n_true_signed = sum(1 for (a, b) in U_full for s in (1, -1) if (a, b, s) in true_signed)

    return dict(
        gene_set=gs, twin_def=twin_def,
        n_pairs_directed=len(U_full), n_true_directed=len(true_directed),
        n_pairs_undirected=len(U_full_und), n_true_undirected=len(true_undirected),
        n_signed_positions=2 * len(U_full), n_true_signed=n_true_signed,
    )


def main():
    ct = pd.read_csv(f"{L.R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    true_signed = L.load_signed_true_edges(ct)

    rows = []
    for twin_def in L.TWIN_DEFS:
        for gs in L.GENE_SETS:
            r = random_baselines(twin_def, gs, CE, true_signed)
            if r:
                rows.append(r)
    base = pd.DataFrame(rows)

    # 'ALL' row per twin_def: size-weighted (by n_pairs/n_positions) mean baseline across gene sets
    all_rows = []
    for twin_def, g in base.groupby("twin_def"):
        all_rows.append(dict(
            gene_set="ALL", twin_def=twin_def,
            n_pairs_directed=g.n_pairs_directed.sum(), n_true_directed=g.n_true_directed.sum(),
            n_pairs_undirected=g.n_pairs_undirected.sum(), n_true_undirected=g.n_true_undirected.sum(),
            n_signed_positions=g.n_signed_positions.sum(), n_true_signed=g.n_true_signed.sum(),
        ))
    base = pd.concat([base, pd.DataFrame(all_rows)], ignore_index=True)

    base["rand_directed_unsigned"] = base.n_true_directed / base.n_pairs_directed
    base["rand_undirected"] = base.n_true_undirected / base.n_pairs_undirected
    base["rand_signed_directed"] = base.n_true_signed / base.n_signed_positions

    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] table = pd.read_csv(f"{HERE}/summary_metrics_table_larry.csv")
    table = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/summary_metrics_table_larry.csv")
    table = table.rename(columns={"dataset_id": "gene_set"})
    rand_long = base.melt(id_vars=["gene_set", "twin_def"],
                           value_vars=["rand_directed_unsigned", "rand_undirected", "rand_signed_directed"],
                           var_name="variant", value_name="random")
    rand_long["variant"] = rand_long["variant"].str.replace("rand_", "", regex=False)

    merged = table.merge(rand_long, on=["gene_set", "twin_def", "variant"], how="left")
    for col in ("auprc", "f1_topk", "precision_topk"):
        merged[f"{col}_x"] = merged[col] / merged["random"]

    merged = merged.rename(columns={"gene_set": "dataset_id"})
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/summary_metrics_table_larry.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/summary_metrics_table_larry.csv"
    merged.round(4).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(merged)} rows, columns: {list(merged.columns)})")
    print(merged[(merged.dataset_id == "ALL") & (merged.variant == "directed_unsigned")]
          [["twin_def", "method", "auprc", "auprc_x", "f1_topk_x"]].to_string(index=False))


if __name__ == "__main__":
    main()
