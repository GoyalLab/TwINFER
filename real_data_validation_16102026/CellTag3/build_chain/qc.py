"""Quality control, applied at load and asserted, so the Methods sentence cannot drift from the data.

THE RULE, as the manuscript states it for LARRY:

    remove cells with more than 20% mitochondrial gene content;
    discard cells expressing fewer than 200 genes.

WHY THIS MODULE EXISTS RATHER THAN A LINE IN EACH SCRIPT
--------------------------------------------------------
1. The stated rule and the delivered data had drifted. LARRY arrives with the mitochondrial ceiling honoured
   exactly (maximum 20.00%) but one cell at 178 genes, below the stated 200 floor. One cell in 72,946 changes
   nothing numerically -- it is unbarcoded, so it never enters a twin statistic -- but a Methods sentence that
   is false of the data is a Methods sentence that has to be fixed. `apply_qc` ASSERTS its own postconditions,
   so the claim can only be wrong if the code refuses to run.

2. The mitochondrial prefix is a real trap and it has already caught someone. Mouse symbols are `mt-Nd1`,
   `mt-Co1`; human are `MT-ND1`. Selecting the wrong case matches nothing, scores every cell at 0%
   mitochondrial content, and removes nobody -- silently, with no error. The published CellTag notebook does
   exactly this (`var_names.str.startswith('MT-')` on mouse data) and its own output reads "high mito%: 0"
   for all three samples. So the prefix here is DERIVED from the gene names and ASSERTED to match something;
   a matrix with no recognisable mitochondrial genes halts instead of quietly passing every cell.

3. The rule above uses no total-count threshold. LARRY's counts are recovered from a normalized matrix, and
   its stored obs['total_counts'] column does not describe its own count matrix: that column takes 7 distinct
   values, all within 0.002 of 2878.75, while the matrix's row sums over 400 sampled cells run from 306 to
   7,890 across 384 distinct values, disagreeing with the stored column by up to 5,011 counts in one cell. A
   threshold read from that column would filter on a quantity the matrix does not have. Whether a count
   threshold computed FROM THE MATRIX should be added is an open question and is not decided here.

WHAT IS DELIBERATELY NOT HERE
  Doublet removal. This module adds none. Where doublet handling sits for each dataset is NOT SETTLED and
  must not be written into the Methods until it is. What is measured, and only that:

    - Weinreb et al. 2020, Supplementary Materials, state that they applied SCRUBLET with thresholds set by
      manual inspection, and that "Doublet filtering eliminated ~10% of cells per" dataset. Whether the
      72,946-cell matrix held here is the set that survived that step is NOT established.
    - datasets/ holds exactly two LARRY matrices, LSK_d2_d4_d6.h5ad and LSK_d2_d4_d6_counts.h5ad. Both
      contain the same 72,946 cells (20,456 / 29,451 / 23,039 by timepoint) and differ only in gene count,
      25,289 against 23,493. There is no LARRY before/after pair here.
    - datasets/ holds two CellTag matrices that differ in cell count: celltag_nosinglet_counts.h5ad with
      60,348 (5,009 day2 / 55,339 day5) and celltag_d2_d5_counts.h5ad with 13,188 (1,471 / 11,717). The
      scripts load the 60,348 one. WHY the two differ has not been verified from primary evidence.
    - "singletCode" occurs zero times in the extracted text of Weinreb et al. 2020 and zero times in that
      of Jindal et al. 2024.
"""
import numpy as np

MITO_PCT_MAX = 20.0      # the manuscript's ceiling; cells strictly above it are removed
MIN_GENES = 200          # the manuscript's floor; cells strictly below it are removed


def mito_mask(var_names):
    """Boolean mask of mitochondrial genes, matching the prefix in ANY case.

    Mouse symbols are `mt-Nd1`, human are `MT-ND1`, and a filter written for one silently removes nothing
    from the other. Rather than choose, this matches `mt-` case-insensitively, so every mitochondrial gene
    present is caught whichever convention the matrix uses. The trailing hyphen is what makes this safe:
    it excludes the metallothioneins (`Mt1`, `Mt2`), which are not mitochondrial and carry no hyphen.

    Returns (mask, prefixes_found). Raises if nothing matches: scoring every cell at 0% and removing nobody
    is the failure mode this exists to prevent, and it is silent by nature.
    """
    v = np.asarray(var_names).astype(str)
    m = np.char.startswith(np.char.lower(v), "mt-")
    if not m.any():
        raise ValueError(
            "no mitochondrial genes found matching 'mt-' in any case. Refusing to continue: a "
            "mitochondrial filter that matches no genes scores every cell at 0% and removes nothing, "
            "without erroring. Check the gene naming before applying quality control.")
    prefixes = sorted({str(g)[:3] for g in v[m]})
    return m, "/".join(prefixes)


def qc_metrics(X, var_names):
    """Per-cell total counts, genes detected and mitochondrial percentage, computed from the matrix."""
    m, prefix = mito_mask(var_names)
    total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.int64)
    mito = np.asarray(X[:, m].sum(axis=1)).ravel().astype(float)
    pct = np.divide(100.0 * mito, total, out=np.zeros_like(total), where=total > 0)
    return {"total_counts": total, "n_genes_by_counts": n_genes, "pct_counts_mt": pct,
            "mito_prefix": prefix, "n_mito_genes": int(m.sum())}


def qc_mask(n_genes, pct_mito, min_genes=MIN_GENES, mito_pct_max=MITO_PCT_MAX):
    """Cells to KEEP, plus the per-criterion breakdown the Methods reports."""
    fail_genes = np.asarray(n_genes) < min_genes
    fail_mito = np.asarray(pct_mito) > mito_pct_max
    keep = ~(fail_genes | fail_mito)
    return keep, {
        "min_genes": min_genes, "mito_pct_max": mito_pct_max,
        "n_cells_in": int(keep.size), "n_cells_kept": int(keep.sum()),
        "removed_total": int((~keep).sum()),
        "removed_fewer_than_min_genes": int(fail_genes.sum()),
        "removed_above_mito_ceiling": int(fail_mito.sum()),
        "removed_by_both": int((fail_genes & fail_mito).sum()),
        "fraction_removed": round(float((~keep).mean()), 6),
    }


def apply_qc(adata, min_genes=MIN_GENES, mito_pct_max=MITO_PCT_MAX, label="", verbose=True):
    """Filter an AnnData in memory and ASSERT the result satisfies the stated rule.

    Metrics are ALWAYS computed from the matrix being filtered, never read from obs.

    A stored `pct_counts_mt` describes whichever matrix it was written from, at whatever precision was
    used then. On LARRY it is float32 and its maximum reads 20.0000019073, while the same quantity
    recomputed from the counts here is exactly 20.0000000000. The rule removes cells STRICTLY above
    20%, so the stored column removes one cell that sits precisely on the ceiling and the recomputed
    one removes none. A filter must be a function of the data it is filtering; a cached column is an
    approximation of a different matrix.
    """
    m = qc_metrics(adata.X, adata.var_names)
    n_genes, pct = m["n_genes_by_counts"], m["pct_counts_mt"]
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["pct_counts_mt"] = pct
    adata.obs["total_counts"] = m["total_counts"]
    source = f"computed from X ({m['n_mito_genes']} '{m['mito_prefix']}' genes)"

    keep, rec = qc_mask(n_genes, pct, min_genes, mito_pct_max)
    rec.update({"dataset": label, "metrics_source": source})
    out = adata[keep].copy()

    # Postconditions. The point of the module: the stated rule is now true of the data, or nothing runs.
    kept_genes = np.asarray(out.obs["n_genes_by_counts"], np.int64)
    kept_pct = np.asarray(out.obs["pct_counts_mt"], float)
    assert kept_genes.min() >= min_genes, \
        f"quality control failed: minimum genes {kept_genes.min()} < {min_genes}"
    assert kept_pct.max() <= mito_pct_max, \
        f"quality control failed: maximum mitochondrial {kept_pct.max():.4f}% > {mito_pct_max}%"
    rec["min_n_genes_after"] = int(kept_genes.min())
    rec["max_pct_mito_after"] = round(float(kept_pct.max()), 4)

    if verbose:
        print(f"  [{label}] quality control: {rec['n_cells_in']} -> {rec['n_cells_kept']} cells "
              f"({rec['removed_total']} removed: {rec['removed_fewer_than_min_genes']} below {min_genes} "
              f"genes, {rec['removed_above_mito_ceiling']} above {mito_pct_max}% mitochondrial); "
              f"metrics from {source}", flush=True)
    return out, rec
