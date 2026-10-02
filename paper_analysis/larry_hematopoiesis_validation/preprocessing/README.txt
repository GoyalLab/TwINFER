This is the pre-processing steps involved in creating the LARRY dataset.
The raw files used are the inDrops count matrix for the LSK library and
LARRY_sorted_and_filtered_barcodes.fastq.gz obtained from authors.
The cell IDs are renamed to actual 16-sequence barcodes using
abundant_barcodes.pickle.

Step 1: Build the raw combined matrix using build_larry_matrix.py, run via
        run_build_larry_matrix.sh. This also does the lineage-barcode processing:
        filters (CellID, Barcode, UMI) combinations to those supported by >=10 reads,
        merges barcodes within Hamming distance 3 (transitive: connected components,
        each collapsing to its highest-abundance member -- matches Weinreb et al.
        2020 Methods 3.3's "graph-connected-components" description), and
        writes umi_table.csv (one row per UMI per (cellID, barcode, sample) triple)
        for singletCode to consume in Step 3.

Steps 2-4 are all applied by one script, build_final_matrix_umi3.py, run via
run_build_final_matrix_umi3.sh, writing the final output to qc_filtered/:

Step 2: Cell-calling whitelist, computed inline in build_final_matrix_umi3.py. Per
        library: Hartigan's dip test on log10(total UMI counts) decides whether
        there's a real second (ambient/empty-droplet) mode at all (p<0.05); if so,
        Otsu's threshold cuts there; if not, every nonzero-count barcode is kept.
        89,813 of 95,584 candidate barcodes pass.

Step 3: Filter multiplets by applying singletCode (min_umi_cutoff=3, not its default
        of 2 to match Weinreb et al. 2020 Methods 3.3's documented UMI threshold) to the
        whitelist-restricted barcode list, then complete singletCode's own
        unfinished 4th singlet criterion (barcode combination recurring in other
        cells ACROSS samples, not just within one) with
        singletcode_cross_sample_rescue.py. The wrapper calls the package's own
        generate_barcode_combo/extract_two_barcode_singlets functions on its full
        cross-sample output rather than reimplementing the logic. 32,728 cells pass
        (957 rescued by the completed 4th criterion specifically).

Step 4: QC the single-cell RNA sequencing data by filtering out cells with more than
        20% mitochondrial genes (mt- prefix, matched case-insensitively) and fewer than
        200 genes detected. The mito/gene-floor criteria are independent of Step 3 (computed
        purely from raw counts and detected-gene numbers, with no reference to a cell's
        Singlet/Multiplet status), but the cell counts are intersected immediately and
        sequentially, not merged later: whitelist -> mito -> singletCode (Step 3) -> gene
        floor, each stage intersected onto the previous one's survivors in build_final_matrix_umi3.py.
        32,395 cells pass all four together.


Final output: qc_filtered/, built end-to-end by build_final_matrix_umi3.py (run via
run_build_final_matrix_umi3.sh) applying Steps 2-4 to Step 1's output (Step 5 retired,
see above). 32,395 cells, 5,387 distinct clones. Validated against the released
LSK_d2_d4_d6.h5ad's own clone_id (24,908 cells have clone_id, 1,221 of which are not
in this qc_filtered/): Adjusted Rand Index = 0.9989 on the 23,687 cells both datasets
call barcoded (76 cells, 0.32%, disagree in exactly how they're grouped into clones --
see singletcode_vs_h5ad_diff_cells.csv for the list).
