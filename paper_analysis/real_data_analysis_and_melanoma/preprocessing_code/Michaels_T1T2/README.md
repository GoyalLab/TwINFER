# Michaels et al. 2024 (human T-cell progenitors, T1 = day 5, T2 = day 10): raw data -> QC'd matrices + clone labels

**Raw data**: GEO GSE248549 (GSM7916614 T1, GSM7916615 T2; `raw_data/Michaels_GEO`) and the TLB lineage-barcode amplicon FASTQs
(SRA SRR26941209-12, -15, -16, -19, -20; `raw_data/Michaels_TLB_fastq`).

**Steps** (launcher `run_michaels_barcodes_and_qc.sbatch`)
1. `mich_qc.py` (rule in `qc.py`): mito <= 20 % (`MT-` genes) and >= 200 genes -> `michaels_T1_qc.h5ad` (8,478 cells), `michaels_T2_qc.h5ad` (5,747 cells).
2. `michaels_barcodes.py`: the 16-mer barcode (AT NNNN CT NNNN AC NNNN TC NNNN GT) from read 2, the paper's filters (UMI merging, singletons,
   >= 2 UMIs per cell-barcode pair, < 50 % of the cell's top barcode removed), cells sharing a barcode linked into a graph, components split by
   min-cut / degree pruning -> `cell_to_clone.csv` (4,118 labelled cells).

**Checks** (optional): `michaels_rebuild_qc_check.py` rebuilds the QC files with an independent implementation; `michaels_rerun_barcodes.py`
(launcher `run_michaels_rerun.sbatch`) re-implements the barcode steps and reports each filter; `michaels_step8_only.py`, `michaels_step8_seeds.py`
rerun the graph step with 20 hash seeds.

**Limitation**: the graph step depends on Python's hash seed, so reruns give 1,359-1,372 clones (the file used has 1,371) with different ids.
