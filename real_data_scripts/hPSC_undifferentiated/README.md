# hPSC_undifferentiated analysis (2026-10-12)
Input: finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad (log1pPF; T0/T10/T15; clone_id). Outputs: analysis_data/hPSC_undifferentiated/.
Run with ~/.conda/envs/twinfer/bin/python -I (01, 03) or twinfer-code (02, 04). Scripts copied unchanged from the session; paths are absolute.
1. 01_de_tfs_timepoints.py - Wilcoxon DE of 891 TFs (analysis_data/hPSC_20260927/gene_modules/all_tfs.txt) for T0-T10, T10-T15, T0-T15 -> de_tfs/
2. 02_cell_cycle_phase.py - Tirosh S/G2M scoring per cell, phase fractions per timepoint -> cell_cycle_phase_per_cell.csv
3. 03_clone_paired_de.py - paired DE within shared clones (clone x timepoint means), genes >=5% cells; TF subset flagged -> clone_de/. Needs de_tfs/ output of step 1 (T0-T15 comparison table).
4. 04_hotspot_modules.{py,sbatch} - Hotspot modules (kNN on PCA, normal model). Job 9678515. Needs hotspot (not installed in any env): sys.path points to ~/.claude-tmp/hs_lib (pip --target copy of hotspotsc 1.1.3). Also needs step 3 and 1 outputs. Submit only after review.
Note: 04 .sbatch/.py in the running job were read from the session scratchpad, this copy is identical apart from paths.
