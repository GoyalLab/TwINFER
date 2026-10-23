# LARRY: dataset QC verification + 5k-HVG PCA (scripts as of 2026-10-16)

Shared scripts copied unchanged from real_data_scripts/dataset_qc_verification/.

QC: python qc_verify.py LARRY
5k-HVG PCA + elbow: python pca5k.py LARRY

PILOT: raw side = per-library counts inside LARRY_data/raw/LSK_d2_1.zip (inDrops quantification read as shipped, gel barcodes mapped with abundant_barcodes.pickle); no FASTQ / barcode calling. Only library LSK_d2_1 is compared to raw.

Outputs: analysis_data/dataset_qc_verification/<dataset>/ (verify.json, qc_*.png, funnel_*.png) and .../pca5k/ (elbow_*.png, *_pca5k.npz).
Python: /home/gzu5140/.conda/envs/twinfer-code/bin/python3. Large datasets need sbatch (login node OOM-killed): run_qc_verify.sbatch, run_pca5k.sbatch, run_petrace_small.sbatch + run_petrace_big.sbatch.
Pair-distance (clonal vs random, all 50 PCs; needs pca5k output): python pair_distance.py <dataset>  (run_pair_distance.sbatch)
