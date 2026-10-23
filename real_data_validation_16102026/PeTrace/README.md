# PeTrace: dataset QC verification + 5k-HVG PCA (scripts as of 2026-10-16)

Shared scripts copied unchanged from real_data_scripts/dataset_qc_verification/.

QC: python qc_verify.py PeTrace_<stage>-R<n>  (15 files: E7.5..E9.5 x R1..R3)
5k-HVG PCA + elbow: python pca5k.py PeTrace_<stage>-R<n>

Final-only (.h5td); run via run_petrace_small.sbatch + run_petrace_big.sbatch. E9.5-R1/R2/R3 re-downloaded 2026-10-15 (MD5 verified).

Outputs: analysis_data/dataset_qc_verification/<dataset>/ (verify.json, qc_*.png, funnel_*.png) and .../pca5k/ (elbow_*.png, *_pca5k.npz).
Python: /home/gzu5140/.conda/envs/twinfer-code/bin/python3. Large datasets need sbatch (login node OOM-killed): run_qc_verify.sbatch, run_pca5k.sbatch, run_petrace_small.sbatch + run_petrace_big.sbatch.
Pair-distance (clonal vs random, all 50 PCs; needs pca5k output): python pair_distance.py <dataset>  (run_pair_distance.sbatch)
