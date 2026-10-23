# Michaels_T1T2: Michaels et al. samples T1/T2 (with clone labels)

QC vs raw (done): python qc_verify.py Michaels_T1T2
PCA + pair distance: sbatch run_michaels_t1t2.sbatch
Clone labels: finalized_data/Michaels_T1T2_data/cell_to_clone.csv (see README_PROVENANCE.md there; derivation not documented).
The hashtag pools A/C/D are the separate Michaels_et_al dataset (no clones).
