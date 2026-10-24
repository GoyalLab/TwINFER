#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_novo
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# Figure 5 preprocessing, Novo Nordisk hPSC: (1) export_seurat.R exports the authors' Seurat object (.qs) to neutral files (binary CSC counts, obs, reductions), (2) build_qc_per_sample.py writes the
# per-sample qc_filtered folders (counts.npz, genes.txt, obs_metadata.csv; cells with a clone.id, nFeature_RNA >= 1000, percent.mt <= 10), (3) every file is compared byte-wise with final_data.
# Output: <REBUILD>/hPSC_Novo_Nordisk/{export,qc_filtered}; final_data is never touched.   Usage: sbatch run_rebuild_novo.sbatch
set -uo pipefail
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
FIG=$ROOT/final_figure_data/real_data_analysis_and_melanoma
RAW=${FIG5_RAW_DATA:-$FIG/raw_data}                                  # links to the raw data (see raw_data/)
FINAL=$FIG/final_data                                                # links to the final objects
REBUILD=${FIG5_REBUILD_DIR:-$FIG/analysis_data/preprocessing_rebuild}  # rebuilt files go here, never into final_data
CODE=$ROOT/code/TwINFER/paper_analysis/real_data_analysis_and_melanoma/preprocessing_code
PY=${TWINFER_PYTHON:-/home/gzu5140/.conda/envs/twinfer-code/bin/python3}   # conda activate does not work in batch jobs: call the environment python directly
export TWINFER_PROJECT_ROOT=$ROOT
W=$REBUILD/hPSC_Novo_Nordisk; mkdir -p "$W/export"
export NOVO_QS=$RAW/Novo_nordisk_hPSC_endo/seurat_lineage_tracing_linneberg_agerholm_2026.qs NOVO_EXPORT_DIR=$W/export
$RAW/external_tools/seurat_qs_env/bin/Rscript "$CODE/hPSC_Novo_Nordisk/export_seurat.R" > "$W/export_seurat.log" 2>&1 || { echo "export failed, see $W/export_seurat.log"; exit 1; }
$PY "$CODE/hPSC_Novo_Nordisk/build_qc_per_sample.py" > "$W/build_qc_per_sample.log" 2>&1 || { echo "build failed, see $W/build_qc_per_sample.log"; exit 1; }
for d in "$FINAL"/hPSC_Novo_Nordisk_data/qc_filtered/*/; do s=$(basename "$d"); for f in counts.npz genes.txt obs_metadata.csv; do cmp -s "$d/$f" "$W/qc_filtered/$s/$f" && echo "$s/$f byte-identical" || echo "$s/$f DIFFERS"; done; done | tee "$W/cmp_vs_final.txt"
