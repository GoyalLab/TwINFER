#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=fm06_adj
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/fm06_adj_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/fm06_adj_%A_%a.out
# pySCENIC ctx + aucell on the TwINFER U and S adjacencies computed with STRATIFIED ranks (rank within strata of the confounders), same settings as the unadjusted FM06 run.
# array task: 0=U/all 1=S/all 2=U/cc 3=S/cc 4=U/cx 5=S/cx   (all = cell cycle + complexity + replicate + mt; cc = cell-cycle phase only; cx = library complexity only)
set -uo pipefail
export LD_LIBRARY_PATH=/home/gzu5140/.conda/envs/grnboost310/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
V=$(echo U S | cut -d' ' -f$((SLURM_ARRAY_TASK_ID%2+1))); T=$(echo all cc cx | cut -d' ' -f$((SLURM_ARRAY_TASK_ID/2+1)))
R=/gpfs/projects/b1255/hzhang/TwINFER_KA; B=$R/envs/pyscenic/bin; C=$R/resources/cistarget; O=$R/analysis_data/fm06/pyscenic; NW=32; N=${V}_st${T}
ADJ=$O/adjacencies_${V}_st${T}_FM06.tsv; echo "[$(date)] $N start"; [ -s $ADJ ] || { echo "missing $ADJ"; exit 1; }
[ -s $O/regulons_${N}_FM06.csv ] || $B/pyscenic ctx $ADJ $C/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather $C/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather --annotations_fname $C/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl --expression_mtx_fname $O/expr_FM06.loom --mode dask_multiprocessing --num_workers $NW --output $O/regulons_${N}_FM06.csv || exit 3
echo "[$(date)] ctx done"; [ -s $O/auc_${N}_FM06.csv ] || $B/pyscenic aucell $O/expr_FM06.loom $O/regulons_${N}_FM06.csv --num_workers $NW -o $O/auc_${N}_FM06.csv || exit 4
echo "[$(date)] $N finished"
