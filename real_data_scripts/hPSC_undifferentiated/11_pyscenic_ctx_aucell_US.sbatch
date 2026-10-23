#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=scenic_US
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/pyscenic/scenicUS_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/pyscenic/scenicUS_%A_%a.out
# pySCENIC ctx (motif pruning) + aucell on the TwINFER U- and S-weighted adjacencies (made by 11_twinfer_US_adjacency.py), same settings as the GRNBoost2 run.
# array task: 0=T0/U 1=T0/S 2=T10/U 3=T10/S 4=T15/U 5=T15/S.  Skips a stage if its output exists.
set -uo pipefail
export LD_LIBRARY_PATH=/home/gzu5140/.conda/envs/grnboost310/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
TP=$(echo T0 T10 T15 | cut -d' ' -f$((SLURM_ARRAY_TASK_ID/2+1))); VAR=$(echo U S | cut -d' ' -f$((SLURM_ARRAY_TASK_ID%2+1)))
R=/gpfs/projects/b1255/hzhang/TwINFER_KA; V=$R/envs/pyscenic/bin; C=$R/resources/cistarget; O=$R/analysis_data/hPSC_undifferentiated/pyscenic/$TP; NW=32
echo "[$(date)] $TP $VAR start"; ADJ=$O/adjacencies_${VAR}_$TP.tsv; [ -s $ADJ ] || { echo "missing $ADJ"; exit 1; }
[ -s $O/regulons_${VAR}_$TP.csv ] || $V/pyscenic ctx $ADJ $C/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather $C/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather --annotations_fname $C/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl --expression_mtx_fname $O/expr_$TP.loom --mode dask_multiprocessing --num_workers $NW --output $O/regulons_${VAR}_$TP.csv || exit 3
echo "[$(date)] ctx done"
[ -s $O/auc_${VAR}_$TP.csv ] || $V/pyscenic aucell $O/expr_$TP.loom $O/regulons_${VAR}_$TP.csv --num_workers $NW -o $O/auc_${VAR}_$TP.csv || exit 4
echo "[$(date)] $TP $VAR finished"; ls -la $O | grep -E "_${VAR}_"
