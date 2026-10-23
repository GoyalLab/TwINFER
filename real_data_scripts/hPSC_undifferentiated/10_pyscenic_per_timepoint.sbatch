#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=scenic_tp
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/pyscenic/scenic_%A_%a.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/pyscenic/scenic_%A_%a.out
# Full pySCENIC (grn -> ctx -> aucell) on hPSC_undifferentiated, one array task per timepoint (0=T0, 1=T10, 2=T15), cells of that timepoint only.
# Genes: expressed in >5% of all cells (11,093). Regulators: hPSC TF list (all_tfs.txt) within that gene set (~846). Each stage is skipped if its output exists (resumable).
set -uo pipefail
export LD_LIBRARY_PATH=/home/gzu5140/.conda/envs/grnboost310/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
TP=$(echo T0 T10 T15 | cut -d' ' -f$((SLURM_ARRAY_TASK_ID+1))); R=/gpfs/projects/b1255/hzhang/TwINFER_KA; V=$R/envs/pyscenic/bin; C=$R/resources/cistarget; O=$R/analysis_data/hPSC_undifferentiated/pyscenic/$TP; NW=32
export TP; mkdir -p $O; echo "[$(date)] $TP start"
[ -s $O/expr_$TP.loom ] || $V/python /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/10_pyscenic_prep_loom.py || exit 1
[ -s $O/adjacencies_$TP.tsv ] || $V/pyscenic grn $O/expr_$TP.loom $O/tfs_in_geneset.txt --method grnboost2 --num_workers $NW --seed 666 -o $O/adjacencies_$TP.tsv || exit 2
echo "[$(date)] grn done: $(wc -l < $O/adjacencies_$TP.tsv) links"
[ -s $O/regulons_$TP.csv ] || $V/pyscenic ctx $O/adjacencies_$TP.tsv $C/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather $C/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather --annotations_fname $C/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl --expression_mtx_fname $O/expr_$TP.loom --mode dask_multiprocessing --num_workers $NW --output $O/regulons_$TP.csv || exit 3
echo "[$(date)] ctx done"
[ -s $O/auc_$TP.csv ] || $V/pyscenic aucell $O/expr_$TP.loom $O/regulons_$TP.csv --num_workers $NW -o $O/auc_$TP.csv || exit 4
echo "[$(date)] $TP finished"; ls -la $O
