#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=fm06_scenic
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/fm06_scenic_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/fm06_scenic_%j.out
# FM06 (melanoma; 12,863 cells, genes >5% of cells = 9,100, log1p(CP10k), 613 regulators): pySCENIC with (a) GRNBoost2 and (b) TwINFER U and (c) TwINFER S adjacencies.
# Stages: grn -> ctx -> aucell (GRNBoost2); then TwINFER U/S adjacencies (same edges per TF as GRNBoost2) -> ctx -> aucell for each. Each stage is skipped if its output exists.
set -uo pipefail
export LD_LIBRARY_PATH=/home/gzu5140/.conda/envs/grnboost310/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
R=/gpfs/projects/b1255/hzhang/TwINFER_KA; V=$R/envs/pyscenic/bin; C=$R/resources/cistarget; O=$R/analysis_data/fm06/pyscenic; NW=32; S=/home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/FM06_pyscenic
DB1=$C/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather; DB2=$C/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather; ANN=$C/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
echo "[$(date)] FM06 start"
[ -s $O/expr_FM06.loom ] || $V/python $S/12_fm06_prep_loom.py || exit 1
[ -s $O/adjacencies_FM06.tsv ] || $V/pyscenic grn $O/expr_FM06.loom $O/tfs_in_geneset.txt --method grnboost2 --num_workers $NW --seed 666 -o $O/adjacencies_FM06.tsv || exit 2
echo "[$(date)] grn done: $(wc -l < $O/adjacencies_FM06.tsv) links"
[ -s $O/adjacencies_U_FM06.tsv ] || TP=FM06 EXPR=$O/expr_FM06.loom GRB=$O/adjacencies_FM06.tsv CLONES_CSV=$O/clones_FM06.csv OUT=$O/ $V/python $S/11_twinfer_US_adjacency.py || exit 5
echo "[$(date)] TwINFER U/S adjacencies done"
for VAR in "" U_ S_; do
  N=FM06; [ -n "$VAR" ] && N=${VAR}FM06; A=$O/adjacencies_$N.tsv; [ "$VAR" = "" ] && A=$O/adjacencies_FM06.tsv
  [ -s $O/regulons_$N.csv ] || $V/pyscenic ctx $A $DB1 $DB2 --annotations_fname $ANN --expression_mtx_fname $O/expr_FM06.loom --mode dask_multiprocessing --num_workers $NW --output $O/regulons_$N.csv || exit 3
  echo "[$(date)] ctx done ($N)"
  [ -s $O/auc_$N.csv ] || $V/pyscenic aucell $O/expr_FM06.loom $O/regulons_$N.csv --num_workers $NW -o $O/auc_$N.csv || exit 4
  echo "[$(date)] aucell done ($N)"
done
echo "[$(date)] FM06 finished"; ls -la $O
