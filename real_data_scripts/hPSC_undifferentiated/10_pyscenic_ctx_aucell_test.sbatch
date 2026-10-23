#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=scenic_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/pyscenic/scenic_test_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/pyscenic/scenic_test_%j.out
# TEST ONLY: ctx + aucell on the small GRN made locally (1,500 T0 cells x 1,096 genes). Checks the databases/annotations/pyscenic plumbing before the real run.
export LD_LIBRARY_PATH=/home/gzu5140/.conda/envs/grnboost310/lib:$LD_LIBRARY_PATH
PS=/gpfs/projects/b1255/hzhang/TwINFER_KA/envs/pyscenic/bin/pyscenic; C=/gpfs/projects/b1255/hzhang/TwINFER_KA/resources/cistarget; T=/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/2cab7687-1986-47a7-8162-62833f2ce066/scratchpad/scenic_test/T0; cd $T
$PS ctx adj.tsv $C/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather $C/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather --annotations_fname $C/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl --expression_mtx_fname expr_T0.loom --mode dask_multiprocessing --num_workers 4 --output regulons.csv
$PS aucell expr_T0.loom regulons.csv --num_workers 4 -o auc.csv
ls -la regulons.csv auc.csv; head -c 400 auc.csv
