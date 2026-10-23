#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --job-name=celltag_67
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_celltag_67_%j.out
# CellTag3 steps 6-7 only (sparse-clone repair + repaired schema), in the existing mirror whose steps 1-5 already reproduced yscher's files.
# yscher's unmodified scripts and interpreter; igraph (and shap, upsetplot, adjustText) installed with pip --user; cassiopeia stays stubbed exactly as yscher's script does.
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA; W=$ROOT/analysis_data/dataset_qc_verification/rebuild/CellTag3; M=$W/mirror
YR="/gpfs/projects/b1255/yscher/Transcriptomic Distance"; PY=/hpc/software/mamba/24.3.0/bin/python3
CMP="/home/gzu5140/.conda/envs/twinfer-code/bin/python3 $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification/h5ad_identity.py"
cd $M; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
D=$M/datasets/celltag; YD="$YR/datasets/celltag"; E=exports/legacy_reproductions/pairing/celltag_clones
step(){ echo "=== $1 $(date)"; shift; "$@" > $W/step67_$(date +%s).log 2>&1; echo "exit $?"; }
step 6_repaired_clones  $PY helpers/celltag_clones_repaired.py
cmp $M/$E/clone_table_repaired.csv "$YR/$E/clone_table_repaired.csv" && echo "clone_table_repaired.csv byte-identical" || echo "clone_table_repaired.csv DIFFERS"
step 7_repaired_schema  $PY helpers/build_celltag_repaired_schema.py
for f in celltag_clones_repaired_nopad.h5ad celltag_clones_repaired_nopad_counts.h5ad; do $CMP $D/$f "$YD/$f" $W/cmp_7_$f.json | tail -2; done
md5sum $D/celltag_clones_repaired_nopad.h5ad $ROOT/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad
echo "done $(date)"
