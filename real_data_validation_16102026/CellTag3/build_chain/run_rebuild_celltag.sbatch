#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=rebuild_celltag
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/logs/rebuild_celltag_%j.out
# CellTag3 (file used: celltag_clones_repaired_nopad.h5ad) rebuilt with yscher's UNMODIFIED scripts and interpreter in a mirror of her
# tree (helpers/ copied; datasets/ and exports/ empty), from the GEO CellRanger matrices + all_samples.csv + the authors' master table.
# Only change: build_celltag_repaired_schema.py's hard-coded os.chdir(<her root>) is pointed at the mirror (sed). Every intermediate is
# compared with her file of the same name (h5ad: element-wise; csv: byte).
set -uo pipefail
ROOT=/gpfs/projects/b1255/hzhang/TwINFER_KA; W=$ROOT/analysis_data/dataset_qc_verification/rebuild/CellTag3; M=$W/mirror
YR="/gpfs/projects/b1255/yscher/Transcriptomic Distance"; PY=/hpc/software/mamba/24.3.0/bin/python3
CMP="/home/gzu5140/.conda/envs/twinfer-code/bin/python3 $ROOT/code/TwINFER/real_data_scripts/dataset_qc_verification/h5ad_identity.py"
mkdir -p $M/datasets/celltag $M/exports/legacy_reproductions/pairing/celltag_clones $M/exports/qc
rsync -a --exclude __pycache__ "$YR/helpers/" $M/helpers/
sed -i "s#os.chdir(\"/gpfs/projects/b1255/yscher/Transcriptomic Distance\")#os.chdir(\"$M\")#" $M/helpers/build_celltag_repaired_schema.py
grep -n "os.chdir" $M/helpers/build_celltag_repaired_schema.py
cd $M; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
step(){ echo "=== $1 $(date)"; shift; "$@" > $W/step_$(date +%s).log 2>&1; echo "exit $?"; }
D=$M/datasets/celltag; YD="$YR/datasets/celltag"; E=exports/legacy_reproductions/pairing/celltag_clones
step 1_build_nosinglet  $PY helpers/build_nosinglet_dataset.py
step 2_finalize         $PY helpers/finalize_nosinglet.py
$CMP $D/celltag_nosinglet_counts.h5ad "$YD/celltag_nosinglet_counts.h5ad" $W/cmp_1_nosinglet.json | tail -3
step 3_qc               env CT_SOURCE=$D/celltag_nosinglet_counts.h5ad $PY helpers/celltag_apply_qc.py
$CMP $D/celltag_nosinglet_counts_mito20_min200genes.h5ad "$YD/celltag_nosinglet_counts_mito20_min200genes.h5ad" $W/cmp_3_qc.json | tail -3
step 4_clones           env CT_SOURCE=$D/celltag_nosinglet_counts_mito20_min200genes.h5ad PAIR_SUFFIX=_mito20_min200genes TAG_MIN=2 TAG_MAX=24 $PY helpers/build_corrected_clones.py
for f in clone_assignments_corrected_mito20_min200genes.csv pairs_barcode_overlap_2to24tags_mito20_min200genes.csv; do cmp $M/$E/$f "$YR/$E/$f" && echo "$f byte-identical" || echo "$f DIFFERS"; done
step 5_published_nopad  env CT_SOURCE=$D/celltag_nosinglet_counts_mito20_min200genes.h5ad OUT_STEM=celltag_clones_published_nopad $PY helpers/build_celltag_schema_clones.py
for f in celltag_clones_published_nopad.h5ad celltag_clones_published_nopad_counts.h5ad; do $CMP $D/$f "$YD/$f" $W/cmp_5_$f.json | tail -2; done
step 6_repaired_clones  $PY helpers/celltag_clones_repaired.py
cmp $M/$E/clone_table_repaired.csv "$YR/$E/clone_table_repaired.csv" && echo "clone_table_repaired.csv byte-identical" || echo "clone_table_repaired.csv DIFFERS"
step 7_repaired_schema  $PY helpers/build_celltag_repaired_schema.py
for f in celltag_clones_repaired_nopad.h5ad celltag_clones_repaired_nopad_counts.h5ad; do $CMP $D/$f "$YD/$f" $W/cmp_7_$f.json | tail -2; done
md5sum $D/celltag_clones_repaired_nopad.h5ad $ROOT/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad
echo "done $(date)"
