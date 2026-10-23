#!/bin/bash
# Resume of run_1k_per_motif_multicore.sh (job 8232091 timed out at 36 h with 2007/3000 replicates).
# Does only the 993 missing replicates listed in remaining_g_2026-10-04.txt, dealt out evenly: task k takes list[k::42] (23-24 replicates each = 3 rounds of 8; 943 of the 993 are the slower Mutual_regulation),
# run as 8 parallel workers x 6 cores (+2 spare cores for I/O) per 50-core task. Existing outputs are skipped, so it is safe to re-run.
# Run `source code/TwINFER/env.sh` before sbatch.
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=60GB
#SBATCH --time=20:00:00
#SBATCH --job-name=motif1k_res
#SBATCH --array=0-41
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2/logs/resume-%A_%a.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2/logs/resume-%A_%a.err
set -eo pipefail
: "${TWINFER_CODE_ROOT:?source code/TwINFER/env.sh before sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=${TWINFER_CODE_ROOT}/package:${PYTHONPATH:-}
OUT=/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2
~/.conda/envs/twinfer-code/bin/python ${TWINFER_CODE_ROOT}/paper_analysis/triplet_motif_discrimination/run_1k_per_motif_multicore.py \
    --slice ${SLURM_ARRAY_TASK_ID} --n-slices 42 --jobs 8 --cores-per 6 --out ${OUT} --g-file ${OUT}/remaining_g_2026-10-04.txt
