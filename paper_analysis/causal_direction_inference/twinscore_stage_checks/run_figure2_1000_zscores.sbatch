#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48GB
#SBATCH --time=1:00:00
#SBATCH --job-name=fig2_1000_z
#SBATCH --output=/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/logs/slurmLog-%A-%x.out
#SBATCH --error=/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/logs/slurmLog-%A-%x.err

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

C=/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/causal_direction_inference/twinscore_stage_checks
D=/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores
~/.conda/envs/twinfer-code/bin/python $C/figure2_1000_zscores.py --reps 0-99 --jobs 24 --out $D/results
