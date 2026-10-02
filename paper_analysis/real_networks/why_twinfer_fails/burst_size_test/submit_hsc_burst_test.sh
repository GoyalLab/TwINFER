#!/bin/bash
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -J hsc_burst_test
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/a2a3302f-b989-404d-a186-76799205cf04/scratchpad/burst_size_test/logs/slurm_%j.out
#SBATCH -e /gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/a2a3302f-b989-404d-a186-76799205cf04/scratchpad/burst_size_test/logs/slurm_%j.err

# /home/gzu5140/.conda/envs/twinfer/bin/python /gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/a2a3302f-b989-404d-a186-76799205cf04/scratchpad/burst_size_test/run_and_score_hsc.py   # [2026-09-30 repointed from an ephemeral Claude scratchpad to the rescued copy]
/home/gzu5140/.conda/envs/twinfer/bin/python ${TWINFER_PROJECT_ROOT}/clean_code/paper_analysis/real_networks/why_twinfer_fails/burst_size_test/run_and_score_hsc.py
