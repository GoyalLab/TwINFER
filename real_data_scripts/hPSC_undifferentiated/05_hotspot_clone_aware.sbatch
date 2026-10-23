#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=hs_clone
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/hotspot_clone_aware/hs_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/hotspot_clone_aware/hs_%j.out
# Clone-aware Hotspot gene modules (clone mates forced into kNN) on hPSC_undifferentiated (55k cells, ~3000 HVG + DE TFs). Output: analysis_data/hPSC_undifferentiated/hotspot_clone_aware/
export OMP_NUM_THREADS=1
/home/gzu5140/.conda/envs/twinfer-code/bin/python -I -c "import sys;sys.path.insert(0,'/gpfs/home/gzu5140/.claude-tmp/hs_lib');exec(open('/home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/05_hotspot_clone_aware.py').read())"
