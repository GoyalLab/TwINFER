#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --job-name=hs_hpsc
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/hotspot/hs_%j.out
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_undifferentiated/hotspot/hs_%j.out
# Hotspot gene modules on hPSC_undifferentiated (55k cells, ~3000 HVG + DE TFs). Output: analysis_data/hPSC_undifferentiated/hotspot/
export OMP_NUM_THREADS=1
/home/gzu5140/.conda/envs/twinfer-code/bin/python -I -c "import sys;sys.path.insert(0,'/gpfs/home/gzu5140/.claude-tmp/hs_lib');exec(open('/home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_undifferentiated/04_hotspot_modules.py').read())"
