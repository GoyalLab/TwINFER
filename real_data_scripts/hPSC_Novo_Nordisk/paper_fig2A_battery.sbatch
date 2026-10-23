#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 01:00:00
#SBATCH -J fig2A_battery
#SBATCH -o /home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction/figA_battery.log
/home/gzu5140/.conda/envs/twinfer-code/bin/python -I /home/gzu5140/TwINFER_KA/code/TwINFER/real_data_scripts/hPSC_Novo_Nordisk/paper_fig2A_battery.py
