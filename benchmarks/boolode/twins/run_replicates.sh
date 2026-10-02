#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=25
#SBATCH --mem 5GB
#SBATCH -t 12:00:00
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/home/gzu5140/Keerthana_b1042/TwINFER/code/BoolODE/twins/logs/real_network_%A_%a.out
#SBATCH --output=/home/gzu5140/Keerthana_b1042/TwINFER/clean_data/benchmarks/boolode/twins/logs/real_network_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/home/gzu5140/Keerthana_b1042/TwINFER/code/BoolODE/twins/logs/real_network_%A_%a.err
#SBATCH --error=/home/gzu5140/Keerthana_b1042/TwINFER/clean_data/benchmarks/boolode/twins/logs/real_network_%A_%a.err
set -eo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BOOLODE_PATH:?source clean_code/env.sh first (full BoolODE install: BoolODE/ package + data/)}"

# Run N independent replicate twin_similarity_sweep.py runs for HSC, VSC,
# and mCAD (GSD deliberately excluded -- it's the expensive one and is
# handled separately).
#
# Each replicate uses a different --seed-offset so it's a genuinely
# independent sample, not a copy of replicate 0 -- rerunning with the same
# --network/--n-pairs/--seed-offset is fully deterministic (seeds only
# depend on pair_id and seed_offset), so looping this script without
# varying --seed-offset would silently produce identical "replicates".
#
# Usage:
#   bash run_replicates.sh
# Adjust N_REPLICATES / N_PAIRS / WORKERS / OUTPUT_BASE below as needed.

set -euo pipefail

# NOTE: deliberately hardcoded, not self-located via ${BASH_SOURCE[0]}.
# sbatch copies the submitted script into a SLURM spool directory
# (/var/spool/slurmd/jobXXXXXXX/) and runs it from there, so
# ${BASH_SOURCE[0]}-based self-location resolves to the spool copy's
# location, not this script's real directory -- breaks under `sbatch`
# even though it works fine under plain `bash run_replicates.sh` (no
# spool-copy involved there). twin_similarity_sweep.py itself doesn't have
# this problem since it locates itself via Python's own __file__ at import
# time, which isn't affected by how the *bash* script that invoked it was
# staged.
# SCRIPT_DIR=/home/gzu5140/Keerthana_b1042/TwINFER/code/BoolODE/twins   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_CODE_ROOT}/benchmarks/boolode/twins
# PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python3}"

NETWORKS=(Pluripotency52N)
N_REPLICATES=20
N_PAIRS=6000
WORKERS=25
# OUTPUT_BASE=/projects/b1042/GoyalLab/Keerthana/TwINFER/simulation_data/boolode_sims_replicates   # [2026-09-30 replaced by env.sh variable]
OUTPUT_BASE=${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates

for NETWORK in "${NETWORKS[@]}"; do
    # Per-network extra flags: GSD is a built-in network (NETWORKS registry
    # in twin_similarity_sweep.py already knows its model/simulation_time,
    # --ics-file here just makes the seed explicit rather than relying on
    # the registry default). Pluripotency52N is a CUSTOM network (not in
    # that registry), so it needs --model-file and --simulation-time
    # explicitly -- --network is just a label in that case.
    #
    # NOTE: --simulation-time 8 for Pluripotency52N has been validated the
    # same way HSC's was -- reran a 40-pair pilot at simulation_time=16 and
    # confirmed the twin/random euclidean+pearson summary stats barely
    # moved (twin pearson 0.933 at t=8 vs 0.938 at t=16, random 0.664 vs
    # 0.652), so the population is already converged by t=8, not still in
    # transit.
    EXTRA_ARGS=()
    case "$NETWORK" in
        # GSD)
        #     EXTRA_ARGS=(--ics-file "/home/gzu5140/Keerthana_b1042/TwINFER/code/BoolODE/data/GSD_ics.txt")
        #     ;;
        Pluripotency52N)
            EXTRA_ARGS=(--model-file "${SCRIPT_DIR}/custom_networks/Pluripotency52N_rules.txt" \
                       --simulation-time 8)
            ;;
    esac

    for ((REP=10; REP<N_REPLICATES; REP++)); do
        # twin_similarity_sweep.py always appends /<network> after
        # --output-dir, so to land directly in the <NETWORK>/replicate_<i>
        # layout (matching convert_twins_to_beeline.py and
        # boolode_to_twinfer_format.py's expected input structure), write
        # to a per-replicate staging dir first, then relocate its
        # <network> subfolder into place and remove the now-empty staging
        # dir.
        STAGING_DIR="${OUTPUT_BASE}/_staging_rep${REP}"
        FINAL_DIR="${OUTPUT_BASE}/${NETWORK}/replicate_${REP}"
        echo "=== ${NETWORK} replicate ${REP} (seed-offset=${REP}) -> ${FINAL_DIR}/ ==="
        "$PYTHON" "${SCRIPT_DIR}/twin_similarity_sweep.py" \
            --network "$NETWORK" \
            --n-pairs "$N_PAIRS" \
            --workers "$WORKERS" \
            --seed-offset "$REP" \
            --output-dir "$STAGING_DIR" \
            "${EXTRA_ARGS[@]}"
        mkdir -p "${OUTPUT_BASE}/${NETWORK}"
        rm -rf "$FINAL_DIR"
        mv "${STAGING_DIR}/${NETWORK}" "$FINAL_DIR"
        rmdir "$STAGING_DIR"
    done
done

echo "All ${#NETWORKS[@]} networks x ${N_REPLICATES} replicates finished."
