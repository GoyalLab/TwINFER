#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH --cpus-per-task=25
#SBATCH --mem 5GB
#SBATCH -t 12:00:00
#SBATCH --array=0-9
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins/logs/real_network_extra_%A_%a.out
#SBATCH --output=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/benchmarks/boolode/twins/logs/real_network_extra_%A_%a.out
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] #SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins/logs/real_network_extra_%A_%a.err
#SBATCH --error=/gpfs/projects/b1255/hzhang/TwINFER_KA/clean_data/benchmarks/boolode/twins/logs/real_network_extra_%A_%a.err
set -euo pipefail
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BOOLODE_PATH:?source clean_code/env.sh first (full BoolODE install: BoolODE/ package + data/)}"

# BoolODE twin-similarity replicate sweep for B_cell_activation, EMT_real,
# and Pluripotent_real -- real networks present under
# analysis_data/paper_analysis/ (TwINFER's own Gillespie-simulated
# topologies, from input_data/real_world_networks/) that had no
# boolode_sims_replicates/ counterpart yet (Circadian_cycle deliberately
# excluded per instructions). NOTE: BoolODE already ships an unrelated
# network called "Pluripotency52N" (52 genes, a different literature
# stem-cell circuit sharing only ~4 genes -- SOX2/SALL4/PRDM14/POU5F1 --
# with TwINFER's own 36-gene Pluripotent network) -- Pluripotent_real here
# is built fresh from TwINFER's actual topology, not a reuse of that.
# Modeled directly on run_replicates.sh (which built Pluripotency52N's
# replicates), with two differences: (1) paths point at THIS repo
# (/gpfs/projects/b1255/hzhang/TwINFER_KA), not the older
# /projects/b1042/GoyalLab/Keerthana/TwINFER copy run_replicates.sh still
# points at; (2) uses a SLURM array with one task PER REPLICATE (10 total
# jobs, not one per network x replicate) -- each job runs all 3 networks'
# given replicate index sequentially, keeping total job count at 10.
#
# Rule/species files: code/BoolODE/twins/custom_networks/
#   B_cell_activation_rules.txt -- from input_data/real_world_networks/B_cell.txt
#     (10-gene signed adjacency matrix), gene 'Pax-5' renamed 'PAX5' (not
#     'Pax_5') to avoid a real bug in twin_similarity_sweep.py's RNA/protein
#     classification, which does substring match ('x_' in varmapper[i])
#     instead of prefix match -- 'Pax_5' contains the literal substring
#     'x_' and got misclassified as a bogus extra protein species. Verified
#     the substitute name produces the correct single-column gene output.
#   EMT_real_rules.txt -- from input_data/real_world_networks/EMT.txt
#     (17-gene signed adjacency matrix, TwINFER's own EMT topology --
#     distinct from BoolODE's stock bundled 52-gene EMT.txt signaling
#     model, which is a different network and NOT used here).
#   Pluripotent_real_rules.txt -- from
#     input_data/real_world_networks/Pluripotent.txt (36-gene signed
#     adjacency matrix). 5 genes (SALL2, TEAD2, ZBTB12, ZNF286A, ZNF286B)
#     have zero edges in the matrix (isolated -- neither source nor
#     target anywhere) and so are absent from edges_to_rules.py's output;
#     added back manually as self-referencing placeholder rules (same
#     convention edges_to_rules.py already uses for source-only nodes) to
#     keep the full 36-gene panel intact.
# All three converted via twins/edges_to_rules.py (default
# activator-OR/repressor-AND-NOT logic; no AND-requirement knowledge
# available from a signed edge list).
#
# simulation_time chosen via the same branch-fraction plateau sweep used to
# originally calibrate GSD/HSC/VSC (found in twins/output/{GSD,HSC,VSC} --
# 7 branch fractions [0.125..0.875] at n=40 pairs, temporarily setting
# BRANCH_FRACTIONS in twin_similarity_sweep.py, looking for where twin
# Pearson plateaus, then fixing production BRANCH_FRACTIONS back to
# [0.5] and setting simulation_time so the 0.5 branch point lands at/after
# the plateau):
#   B_cell_activation: pearson 0.917 (t=2) -> 0.951 (t=4) -> ~0.95 flat
#     through t=14. Plateaus by t=4. Using simulation_time=8 (branch at
#     t=4, right at plateau onset).
#   EMT_real: pearson flat (0.91-0.93, within noise) across the ENTIRE
#     range t=4 to t=28 -- no real trend, plateaus effectively immediately.
#     Using simulation_time=16 (branch at t=8) -- conservative, well past
#     the (already-immediate) plateau.
#   Pluripotent_real: pearson 0.857 (t=2) -> 0.926 (t=4) -> noisy plateau
#     0.91-0.945 from t=4 to t=14. Using simulation_time=16 (branch at
#     t=8, pearson=0.939) for a solid margin into the plateau.
# (For comparison: this same sweep applied to mCAD and Pluripotency52N --
# both pre-existing/committed data, deliberately left as-is -- shows mCAD's
# branch point is genuinely pre-steady-state (Emx2 42% off its tail value)
# and Pluripotency52N's drifts mildly through t=14 rather than hard
# plateauing; per instruction, non-steady-state branch points are
# acceptable for this new sweep too, so B_cell/EMT/Pluripotent_real were
# not forced to reach true steady state, only past the twin-similarity
# plateau.)
#
# NOTE: hardcoded absolute paths, not self-located via ${BASH_SOURCE[0]} --
# sbatch stages the submitted script into a SLURM spool dir, so
# BASH_SOURCE-based self-location would resolve to the spool copy, not this
# file's real directory (same footgun documented in run_replicates.sh).
# SCRIPT_DIR=/gpfs/projects/b1255/hzhang/TwINFER_KA/code/BoolODE/twins   # [2026-09-30 replaced by env.sh variable]
SCRIPT_DIR=${TWINFER_CODE_ROOT}/benchmarks/boolode/twins
# PYTHON=/home/gzu5140/.conda/envs/BEELINE/bin/python3   # [2026-09-30 replaced by env.sh variable]
PYTHON="${BEELINE_PYTHON:-/home/gzu5140/.conda/envs/BEELINE/bin/python3}"
# OUTPUT_BASE=/gpfs/projects/b1255/hzhang/TwINFER_KA/simulation_data/boolode_sims_replicates   # [2026-09-30 replaced by env.sh variable]
OUTPUT_BASE=${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates

N_PAIRS=6000
WORKERS=25

# Array layout: 10 jobs total (REP 0..9); each job runs all 3 networks
# sequentially for its replicate index.
NETWORKS=(B_cell_activation EMT_real Pluripotent_real)

REP=${SLURM_ARRAY_TASK_ID:-0}

for NETWORK in "${NETWORKS[@]}"; do
    case "$NETWORK" in
        B_cell_activation)
            EXTRA_ARGS=(--model-file "${SCRIPT_DIR}/custom_networks/B_cell_activation_rules.txt" \
                       --simulation-time 8)
            ;;
        EMT_real)
            EXTRA_ARGS=(--model-file "${SCRIPT_DIR}/custom_networks/EMT_real_rules.txt" \
                       --simulation-time 16)
            ;;
        Pluripotent_real)
            EXTRA_ARGS=(--model-file "${SCRIPT_DIR}/custom_networks/Pluripotent_real_rules.txt" \
                       --simulation-time 16)
            ;;
    esac

    # twin_similarity_sweep.py always appends /<network> after --output-dir,
    # so to land directly in the <NETWORK>/replicate_<i>/ layout (matching
    # the existing GSD/HSC/mCAD/VSC/Pluripotency52N structure), write to a
    # per-task staging dir first, then relocate.
    STAGING_DIR="${OUTPUT_BASE}/_staging_extra_${SLURM_ARRAY_JOB_ID:-manual}_${REP}_${NETWORK}"
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
    echo "Done: ${NETWORK} replicate ${REP}"
done
