# cd /home/gzu5140/Keerthana_b1042/TwINFER/code/BoolODE/twins   # [2026-09-30 replaced by env.sh variable]
# [2026-09-30 note: #SBATCH paths cannot use variables; check them before submitting. Source clean_code/env.sh before sbatch (sbatch exports the environment).]
: "${TWINFER_CODE_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_PROJECT_ROOT:?source clean_code/env.sh before running or submitting this script}"
: "${TWINFER_BOOLODE_PATH:?source clean_code/env.sh first (full BoolODE install: BoolODE/ package + data/)}"
cd ${TWINFER_CODE_ROOT}/benchmarks/boolode/twins
# python twin_similarity_sweep.py --network mCAD --n-pairs 6000 --workers 20 --output-dir /home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/boolode_sims/   # [2026-09-30 replaced by env.sh variable]
python twin_similarity_sweep.py --network mCAD --n-pairs 6000 --workers 20 --output-dir ${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims/
# python twin_similarity_sweep.py --network HSC  --n-pairs 6000 --workers 20 --output-dir /home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/boolode_sims/   # [2026-09-30 replaced by env.sh variable]
python twin_similarity_sweep.py --network HSC  --n-pairs 6000 --workers 20 --output-dir ${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims/
# python twin_similarity_sweep.py --network VSC  --n-pairs 6000 --workers 20 --output-dir /home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/boolode_sims/   # [2026-09-30 replaced by env.sh variable]
python twin_similarity_sweep.py --network VSC  --n-pairs 6000 --workers 20 --output-dir ${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims/
# python twin_similarity_sweep.py --network GSD  --n-pairs 6000 --workers 20 --output-dir /home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/boolode_sims/   # [2026-09-30 replaced by env.sh variable]
python twin_similarity_sweep.py --network GSD  --n-pairs 6000 --workers 20 --output-dir ${TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims/
