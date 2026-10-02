from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import re
from pathlib import Path
S = Path(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/triplet_motif_discrimination')
src = (S / "simulate.py").read_text()
n = lambda pat: len(re.findall(pat, src))

# 1. paths: inputs from TwINFER_KA, outputs into the new 1k folder. Never Keerthana's tree.
src = src.replace('path_to_code_repo = "/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER"',
                  f'path_to_code_repo = "{TWINFER_PROJECT_ROOT}/code/TwINFER"')
src = src.replace('path_to_output = "/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/figure_4/"',
                  # [2026-10-01 commented out: scratch purged; output goes to simulation_data/figure_4/1k_per_motif]
                  # 'path_to_output = "/scratch/gzu5140/motif_1k/"')
                  'path_to_output = f"{TWINFER_PROJECT_ROOT}/simulation_data/figure_4/1k_per_motif/"')
assert "Keerthana" not in src, "a Keerthana path survived"

# 2. THE FIX: the old TwINFER_function_scripts module is gone; the maintained package
#    exposes the identical process_param_set(rows, label, base_config).
old_imports = [
    "from TwINFER_function_scripts import gillespie_script_variations",
    "importlib.reload(gillespie_script_variations)",
    "from TwINFER_function_scripts.gillespie_script_variations import process_param_set",
]
for line in old_imports:
    assert line in src, f"missing: {line}"
src = src.replace(old_imports[0], "from twinfer.simulation.gillespie_simulations import process_param_set")
src = src.replace(old_imports[1], "")
src = src.replace(old_imports[2], "")

# 3. per-task CLI: which motif, how many replicates, numbering offset, cores
src = src.replace('parser.add_argument("--config_index", type=int, default=0)',
                  'parser.add_argument("--config_index", type=int, default=0)\n'
                  'parser.add_argument("--n_reps", type=int, default=10)\n'
                  'parser.add_argument("--rep_offset", type=int, default=0)\n'
                  'parser.add_argument("--cores", type=int, default=8)')
assert n(r'"number_of_cores_per_parameter":\s*48') == 3
src = re.sub(r'"number_of_cores_per_parameter":\s*48',
             '"number_of_cores_per_parameter": args.cores', src)

# 4. replicate loop -> this task's slice of 0..999
assert src.count("for i in range(10):") == 1
src = src.replace("for i in range(10):",
                  "for i in range(args.rep_offset, args.rep_offset + args.n_reps):")

(S / "simulate_1k.py").write_text(src)
print("wrote simulate_1k.py")
for l in src.split("\n"):
    if any(k in l for k in ("path_to_code_repo =", "path_to_output =", "import process_param_set",
                            "for i in range", "gillespie_script_variations")):
        print("   |", l.strip()[:100])
