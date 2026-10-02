"""2026-09-18: BEELINE configs for the t1=10,t2=20 twin_paired-only reruns of all 4 simulated
benchmark families (e13_pos100, network_sweep_final broader, mixed_network_sweep, real_networks).
Each *_t1_10 input root (built by build_t1_10_beeline_inputs.py) contains ONLY simrep<k>_twin_paired
directories -- no spread -- so scan_run_subdirectories naturally picks up just the new t1=10 runs,
nothing from the existing t1=1 data.

Same MAIN_ALGORITHMS list as generate_mixed_network_sweep_configs.py/generate_e13_pos100_configs.py
(verified identical, 2026-09-18): PIDC/PPCOR/SCODE/SCSGL/PEARSON in the main config,
GENIE3/GRNBOOST2 split into their own config per the existing memory-safety convention.

    python generate_t1_10_beeline_configs.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
from twinfer.utils.paths import get_external_repo_path
from pathlib import Path

import yaml

# BEELINE_DIR = Path(__file__).parent   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
BEELINE_DIR = get_external_repo_path("beeline")
ANALYSIS_ROOT = Path(f'{TWINFER_PROJECT_ROOT}/analysis_data')

MAIN_ALGORITHMS = [
    {"algorithm_id": "PIDC", "image": "grnbeeline/pidc:base", "should_run": [True]},
    {"algorithm_id": "GENIE3", "image": "grnbeeline/arboreto:base", "should_run": [False]},
    {"algorithm_id": "GRNBOOST2", "image": "grnbeeline/arboreto:base", "should_run": [False]},
    {"algorithm_id": "PPCOR", "image": "grnbeeline/ppcor:base", "should_run": [True],
     "params": {"pVal": [0.01]}},
    {"algorithm_id": "SCODE", "image": "grnbeeline/scode:base", "should_run": [True],
     "params": {"z": [4], "nIter": [200], "nRep": [2]}},
    {"algorithm_id": "SINCERITIES", "image": "grnbeeline/sincerities:base",
     "should_run": [False], "params": {"nBins": [10]}},
    {"algorithm_id": "LEAP", "image": "grnbeeline/leap:base", "should_run": [False],
     "params": {"maxLag": [0.33]}},
    {"algorithm_id": "SCSGL", "image": "grnbeeline/scsgl:base", "should_run": [True],
     "params": {"pos_density": [0.45], "neg_density": [0.45], "assoc": ["correlation"]}},
    {"algorithm_id": "PEARSON", "image": "local", "should_run": [True]},
]

# (config_stem, beeline_input_subdir, analysis_output_subdir)
FAMILIES = [
    ("e13_pos100", "e13_pos100_t1_10",
     "network_sweep_final/e13_pos100/beeline_inference_t1_10"),
    ("network_sweep_final", "network_sweep_final_t1_10",
     "network_sweep_final/beeline_inference_t1_10"),
    ("mixed_network_sweep", "mixed_network_sweep_t1_10",
     "mixed_network_sweep/beeline_inference_t1_10"),
    ("real_networks", "real_networks_t1_10",
     "paper_analysis/real_networks/beeline_inference_t1_10"),
]


def _algos_genie3_grnboost2():
    out = []
    for a in MAIN_ALGORITHMS:
        a = {k: (list(v) if isinstance(v, list) else v) for k, v in a.items()}
        a["should_run"] = [a["algorithm_id"] in ("GENIE3", "GRNBOOST2")]
        out.append(a)
    return out


def _datasets(input_dir: Path):
    nets = sorted(p.name for p in input_dir.iterdir()
                  if p.is_dir() and (p / "GroundTruthNetwork.csv").exists())
    return [
        {"dataset_id": net, "should_run": [True],
         "scan_run_subdirectories": True,
         "groundTruthNetwork": "GroundTruthNetwork.csv"}
        for net in nets
    ]


def _write(path, input_subdir, output_dir, algorithms):
    input_dir = BEELINE_DIR / "inputs" / input_subdir
    config = {
        "input_settings": {
            "input_dir": f"inputs/{input_subdir}",
            "algorithms": algorithms,
            "datasets": _datasets(input_dir),
        },
        "output_settings": {"output_dir": str(output_dir), "experiment_id": ""},
    }
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"wrote {path}  ({len(config['input_settings']['datasets'])} datasets)")


def main():
    cfg_dir = BEELINE_DIR / "config-files"
    for stem, input_subdir, output_subdir in FAMILIES:
        output_dir = ANALYSIS_ROOT / output_subdir
        _write(cfg_dir / f"config_{stem}_t1_10.yaml", input_subdir, output_dir, MAIN_ALGORITHMS)
        _write(cfg_dir / f"config_{stem}_t1_10_genie3_grnboost2.yaml", input_subdir, output_dir,
               _algos_genie3_grnboost2())


if __name__ == "__main__":
    main()
