"""
Generate BEELINE configs for the e13_pos100 benchmark (3 datasets:
n6_e13_pos100_rep{1,2,3}), same algorithm set/layout as
generate_mixed_network_sweep_configs.py.

    python generate_e13_pos100_configs.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
from twinfer.utils.paths import get_external_repo_path
from pathlib import Path

import yaml

# BEELINE_DIR = Path(__file__).parent   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
BEELINE_DIR = get_external_repo_path("beeline")
INPUT_DIR = BEELINE_DIR / "inputs" / "e13_pos100"
OUTPUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_inference'

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


def _algos_genie3_grnboost2():
    out = []
    for a in MAIN_ALGORITHMS:
        a = {k: (list(v) if isinstance(v, list) else v) for k, v in a.items()}
        a["should_run"] = [a["algorithm_id"] in ("GENIE3", "GRNBOOST2")]
        out.append(a)
    return out


def _datasets():
    nets = sorted(p.name for p in INPUT_DIR.iterdir()
                  if p.is_dir() and (p / "GroundTruthNetwork.csv").exists())
    return [
        {"dataset_id": net, "should_run": [True],
         "scan_run_subdirectories": True,
         "groundTruthNetwork": "GroundTruthNetwork.csv"}
        for net in nets
    ]


def _write(path, algorithms):
    config = {
        "input_settings": {
            "input_dir": "inputs/e13_pos100",
            "algorithms": algorithms,
            "datasets": _datasets(),
        },
        "output_settings": {"output_dir": OUTPUT_DIR, "experiment_id": ""},
    }
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"wrote {path}  ({len(config['input_settings']['datasets'])} datasets)")


def main():
    cfg_dir = BEELINE_DIR / "config-files"
    _write(cfg_dir / "config_e13_pos100.yaml", MAIN_ALGORITHMS)
    _write(cfg_dir / "config_e13_pos100_genie3_grnboost2.yaml", _algos_genie3_grnboost2())


if __name__ == "__main__":
    main()
