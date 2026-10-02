"""
Generates one BLRunner config per dataset for the cyclic_g3/g4/g5/g6 datasets (simple
directed cycles, 3/4/5/6 genes) -- same 7 enabled algorithms (PIDC, GENIE3, GRNBOOST2,
PPCOR, SCODE, SCSGL, PEARSON) and input_dir convention (inputs/real_networks/<dataset_id>/)
as the existing config-files/_real_networks_per_run/<dataset>__<run_id>.yaml configs,
but using BLRunner's `scan_run_subdirectories: true` (see BLRunner.py's get_datasets())
to auto-discover all simrep{N}_{twin_paired,spread} run subdirectories instead of
listing each run_id explicitly -- one config file per dataset instead of one per
(dataset, run) pair, since BLRunner itself has no internal parallelism (all
parallelism in this repo comes from running separate config files as separate
processes), so per-dataset granularity loses nothing dataset-level while cutting
file count 20x.

Also writes the matching chunk-list file (one config path per line) that
run_real_networks_chunk.sh expects as its $1 argument.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
from twinfer.utils.paths import get_external_repo_path

from pathlib import Path

DATASETS = ["cyclic_g3", "cyclic_g4", "cyclic_g5", "cyclic_g6"]

OUTPUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/cyclic_g3456/beeline_inference'

# CONFIG_DIR = Path(__file__).parent / "config-files" / "_cyclic_g3456_per_run"   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
CONFIG_DIR = get_external_repo_path("beeline") / "config-files" / "_cyclic_g3456_per_run"
# CHUNK_LIST_PATH = Path(__file__).parent / "config-files" / "cyclic_g3456_chunk_list.txt"   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
CHUNK_LIST_PATH = get_external_repo_path("beeline") / "config-files" / "cyclic_g3456_chunk_list.txt"

TEMPLATE = """input_settings:
  input_dir: inputs/real_networks/
  algorithms:
  - algorithm_id: PIDC
    image: grnbeeline/pidc:base
    should_run:
    - true
  - algorithm_id: GENIE3
    image: grnbeeline/arboreto:base
    should_run:
    - true
  - algorithm_id: GRNBOOST2
    image: grnbeeline/arboreto:base
    should_run:
    - true
  - algorithm_id: PPCOR
    image: grnbeeline/ppcor:base
    should_run:
    - true
    params:
      pVal:
      - 0.01
  - algorithm_id: SCODE
    image: grnbeeline/scode:base
    should_run:
    - true
    params:
      z:
      - 4
      nIter:
      - 200
      nRep:
      - 6
  - algorithm_id: SINCERITIES
    image: grnbeeline/sincerities:base
    should_run:
    - false
    params:
      nBins:
      - 10
  - algorithm_id: LEAP
    image: grnbeeline/leap:base
    should_run:
    - false
    params:
      maxLag:
      - 0.33
  - algorithm_id: SCSGL
    image: grnbeeline/scsgl:base
    should_run:
    - true
    params:
      pos_density:
      - 0.45
      neg_density:
      - 0.45
      assoc:
      - correlation
  - algorithm_id: PEARSON
    image: local
    should_run:
    - true
  datasets:
  - dataset_id: {dataset_id}
    should_run:
    - true
    scan_run_subdirectories: true
    groundTruthNetwork: GroundTruthNetwork.csv
output_settings:
  output_dir: {output_dir}
  experiment_id: ''
"""


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_paths = []

    for dataset_id in DATASETS:
        cfg_text = TEMPLATE.format(dataset_id=dataset_id, output_dir=OUTPUT_DIR)
        cfg_path = CONFIG_DIR / f"{dataset_id}.yaml"
        cfg_path.write_text(cfg_text)
        config_paths.append(cfg_path)

    CHUNK_LIST_PATH.write_text(
        "\n".join(f"config-files/_cyclic_g3456_per_run/{p.name}" for p in config_paths) + "\n"
    )

    print(f"Wrote {len(config_paths)} config(s) to {CONFIG_DIR}")
    print(f"Wrote chunk list ({len(config_paths)} lines) to {CHUNK_LIST_PATH}")


if __name__ == "__main__":
    main()
