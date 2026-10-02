"""
Generates one BLRunner config per (dataset, run) pair for the EMT/Pluripotent
real-network datasets -- same 7 enabled algorithms (PIDC, GENIE3, GRNBOOST2,
PPCOR, SCODE, SCSGL, PEARSON) and output_dir as config-files/config_real_networks.yaml
(which also lists these two datasets alongside the already-populated
B_cell_activation and Circadian_cycle), split at (dataset, run) granularity --
same pattern as the existing config-files/_real_networks_per_run/<dataset>__<run_id>.yaml
configs used for B_cell_activation/Circadian_cycle -- since BLRunner has no
internal parallelism, per-run config files are what let run_real_networks_chunk.sh
fan work out via xargs -P (per-dataset granularity, tried first, serialized
all of a dataset's runs x algorithms into one process -- too slow for EMT's
16 runs).

Run IDs are discovered by listing subdirectories of each dataset's input dir
(inputs/real_networks/<dataset_id>/) rather than hardcoded.

Also writes the matching chunk-list file (one config path per line) that
run_real_networks_chunk.sh expects as its $1 argument.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
from twinfer.utils.paths import get_external_repo_path

from pathlib import Path

DATASETS = ["EMT", "Pluripotent"]

# INPUT_ROOT = Path(__file__).parent / "inputs" / "real_networks"   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
INPUT_ROOT = get_external_repo_path("beeline") / "inputs" / "real_networks"
OUTPUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference'

# CONFIG_DIR = Path(__file__).parent / "config-files" / "_emt_pluripotent_per_run"   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
CONFIG_DIR = get_external_repo_path("beeline") / "config-files" / "_emt_pluripotent_per_run"
# CHUNK_LIST_PATH = Path(__file__).parent / "config-files" / "emt_pluripotent_chunk_list.txt"   [2026-09-30 replaced: the script used to sit in the Beeline install dir; BLRunner reads config-files/ and inputs/ relative to the full install]
CHUNK_LIST_PATH = get_external_repo_path("beeline") / "config-files" / "emt_pluripotent_chunk_list.txt"

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
    runs:
    - run_id: {run_id}
    groundTruthNetwork: GroundTruthNetwork.csv
output_settings:
  output_dir: {output_dir}
  experiment_id: ''
"""


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Clear out the stale per-dataset configs from the previous (too-serial) attempt.
    for stale in CONFIG_DIR.glob("*.yaml"):
        stale.unlink()

    config_paths = []
    for dataset_id in DATASETS:
        run_ids = sorted(d.name for d in (INPUT_ROOT / dataset_id).iterdir() if d.is_dir())
        for run_id in run_ids:
            cfg_text = TEMPLATE.format(dataset_id=dataset_id, run_id=run_id, output_dir=OUTPUT_DIR)
            cfg_path = CONFIG_DIR / f"{dataset_id}__{run_id}.yaml"
            cfg_path.write_text(cfg_text)
            config_paths.append(cfg_path)

    CHUNK_LIST_PATH.write_text(
        "\n".join(f"config-files/_emt_pluripotent_per_run/{p.name}" for p in config_paths) + "\n"
    )

    print(f"Wrote {len(config_paths)} config(s) to {CONFIG_DIR}")
    print(f"Wrote chunk list ({len(config_paths)} lines) to {CHUNK_LIST_PATH}")


if __name__ == "__main__":
    main()
