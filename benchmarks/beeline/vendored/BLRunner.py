import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

from BLRun.genie3Runner import GENIE3Runner
from BLRun.grnboost2Runner import GRNBoost2Runner
from BLRun.grisliRunner import GRISLIRunner
from BLRun.grnvbemRunner import GRNVBEMRunner
from BLRun.jump3Runner import JUMP3Runner
from BLRun.leapRunner import LEAPRunner
from BLRun.pidcRunner import PIDCRunner
from BLRun.ppcorRunner import PPCORRunner
from BLRun.scodeRunner import SCODERunner
from BLRun.scribeRunner import SCRIBERunner
from BLRun.scsglRunner import SCSGLRunner
from BLRun.sinceritiesRunner import SINCERITIESRunner
from BLRun.singeRunner import SINGERunner
from BLRun.pearsonRunner import PearsonRunner

RUNNERS = {
    'GENIE3':       GENIE3Runner,
    'GRNBOOST2':    GRNBoost2Runner,
    'GRISLI':       GRISLIRunner,
    'GRNVBEM':      GRNVBEMRunner,
    'JUMP3':        JUMP3Runner,
    'LEAP':         LEAPRunner,
    'PEARSON':      PearsonRunner,
    'PIDC':         PIDCRunner,
    'PPCOR':        PPCORRunner,
    'SCODE':        SCODERunner,
    'SCRIBE':       SCRIBERunner,
    'SCSGL':        SCSGLRunner,
    'SINCERITIES':  SINCERITIESRunner,
    'SINGE':        SINGERunner,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='BLRunner: Run GRN inference algorithms using BEELINE.'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to the configuration file used to run the inference algorithms.'
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Process runners in reverse order instead of forward. Run one instance '
             'without this flag and one with it to split the work across two '
             'invocations that meet in the middle.'
    )
    parser.add_argument(
        '--skip-populated',
        action='store_true',
        dest='skip_populated',
        help='Skip dataset/algorithm combinations whose working_dir already exists '
             'and is non-empty, instead of overwriting them. Combine with --reverse '
             'to run two invocations that split only the remaining (not-yet-run) work.'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Overwrite populated working directories without the interactive '
             'confirmation prompt. Required when running non-interactively (e.g. '
             'under sbatch/SLURM), since there is no terminal there to answer the '
             'prompt -- use this or --skip-populated, not the bare overwrite prompt.'
    )
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_datasets(input_settings):
    """
    Return a flat list of dataset dicts from input_settings.

    Each dataset entry in the config becomes a separate runnable dataset, with
    each run forming a flat entry keyed by run_id. The dataset_id is used
    directly as the path segment beneath input_dir.

    If a dataset has 'scan_run_subdirectories: true', subdirectories of
    input_dir/dataset_id/ are discovered at runtime and used as runs instead
    of an explicit 'runs' list.

    If 'datasets' is absent, returns an empty list (caller handles auto-discovery).
    """
    if 'datasets' not in input_settings:
        return []

    datasets = []
    input_dir = Path.cwd() / input_settings['input_dir']

    for ds in input_settings['datasets']:
        if not ds.get('should_run', [True])[0]:
            continue

        if ds.get('scan_run_subdirectories'):
            # Discover runs by scanning subdirectories of the dataset input path.
            # ds_input_path : Path — input_dir/dataset_id/
            ds_input_path = input_dir / ds['dataset_id']
            if not ds_input_path.is_dir():
                raise FileNotFoundError(
                    f"scan_run_subdirectories is set for dataset '{ds['dataset_id']}' "
                    f"but input directory '{ds_input_path}' does not exist."
                )
            runs = [{'run_id': d.name} for d in sorted(ds_input_path.iterdir()) if d.is_dir()]
            if not runs:
                raise RuntimeError(
                    f"scan_run_subdirectories is set for dataset '{ds['dataset_id']}' "
                    f"but no subdirectories were found in '{ds_input_path}'."
                )
        else:
            if 'runs' not in ds:
                raise KeyError(f"Dataset '{ds['dataset_id']}' is missing required 'runs' field.")
            runs = ds['runs']
        # runs may be a single dict or a list of dicts
        if isinstance(runs, dict):
            runs = [runs]

        for run in runs:
            datasets.append({
                'dataset_id':        ds['dataset_id'],
                'run_id':            run['run_id'],
                'exprData':          run.get('exprData', 'ExpressionData.csv'),
                'pseudoTimeData':    run.get('pseudoTimeData', 'PseudoTime.csv'),
                'groundTruthNetwork': ds.get('groundTruthNetwork', 'GroundTruthNetwork.csv'),
            })

    return datasets


def build_runner(algo_name, image, params, dataset, input_settings, output_settings):
    if algo_name not in RUNNERS:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(RUNNERS)}")

    runner_config = {
        'input': {
            'input_dir': input_settings['input_dir'],
        },
        'dataset': {
            'dataset_id':          dataset['dataset_id'],
            'run_id':              dataset['run_id'],
            'exprData':            dataset['exprData'],
            'pseudoTimeData':      dataset['pseudoTimeData'],
            'groundTruthNetwork':  dataset['groundTruthNetwork'],
        },
        'output_settings': {
            'output_dir':      output_settings['output_dir'],
            'experiment_id':   output_settings.get('experiment_id', ''),
        },
        'algo_name': algo_name,
        'image': image,
        'params': params,
    }

    return RUNNERS[algo_name](Path.cwd(), runner_config)


def enumerate_dataset_algo_pairs(config):
    """
    Yield (dataset, algo, working_dir) for every enabled dataset/algorithm
    combination, in the canonical build order.

    Single source of truth for both get_working_dirs (path-only, no Runner
    side effects -- safe to call before deciding what to build) and
    build_runners (which must avoid constructing a Runner for anything meant
    to be skipped, since Runner.__init__ erases its working_dir).

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary.

    Yields
    ------
    (dict, dict, Path)
        dataset entry, algo entry, and that combination's working_dir path.
    """
    root            = Path.cwd()
    input_settings  = config['input_settings']
    output_settings = config['output_settings']
    experiment_id   = output_settings.get('experiment_id', '')
    output_dir      = Path(output_settings['output_dir'])
    datasets        = get_datasets(input_settings)
    algorithms      = input_settings.get('algorithms', [])

    for dataset in datasets:
        for algo in algorithms:
            if not algo.get('should_run', [False])[0]:
                continue
            base_output = output_dir if output_dir.is_absolute() else root / output_dir
            if experiment_id:
                base_output = base_output / experiment_id
            base_output = base_output / dataset['dataset_id'] / dataset['run_id'] / algo['algorithm_id']
            yield dataset, algo, base_output / 'working_dir'


def build_runners(config, skip_populated=False):
    """
    Build Runner objects for every enabled dataset/algorithm combination.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary.
    skip_populated : bool, default False
        If True, don't build (and don't erase the working_dir of) any
        combination whose rankedEdges.csv already exists -- i.e. resume
        rather than overwrite. Checked before construction, since
        Runner.__init__ erases its working_dir as a side effect.

        Deliberately keyed on rankedEdges.csv (the actual parsed inference
        output), not on working_dir being non-empty: working_dir fills up
        with intermediate files (copied ExpressionData.csv, partial
        algorithm output, etc.) as soon as a run starts, so a run killed
        mid-way (e.g. OOM) leaves a non-empty working_dir with no
        rankedEdges.csv -- checking working_dir alone would treat that as
        "already done" and skip it forever instead of resuming it.

    Returns
    -------
    list of Runner
    """
    input_settings  = config['input_settings']
    output_settings = config['output_settings']

    runners = []
    for dataset, algo, working_dir in enumerate_dataset_algo_pairs(config):
        if skip_populated and (working_dir.parent / 'rankedEdges.csv').exists():
            continue
        params = algo.get('params', {})
        runners.append(build_runner(algo['algorithm_id'], algo['image'], params, dataset, input_settings, output_settings))
    return runners

def get_working_dirs(config):
    """
    Compute expected working_dir paths from config without constructing Runner objects.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary.

    Returns
    -------
    list of Path
        One working_dir path per enabled dataset/algorithm combination.
    """
    return [working_dir for _, _, working_dir in enumerate_dataset_algo_pairs(config)]

def warn_if_populated(working_dirs):
    """
    Warn and prompt the user if any working directory already contains files.

    Parameters
    ----------
    working_dirs : list of Path
        Working directory paths to check for existing content.

    Returns
    -------
    bool
        True if the user confirms they want to proceed, False otherwise.
    """
    populated_dirs = [p for p in working_dirs if p.exists() and any(p.iterdir())]
    n_populated = len(populated_dirs)
    if n_populated == 0:
        return True
    print(f"Example populated working dir: {populated_dirs[0]}")

    prompt = (
        f"Warning: {n_populated} working director{'y' if n_populated == 1 else 'ies'} "
        f"already exist and will be overwritten. Proceed? [y/n]: "
    )
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        raise RuntimeError(
            f"{prompt}\nNo interactive terminal available to answer this prompt "
            "(e.g. running under sbatch/SLURM). Re-run with --yes to overwrite "
            "without prompting, or --skip-populated to skip already-populated "
            "directories instead."
        ) from None
    return answer == 'y'

def main():
    args = parse_args()
    config = load_config(args.config)

    if args.skip_populated:
        runners = build_runners(config, skip_populated=True)
    else:
        if not args.yes and not warn_if_populated(get_working_dirs(config)):
            print("Aborted.")
            return
        runners = build_runners(config)

    if args.reverse:
        runners = runners[::-1]

    failures = []
    for runner in tqdm(runners):
        tqdm.write(runner.running_message)
        try:
            runner.generateInputs()
            runner.run()
            runner.parseOutput()
        except Exception as e:
            failures.append(runner.running_message)
            tqdm.write(f"FAILED: {runner.running_message}\n  {type(e).__name__}: {e}")

    if failures:
        print(f"\n{len(failures)} of {len(runners)} run(s) failed:")
        for message in failures:
            print(f"  - {message}")

if __name__ == '__main__':
    main()
