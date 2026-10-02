"""
Single source of truth for every path referenced across this repo. Nothing
outside this module should hardcode an absolute path, insert onto
sys.path, or hand-type an output directory. See the root README
for installation and environment-variable configuration.
"""

import os
from datetime import datetime
from pathlib import Path


def get_repo_root() -> Path:
    """
    Root directory of the TwINFER repository.

    Resolved by walking up from this file's own location until a directory
    containing package/pyproject.toml (the repo root's marker) is found.
    This only works for an editable install (`pip install -e package/`),
    since this file must resolve to
    <repo_root>/package/twinfer/utils/paths.py; a regular (non-editable)
    install copies the source into site-packages, severing the link back to
    the repo. Set TWINFER_REPO_ROOT to bypass detection entirely, e.g. when
    twinfer is installed non-editably.

    Returns:
        Path: TWINFER_REPO_ROOT if set, otherwise the detected repository root.

    Raises:
        RuntimeError: If no TWINFER_REPO_ROOT is set and no repo root marker
            is found among this file's parent directories.
    """
    override = os.environ.get("TWINFER_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    start = Path(__file__).resolve()
    for candidate in start.parents:
        if (candidate / "package" / "pyproject.toml").is_file():
            return candidate

    raise RuntimeError(
        f"Could not locate the TwINFER repository root from {start}. "
        "This typically happens when twinfer is installed as a regular (non-editable) "
        "package, so its source no longer lives inside the repo. Either run "
        "`pip install -e package/` from the repo root, or set TWINFER_REPO_ROOT explicitly."
    )


def get_data_root() -> Path:
    """
    Root directory for repository data.

    Defaults to <repo_root's parent's parent>/analysis_data, i.e. a sibling
    of the code/ directory that also holds this repo and the Beeline/BoolODE
    sibling repos (get_external_repo_path) -- matching the layout this
    project already uses on disk, without hardcoding a user-specific path.

    Returns:
        Path: TWINFER_DATA_ROOT if set, otherwise
            <repo_root>/../../analysis_data.
    """
    override = os.environ.get("TWINFER_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return get_repo_root().parent.parent / "analysis_data"


def stage_dir(figure_name: str, stage: str, run_tag: str | None = None, *, make_latest: bool = True) -> Path:
    """
    Output directory for one stage of one figure's pipeline.

    Returns <data_root>/paper_analysis/<figure_name>/<stage>/<run_tag>/.

    Args:
        figure_name (str): Figure or analysis identifier, e.g. "figure_2".
        stage (str): Pipeline stage, e.g. "simulation", "analysis", "plot".
        run_tag (str, optional): Which run to use.
            - None (default): generate a new timestamped run
              (YYYYMMDD_HHMMSS) and repoint <stage>/latest at it, unless
              make_latest=False. Reads the TWINFER_RUN_TAG environment
              variable first if set, so a runner script can pin every
              stage of one pipeline invocation to the same tag.
            - "latest": resolve the existing <stage>/latest symlink.
            - any other string: used verbatim, e.g. to reproduce or
              inspect a specific past run.
        make_latest (bool, optional): Update the latest symlink after
            creating a new run_tag directory. Defaults to True. Ignored
            when run_tag="latest".

    Returns:
        Path: The resolved stage directory.

    Raises:
        FileNotFoundError: If run_tag="latest" and no run exists yet.
    """
    base = get_data_root() / "paper_analysis" / figure_name / stage
    latest_link = base / "latest"

    if run_tag == "latest":
        if not latest_link.exists():
            raise FileNotFoundError(
                f"No run found yet for {figure_name}/{stage} -- run the earlier stage first, "
                f"or pass an explicit run_tag."
            )
        return latest_link.resolve()

    if run_tag is None:
        run_tag = os.environ.get("TWINFER_RUN_TAG") or datetime.now().strftime("%Y%m%d_%H%M%S")

    path = base / run_tag
    path.mkdir(parents=True, exist_ok=True)

    if make_latest:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path.name)

    return path


def get_external_repo_path(name: str) -> Path:
    """
    Path to a sibling repository this project depends on but does not
    vendor. Currently BEELINE and BoolODE, used by paper_analysis/benchmark/.

    Args:
        name (str): "beeline" or "boolode" (case-insensitive).

    Returns:
        Path: TWINFER_<NAME>_PATH if set, otherwise
            <repo_root's parent>/<Beeline|BoolODE>.

    Raises:
        ValueError: If name is not a recognized external repo.
    """
    env_var = f"TWINFER_{name.upper()}_PATH"
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()

    default_siblings = {"beeline": "Beeline", "boolode": "BoolODE"}
    key = name.lower()
    if key not in default_siblings:
        raise ValueError(f"Unknown external repo {name!r}; set {env_var} explicitly.")
    return get_repo_root().parent / default_siblings[key]


def get_larry_dataset_dir() -> Path:
    """
    Working directory of the LARRY preprocessing pipeline (raw per-library zips + barcode fastq + processed/ QC outputs).
    [2026-10-01 added: replaces the literal /scratch/gzu5140/ka_twinfer/larry_dataset in ~25 scripts.]

    Returns:
        Path: TWINFER_LARRY_DATASET if set, otherwise /scratch/gzu5140/ka_twinfer/larry_dataset (the location the pipeline was
            run in; scratch storage is purged, so set TWINFER_LARRY_DATASET to a shared-storage copy to keep results).
    """
    override = os.environ.get("TWINFER_LARRY_DATASET")
    if override:
        return Path(override).expanduser()
    return Path("/scratch/gzu5140/ka_twinfer/larry_dataset")
