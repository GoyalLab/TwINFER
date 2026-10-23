"""Single source of truth for where a dataset file lives under datasets/.

datasets/ is grouped by arm, one subfolder per dataset:

    datasets/celltag/              the CellTag arm
    datasets/larry/                the LARRY arm
    datasets/shared/               files used by both
    datasets/twinfer_2026scrnaseq/ the 2026 scRNAseq run (raw data only: fastq, bcl, cellranger)

Scripts receive a dataset as a BARE FILENAME, usually through an environment variable
(SOURCE, COUNTS, CT_SOURCE, ...), and used to build the path as
os.path.join(ROOT, "datasets", NAME). That only resolves while the flat top-level names still
exist. Call dpath(NAME) instead, so the grouping is defined once here.

Resolution order:
  1. datasets/<name> if it exists  -- the back-compat symlinks, and any file not yet grouped
  2. datasets/<group>/<name> for each group, if it exists
  3. for a file that does not exist yet (an OUTPUT being written), the group is derived from
     the name prefix

An unmappable name raises rather than silently returning a path that nothing will read.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATASETS = os.path.join(ROOT, "datasets")

# EVERY subfolder of datasets/ is a group. The list used to be written out by hand and went stale as
# arms were added: michaels2024, twinfer2026, twinfer2026_run1plus2 and
# twinfer_2026scrnaseq_run1plus2 all existed on disk while dpath still refused to look in them.
GROUPS = tuple(sorted(d for d in os.listdir(DATASETS) if os.path.isdir(os.path.join(DATASETS, d)))) \
    if os.path.isdir(DATASETS) else ("celltag", "larry", "shared", "twinfer_2026scrnaseq")

# Prefix -> group, for outputs that do not exist yet. Derived from the naming already in use;
# it is a fallback only, never consulted while the file itself can be found on disk.
PREFIXES = (
    ("celltag_", "celltag"),
    ("larry_",   "larry"),
    ("LSK_",     "larry"),
)


def group_for(name):
    """The subfolder a dataset name belongs to, or None if the name maps to no group."""
    base = os.path.basename(name)
    for pre, g in PREFIXES:
        if base.startswith(pre):
            return g
    return None


def dpath(name, *parts):
    """Absolute path to a dataset file, wherever its group folder puts it.

    dpath("LSK_d2_d4_d6.h5ad")            -> .../datasets/larry/LSK_d2_d4_d6.h5ad
    dpath("celltag_nosinglet_counts.h5ad")-> .../datasets/celltag/celltag_nosinglet_counts.h5ad
    """
    if parts:
        name = os.path.join(name, *parts)
    if os.path.isabs(name):
        return name
    # ALREADY CARRIES ITS GROUP, with or without the leading "datasets/". A caller that passes the
    # repo-relative path it reads in the sbatch -- datasets/larry/x.h5ad -- used to fall past this
    # test, because the first component is "datasets" and not a group name; the prefix fallback then
    # returned datasets/larry/datasets/larry/x.h5ad and every CellTag competitor in job 4925320 died
    # on FileNotFoundError against that doubled path.
    parts_ = name.split(os.sep)
    if parts_ and parts_[0] == os.path.basename(DATASETS):
        parts_ = parts_[1:]
        name = os.path.join(*parts_) if parts_ else name
    if parts_ and parts_[0] in GROUPS:
        return os.path.join(DATASETS, name)
    flat = os.path.join(DATASETS, name)
    if os.path.exists(flat):
        return flat
    for g in GROUPS:
        p = os.path.join(DATASETS, g, name)
        if os.path.exists(p):
            return p
    g = group_for(name)
    if g is None:
        raise ValueError(
            f"cannot map dataset {name!r} to a group under datasets/. "
            f"Groups: {', '.join(GROUPS)}. Add a prefix rule in helpers/dataset_paths.py."
        )
    return os.path.join(DATASETS, g, name)


def dfind(name, *parts):
    """Like dpath, but returns None instead of raising when the name maps to no group.

    For callers that are ASKING whether a file is there (existence tests, currency scans),
    where an unmappable name is an answer of "not found" rather than an error.
    """
    try:
        p = dpath(name, *parts)
    except ValueError:
        return None
    return p if os.path.exists(p) else None
