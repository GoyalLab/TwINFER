"""
Cleanup for network_sweep_final's simulation_data directory, per the finalized
150-file manifest (source_manifest_150.json):

1. before_division files matching a kept run -> gzip (verify, THEN delete original)
2. before_division files NOT matching a kept run (retry waste / never-completed) -> delete
3. complete df_*.csv simulation files NOT in the kept 150 (excess retries) -> delete

Nothing is deleted until its replacement (the .gz) is verified, or until it's
confirmed to have no bearing on the kept manifest.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import gzip
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

MANIFEST_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/source_manifest_150.json'
SIM_DIR = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/network_sweep_final')


def load_kept_basenames():
    manifest = json.load(open(MANIFEST_PATH))
    manifest.pop("logs", None)
    kept = set()
    for ds, labels in manifest.items():
        for label, info in labels.items():
            if "source" in info:
                kept.add(Path(info["source"]).name)
    return kept


def compress_and_verify(path_str):
    """gzip one file, verify the .gz decompresses to something the same size as
    the original, and only then delete the original. Returns (path, status)."""
    path = Path(path_str)
    gz_path = path.with_suffix(path.suffix + ".gz")
    orig_size = path.stat().st_size
    try:
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024 * 16)
        # verify: decompressed size must match original exactly
        result = subprocess.run(["gzip", "-t", str(gz_path)], capture_output=True)
        if result.returncode != 0:
            gz_path.unlink(missing_ok=True)
            return path_str, f"FAILED gzip -t: {result.stderr.decode()[:200]}"
        with gzip.open(gz_path, "rb") as f:
            decompressed_size = sum(len(chunk) for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""))
        if decompressed_size != orig_size:
            gz_path.unlink(missing_ok=True)
            return path_str, f"FAILED size mismatch: orig={orig_size} decompressed={decompressed_size}"
        path.unlink()
        return path_str, f"OK ({orig_size/1e9:.2f}GB -> {gz_path.stat().st_size/1e9:.2f}GB)"
    except Exception as e:
        return path_str, f"FAILED exception: {e}"


def main():
    kept = load_kept_basenames()
    before_div = list(SIM_DIR.glob("simulation_before_division_*.csv"))
    matched = [f for f in before_div if f.name.replace("simulation_before_division_", "", 1) in kept]
    unmatched = [f for f in before_div if f.name.replace("simulation_before_division_", "", 1) not in kept]

    all_complete = [f for f in SIM_DIR.glob("df_grn_n6_*_ncells_*.csv")
                    if not f.name.startswith("simulation_before_division")]
    excess_complete = [f for f in all_complete if f.name not in kept]

    print(f"Matched before_division to compress: {len(matched)}")
    print(f"Unmatched before_division to delete: {len(unmatched)}")
    print(f"Excess complete simulation files to delete: {len(excess_complete)}")

    # Step 1: compress + verify + delete-original, matched files, in parallel
    print("\n=== Compressing matched before_division files ===", flush=True)
    n_ok, n_failed = 0, 0
    with ProcessPoolExecutor(max_workers=24) as ex:
        futures = {ex.submit(compress_and_verify, str(f)): f for f in matched}
        for i, fut in enumerate(as_completed(futures)):
            path_str, status = fut.result()
            ok = status.startswith("OK")
            n_ok += ok
            n_failed += not ok
            print(f"[{i+1}/{len(matched)}] {Path(path_str).name}: {status}", flush=True)
    print(f"\nCompression done: {n_ok} OK, {n_failed} FAILED (left uncompressed, not deleted)")

    # Step 2: delete unmatched before_division files
    print("\n=== Deleting unmatched before_division files ===", flush=True)
    for f in unmatched:
        f.unlink()
    print(f"Deleted {len(unmatched)} unmatched before_division files")

    # Step 3: delete excess complete simulation files
    print("\n=== Deleting excess complete simulation files ===", flush=True)
    for f in excess_complete:
        f.unlink()
    print(f"Deleted {len(excess_complete)} excess complete simulation files")

    print("\nCleanup finished.")


if __name__ == "__main__":
    main()
