"""
Compress the before_division checkpoint files for cyclic (g3/g4/g5/cyclic_6_nodes)
and autoregulation -- unlike network_sweep_final, every one of these matches a
kept run 1:1 (no retries happened in either benchmark), so nothing is deleted
here, only compressed with the same compress-verify-then-remove-original
approach as the network_sweep_final cleanup.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import gzip
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DIRS = [
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g2_pos',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g2_neg',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g3_pos',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/autoreg_g3_neg',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/cyclic_g3',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/cyclic_g4',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/synthetic_network/cyclic_g5',
    f'{TWINFER_PROJECT_ROOT}/simulation_data/cyclic_6_nodes',
]


def compress_and_verify(path_str):
    path = Path(path_str)
    gz_path = path.with_suffix(path.suffix + ".gz")
    orig_size = path.stat().st_size
    try:
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024 * 16)
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
    files = []
    for d in DIRS:
        files.extend(Path(d).glob("simulation_before_division_*.csv"))
    print(f"{len(files)} before_division files to compress across {len(DIRS)} directories")

    n_ok, n_failed = 0, 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(compress_and_verify, str(f)): f for f in files}
        for i, fut in enumerate(as_completed(futures)):
            path_str, status = fut.result()
            ok = status.startswith("OK")
            n_ok += ok
            n_failed += not ok
            print(f"[{i+1}/{len(files)}] {Path(path_str).name}: {status}", flush=True)

    print(f"\nDone: {n_ok} OK, {n_failed} FAILED (left uncompressed)")


if __name__ == "__main__":
    main()
