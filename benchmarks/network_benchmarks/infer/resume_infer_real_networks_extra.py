"""Resume pass for infer_real_networks_extra.py: only (re)runs tasks whose
fanout_on/fanout_off JSON pair isn't already a *regenerated* (current-code)
record, so a partial/interrupted run can be continued without redoing
completed work.

NOTE: for the old 4 networks (GSD/HSC/VSC/mCAD) the target files already
existed before this script ever ran (in-place overwrite of the legacy
JSONs) -- so mere file *existence* is not a valid "already done" signal.
Instead this checks for the "gene_names" key, which only the current code
(infer_real_networks_extra.run_one) writes; the legacy generator's JSONs
never had it.
"""
import json
import os
import warnings
from collections import Counter

from joblib import Parallel, delayed

from benchmarks.network_benchmarks.infer import infer_real_networks_extra as m

warnings.filterwarnings("ignore")


def is_regenerated(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
        return "gene_names" in d
    except Exception:
        return False


def main():
    tasks = m.build_tasks()
    remaining = [
        t for t in tasks
        if not (
            is_regenerated(os.path.join(m.FANOUT_ON, f"{t[1]}_all_results.json"))
            and is_regenerated(os.path.join(m.FANOUT_OFF, f"{t[1]}_all_results.json"))
        )
    ]
    print("remaining tasks:", len(remaining), flush=True)
    print(Counter(t[0] for t in remaining), flush=True)

    results = Parallel(n_jobs=4, backend="loky")(
        delayed(m.run_one)(net, analysis_key, raw_csv_path, base_config)
        for net, analysis_key, raw_csv_path, base_config in remaining
    )
    n_ok = sum(1 for r in results if r.endswith("OK"))
    print(f"done: {n_ok} ok, {len(results) - n_ok} failed", flush=True)
    for r in results:
        if not r.endswith("OK"):
            print(r, flush=True)


if __name__ == "__main__":
    main()
