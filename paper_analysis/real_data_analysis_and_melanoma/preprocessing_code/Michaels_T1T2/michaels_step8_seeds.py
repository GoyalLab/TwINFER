#!/usr/bin/env python3
"""Step 8 of yscher's michaels_barcodes.py (min-cut / degree pruning) depends on PYTHONHASHSEED. Run michaels_step8_only.py for seeds 0-19 with yscher's interpreter
(networkx 3.4.2) and compare each result with yscher's cell_to_clone.csv: clones, cells, identical rows, Adjusted Rand Index on the shared cells.
Writes <REBUILD>/Michaels_T1T2/michaels_rerun/step8_seeds.json  (run with the python that has sklearn; the per-seed runs use raw_data/external_tools/mamba_python3.10)"""
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import MAMBA_PYTHON, MICHAELS_CLONES, REBUILD      # noqa: E402
OUT = os.path.join(REBUILD, "Michaels_T1T2", "michaels_rerun")
os.makedirs(f"{OUT}/seeds", exist_ok=True)
YPY = MAMBA_PYTHON                                  # python 3.10 of the shared mamba (raw_data/external_tools link), as used for yscher's script
Y = MICHAELS_CLONES
ref = {r[0]: r[2] for r in list(csv.reader(open(Y)))[1:]}


def run(seed):
    out = f"{OUT}/seeds/seed_{seed}.csv"
    p = subprocess.run([YPY, f"{HERE}/michaels_step8_only.py", out], env={**os.environ, "PYTHONHASHSEED": str(seed)}, capture_output=True, text=True)
    rows = list(csv.reader(open(out)))[1:]
    mine = {r[0]: r[2] for r in rows}
    shared = sorted(set(mine) & set(ref))
    ari = adjusted_rand_score([ref[c] for c in shared], [mine[c] for c in shared])
    return dict(seed=seed, clones=len({r[2] for r in rows}), cells=len(rows), shared_cells=len(shared), only_rerun=len(set(mine) - set(ref)), only_yscher=len(set(ref) - set(mine)),
                ARI_shared=round(ari, 5), rows_identical_to_yscher="rows identical incl. ids and order: True" in p.stdout, networkx=p.stdout.split()[1] if p.stdout else None)


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(run, range(20)))
summ = dict(yscher_clones=len({v for v in ref.values()}), yscher_cells=len(ref), seeds=res, clones_min=min(r["clones"] for r in res), clones_max=max(r["clones"] for r in res),
            cells_min=min(r["cells"] for r in res), cells_max=max(r["cells"] for r in res), ARI_min=min(r["ARI_shared"] for r in res), ARI_max=max(r["ARI_shared"] for r in res),
            seeds_identical_to_yscher=sum(r["rows_identical_to_yscher"] for r in res))
json.dump(summ, open(f"{OUT}/step8_seeds.json", "w"), indent=1)
print({k: v for k, v in summ.items() if k != "seeds"})
