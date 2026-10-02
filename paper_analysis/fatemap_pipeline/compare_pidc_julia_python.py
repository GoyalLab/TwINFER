#!/usr/bin/env python3
"""Compare the Julia PIDC (NetworkInference.jl) with the Python PIDC competitor (run_competitors_fatemap.py) on FM06.
usage: compare_pidc_julia_python.py [gene_set]     (default correlation_high)

Julia output = tab-separated, one row per UNORDERED gene pair (Gene1, Gene2, weight); it is mirrored to both directions.
Python output = <BASE>/analysis_data/fm06/data/networks/pidc_<gs>_allgenes.csv (directed, TF/target/importance).
Reports rank agreement and, against CollecTRI, AUPRC / AUPRC-x / true hits at k for Julia PIDC, Python PIDC and the final TwinFER score,
in the all-pairs and CollecTRI-source universes; and the run time / peak memory from time.txt.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline.final_fm06_tables import COLLECTRI, SCORED, metrics  # noqa: E402

BASE = f'{TWINFER_PROJECT_ROOT}'
gs = sys.argv[1] if len(sys.argv) > 1 else "correlation_high"
J = f"{BASE}/analysis_data/fm06/data/pidc_julia/{gs}"

p = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
py = pd.read_csv(f"{BASE}/analysis_data/fm06/data/networks/pidc_{gs}_allgenes.csv")
pyd = {(a, b): v for a, b, v in zip(py.TF, py.target, py.importance)}
jl = pd.read_csv(f"{J}/outFile.txt", sep="\t", header=None, names=["g1", "g2", "w"])
jd = {}
for a, b, v in zip(jl.g1, jl.g2, jl.w):
    jd[(a, b)] = v
    jd[(b, a)] = v
p["PIDC_py"] = [pyd.get((a, b), 0.0) for a, b in zip(p.gene_1, p.gene_2)]
p["PIDC_jl"] = [jd.get((a, b), np.nan) for a, b in zip(p.gene_1, p.gene_2)]
print(f"[{gs}] pairs {len(p)}; Julia pairs found {int(p.PIDC_jl.notna().sum())}; Julia unordered edges {len(jl)}")
p["PIDC_jl"] = p.PIDC_jl.fillna(0.0)

print("\n== rank agreement ==")
print("Spearman Julia (mirrored) vs Python, all ordered pairs: %.3f" % spearmanr(p.PIDC_jl, p.PIDC_py)[0])
u = p[p.gene_1 < p.gene_2].copy()
rev = p.set_index(["gene_1", "gene_2"]).PIDC_py
u["py_mean"] = [(pyd.get((a, b), 0) + pyd.get((b, a), 0)) / 2 for a, b in zip(u.gene_1, u.gene_2)]
print("Spearman Julia vs Python (mean of the two directions), unordered pairs: %.3f" % spearmanr(u.PIDC_jl, u.py_mean)[0])
sym = spearmanr([pyd.get((a, b), 0) for a, b in zip(u.gene_1, u.gene_2)], [pyd.get((b, a), 0) for a, b in zip(u.gene_1, u.gene_2)])[0]
print("Python PIDC: Spearman between the two directions of a pair: %.3f" % sym)
k = int(p.collectri_edge.sum())
top = lambda c: set(p.assign(pair=p.gene_1 + "->" + p.gene_2).nlargest(k, c).pair)
print(f"top-{k} directed pairs shared, Julia vs Python: {len(top('PIDC_jl') & top('PIDC_py'))}; "
      f"Julia vs TwinFER: {len(top('PIDC_jl') & top('TwinScore'))}; Python vs TwinFER: {len(top('PIDC_py') & top('TwinScore'))}")
print("Spearman with the in-pipeline D term: Julia %.3f, Python competitor %.3f" % (spearmanr(p.PIDC_jl, p.D)[0], spearmanr(p.PIDC_py, p.D)[0]))

ct = pd.read_csv(COLLECTRI, sep="\t")
sources = set(ct[ct.source_genesymbol != ct.target_genesymbol].source_genesymbol)
y = p.collectri_edge.to_numpy()
print("\n== accuracy against CollecTRI ==")
rows = []
for uname, mask in (("all pairs", np.ones(len(p), bool)), ("CollecTRI-source pairs", p.gene_1.isin(sources).to_numpy())):
    for name, col in (("Julia PIDC (mirrored)", "PIDC_jl"), ("Python PIDC", "PIDC_py"), ("TwinFER (final)", "TwinScore")):
        m = metrics(p[col].to_numpy()[mask], y[mask])
        rows.append((uname, name, round(m["AUPRC"], 4), round(m["AUPRC_x"], 3), m["TP_at_k"], m["k"], round(m["AUROC"], 3)))
print(pd.DataFrame(rows, columns=["universe", "method", "AUPRC", "AUPRC-x", "true hits at k", "k", "AUROC"]).to_string(index=False))

print("\n== usage of the Julia run ==")
try:
    t = open(f"{J}/time.txt").read()
    for key in ("Elapsed (wall clock)", "Maximum resident set size", "User time", "System time", "Percent of CPU"):
        m = re.search(rf"{re.escape(key)}[^:]*:\s*(.*)", t)
        print(f"  {key}: {m.group(1).strip() if m else 'n/a'}")
except FileNotFoundError:
    print("  time.txt not found")
