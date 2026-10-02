#!/usr/bin/env python3
"""TODO4v2 (apply_todo4v2_vs_collectri.py's formula, unchanged) on the FM06 A/B split (t1 = A, t2 = B), every ordered pair of a gene set, vs CollecTRI.
   score = z_abs_rho_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s(|z_dagger|), gate: max(z_abs_t1, z_abs_t2) > 2.576 and z_reg_gated > 2.326
Inputs  (analysis_data/fm06/data/)
   ranked_edges_fm06_<gs>_absplit_allpairs.csv   all-pairs A/B inference (relaxed gates): the z columns for every ordered pair
   z_reg_gated_fm06_<gs>_absplit.json            TwINFER's gated regulation z (gated A/B run, run_infer_fatemap_ab_split.py); a pair missing there fails the gate
   z_dagger_fm06_<gs>_absplit_allpairs.json      empirical-null z of rho_cross(x->y) for every ordered pair
Writes  twinscore_spec/todo4v2_absplit_fm06_<gs>_scores.csv  (gene_1, gene_2, TODO4v2, gated)  and prints AUPRC / AUPRC-x / hits, all pairs and source-limited.
usage: apply_todo4v2_absplit_fm06.py <gene_set>
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
from paper_analysis.fatemap_pipeline.apply_todo4v2_vs_collectri import COLLECTRI_PATH, full_report, todo4v2_score  # noqa: E402

BASE = f'{TWINFER_PROJECT_ROOT}'
D = f"{BASE}/analysis_data/fm06/data"
gs = sys.argv[1]
dd = pd.read_csv(f"{D}/ranked_edges_fm06_{gs}_absplit_allpairs.csv")
dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
zreg = json.load(open(f"{D}/z_reg_gated_fm06_{gs}_absplit.json"))
zdag = json.load(open(f"{D}/z_dagger_fm06_{gs}_absplit_allpairs.json"))
score, n_gate = todo4v2_score(dd, zreg, zdag)
ct = pd.read_csv(COLLECTRI_PATH, sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE, sources = set(zip(ct.source_genesymbol, ct.target_genesymbol)), set(ct.source_genesymbol)
y = np.array([int(p in CE) for p in zip(dd.gene_1, dd.gene_2)])
out = dd[["gene_1", "gene_2"]].copy()
out["TODO4v2"], out["gated"], out["collectri_edge"] = score, np.isfinite(score), y
os.makedirs(f"{D}/twinscore_spec", exist_ok=True)
out.to_csv(f"{D}/twinscore_spec/todo4v2_absplit_fm06_{gs}_scores.csv", index=False)
print(f"{gs}: {len(dd)} ordered pairs, {int(y.sum())} true; pass the gate: {n_gate}; z_reg_gated entries {len(zreg)}, z_dagger entries {len(zdag)}")
for uname, mask in (("all pairs", np.ones(len(dd), bool)), ("source-limited", dd.gene_1.isin(sources).to_numpy())):
    sc, yy = score[mask], y[mask]
    m = full_report(sc, yy)
    g = np.isfinite(sc)
    print(f"  {uname}: {mask.sum()} pairs, {int(yy.sum())} true | AUPRC {m['auprc']:.4f}  AUPRC-x {m['auprc_x']:.3f}  hits@k {m['tp']}/{m['k']}  | gate passes {int(g.sum())}: precision {yy[g].mean() if g.any() else float('nan'):.3f}  recall {yy[g].sum() / yy.sum():.3f}")
