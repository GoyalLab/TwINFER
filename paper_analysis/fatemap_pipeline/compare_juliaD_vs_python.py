#!/usr/bin/env python3
"""FM06: what changes when PIDC is the Julia implementation (NetworkInference.jl) instead of the Python port.
usage: compare_juliaD_vs_python.py

(a) TwinFER rescored with the Julia PIDC as the D term  (twinscore_supp_gated_bootstrap_juliaD/)  vs  the final TwinFER (Python D)
(b) the PIDC competitor: Julia (pidc_julia/<gs>/outFile.txt, mirrored to both directions) vs Python (networks/pidc_<gs>_allgenes.csv)
    and the best competitor recomputed with each.
Writes <BASE>/analysis_data/fatemap_comparison/data/final_fm06/juliaD_vs_python_{summary,competitor}.csv and prints the tables.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline.final_fm06_tables import COLLECTRI, COMPETITORS, GENE_SETS, OUT, SCORED, competitor_scores, metrics  # noqa: E402

BASE = f'{TWINFER_PROJECT_ROOT}'
JD = f"{BASE}/analysis_data/fm06/data/twinscore_supp_gated_bootstrap_juliaD"
ct = pd.read_csv(COLLECTRI, sep="\t")
sources = set(ct[ct.source_genesymbol != ct.target_genesymbol].source_genesymbol)
rows, crow = [], []
for gs in GENE_SETS:
    f_new = f"{JD}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv"
    try:
        new = pd.read_csv(f_new)
    except FileNotFoundError:
        print(f"[{gs}] not finished yet"); continue
    old = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
    assert (old.gene_1 == new.gene_1).all() and (old.gene_2 == new.gene_2).all()
    dg_o = json.load(open(f"{SCORED}/run_params_fm06_{gs}_absplit_gated_bootstrap.json"))["diagnostics"]
    dg_n = json.load(open(f"{JD}/run_params_fm06_{gs}_absplit_gated_bootstrap.json"))["diagnostics"]
    y = old.collectri_edge.to_numpy()
    comp = competitor_scores(gs, old)
    jl = pd.read_csv(f"{BASE}/analysis_data/fm06/data/pidc_julia/{gs}/outFile.txt", sep="\t", header=None, names=["a", "b", "w"])
    jd = {}
    for a, b, v in zip(jl.a, jl.b, jl.w):
        jd[(a, b)] = v; jd[(b, a)] = v
    comp_j = dict(comp)
    comp_j["pidc"] = np.array([jd.get((a, b), 0.0) for a, b in zip(old.gene_1, old.gene_2)])
    for uname, mask in (("allpairs", np.ones(len(old), bool)), ("collectri_sources", old.gene_1.isin(sources).to_numpy())):
        yy = y[mask]
        m_o, m_n = metrics(old.TwinScore.to_numpy()[mask], yy), metrics(new.TwinScore.to_numpy()[mask], yy)
        res_py = {m: metrics(comp[m][mask], yy) for m in COMPETITORS}
        res_jl = {m: metrics(comp_j[m][mask], yy) for m in COMPETITORS}
        bp = max(COMPETITORS, key=lambda m: res_py[m]["AUPRC_x"]); bj = max(COMPETITORS, key=lambda m: res_jl[m]["AUPRC_x"])
        k = int(yy.sum())
        top = lambda s_: set(np.argsort(-np.asarray(s_)[mask], kind="stable")[:k])
        rows.append(dict(gene_set=gs, universe=uname, n_true=k, w_python_D=round(dg_o["w"], 4), w_julia_D=round(dg_n["w"], 4),
                         same_gated_genes=set(dg_o["gated_genes"]) == set(dg_n["gated_genes"]),
                         spearman_D_python_vs_julia=round(spearmanr(old.D[mask], new.D[mask])[0], 3),
                         spearman_score_old_vs_new=round(spearmanr(old.TwinScore[mask], new.TwinScore[mask])[0], 3),
                         topk_overlap_old_new=len(top(old.TwinScore) & top(new.TwinScore)),
                         TwinFER_pyD_AUPRC=round(m_o["AUPRC"], 4), TwinFER_pyD_AUPRC_x=round(m_o["AUPRC_x"], 3), TwinFER_pyD_hits=m_o["TP_at_k"],
                         TwinFER_jlD_AUPRC=round(m_n["AUPRC"], 4), TwinFER_jlD_AUPRC_x=round(m_n["AUPRC_x"], 3), TwinFER_jlD_hits=m_n["TP_at_k"],
                         delta_AUPRC_x=round(m_n["AUPRC_x"] - m_o["AUPRC_x"], 3),
                         best_comp_python_PIDC=bp, best_x_python_PIDC=round(res_py[bp]["AUPRC_x"], 3),
                         best_comp_julia_PIDC=bj, best_x_julia_PIDC=round(res_jl[bj]["AUPRC_x"], 3)))
        crow.append(dict(gene_set=gs, universe=uname, python_PIDC_AUPRC_x=round(res_py["pidc"]["AUPRC_x"], 3), python_PIDC_hits=res_py["pidc"]["TP_at_k"],
                         julia_PIDC_AUPRC_x=round(res_jl["pidc"]["AUPRC_x"], 3), julia_PIDC_hits=res_jl["pidc"]["TP_at_k"],
                         spearman_python_vs_julia=round(spearmanr(comp["pidc"][mask], comp_j["pidc"][mask])[0], 3)))
S, C = pd.DataFrame(rows), pd.DataFrame(crow)
S.to_csv(f"{OUT}/juliaD_vs_python_summary.csv", index=False); C.to_csv(f"{OUT}/juliaD_vs_python_competitor.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
for u in ("allpairs", "collectri_sources"):
    x = S[S.universe == u]
    print(f"\n== TwinFER, Python D vs Julia D -- universe {u} ==")
    print(x[["gene_set", "w_python_D", "w_julia_D", "same_gated_genes", "spearman_D_python_vs_julia", "spearman_score_old_vs_new", "topk_overlap_old_new",
             "TwinFER_pyD_AUPRC_x", "TwinFER_pyD_hits", "TwinFER_jlD_AUPRC_x", "TwinFER_jlD_hits", "delta_AUPRC_x"]].to_string(index=False))
    print("mean AUPRC-x: Python D %.3f | Julia D %.3f | delta %.3f (n=%d sets)" % (x.TwinFER_pyD_AUPRC_x.mean(), x.TwinFER_jlD_AUPRC_x.mean(), x.delta_AUPRC_x.mean(), len(x)))
    print(f"-- best competitor (with Python PIDC vs with Julia PIDC) --")
    print(x[["gene_set", "best_comp_python_PIDC", "best_x_python_PIDC", "best_comp_julia_PIDC", "best_x_julia_PIDC"]].to_string(index=False))
    c = C[C.universe == u]
    print(f"-- PIDC competitor, Python vs Julia --"); print(c.drop(columns="universe").to_string(index=False))
