#!/usr/bin/env python3
"""FM06: three PIDC implementations, as (a) the D term of TwinFER and (b) the PIDC competitor.
  python   = Python PIDC (helpers/pidc.py of yscher, ported in run_competitors_fatemap.py / twinscore_supp_helpers.py); directed scores
  julia_sym = NetworkInference.jl PIDCNetworkInference (run_pidc_julia.jl); one weight per unordered pair, mirrored
  julia_dir = directed variant on the same package internals (run_pidc_julia_directed.jl)
(a) TwinFER scored with each D:  twinscore_supp_gated_bootstrap/ (python), ..._juliaD/ (julia_sym), ..._juliaD_directed/ (julia_dir), all under
    <BASE>/analysis_data/fm06/data/ .
(b) PIDC competitor scores: networks/pidc_<gs>_allgenes.csv (python), pidc_julia/<gs>/outFile.txt (julia_sym), pidc_julia/<gs>/outFile_directed.txt (julia_dir).
Writes final_fm06/julia_pidc_variants_{twinfer,competitor}.csv and prints the tables (AUPRC-x with true hits at k; universes: all pairs / CollecTRI-source pairs).
usage: compare_julia_pidc_variants.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline.final_fm06_tables import COLLECTRI, COMPETITORS, GENE_SETS, OUT, SCORED, competitor_scores, metrics  # noqa: E402

DATA = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
FOLD = {"python": SCORED, "julia_sym": f"{DATA}/twinscore_supp_gated_bootstrap_juliaD", "julia_dir": f"{DATA}/twinscore_supp_gated_bootstrap_juliaD_directed"}
ct = pd.read_csv(COLLECTRI, sep="\t")
sources = set(ct[ct.source_genesymbol != ct.target_genesymbol].source_genesymbol)


def jl_scores(path, p, mirror):
    d = pd.read_csv(path, sep="\t", header=None, names=["a", "b", "w"])
    m = {}
    for a, b, v in zip(d.a, d.b, d.w):
        m[(a, b)] = v
        if mirror:
            m[(b, a)] = v
    return np.array([m.get((a, b), 0.0) for a, b in zip(p.gene_1, p.gene_2)])


tw, cp = [], []
for gs in GENE_SETS:
    ps = {}
    for k, f in FOLD.items():
        try:
            ps[k] = pd.read_csv(f"{f}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
        except FileNotFoundError:
            pass
    base = ps["python"]
    y = base.collectri_edge.to_numpy()
    comp = competitor_scores(gs, base)
    pid = {"python": comp["pidc"], "julia_sym": jl_scores(f"{DATA}/pidc_julia/{gs}/outFile.txt", base, True)}
    try:
        pid["julia_dir"] = jl_scores(f"{DATA}/pidc_julia/{gs}/outFile_directed.txt", base, False)
    except FileNotFoundError:
        pass
    for uname, mask in (("allpairs", np.ones(len(base), bool)), ("collectri_sources", base.gene_1.isin(sources).to_numpy())):
        yy = y[mask]
        r = dict(gene_set=gs, universe=uname, n_true=int(yy.sum()))
        for k, p in ps.items():
            m = metrics(p.TwinScore.to_numpy()[mask], yy)
            r[f"TwinFER_{k}_AUPRC"] = round(m["AUPRC"], 4); r[f"TwinFER_{k}_AUPRC_x"] = round(m["AUPRC_x"], 3); r[f"TwinFER_{k}_hits"] = m["TP_at_k"]
        tw.append(r)
        c = dict(gene_set=gs, universe=uname, n_true=int(yy.sum()))
        others = {m_: metrics(comp[m_][mask], yy)["AUPRC_x"] for m_ in COMPETITORS if m_ != "pidc"}
        for k, s_ in pid.items():
            m = metrics(s_[mask], yy)
            c[f"PIDC_{k}_AUPRC"] = round(m["AUPRC"], 4); c[f"PIDC_{k}_AUPRC_x"] = round(m["AUPRC_x"], 3); c[f"PIDC_{k}_hits"] = m["TP_at_k"]
            best_other = max(others.values())
            c[f"best_competitor_with_PIDC_{k}_AUPRC_x"] = round(max(best_other, m["AUPRC_x"]), 3)
        c["spearman_python_vs_julia_sym"] = round(spearmanr(pid["python"][mask], pid["julia_sym"][mask])[0], 3)
        if "julia_dir" in pid:
            c["spearman_python_vs_julia_dir"] = round(spearmanr(pid["python"][mask], pid["julia_dir"][mask])[0], 3)
        cp.append(c)
T, C = pd.DataFrame(tw), pd.DataFrame(cp)
T.to_csv(f"{OUT}/julia_pidc_variants_twinfer.csv", index=False)
C.to_csv(f"{OUT}/julia_pidc_variants_competitor.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for u in ("allpairs", "collectri_sources"):
    x = T[T.universe == u]
    cols = ["gene_set", "n_true"] + [c for c in x.columns if c.endswith("_AUPRC_x") or c.endswith("_hits")]
    print(f"\n== TwinFER by D implementation, universe {u} ==")
    print(x[cols].to_string(index=False))
    print("mean AUPRC-x:", {c.replace("TwinFER_", "").replace("_AUPRC_x", ""): round(x[c].mean(), 3) for c in x.columns if c.endswith("_AUPRC_x")})
    y_ = C[C.universe == u]
    cols = ["gene_set", "n_true"] + [c for c in y_.columns if c.startswith("PIDC_") and (c.endswith("_AUPRC_x") or c.endswith("_hits"))] + [c for c in y_.columns if c.startswith("spearman")]
    print(f"\n== PIDC competitor, universe {u} ==")
    print(y_[cols].to_string(index=False))
    print("mean AUPRC-x:", {c.replace("PIDC_", "").replace("_AUPRC_x", ""): round(y_[c].mean(), 3) for c in y_.columns if c.startswith("PIDC_") and c.endswith("_AUPRC_x")})
