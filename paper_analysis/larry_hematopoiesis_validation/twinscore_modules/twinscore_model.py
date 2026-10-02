# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""The TwinScore: calculate_twin_score as the gzu module writes it, with one term dropped for
a measured reason, two declared filters, and one added penalty.

THE MODULE'S OWN SCORE (correlation_functions.py, calculate_twin_score)

    TwinScore(x->y) = z(|rho(t1)|) + z(|rho(t2)|)                       existence, both timepoints
                  - z_stable - s(z_drift)                            the change axis
                  + z_flux,  z_flux = reg(x) - reg(y)                direction
                  + I(max(z_het(t1), z_het(t2)) < -2.326) * max(z_het(t1), z_het(t2))
                  - I(min|z_dagger| > 2.326) * min|z_dagger|,   min|z_dagger| = min(|z_rho_dagger_x->y|, |z_rho_dagger_y->x|)
"""
import os, sys, glob
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT = "/gpfs/projects/b1255/yscher/Transcriptomic Distance"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/417a932c-bfcb-4ca6-98ea-915df4b1cf2e/scratchpad")
from paper_analysis.larry_hematopoiesis_validation.twinscore_modules.module_twinscore import panel, module_score
TERMS = ("rho_max", "stable_hinge_s", "flux_dagger", "sym_hinge_s", "hinge_max_s",
         "F_exist", "F_reg_signed", "alpha01_reg")
LABEL = {
 "larry_log1pPF_stable24_det5_p4a01_corrhigh_24": "high co-expression",
 "larry_log1pPF_stable24_det5_p4a01_corrmid_24":  "mid co-expression",
 "larry_log1pPF_stable24_det5_p4a01_corrlow_24":  "low co-expression",
 "larry_log1pPF_stable24_det5_p4a01_detect_q90100_24": "detection 90-100%",
 "larry_log1pPF_stable24_det5_p4a01_detect_q7590_24":  "detection 75-90%",
 "larry_log1pPF_stable24_det5_p4a01_detect_q5075_24":  "detection 50-75%",
 "larry_log1pPF_stable24_det5_p4a01_hvg_q67100_24": "spread high",
 "larry_log1pPF_stable24_det5_p4a01_hvg_q3367_24":  "spread mid",
 "larry_log1pPF_stable24_det5_p4a01_hvg_q0033_24":  "spread low",
 "larry_log1pPF_stable24_det5_p4a01_panel12_24":    "12-gene panel",
}

def tie_hits(sc, y, R):
    o = np.argsort(-sc, kind="stable"); sv, yv = sc[o], y[o]
    e = np.empty_like(yv); i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]: j += 1
        e[i:j] = yv[i:j].mean(); i = j
    return float(e[:R].sum())

def row(BN, comp_token=None):
    """comp_token: when given, a competitor file is admitted only if its name contains it (the window's t1
    timepoint tag, e.g. "_T4_"), so a window is never scored against a competitor run on other cells."""
    d = panel(BN); sc, keep = module_score(d, TERMS, return_keep=True)
    y = d["y"]; R = d["R"]
    h = tie_hits(sc, y, R)
    best = None
    FULL = {(a, b) for a in d["genes"] for b in d["genes"] if a != b}
    for nm, stem in (("PIDC","pidc"),("GRNBoost2","grnboost"),("ppcor","ppcor"),("GENIE3","genie3")):
        for f in sorted(glob.glob(f"{ROOT}/exports/networks/*.csv")):
            if stem not in os.path.basename(f).lower(): continue
            if comp_token and comp_token not in os.path.basename(f): continue
            try: d0 = pd.read_csv(f)
            except Exception: continue
            c = {x.lower(): x for x in d0.columns}
            cs = c.get("tf") or d0.columns[0]; ct = c.get("target") or d0.columns[1]
            cw = c.get("importance") or c.get("weight") or c.get("score") or d0.columns[2]
            d0 = d0[[cs, ct, cw]].dropna(); d0.columns = ["TF", "target", "w"]
            S = set(zip(d0.TF, d0.target))
            # A competitor is admitted only if it saw the panel's whole gene list and scored
            # nothing outside it. GRNBoost2 omits its zero-importance edges rather than writing
            # them, so an absent pair is scored 0 and ranks last, the same rule the TwinScore's
            # own called-NO pairs follow. Nothing here gives a method a smaller list.
            if (S - FULL) or (set(d0.TF) | set(d0.target)) != set(d["genes"]): continue
            mm = {(a, b): float(v) for a, b, v in d0.itertuples(index=False)}
            u = np.array([mm.get(p, 0.0) for p in d["univ"]], float)
            xb = (tie_hits(u, y, R) / R) / d["base"]
            ab = average_precision_score(y, u) / d["base"]
            aP = average_precision_score(y, u)
            # ranked by AUPRC, the same metric the TwinScore is ranked by
            if best is None or ab > best[3]: best = (nm, aP, xb, ab)
            break
    nof = int(np.sum(~keep))
    ap = average_precision_score(y, sc)
    return dict(calls=R, hits=h, prec=100*h/R, no=nof, auroc=roc_auc_score(y, sc),
                auprc=ap, auprc_base=ap/d["base"], x_base=(h/R)/d["base"],
                best=("%s, %.3f (%.2fx/base %.2f)" % best) if best else "no full-universe run")

if __name__ == "__main__":
    only = os.environ.get("PANELS", "")
    keys = [b for b in LABEL if not only or any(k in b for k in only.split(","))]
    rows = [(LABEL[b], row(b)) for b in keys]
    print("""
Filters, applied to every ordered pair; a pair failing either is CALLED NO and ranks last:
  existence   |z_rho|        > 2.576 at t1 OR t2
  regulation   z_reg_gated  > 2.326 at t1 OR t2   (signed)

TwinScore(x->y) = max_t s(z(|rho(t)|))                                 existence
                  + s(z_flux)                                           direction
                  - I(z_stable > 2.326) * s(z_stable)                   change
                  + I(max(z_het(t1), z_het(t2)) < -2.326) * s(max(z_het(t1), z_het(t2)))
                  - I(min|z_dagger| > 2.326) * s(min|z_dagger|)         inherited channel
  s(.) = panel standardisation of the z over all ordered pairs

  z_flux = reg(x) - reg(y),  reg(g) = mean_w z_rho_dagger(g->w) - mean_w z_rho_dagger(w->g)
  z_stable = z(|rho(t2) - rho(t1)|),  min|z_dagger| = min(|z_rho_dagger_x->y|, |z_rho_dagger_y->x|)
""")
    print("TwinScore")
    print("%-20s %5s %6s %6s %7s %6s %6s %7s %7s %-16s %s"
          % ("panel","days","calls","hits","prec %","rec %","no","AUROC","AUPRC",
             "x base precision",
             "best competitor"))
    for lab, d in rows:
        print("%-20s %5s %6d %6.1f %7.2f %6.2f %6d %7.3f %7.3f %-16s %s"
              % (lab, "2-4", d["calls"], d["hits"], d["prec"], d["prec"], d["no"],
                 d["auroc"], d["auprc"], "%.2fx" % d["x_base"], d["best"]))
    nine = [d for lab, d in rows if lab != "12-gene panel"]
    if only:
        print("%-20s mean over these %d panels: AUPRC/base %.3f | x base precision %.3f"
              % ("", len(nine), float(np.mean([d["auprc_base"] for d in nine])),
                 float(np.mean([d["x_base"] for d in nine])))); raise SystemExit
    print("%-20s mean over the nine scoring panels: AUPRC/base %.3f | x base precision %.3f"
          % ("", float(np.mean([d["auprc_base"] for d in nine])),
             float(np.mean([d["x_base"] for d in nine]))))
