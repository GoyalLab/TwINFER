# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""The LARRY 2-4 table, in the approved format.

UNIVERSE   every ordered pair of the panel's genes (curr_gene_list). Nothing is removed.
           A gene constant in a table the score reads has no correlation there, so its pairs
           have no z-score -- those pairs are CALLED NO and rank last, exactly like a pair
           that fails a filter. They are never dropped from the universe.
FILTERS    declared before the score, applied to every pair:
             existence   |z_rho(t1)| > 2.576 OR |z_rho(t2)| > 2.576   (alpha 0.01 two-sided)
             regulation  |z_reg_gated(t1)| > 1.645
           A pair failing either is CALLED NO and ranked last. Nothing abstains.
TERMS      every one a z-score against its own null, then standardised across the panel:
             z_rho(t1), z_rho(t2), z_het, z_div, z_rho_cross  -- gzu module outputs
             z_reg_gated                                      -- calculate_gated_regulation_statistic
             z_gamma, z_change                                -- _null_unit's construction, module draws
             z_drift = |<rho_Delta>(t2)-<rho_Delta>(t1)| / SD  -- generate_difference_shuffle
             z_D                                              -- same paired null, signed
COMPETITOR admitted only if the file scored EXACTLY the panel's ordered universe.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, glob, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.abspath("helpers"))
from panel_live_genes import live_genes

ROOT = "/gpfs/projects/b1255/yscher/Transcriptomic Distance"
YCOPY = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports"  # [2026-10-01 added: copy of yscher exports (twinscore_script, twinscore_gated, panels/tf_target_panel, _probe_tmp gene_flags)]
LABEL = {
 "larry_log1pPF_stable24_det5_p4a01_corrhigh_24": "high co-expression",
 "larry_log1pPF_stable24_det5_p4a01_corrmid_24":  "mid co-expression",
 "larry_log1pPF_stable24_det5_p4a01_corrlow_24":  "low co-expression",
 "larry_log1pPF_stable24_det5_p4a01_detect_q5075_24":  "detection 50-75%",
 "larry_log1pPF_stable24_det5_p4a01_detect_q7590_24":  "detection 75-90%",
 "larry_log1pPF_stable24_det5_p4a01_detect_q90100_24": "detection 90-100%",
 "larry_log1pPF_stable24_det5_p4a01_hvg_q0033_24":  "spread low",
 "larry_log1pPF_stable24_det5_p4a01_hvg_q3367_24":  "spread mid",
 "larry_log1pPF_stable24_det5_p4a01_hvg_q67100_24": "spread high",
 "larry_log1pPF_stable24_det5_p4a01_panel12_24":    "12-gene panel",
}
ORDER = list(LABEL.values())
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] Cm = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_mouse.tsv", sep="\t")
Cm = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_mouse.tsv", sep="\t")
def s(v):
    v = np.asarray(v, float); f = np.isfinite(v)
    o = (v - np.nanmean(v)) / max(np.nanstd(v, ddof=1), 1e-12)
    return np.where(f, o, np.nanmin(o[f]) - 1.0)

def score(BN):
    ck = pickle.load(open(f"{ROOT}/exports/runs/{BN}/stage_inputs_checkpoint.pkl", "rb"))
    genes = sorted(ck["curr_gene_list"]); gset = set(genes)
    cur = {(a, b) for a, b in zip(Cm["source_genesymbol"], Cm["target_genesymbol"])
           if a in gset and b in gset and a != b}
    univ = [(a, b) for a in genes for b in genes if a != b]
    R = sum(1 for p in univ if p in cur)
    if R < 3: return None
    base = R / len(univ); y = np.array([1.0 if p in cur else 0.0 for p in univ])
    # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] J = json.load(open(f"{ROOT}/exports/twinscore_script/{BN}.json"))
    J = json.load(open(f"{YCOPY}/twinscore_script/{BN}.json"))
    assert "gzu5140" in J["src"]
    # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] S = json.load(open(f"{ROOT}/exports/twinscore_script/SCREEN_{BN}.json"))
    S = json.load(open(f"{YCOPY}/twinscore_script/SCREEN_{BN}.json"))
    # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] Z = pd.read_csv(f"{ROOT}/exports/twinscore_script/ZDDAG_{BN}.csv").set_index(["gene_1","gene_2"])
    Z = pd.read_csv(f"{YCOPY}/twinscore_script/ZDDAG_{BN}.csv").set_index(["gene_1","gene_2"])
    # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] CH = pd.read_csv(f"{ROOT}/exports/twinscore_script/CHANGE_{BN}.csv").set_index(["gene_1","gene_2"])
    CH = pd.read_csv(f"{YCOPY}/twinscore_script/CHANGE_{BN}.csv").set_index(["gene_1","gene_2"])
    # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] GG = pd.read_csv(f"{ROOT}/exports/twinscore_script/GZUGATED_{BN}.csv").set_index(["gene_1","gene_2"])
    GG = pd.read_csv(f"{YCOPY}/twinscore_script/GZUGATED_{BN}.csv").set_index(["gene_1","gene_2"])
    P = [tuple(p) for p in J["pairs"]]; idx = {p: i for i, p in enumerate(P)}
    ja = lambda k: np.array(J[k], float)
    col = lambda df, c: np.array([df[c].loc[p] if p in df.index else np.nan for p in P], float)
    zr1 = np.abs(ja("z_rho_t1")); zr2 = np.abs(ja("z_rho_t2"))
    ZD = col(Z, "z_Ddag"); GAM = col(Z, "z_gamma")
    CHv = col(CH, "z_change"); GZ = col(GG, "z_reg_gated")
    INH = np.minimum(np.abs(ja("z_fwd")), np.abs(ja("z_rev")))
    # the reference change rho_Delta(t2)-rho_Delta(t1), from generate_difference_shuffle:
    # its mean over the module's draws, in units of its own spread over those draws
    # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] DH = pd.read_csv(f"{ROOT}/exports/twinscore_script/DHET_{BN}.csv").set_index(["gene_1","gene_2"])
    DH = pd.read_csv(f"{YCOPY}/twinscore_script/DHET_{BN}.csv").set_index(["gene_1","gene_2"])
    dm = np.array([DH["ref_change_mean"].loc[p_] if p_ in DH.index else np.nan for p_ in P], float)
    dsd = np.array([DH["ref_change_sd"].loc[p_] if p_ in DH.index else np.nan for p_ in P], float)
    DRIFT = np.abs(dm) / np.where(dsd > 0, dsd, np.nan)
    zdd = {}
    for i, (a, b) in enumerate(P):
        if np.isfinite(ZD[i]): zdd[(a, b)] = ZD[i]; zdd[(b, a)] = -ZD[i]
    reg = {x: np.nanmean([zdd[(x, w)] for w in genes if (x, w) in zdd]) for x in genes}
    rv = np.array([v for v in reg.values() if np.isfinite(v)])
    RG = {x: ((v - rv.mean()) / max(rv.std(ddof=1), 1e-12) if np.isfinite(v) else 0.0)
          for x, v in reg.items()}
    SYM = s(np.nan_to_num(np.minimum(zr1, zr2))) - s(np.nan_to_num(np.abs(CHv))) - s(np.nan_to_num(DRIFT)) \
          - s(np.nan_to_num(INH)) + s(np.nan_to_num(np.abs(GAM)))
    ANTI = 0.5 * s(np.nan_to_num(ZD))
    # a pair whose quantities are undefined (a constant gene) cannot pass a filter, so it is
    # called NO along with every pair that fails one
    defined = np.isfinite(zr1) & np.isfinite(CHv) & np.isfinite(DRIFT) & np.isfinite(INH) \
              & np.isfinite(GAM) & np.isfinite(ZD) & np.isfinite(GZ)
    PASS = defined & ((zr1 > 2.576) | (zr2 > 2.576)) & (np.abs(GZ) > 1.645)
    sc = np.empty(len(univ))
    for u, (x, w) in enumerate(univ):
        i = idx.get((x, w)); sg = 1.0
        if i is None: i = idx[(w, x)]; sg = -1.0
        sc[u] = (SYM[i] + sg * ANTI[i]) if PASS[i] else -1e12
    o = np.argsort(-sc, kind="stable"); sv, yv = sc[o], y[o]
    e = np.empty_like(yv); i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]: j += 1
        e[i:j] = yv[i:j].mean(); i = j
    h = e[:R].sum(); ap = average_precision_score(y, sc)
    best = None; FULL = {(a, b) for a in gset for b in gset if a != b}
    for nm, stem in (("PIDC","pidc"),("GRNBoost2","grnboost"),("ppcor","ppcor"),("GENIE3","genie3")):
        for f in sorted(glob.glob(f"{ROOT}/exports/networks/*.csv")):
            if stem not in os.path.basename(f).lower(): continue
            try: d0 = pd.read_csv(f)
            except Exception: continue
            c = {x.lower(): x for x in d0.columns}
            cs = c.get("tf") or c.get("source") or d0.columns[0]
            ct = c.get("target") or d0.columns[1]
            cw = c.get("importance") or c.get("weight") or c.get("score") or d0.columns[2]
            d0 = d0[[cs, ct, cw]].dropna(); d0.columns = ["TF","target","w"]
            if set(zip(d0.TF, d0.target)) != FULL: continue
            mm = {(a, b): float(v) for a, b, v in d0.itertuples(index=False)}
            u = np.array([mm[p] for p in univ], float)
            oo = np.argsort(-u, kind="stable"); uu, yy = u[oo], y[oo]
            e2 = np.empty_like(yy); ii = 0
            while ii < len(uu):
                jj = ii
                while jj < len(uu) and uu[jj] == uu[ii]: jj += 1
                e2[ii:jj] = yy[ii:jj].mean(); ii = jj
            xb = (e2[:R].sum()/R)/base
            if best is None or xb > best[1]: best = (nm, xb, average_precision_score(y, u))
            break
    return dict(genes=len(genes), calls=R, hits=h, prec=100*h/R, rec=100*h/R,
                no=int(2*(~PASS).sum()), auroc=roc_auc_score(y, sc), auprc=ap,
                auprc_base=ap/base, x_base=(h/R)/base,
                best=("%s %.2fx (AUPRC %.3f)" % best) if best else "no full-universe run")

rows = [(LABEL[b], b, score(b)) for b in LABEL]
rows = [r for r in rows if r[2]]; rows.sort(key=lambda r: ORDER.index(r[0]))
print("\nTwinScore")
print("%-20s %5s %6s %6s %7s %6s %6s %7s %7s %-16s %s"
      % ("panel","days","calls","hits","prec %","rec %","no","AUROC","AUPRC",
         "x base precision","best competitor"))
for lab, BN, d in rows:
    print("%-20s %5s %6d %6.1f %7.2f %6.2f %6d %7.3f %7.3f %-16s %s"
          % (lab,"2-4",d["calls"],d["hits"],d["prec"],d["rec"],d["no"],
             d["auroc"],d["auprc"],"%.2fx"%d["x_base"],d["best"]))
nine = [d for lab, BN, d in rows if lab != "12-gene panel"]
print("%-20s mean over the nine scoring panels: AUPRC/base %.3f | x base precision %.3f"
      % ("", float(np.mean([d["auprc_base"] for d in nine])),
         float(np.mean([d["x_base"] for d in nine]))))
