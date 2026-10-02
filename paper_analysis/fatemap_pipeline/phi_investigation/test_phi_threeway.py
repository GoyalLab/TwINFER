# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

MEFF = {"fm06": (3949.0, 3159.0), "fm08": (710.0, 2498.0)}


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    return auprc / rand if rand > 0 else float("nan")


def three_way_s(phi_x, cls):
    """cls: array of 'real'/'noise'/'real_neg' per pair.
    'real'     -> standard z-score of the (finite) phi value
    'noise'    -> 0 (either h is within its own noise band, sign-agnostic)
    'real_neg' -> worst finite z-score - 1 (statistically real negative persistence)"""
    v = np.asarray(phi_x, float)
    real_mask = (cls == "real") & np.isfinite(v)
    o = np.zeros(len(v))
    if real_mask.sum() > 1:
        o[real_mask] = (v[real_mask] - v[real_mask].mean()) / max(v[real_mask].std(ddof=1), 1e-12)
    worst = (o[real_mask].min() - 1.0) if real_mask.any() else -1.0
    o[cls == "noise"] = 0.0
    o[cls == "real_neg"] = worst
    return o


rows = []
for dataset in ["fm06", "fm08"]:
    m1, m2 = MEFF[dataset]
    nsd1, nsd2 = 1 / np.sqrt(m1 - 1), 1 / np.sqrt(m2 - 1)
    pair_files = sorted(glob.glob(f"analysis_data/{dataset}/data/twinscore_supplement_{dataset}_*_absplit_allpairs_pair_terms.csv"))
    for pf in pair_files:
        gs = pf.split(f"twinscore_supplement_{dataset}_")[1].split("_absplit_allpairs")[0]
        gf = pf.replace("_pair_terms.csv", "_gene_terms.csv")
        pdf = pd.read_csv(pf)
        gdf = pd.read_csv(gf).set_index("gene")

        y = pdf.collectri_edge.to_numpy()

        h1 = pdf.gene_1.map(gdf["h_t1"]).to_numpy()
        h2 = pdf.gene_1.map(gdf["h_t2"]).to_numpy()

        real_neg = (h1 < -nsd1) | (h2 < -nsd2)
        noise = (~real_neg) & ((np.abs(h1) < nsd1) | (np.abs(h2) < nsd2))
        real = ~real_neg & ~noise
        cls = np.where(real_neg, "real_neg", np.where(noise, "noise", "real"))

        # recover w_rel exactly as before (from the as-saved TwinScore/PAIR/direction_term/phi_x)
        s_pair = s(pdf.PAIR.to_numpy())
        s_orig = s(pdf.phi_x.to_numpy())
        resid = pdf.TwinScore.to_numpy() - s_pair - pdf.direction_term.to_numpy()
        mask = np.abs(s_orig) > 1e-9
        w_rel_est = np.median(resid[mask] / s_orig[mask]) if mask.any() else 0.0
        recon_err = np.max(np.abs(w_rel_est * s_orig + s_pair + pdf.direction_term.to_numpy() - pdf.TwinScore.to_numpy()))

        auprc_x_orig = full_report(pdf.TwinScore.to_numpy(), y)

        s_3way = three_way_s(pdf.phi_x.to_numpy(), cls)
        score_3way = w_rel_est * s_3way + s_pair + pdf.direction_term.to_numpy()
        auprc_x_3way = full_report(score_3way, y)

        rows.append(dict(dataset=dataset.upper(), gene_set=gs, n_pairs=len(pdf), n_true=int(y.sum()),
                          n_real=int((cls == "real").sum()), n_noise=int((cls == "noise").sum()),
                          n_real_neg=int((cls == "real_neg").sum()),
                          recon_err=recon_err,
                          auprc_x_orig=round(auprc_x_orig, 3),
                          auprc_x_3way=round(auprc_x_3way, 3),
                          delta=round(auprc_x_3way - auprc_x_orig, 3)))

out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(out.to_string(index=False))

is_dv = out.gene_set.str.startswith(("detection", "variability"))
is_corr = out.gene_set.str.startswith("correlation")
print()
print("mean delta (3way - orig) on detection_*/variability_*:", out.loc[is_dv, "delta"].mean())
print("mean delta (3way - orig) on correlation_*             :", out.loc[is_corr, "delta"].mean())
print("mean orig  auprc_x on detection_*/variability_*:", out.loc[is_dv, "auprc_x_orig"].mean())
print("mean 3way  auprc_x on detection_*/variability_*:", out.loc[is_dv, "auprc_x_3way"].mean())
print("mean orig  auprc_x on correlation_*:", out.loc[is_corr, "auprc_x_orig"].mean())
print("mean 3way  auprc_x on correlation_*:", out.loc[is_corr, "auprc_x_3way"].mean())
print("FM06 mean orig:", out.loc[out.dataset=="FM06","auprc_x_orig"].mean(), " FM06 mean 3way:", out.loc[out.dataset=="FM06","auprc_x_3way"].mean())
print("FM08 mean orig:", out.loc[out.dataset=="FM08","auprc_x_orig"].mean(), " FM08 mean 3way:", out.loc[out.dataset=="FM08","auprc_x_3way"].mean())
print()
print("max reconstruction err:", out.recon_err.max())
