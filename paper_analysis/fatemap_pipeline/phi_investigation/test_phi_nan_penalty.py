# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def s_neutral(v):
    """Alt: NaN phi contributes 0 (neutral) instead of worst-1."""
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = 0.0
    return o


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    return auprc / rand if rand > 0 else float("nan")


rows = []
for dataset in ["fm06", "fm08"]:
    files = sorted(glob.glob(f"analysis_data/{dataset}/data/twinscore_supplement_{dataset}_*_absplit_allpairs_pair_terms.csv"))
    for f in files:
        gs = f.split(f"twinscore_supplement_{dataset}_")[1].split("_absplit_allpairs")[0]
        df = pd.read_csv(f)
        y = df.collectri_edge.to_numpy()
        n_true = int(y.sum())
        n_nan_phi = df.phi_x.isna().sum()
        pct_nan = 100 * n_nan_phi / len(df)

        # original, as-saved TwinScore
        auprc_x_orig_saved = full_report(df.TwinScore.to_numpy(), y)

        # score = w_rel*s(phi_x) + s(PAIR) + direction_term  (note: s() applied to PAIR too)
        s_pair = s(df.PAIR.to_numpy())
        s_orig = s(df.phi_x.to_numpy())
        resid = (df.TwinScore.to_numpy() - s_pair - df.direction_term.to_numpy())
        mask = np.abs(s_orig) > 1e-9
        w_rel_est = np.median(resid[mask] / s_orig[mask]) if mask.any() else 0.0

        # sanity: reconstruct TwinScore with recovered w_rel and original s()
        recon_orig = w_rel_est * s_orig + s_pair + df.direction_term.to_numpy()
        max_abs_err = np.max(np.abs(recon_orig - df.TwinScore.to_numpy()))

        # alt score: NaN-phi genes get neutral (0) contribution instead of worst-1
        s_alt = s_neutral(df.phi_x.to_numpy())
        score_alt = w_rel_est * s_alt + s_pair + df.direction_term.to_numpy()
        auprc_x_alt = full_report(score_alt, y)

        # PAIR-alone (no phi at all) for reference
        auprc_x_pair_alone = full_report(s_pair, y)

        rows.append(dict(dataset=dataset.upper(), gene_set=gs, n_pairs=len(df), n_true=n_true,
                          pct_nan_phi=round(pct_nan, 1), w_rel=round(w_rel_est, 3),
                          recon_max_err=max_abs_err,
                          auprc_x_orig_saved=round(auprc_x_orig_saved, 3),
                          auprc_x_alt_neutral=round(auprc_x_alt, 3),
                          delta=round(auprc_x_alt - auprc_x_orig_saved, 3),
                          auprc_x_pair_alone=round(auprc_x_pair_alone, 3)))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))

is_dv = out.gene_set.str.startswith(("detection", "variability"))
is_corr = out.gene_set.str.startswith("correlation")
print()
print("mean delta (alt-neutral minus orig) on detection_*/variability_*:", out.loc[is_dv, "delta"].mean())
print("mean delta (alt-neutral minus orig) on correlation_*          :", out.loc[is_corr, "delta"].mean())
print("mean orig  auprc_x on detection_*/variability_*:", out.loc[is_dv, "auprc_x_orig_saved"].mean())
print("mean alt   auprc_x on detection_*/variability_*:", out.loc[is_dv, "auprc_x_alt_neutral"].mean())
print("mean orig  auprc_x on correlation_*:", out.loc[is_corr, "auprc_x_orig_saved"].mean())
print("mean alt   auprc_x on correlation_*:", out.loc[is_corr, "auprc_x_alt_neutral"].mean())
print()
print("max reconstruction error across all gene sets:", out.recon_max_err.max())
