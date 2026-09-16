"""
Rebuild real_networks_all_nofilter_heatmap.pdf with the TwINFER row computed
from ANALYTIC z-scores instead of the 10k-shuffle permutation z-scores.

Target row: "TwINFER (no-filter, z_het-fixed)" (and its signed twin) -- the
variant that ranks #1 in the permutation heatmap:

    twinScore = s(z|rho_t1|) + s(z|rho_t2|) - s(z|rho_t2-rho_t1|) + pi
                - divergence_penalty - heterogeneity_penalty + gamma_bonus
    z_het-fix: + heterogeneity_penalty + panel_z(|z_het|)      (score_all_nofilter_combined.py)

Every z-score is recomputed in closed form from the correlation matrices stored
in each nofilter replicate JSON (correlations.{rho_t1,rho_t2,rho_delta_t1,
rho_delta_t2,random_delta_t1,random_delta_t2,rho_cross}), using
analytic_zscores.py's model:

    z_abs(rho, sd)      = (|rho| - sd*sqrt(2/pi)) / (sd*sqrt(1-2/pi))
    z_signed(rho,sd,c)  = (rho - c) / sd
    z_gamma(g, sd_x)    = g / (sd_x*sqrt(2*(1-2/pi))),  g = |rho_cross_xy|-|rho_cross_yx|

One null SD per (replicate, statistic) is CALIBRATED from that replicate's own
reported per-pair z's (ANALYTIC_ZSCORES.md step 2 -- the ratio |rho|/(z*sqrt(1-2/pi)
+ sqrt(2/pi)) is near-constant across pairs, ~0.5% spread), except sd_cross which
is the median of direction.rho_cross_null[*].null_std. The z_het centre is the
random-pair rho_delta (well-estimated here: 10k draws).

Everything else in the heatmap -- BEELINE rows, all other TwINFER variants,
layout, 16 pages -- is copied verbatim from
real_networks_all_nofilter_combined_scores.csv.

Output: real_networks_analytic_zhet_fixed_combined_scores.csv
        real_networks_analytic_zhet_fixed_heatmap.pdf
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from twinfer.utils.paths import get_data_root
import plot_all_nofilter_heatmap as H

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
TOPO_ROOT = PROJECT_ROOT / "input_data" / "real_world_networks"
REAL_NET_ROOT = DATA_ROOT / "paper_analysis" / "real_networks"
OLD_DIR = REAL_NET_ROOT / "twinfer_inference_nofilter"
NEW_DIR = REAL_NET_ROOT / "twinfer_inference_multistate_nofilter"
COMBINED_CSV = REAL_NET_ROOT / "real_networks_all_nofilter_combined_scores.csv"

# Circadian_cycle is excluded (as in plot_all_nofilter_heatmap.ORIGINAL_DATASETS):
# its oscillator dynamics make it an outlier that skews the pooled means.
OLD_TOPO = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
            "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt",
            "B_cell_activation": "B_cell.txt"}
NEW_BASE_TOPO = {"GSD": "GSD.txt", "HSC": "HSC.txt", "EMT": "EMT.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt"}
NEW_DATASETS = ["GSD_multistate", "GSD_seeded", "HSC_multistate", "HSC_seeded",
                "EMT_multistate", "EMT_seeded", "VSC_seeded", "mCAD_seeded"]

ROW_UNSIGNED = "TwINFER (no-filter, z_het-fixed)"
ROW_SIGNED = "TwINFER (no-filter, z_het-fixed, signed)"

S2PI = np.sqrt(2.0 / np.pi)
VH = 1.0 - 2.0 / np.pi


# ----------------------------------------------------------------- scoring bits
def true_edges_from_topo(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    g = [f"gene_{i + 1}" for i in range(n)]
    true = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    poss = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j}
    tsign = {(g[i], g[j]): ("+" if M[i, j] > 0 else "-")
             for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    return true, poss, tsign


def auprc_from_scored_pairs(pair_scores, true_edges, possible_edges):
    from sklearn.metrics import auc, precision_recall_curve
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    y = np.fromiter((1 if p in true_edges else 0 for p in possible_edges), int)
    s = np.fromiter((pair_scores.get(p, floor) for p in possible_edges), float)
    prec, rec, _ = precision_recall_curve(y, s)
    return float(auc(rec, prec))


def f1_topk(pair_scores, true_edges, possible_edges):
    k = len(true_edges)
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    scored = sorted(possible_edges, key=lambda p: pair_scores.get(p, floor), reverse=True)
    if k <= 0 or not scored:
        return 0.0
    k = min(k, len(scored))
    boundary = pair_scores.get(scored[k - 1], floor)
    sel = {p for p in scored if pair_scores.get(p, floor) >= boundary}
    tp = len(sel & true_edges)
    if not sel or not true_edges:
        return 0.0
    p, r = tp / len(sel), tp / len(true_edges)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _sign_gate(mag, sign, true_sign, possible_edges):
    g = {}
    for p in possible_edges:
        if p in true_sign:
            if sign.get(p) == true_sign[p] and p in mag:
                g[p] = mag[p]
        elif p in mag:
            g[p] = mag[p]
    return g


# --------------------------------------------------------- analytic twinScore
def _mat(block):
    if block is None:
        return None
    return pd.DataFrame(block["data"], index=block["index"], columns=block["columns"])


def _calib(num, z, lo):
    """median  num / (|z|*sqrt(VH) + S2PI)   over pairs with |z| > lo  (abs terms)."""
    num, z = np.asarray(num, float), np.asarray(z, float)
    m = np.isfinite(num) & np.isfinite(z) & (np.abs(z) > lo)
    if m.sum() < 5:
        return np.nan
    return float(np.median(num[m] / (np.abs(z[m]) * np.sqrt(VH) + S2PI)))


def _calib_signed(num, z, lo):
    """median |num| / |z|   over pairs with |z| > lo  (signed terms: sd = num/z)."""
    num, z = np.asarray(num, float), np.asarray(z, float)
    m = np.isfinite(num) & np.isfinite(z) & (np.abs(z) > lo)
    if m.sum() < 5:
        return np.nan
    return float(np.median(np.abs(num[m]) / np.abs(z[m])))


def analytic_zhet_fixed(json_path):
    """(mag, sign) dicts over every directed pair, or None if a matrix is missing."""
    d = json.load(open(json_path))
    C = d["correlations"]
    get = lambda *ks: next((C[k] for k in ks if C.get(k) is not None), None)
    blocks = dict(
        R1=get("rho_t1", "gene_t1"), R2=get("rho_t2", "gene_t2"),
        D1=get("rho_delta_t1", "twin_delta_t1"), D2=get("rho_delta_t2", "twin_delta_t2"),
        RR1=get("random_delta_t1"), RR2=get("random_delta_t2"),
        XC=get("rho_cross", "direction"),
    )
    if any(v is None for v in blocks.values()):
        return None
    R1, R2, D1, D2, RR1, RR2, XC = (_mat(blocks[k]) for k in
                                    ["R1", "R2", "D1", "D2", "RR1", "RR2", "XC"])
    genes = list(R1.columns)
    red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])

    # --- one calibrated null SD per statistic, from this replicate's own z's
    sd1 = _calib(red.rho_t1.abs(), red.z_abs_rho_t1, 0.7)
    sd2 = _calib(red.rho_t2.abs(), red.z_abs_rho_t2, 0.7)
    sdc = _calib((red.rho_t2 - red.rho_t1).abs(), red.z_abs_rho_change, 0.7)
    ncross = [v["null_std"] for v in d["direction"]["rho_cross_null"].values()]
    sdx = float(np.median(ncross)) if ncross else np.nan
    # signed twin-delta nulls
    red_key = red.set_index(["gene_1", "gene_2"])
    het_num = [D1.loc[a, b] - RR1.loc[a, b] for a, b in red_key.index]
    sd_het = _calib_signed(het_num, red.z_het.values, 0.5)
    div_num = [D1.loc[a, b] for a, b in red_key.index]
    sd_div = _calib_signed(div_num, red.z_div.values, 0.3)
    d_obs = np.array([(D2.loc[a, b] - D1.loc[a, b]) for a, b in red_key.index])
    d_cen = np.array([(RR2.loc[a, b] - RR1.loc[a, b]) for a, b in red_key.index])
    sd_d = _calib_signed(d_obs - d_cen, red.z_d_het.values, 0.5)
    # fallbacks
    sd1 = sd1 if np.isfinite(sd1) else 0.026
    sd2 = sd2 if np.isfinite(sd2) else sd1
    sdc = sdc if np.isfinite(sdc) else sd1 * np.sqrt(2)
    sdx = sdx if np.isfinite(sdx) else 0.018
    sd_het = sd_het if np.isfinite(sd_het) else 0.030
    sd_div = sd_div if np.isfinite(sd_div) else sd_het
    sd_d = sd_d if np.isfinite(sd_d) else 0.030

    def pz(v):
        v = np.asarray(v, float)
        f = np.isfinite(v)
        o = np.full(v.shape, np.nan)
        if f.any():
            x = v[f]
            sd = x.std(ddof=0)
            o[f] = 0.0 if sd == 0 else (x - x.mean()) / sd
        return o

    a_idx, b_idx = [], []
    rt1, rt2, rxy, ryx = [], [], [], []
    z_abs_t1, z_abs_t2, z_abs_chg, z_gam = [], [], [], []
    z_het, z_div, z_d_het = [], [], []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            r1, r2 = R1.loc[a, b], R2.loc[a, b]
            cxy, cyx = XC.loc[a, b], XC.loc[b, a]
            a_idx.append(a); b_idx.append(b)
            rt1.append(r1); rt2.append(r2); rxy.append(cxy); ryx.append(cyx)
            z_abs_t1.append((abs(r1) - sd1 * S2PI) / (sd1 * np.sqrt(VH)))
            z_abs_t2.append((abs(r2) - sd2 * S2PI) / (sd2 * np.sqrt(VH)))
            z_abs_chg.append((abs(r2 - r1) - sdc * S2PI) / (sdc * np.sqrt(VH)))
            g = abs(cxy) - abs(cyx)
            z_gam.append(g / (sdx * np.sqrt(2 * VH)))
            z_het.append((D1.loc[a, b] - RR1.loc[a, b]) / sd_het)
            z_div.append(D1.loc[a, b] / sd_div)
            z_d_het.append(((D2.loc[a, b] - D1.loc[a, b]) - (RR2.loc[a, b] - RR1.loc[a, b])) / sd_d)

    out = pd.DataFrame(dict(gene_1=a_idx, gene_2=b_idx, rho_t1=rt1, rho_t2=rt2,
                            z_abs_rho_t1=z_abs_t1, z_abs_rho_t2=z_abs_t2,
                            z_abs_rho_change=z_abs_chg, z_gamma=z_gam,
                            z_het=z_het, z_div=z_div, z_d_het=z_d_het,
                            gamma=np.abs(rxy) - np.abs(ryx)))

    s1 = pz(out.z_abs_rho_t1); s2 = pz(out.z_abs_rho_t2); sc = pz(out.z_abs_rho_change)
    sg = pz(out.z_gamma)
    div_gate = np.isfinite(out.z_het) & (out.z_het < -2.326)
    div_pen = np.where(div_gate & np.isfinite(out.z_div), np.abs(out.z_div), 0.0)
    het_gate = np.isfinite(out.z_d_het) & (out.z_d_het > 0.0)
    het_pen = np.where(het_gate & np.isfinite(out.z_het), out.z_het, 0.0)
    gamma_bonus = 0.5 * np.sign(out.gamma.to_numpy()) * (np.abs(sg) >= 1.0)

    twin = s1 + s2 - sc + np.pi - div_pen - het_pen + gamma_bonus
    twin_zf = twin + het_pen + pz(np.abs(out.z_het))     # z_het-fix

    mag = {(a, b): abs(float(v)) for a, b, v in zip(a_idx, b_idx, twin_zf) if np.isfinite(v)}
    sign = {}
    for a, b, r2, r1 in zip(a_idx, b_idx, out.rho_t2, out.rho_t1):
        s = r2 if np.isfinite(r2) else r1
        sign[(a, b)] = "+" if (np.isfinite(s) and s >= 0) else "-"
    return mag, sign


# ----------------------------------------------------------------------- driver
def score_dataset(json_paths, topo_path):
    true_e, poss_e, tsign = true_edges_from_topo(topo_path)
    au, f1, aus, f1s = [], [], [], []
    n_used = 0
    for jp in json_paths:
        res = analytic_zhet_fixed(jp)
        if res is None:
            continue
        n_used += 1
        mag, sign = res
        au.append(auprc_from_scored_pairs(mag, true_e, poss_e))
        f1.append(f1_topk(mag, true_e, poss_e))
        gated = _sign_gate(mag, sign, tsign, poss_e)
        aus.append(auprc_from_scored_pairs(gated, set(tsign), poss_e))
        f1s.append(f1_topk(gated, set(tsign), poss_e))
    m = lambda x: float(np.mean(x)) if x else np.nan
    return n_used, m(au), m(f1), m(aus), m(f1s)


def main():
    df = pd.read_csv(COMBINED_CSV)
    prevalence = df.set_index(["dataset"]).prevalence.groupby("dataset").first().to_dict()

    updates = {}   # (dataset, algorithm) -> (auprc, f1)
    for net, topo in OLD_TOPO.items():
        jps = sorted(glob.glob(str(OLD_DIR / f"{net}_rep_*_all_results.json")))
        n, a, f, asg, fsg = score_dataset(jps, TOPO_ROOT / topo)
        updates[(net, ROW_UNSIGNED)] = (a, f)
        updates[(net, ROW_SIGNED)] = (asg, fsg)
        print(f"[old:{net}] {n}/{len(jps)} reps  AUPRC {a:.3f} (signed {asg:.3f})  F1 {f:.3f}", flush=True)

    for ds in NEW_DATASETS:
        base = ds.split("_")[0]
        jps = sorted(glob.glob(str(NEW_DIR / f"{ds}_rep_*_all_results.json")))
        n, a, f, asg, fsg = score_dataset(jps, TOPO_ROOT / NEW_BASE_TOPO[base])
        updates[(ds, ROW_UNSIGNED)] = (a, f)
        updates[(ds, ROW_SIGNED)] = (asg, fsg)
        print(f"[new:{ds}] {n}/{len(jps)} reps  AUPRC {a:.3f} (signed {asg:.3f})  F1 {f:.3f}", flush=True)

    for (ds, alg), (a, f) in updates.items():
        mask = (df.dataset == ds) & (df.algorithm == alg)
        if not mask.any():
            df = pd.concat([df, pd.DataFrame([dict(dataset=ds, track="?", algorithm=alg)])],
                           ignore_index=True)
            mask = (df.dataset == ds) & (df.algorithm == alg)
        prev = prevalence.get(ds, np.nan)
        df.loc[mask, ["auprc", "f1", "prevalence"]] = [a, f, prev]
        df.loc[mask, "ratio_auprc"] = a / prev
        df.loc[mask, "ratio_f1"] = f / prev

    out_csv = REAL_NET_ROOT / "real_networks_analytic_zhet_fixed_combined_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ---- heatmap, reusing plot_all_nofilter_heatmap's panel builder & layout
    out_pdf = REAL_NET_ROOT / "real_networks_analytic_zhet_fixed_heatmap.pdf"
    with PdfPages(out_pdf) as pdf:
        for group_name, columns in H.DATASET_GROUPS:
            for sign_name, methods in H.SIGN_MODES:
                for metric, mlabel, vmin, vmax, fmt, cbar_label in H.METRICS:
                    sub = df[df.algorithm.isin(methods) & df.dataset.isin(columns)]
                    sub = sub[["dataset", "algorithm", metric]]
                    if sub.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(10.5, 6.5))
                    title = f"{mlabel}  --  {group_name}  --  {sign_name}  [ANALYTIC z]"
                    im = H.make_panel(ax, sub, metric, columns, title, vmin=vmin, vmax=vmax, fmt=fmt)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                    cbar.set_label(cbar_label, fontsize=9)
                    if metric.startswith("ratio"):
                        cbar.ax.axhline(1.0, color="red", lw=1.2)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
