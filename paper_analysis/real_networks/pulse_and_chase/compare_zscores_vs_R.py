"""After the pulse_chase array: for every network, correlate each TwINFER
z-score with the local causal response R (from the pulse-and-chase), and
summarize across networks.

  Spearman( |R| , |z| )   -- does the z-score magnitude track causal-link strength
  Spearman(  R  ,  z  )   -- does it track the signed effect
plus |R| true-vs-nonedge separation and sign-recovery of R itself.

Outputs: zscore_vs_R_summary.csv, zscore_vs_R.png
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/paper_analysis/real_networks/pulse_and_chase and read/write next to itself; that dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/pulse_and_chase'
OUT = f"{HERE}/out"
NETS = ["mCAD", "VSC", "HSC", "B_cell_activation", "GSD", "EMT", "Pluripotent"]
ZC = ["z_abs_rho_t1", "z_abs_rho_t2", "z_het", "z_div", "z_d_het", "z_gamma", "twinScore"]
ZLAB = {"z_abs_rho_t1": r"$z_{|\rho|t_1}$", "z_abs_rho_t2": r"$z_{|\rho|t_2}$",
        "z_het": r"$z_{\rm het}$", "z_div": r"$z_{\rm div}$", "z_d_het": r"$z_{d,\rm het}$",
        "z_gamma": r"$z_\gamma$", "twinScore": "twinScore"}

rows = []
per_net = {}
for net in NETS:
    zf = f"{OUT}/{net}_vs_zscores.csv"
    rmf = f"{OUT}/{net}_response_mrna.csv"
    rpf = f"{OUT}/{net}_response_protein.csv"
    if not (os.path.exists(zf) and os.path.exists(rmf)):
        print(f"[skip] {net}: missing outputs")
        continue
    tw = pd.read_csv(zf)[["source_g", "target_g"] + ZC].drop_duplicates("source_g target_g".split())
    for kind, rf in (("mrna", rmf), ("protein", rpf)):
        rr = pd.read_csv(rf)
        rr = rr[rr["self"] == 0]
        for tau in sorted(rr.tau.unique()):
            sub = rr[rr.tau == tau].merge(tw, on=["source_g", "target_g"], how="inner")
            n = len(sub)
            if n < 6:
                continue
            for z in ZC:
                s_absR_absZ = spearmanr(sub.R.abs(), sub[z].abs(), nan_policy="omit")[0]
                s_R_Z = spearmanr(sub.R, sub[z], nan_policy="omit")[0]
                rows.append(dict(net=net, kind=kind, tau=tau, z=z, n=n,
                                 rho_absR_absZ=s_absR_absZ, rho_R_Z=s_R_Z))
            # store one merged frame per (net, kind) at a mid tau for scatter
            if kind == "mrna" and abs(tau - 12) < 1e-6:
                per_net[net] = sub

df = pd.DataFrame(rows)
df.to_csv(f"{HERE}/zscore_vs_R_summary.csv", index=False)

# ---- headline: mean |Spearman(|R|,|z|)| over tau, per (net, z), mRNA readout ----
piv = (df[df.kind == "mrna"].groupby(["net", "z"]).rho_absR_absZ.mean()
       .unstack().reindex(index=[n for n in NETS if n in df.net.unique()], columns=ZC))
piv.loc["MEAN"] = piv.mean()
print("\n=== Spearman(|R_mRNA|, |z|), averaged over chase times ===")
print(piv.round(2).to_string())

piv_p = (df[df.kind == "protein"].groupby(["net", "z"]).rho_absR_absZ.mean()
         .unstack().reindex(index=[n for n in NETS if n in df.net.unique()], columns=ZC))
piv_p.loc["MEAN"] = piv_p.mean()
print("\n=== Spearman(|R_protein|, |z|), averaged over chase times ===")
print(piv_p.round(2).to_string())

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, (P, ttl) in zip(axes, [(piv, "mRNA response"), (piv_p, "protein response")]):
    A = P.to_numpy(float)
    im = ax.imshow(A, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(ZC))); ax.set_xticklabels([ZLAB[z] for z in ZC], fontsize=11)
    ax.set_yticks(range(len(P.index))); ax.set_yticklabels(P.index, fontsize=9)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(A[i, j]) > 0.4 else "0.1")
    ax.axhline(len(P.index) - 1.5, color="k", lw=1)
    ax.set_title(f"Spearman(|R|, |z|)  ·  {ttl}  ·  mean over $\\tau$", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle("Which TwINFER z-score tracks the pulse-and-chase causal response, per network",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{HERE}/zscore_vs_R.png", dpi=140, bbox_inches="tight")
print(f"\nwrote zscore_vs_R_summary.csv and zscore_vs_R.png")

# ---- also: |R| true vs non-edge, and R sign recovery, per net ----
print("\n=== |R_mRNA| (tau=12) true vs non-edge, and R-sign match to topology ===")
for net, sub in per_net.items():
    t = sub[sub.true_edge == 1]; f = sub[sub.true_edge == 0]
    sm = (np.sign(t.R) == t.edge_sign).mean() if len(t) else np.nan
    print(f"  {net:18s} |R| true {t.R.abs().median():.3g} vs non-edge {f.R.abs().median():.3g}"
          f"   ratio {t.R.abs().median()/max(f.R.abs().median(),1e-9):.1f}x   R-sign match {sm:.2f}  (n_true={len(t)})")
