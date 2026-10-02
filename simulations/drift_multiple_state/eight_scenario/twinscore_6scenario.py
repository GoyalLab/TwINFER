"""twinScore on the 6 drift/regulation scenarios, both edge directions (g1->g2, g2->g1).

Two variants, from the per-rep JSONs (2-gene systems, so the "panel" for the s()
standardization is the pool of all scenario x direction x rep rows):

  tuned      : yscher helpers/twinscore_current.py, faithfully:
               S(x->y) = 1[|z_reg(t1)|>1.645] * ( s(z_rho(t1)) - s(|z_change|) - s(|rho_D(t1)|)
                   - s(|<rho_D>(t2)-<rho_D>(t1)|) - s(min(|rho_dag_fwd|,|rho_dag_rev|))
                   - s(|z_gamma|) + 0.5 s(z_Ddag) + [reg(x)-reg(y)] )
                 - 1[|z_reg(t1)|<1.645] * s(z_rho(t1))      # gate-fail -> -E, NOT dropped, NOT abs
               s() = mean-centre / std(ddof=1), non-finite -> min(finite)-1  (yscher's s()).
               reg(g) = mean_w z_Ddag(g,w) z-scored across genes; 2-gene reduction gives
               REG = sqrt(2)*sign(z_Ddag(x->y)).  z_change/z_Ddag/z_rho are analytic
               (analytic_zscores.py; validated <0.1 z vs permutation). Residual differences
               from yscher: rho is the sim's clone-weighted (not u_abs_rho/u_gamma unweighted);
               <rho_D>_random is a single draw here, not an averaged reference.
  zhet_fixed : shipped calculate_twin_score
               s z|rho_t1| + s z|rho_t2| - s z|drho| + pi - div_pen - het_pen + gamma_bonus
               then  + het_pen + panel_z(|z_het|)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario'
# OUT = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/drift_inference/eight_scenario and read/write next to itself; that dir is now referenced explicitly]
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario'
SCEN = ["no_regulation", "A_to_B", "multistate_A_to_B", "multistate_A_B", "kramp_A_to_B", "kramp_A_B"]
SCEN_LBL = {"no_regulation": "A, B", "A_to_B": "A→B",
            "multistate_A_to_B": "A→B\nmultistate", "multistate_A_B": "A, B\nmultistate",
            "kramp_A_to_B": "A→B\ndrift", "kramp_A_B": "A, B\ndrift"}
S2PI, VH = np.sqrt(2 / np.pi), 1 - 2 / np.pi
Z1 = 2.326          # divergence gate
GATE = 1.645        # z_reg_gated gate


def g(d, k, default=np.nan):
    return d.get(k, default)


T1 = os.environ.get("T1", "10")                     # "1", "10", or "1,10" to pool
SUBS = tuple(f"t1_{x}_t2_20" for x in T1.split(","))
TAG = "_t1_" + T1.replace(",", "+")

rows = []
for sub in SUBS:
    for jf in sorted(glob.glob(f"{ROOT}/{sub}/*_rep_*.json")):
        d = json.load(open(jf))
        scen = d.get("scenario") or os.path.basename(jf).rsplit("_rep_", 1)[0]
        if scen not in SCEN:
            continue
        rho_t1 = d["gene_t1_gene_1_gene_2"]; rho_t2 = d["gene_t2_gene_1_gene_2"]
        rd1 = d["twin_delta_t1_gene_1_gene_2"]
        drr = d["random_delta_t2_gene_1_gene_2"] - d["random_delta_t1_gene_1_gene_2"]
        rxy0 = d["step4_rho_cross_1to2"]; ryx0 = d["step4_rho_cross_2to1"]
        s1, s2 = d["step1_null_std_t1"], d["step1_null_std_t2"]
        z_het = d["step2_z_het_t1"]; z_div = d["z_div_t1"]; z_d_het = d["step3_z_d"]
        z_reg = d["z_reg_gated_t1"]
        # analytic z-scores (analytic_zscores.py: validated <0.1 z vs permutation)
        za_t1 = (abs(rho_t1) - s1 * S2PI) / (s1 * np.sqrt(VH))
        za_t2 = (abs(rho_t2) - s2 * S2PI) / (s2 * np.sqrt(VH))
        # z_change = SIGNED d-rho over its own (time-label-shuffle) null SD, centre 0
        sd_ch = np.sqrt(s1 ** 2 + s2 ** 2)
        z_change = (rho_t2 - rho_t1) / sd_ch                              # signed (yscher CHANGE)
        za_ch = (abs(rho_t2 - rho_t1) - sd_ch * S2PI) / (sd_ch * np.sqrt(VH))   # |.| z (shipped)
        # cross-corr null SD from step4 (|rho_cross / z_rho_cross|), for z_Ddag
        sx = np.nanmean([abs(rxy0 / d["step4_z_1to2"]) if d["step4_z_1to2"] else np.nan,
                         abs(ryx0 / d["step4_z_2to1"]) if d["step4_z_2to1"] else np.nan])
        sd_ddag = (sx if np.isfinite(sx) else 0.018) * np.sqrt(2.0)
        for dirn, rxy, ryx in (("g1→g2", rxy0, ryx0), ("g2→g1", ryx0, rxy0)):
            zg = d["z_gamma_1to2"] if dirn == "g1→g2" else d["z_gamma_2to1"]
            rows.append(dict(
                scen=scen, sub=sub, dirn=dirn, rep=os.path.basename(jf),
                gene_1=("A" if dirn == "g1→g2" else "B"),
                gene_2=("B" if dirn == "g1→g2" else "A"),
                rho_t1=rho_t1, rho_t2=rho_t2, rd1=rd1, drr=drr, rxy=rxy, ryx=ryx,
                u_gamma=abs(rxy) - abs(ryx),          # yscher's u_gamma  (|fwd|-|rev|)
                asym=rxy - ryx,                       # signed rho_dagger difference
                z_change=z_change, za_ch=za_ch, z_ddag=(rxy - ryx) / sd_ddag,
                za_t1=za_t1, za_t2=za_t2, z_gamma=zg,
                z_het=z_het, z_div=z_div, z_d_het=z_d_het, z_reg=z_reg))
df = pd.DataFrame(rows)
print(f"{len(df)} rows  ({df.rep.nunique()} reps x 6 scen x 2 dir)")


def s(v):
    """yscher twinscore_current.py s(): mean-centre, / std(ddof=1); non-finite -> min(finite)-1."""
    v = np.asarray(v, float); f = np.isfinite(v)
    o = (v - np.nanmean(v)) / max(np.nanstd(v, ddof=1), 1e-12)
    return np.where(f, o, (np.nanmin(o[f]) - 1.0) if f.any() else 0.0)


# ---- tuned = yscher twinscore_current.py, faithfully -----------------------
# TwinScore(x->y) = 1[|z_reg(t1)|>1.645] * ( s(z_rho(t1)) - s(|z_change|) - s(|rho_hat_Delta(t1)|)
#     - s(|<rho_D>(t2)-<rho_D>(t1)|) - s(min(|rho_dag_fwd|,|rho_dag_rev|)) - s(|z_gamma|)
#     + 0.5 s(z_Ddag) + [reg(x)-reg(y)] )   -   1[|z_reg(t1)|<1.645] * s(z_rho(t1))
E = s(np.abs(df.rho_t1))                                     # s(z_rho(t1)); |rho| magnitude
# reg(g) = mean_w z_Ddag(g,w), standardized across genes; REG = reg(x)-reg(y).
# 2-gene reduction: {reg(A),reg(B)}={q,-q} -> RG={sign(q)/sqrt2, -sign(q)/sqrt2}
#   -> REG(x->y row) = sqrt(2) * sign(z_Ddag(x->y)) = sqrt(2) * sign(z_ddag[row])
REG = np.sqrt(2.0) * np.sign(df.z_ddag.to_numpy())
TS = (E
      - s(np.abs(df.z_change))
      - s(np.abs(df.rd1))
      - s(np.abs(df.drr))
      - s(np.minimum(np.abs(df.rxy), np.abs(df.ryx)))
      - s(np.abs(df.u_gamma))
      + 0.5 * s(df.z_ddag)
      + REG)
IN = np.isfinite(df.z_reg.to_numpy()) & (np.abs(df.z_reg.to_numpy()) > GATE)
df["tuned"] = np.where(IN, TS, 0.0) - np.where(IN, 0.0, E)   # TS if gated-in, else -E  (NOT abs, NOT dropped)

# ---- shipped + z_het fix -------------------------------------------------
sza1, sza2, szach = s(df.za_t1), s(df.za_t2), s(df.za_ch)
szg = s(df.z_gamma)
div_pen = np.where(df.z_het.to_numpy() < -Z1, np.abs(df.z_div.to_numpy()), 0.0)
het_pen = np.where(df.z_d_het.to_numpy() > 0.0, df.z_het.to_numpy(), 0.0)
gamma_bonus = 0.5 * np.sign(df.u_gamma.to_numpy()) * (np.abs(szg) >= 1.0)
df["_gated_in"] = IN
shipped = sza1 + sza2 - szach + np.pi - div_pen - het_pen + gamma_bonus
df["zhet_fixed"] = shipped + het_pen + s(np.abs(df.z_het))

# ---- plot ---------------------------------------------------------------
C = {"g1→g2": "#0072B2", "g2→g1": "#D55E00"}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
for ax, col, ttl in zip(axes, ["tuned", "zhet_fixed"],
                        ["tuned analytic twinScore", "shipped twinScore + z_het-fix"]):
    for k, sc_ in enumerate(SCEN):
        for dx, dirn in ((-0.19, "g1→g2"), (0.19, "g2→g1")):
            v = df[(df.scen == sc_) & (df.dirn == dirn)][col].replace([np.inf, -np.inf], np.nan).dropna()
            if not len(v):
                continue
            ax.boxplot(v, positions=[k + dx], widths=0.33, patch_artist=True, showfliers=True,
                       medianprops=dict(color=C[dirn], lw=2),
                       boxprops=dict(facecolor=C[dirn], alpha=0.18, edgecolor=C[dirn], lw=1.2),
                       whiskerprops=dict(color=C[dirn], lw=1.1), capprops=dict(color=C[dirn], lw=1.1),
                       flierprops=dict(marker="o", ms=3, mfc=C[dirn], mec=C[dirn], alpha=.5))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xticks(range(6)); ax.set_xticklabels([SCEN_LBL[s_] for s_ in SCEN], fontsize=8)
    ax.set_xlim(-0.7, 5.3); ax.set_ylabel("twinScore"); ax.set_title(ttl, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(handles=[Patch(facecolor=C["g1→g2"], alpha=.5, label="g1→g2"),
                    Patch(facecolor=C["g2→g1"], alpha=.5, label="g2→g1")],
           loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("twinScore on the 6 drift/regulation scenarios, both directions  "
             f"($t_1$={T1.replace(',', '+')} h, $t_2$=20; panel standardised within this plot)",
             fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(f"{OUT}/twinscore_6scenario{TAG}.png", dpi=140, bbox_inches="tight")
print("wrote", f"{OUT}/twinscore_6scenario{TAG}.png")

print("\n=== median twinScore by scenario x direction ===")
print(df.groupby(["scen", "dirn"])[["tuned", "zhet_fixed"]].median().round(2)
      .reindex(SCEN, level=0).to_string())
print("\n=== gate pass rate (|z_reg(t1)|>1.645) by scenario ===")
print(df.groupby("scen")._gated_in.mean().reindex(SCEN).round(2).to_string())
