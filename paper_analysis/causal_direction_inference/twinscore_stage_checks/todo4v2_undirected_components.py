"""Which TODO4v2 (nogate) terms survive without direction, and how do they relate to / add to the three z-scores (z_rho, z_C, z_U)?
Per unordered gene pair (all pairs) from twinfer_inference_allpairs/*_all_results.json (t1 = 1 h, t2 = 20 h, infer.py partition 1500+1500 twins + 3000 cross-time clones), merged on (network, sim, pair)
with the Figure-2-design z_rho / z_C / z_U (sample B = 20 h).  Undirected components (symmetric in x,y, or made symmetric by max over the two directions):
  rho1   = z_abs_rho_t1              (|rho| at t1)                        ~ |z_rho|
  Cc     = -s(z_abs_rho_change)      (|rho(t2)-rho(t1)| over time)         no 3-z counterpart (needs two samples)
  divp   = -|z_div| * 1[z_het < -2.326]   (twin rho_Delta well below random-pair rho_Delta: rho carried by the shared part)   ~ -|z_C| penalty
  hinge_het = |z_het| * 1[|z_het| > 2.576]   (rho_Delta_twin vs random pair rho_Delta, either sign)                         ~ |rho_hat_Delta - rho|
  hinge_stable = -z_stable*1[z_stable>2.326], z_stable = -z_rho_change   (rho fell from t1 to t2)
  zdag   = s(max(|z_dagger_xy|, |z_dagger_yx|))   (cross-sample coupling, direction-free)
  z_div  ~ |z_U| (fixed-Delta twin test), z_het ~ (rho_hat_Delta - rho)
Dropped (need direction): new_gamma = s(z_gamma), s(z_flux).  s() = panel z-standardisation per (network, sim) panel, as in todo4v2.s.
Metric: undirected AUPRC x base per group (same positives/base as compare_old_auprc_x_undirected.py), mean over groups."""
import glob, json, re, os, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from twinfer.scoring.todo4v2 import s, Z_TWO_SIDED, Z_ONE_SIDED, Z_HET_THR_NEW
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0])     # d: pair z-scores + rel (edge/non-edge), columns raw, sim, x, y
A_ = A + "analysis_data/"
dirs = [(A_ + "mixed_network_sweep/twinfer_inference_allpairs", "mixed_network_sweep"), (A_ + "network_sweep_final/twinfer_inference_allpairs", "network_sweep_final"), (A_ + "network_sweep_final/e13_pos100/twinfer_inference_allpairs", "e13_pos100")]
feat = []
for jd, _ in dirs:
    for f in sorted(glob.glob(jd + "/*_all_results.json")):
        m = re.match(r"(.+_rep\d+)_rep(\d+)_all_results\.json", os.path.basename(f))
        if not m: continue
        raw, sim = m.group(1), int(m.group(2)); j = json.load(open(f)); t = j.get("twin_score_inputs")
        if t is None: continue
        dd = pd.DataFrame(t["data"], columns=t["columns"]); dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd["a"] = dd.gene_1.str.split("_").str[1].astype(int); dd["b"] = dd.gene_2.str.split("_").str[1].astype(int)
        dd["zdag"] = np.abs(z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"]))
        dd["key"] = [frozenset(p) for p in zip(dd.a, dd.b)]
        g = dd.groupby("key").agg(z_abs_rho_t1=("z_abs_rho_t1", "first"), z_abs_rho_change=("z_abs_rho_change", "first"), z_het=("z_het", "first"), z_div=("z_div", "first"), z_rho_change=("z_rho_change", "first"), zdag_max=("zdag", "max")).reset_index()
        g["raw"] = raw; g["sim"] = sim; feat.append(g)
F = pd.concat(feat, ignore_index=True); print("pairs with TODO4v2 inputs:", len(F), "| panels:", F.groupby(["raw", "sim"]).ngroups)
d["key"] = [frozenset((int(a), int(b))) for a, b in zip(d.x, d.y)]
D = d.merge(F, on=["raw", "sim", "key"], how="inner"); print("merged pairs:", len(D), "of", len(d))
D["edge"] = D.rel.str.startswith("edge").astype(int); D["azU"] = D.z_U.abs(); D["azC"] = D.z_C.abs(); D["azr"] = D.z_rho.abs(); D["new3"] = D.azr - D.azC + D.azU
def comps(x):
    zst = -x.z_rho_change.to_numpy()
    c = pd.DataFrame(index=x.index)
    c["rho1"] = x.z_abs_rho_t1; c["Cc"] = -s(x.z_abs_rho_change); c["divp"] = -np.abs(x.z_div) * (x.z_het < Z_HET_THR_NEW).astype(float)
    c["hinge_het"] = np.where(np.abs(x.z_het) > Z_TWO_SIDED, np.abs(x.z_het), 0.0); c["hinge_stable"] = -np.where(zst > Z_ONE_SIDED, zst, 0.0); c["zdag"] = s(x.zdag_max)
    return c
out = []
for (raw, sim), x in D.groupby(["raw", "sim"]):
    if x.edge.sum() == 0 or x.edge.sum() == len(x): continue
    c = comps(x); base = x.edge.mean(); sc = {}
    for k in c: sc["term: " + k] = c[k]
    sc["term: |z_div| (~|z_U|)"] = np.abs(x.z_div); sc["term: z_het"] = x.z_het; sc["term: -z_het"] = -x.z_het; sc["term: |z_het|"] = np.abs(x.z_het)
    sc["new3: |z_rho|-|z_C|+|z_U|"] = x.new3; sc["|z_rho| (B,20h)"] = x.azr; sc["|z_U| (B,20h)"] = x.azU; sc["|z_C| (B,20h)"] = x.azC
    und = c.sum(axis=1); sc["TODO4v2 undirected terms (no gamma, no flux)"] = und
    sc["  + new3 (z-scale)"] = und + x.new3; sc["  + 2*new3"] = und + 2 * x.new3
    sc["rho1 + new3"] = c.rho1 + x.new3; sc["rho1 + Cc + zdag + new3"] = c.rho1 + c.Cc + c.zdag + x.new3
    zd = c.zdag; sn = pd.Series(s(x.new3), index=x.index)
    sc["s(zdag) + s(new3)"] = zd + sn; sc["s(zdag) + 0.5 s(new3)"] = zd + 0.5 * sn; sc["s(zdag) + 2 s(new3)"] = zd + 2 * sn
    sc["s(zdag) + s(new3) + Cc"] = zd + sn + c.Cc; sc["s(zdag) + s(new3) + hinge_het"] = zd + sn + c.hinge_het
    sc["s(zdag) + s(|z_rho|)"] = zd + s(x.azr); sc["s(zdag) + s(|z_rho|) + s(|z_U|) - s(|z_C|)"] = zd + s(x.azr) + s(x.azU) - s(x.azC)
    r = dict(raw=raw, group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0])
    for k, v in sc.items():
        v = np.asarray(v, float); f = np.isfinite(v); v = np.where(f, v, (v[f].min() - 1.0) if f.any() else 0.0)     # non-finite -> below panel minimum (as todo4v2.s)
        r[k] = average_precision_score(x.edge, v) / base
    out.append(r)
R = pd.DataFrame(out); G = R.groupby(["fam", "group"]).mean(numeric_only=True)
T = pd.DataFrame({"mixed (16)": G.loc["mixed"].mean(), "sweep (%d)" % len(G.loc["sweep"]): G.loc["sweep"].mean()}); T["avg"] = T.mean(axis=1)
pd.set_option("display.width", 220); print(T.round(3).to_string())
R.to_csv(A_ + "paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/todo4v2_undirected_components_replicates.csv", index=False)
