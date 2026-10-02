# [copied 2026-09-30 from /gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp/hyp_check/self/twinscore_mega_A.py (author/owner: yscher/laj2116), md5 9ccd75ef692661e7cf3a3e079f19e831; unmodified below this header]
"""TwinScore + gated gamma (form A, twinscore_gamma.py), approved table, on: six corrhigh lists, every dataset's corrmid / corrlow panels (LARRY 4-6 on the p4a46 panels), the two Perturb-seq lists. Specification unchanged. Competitor tag candidates per sample are tried in order; the first file whose gene set equals the panel is used."""
import os, sys, glob, pickle, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# os.chdir("/gpfs/projects/b1255/yscher/Transcriptomic Distance"); sys.path.insert(0, "helpers"); sys.path.insert(0, "exports/_probe_tmp")
# os.chdir("/gpfs/projects/b1255/yscher/Transcriptomic Distance"); sys.path.insert(0, "helpers"); sys.path.insert(0, "exports/_probe_tmp")
from sklearn.metrics import average_precision_score as aps, roc_auc_score as auc
from scipy.stats import norm
import eval_x_real as E, standing_plus as SP
from twinscore_model import tie_hits
s_ = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) if np.nanstd(v) > 0 else 1.0); P0 = pickle.load(open("exports/_probe_tmp/brainstorm/B_pooled_rho_tw.pkl", "rb")); rows = []; rows2 = []
def clr_vec(v, U, G):
    gi = {g: i for i, g in enumerate(G)}; M = np.zeros((len(G), len(G)))
    for (a, b), x in zip(U, np.nan_to_num(v)): M[gi[a], gi[b]] = max(x, 0.0)
    C = SP.clr(M); return np.nan_to_num(np.array([C[gi[a], gi[b]] for a, b in U]))
L = [(a, b, "corrhigh", c, ([d[0]], [d[1]])) for a, b, c, d in E.ROWS]
for band in ("corrmid", "corrlow"):
    L.append(("LARRY", "day 2 - day 4", band, f"larry_log1pPF_stable24_det5_p4a01_{band}_24", ([f"p4a01_{band}_T2_det5"], [f"p4a01_{band}_T4_det5"])))
    L.append(("LARRY", "day 4 - day 6", band, f"larry_log1pPF_stable24_det5_p4a46_{band}_46", ([f"p4a46_{band}_T4_det5"], [f"p4a46_{band}_6_det5"])))
    L.append(("michaels", "day 5 - day 10", band, f"michaels_mic5a01_{band}_panel_lines_T1_T2_det5", ([f"mic5a01_{band}_5_det5"], [f"mic5a01_{band}_10_det5"])))
    L.append(("hPSC", "T0 - T10", band, f"net_t26r12a01_{band}_panel_lines_T0_T10_det5", ([f"t26r12a01_{band}_T0_det5"], [f"t26r12a01_{band}_T10_det5"])))
    L.append(("hPSC", "T10 - T15", band, f"net_t26r12a01_{band}_panel_lines_T10_T15_det5", ([f"t26r12a01_{band}_T10_det5"], [f"t26r12a01_{band}_T15_det5"])))
    L.append(("CellTag", "day 2.5 - day 5", band, f"repaired_clone_nopad_ct5a01_{band}__log1pPF_stable24", ([f"ct5a01_{band}_qc_day2_det5"], [f"ct5a01_{band}_qc_day5_det5"])))
L.append(("CellTag (notebook filter: unlabelled day-5 cells kept)", "day 2.5 - day 5", "corrhigh", "repaired_clone_nopad_nbfilter_ct5a01_corrhigh__log1pPF_stable24", (["ct5a01_corrhigh_qc_day2_det5"], ["ct5a01_corrhigh_qc_day5_det5"])))
L += [("LARRY Perturb-seq", "day 2 - day 4", "top100", "larry_log1pPF_stable24_det5_ps24_top100_24", (["ps24_top100_T2_det5"], ["ps24_top100_T4_det5"])), ("LARRY Perturb-seq", "day 4 - day 6", "top100", "larry_log1pPF_stable24_det5_ps46_top100_46", (["ps46_top100_T4_det5"], ["ps46_top100_T6_det5"]))]
for ds, win, band, BN, tagc in L:
    fA = f"exports/_probe_tmp/analytic_smart/nosplit__{BN}.pkl"; fA = fA if os.path.exists(fA) else fA.replace("nosplit__", "asis__")
    if not os.path.exists(fA) or not os.path.exists(f"exports/_probe_tmp/regdet/realMem_{BN}.pkl"): rows.append(dict(panel=ds, days=win, band=band, note="inputs missing")); continue
    try:
        d, _, _ = SP.build(BN, "nosplit")
    except Exception as e: rows.append(dict(panel=ds, days=win, band=band, note=f"build failed: {str(e)[:60]}")); continue
    y = d["y"]; U = d["univ"]; G = d["genes"]; gi = {g: i for i, g in enumerate(G)}; base = d["base"]; A_ = pickle.load(open(fA, "rb")); O = {p: i for i, p in enumerate(A_["pairs"])}; SD = A_["SD"]
    g = lambda k: np.nan_to_num(np.array([A_[k][O[q]] if q in O else A_[k][O[(q[1], q[0])]] for q in U], float)); ck = pickle.load(open(glob.glob(f"exports/*/{BN}/stage_inputs_checkpoint.pkl")[0], "rb")); n1, n2 = len(ck["all_t1_measurements"]), len(ck["all_t2_measurements"]); big = "t1" if n1 > n2 else "t2"
    comp = {}
    for nm, pat in E.METHODS.items():
        for i, cands in enumerate(tagc):
            for t_ in cands:
                c_ = E.comp_vec(pat.format(tag=t_), {"genes": G, "univ": U})
                if c_ is not None: comp[f"{nm} t{i+1}"] = c_; break
    if f"PIDC {big}" not in comp: rows.append(dict(panel=ds, days=win, band=band, note=f"no PIDC at {big}: {sorted(comp)}")); continue
    Dv = s_(np.nan_to_num(comp[f"PIDC {big}"])); Rv = s_(clr_vec(g("zreg1" if big == "t1" else "zreg2"), U, G)); up = np.array([gi[a] < gi[b] for a, b in U])
    fw = f"exports/_probe_tmp/twin_layers/real/{BN}.pkl"
    if os.path.exists(fw):
        Lr = pickle.load(open(fw, "rb")); g3 = {q: i for i, q in enumerate(Lr["genes"])}; ws = "t1" if Lr["t1"]["shrink_W"] < Lr["t2"]["shrink_W"] else "t2"; Wz = np.nan_to_num(np.array([Lr[ws]["pcW"][g3[a], g3[b]] / max(Lr[ws]["pcW_se"][g3[a], g3[b]], 1e-9) for a, b in U])); vv = float(np.var(Wz[up])); v = max(0.0, 1 - 1 / vv) if vv > 0 else 0.0
    else: Wz = np.zeros(len(U)); v = 0.0
    zC = g("rcross1" if big == "t1" else "rcross2") / float(SD["het_t1"] if big == "t1" else SD["het_t2"]); vc = float(np.var(zC[up])); gg = max(0.0, 1 - 1 / vc) if vc > 0 else 0.0; PAIR = s_(Dv) + gg * s_(Rv) + gg * v * s_(Wz)
    mem = pickle.load(open(f"exports/_probe_tmp/regdet/realMem_{BN}.pkl", "rb"))["mem"]; dg = np.array([mem[x] for x in G]); sdx = float(SD["cross"])
    if BN not in P0 and not os.path.exists(f"exports/_probe_tmp/analytic_smart/icc_nosplit__{BN}.pkl") and not os.path.exists(f"exports/_probe_tmp/analytic_smart/icc_asis__{BN}.pkl"): rows.append(dict(panel=ds, days=win, band=band, note="twin-twin correlation file missing")); continue
    if BN in P0: r1 = np.asarray(P0[BN]["t1"]["rho_tw"], float); r2 = np.asarray(P0[BN]["t2"]["rho_tw"], float); sd1 = P0[BN]["t1"]["sd"]; sd2 = P0[BN]["t2"]["sd"]
    else: fi = f"exports/_probe_tmp/analytic_smart/icc_nosplit__{BN}.pkl"; I = pickle.load(open(fi if os.path.exists(fi) else fi.replace("icc_nosplit__", "icc_asis__"), "rb")); ig = {q: i for i, q in enumerate(I["genes"])}; r1 = np.array([I["t1"][ig[q]] for q in G]); r2 = np.array([I["t2"][ig[q]] for q in G]); sd1, sd2 = SD["het_t1"], SD["het_t2"]
    ok = (r1 > 2 * sd1) & (r2 > 2 * sd2); phi = np.where(ok, dg / np.sqrt(np.maximum(r1 * r2, 1e-9)), np.nan); vn = phi[ok] ** 2 * ((sdx / np.where(np.abs(dg[ok]) > 1e-9, dg[ok], 1e-9)) ** 2 + (sd1 / r1[ok]) ** 2 / 4 + (sd2 / r2[ok]) ** 2 / 4) if ok.sum() else np.array([np.nan]); floor = float(np.nanmedian(vn)); vg = float(np.nanvar(phi)); w = max(0.0, 1 - floor / vg) if vg > 0 and np.isfinite(floor) else 0.0; phi = np.where(np.isnan(phi), np.nanmedian(phi), phi); xs = np.array([gi[a] for a, b in U])
    zf = np.nan_to_num(np.array([A_["rxy"][O[q]] if q in O else A_["ryx"][O[(q[1], q[0])]] for q in U], float)) / sdx; zr = np.nan_to_num(np.array([A_["ryx"][O[q]] if q in O else A_["rxy"][O[(q[1], q[0])]] for q in U], float)) / sdx; gam = (np.abs(zf) - np.abs(zr)) / np.sqrt(2 * (1 - 2 / np.pi)); vga = float(np.var(gam[up])); kg = max(0.0, 1 - 1 / vga) if vga > 0 else 0.0; sig = np.maximum(np.abs(zf), np.abs(zr)) > 2.576
    r0 = 0.69; qf = np.log(0.5 + (r0 - 0.5) * (2 * norm.cdf(np.abs(gam)) - 1) * np.sign(gam)); h1 = float(np.median(r1)); Am = np.zeros((len(G), len(G)))
    for (a, b), v_ in zip(U, np.abs(zf + zr) / 2): Am[gi[a], gi[b]] = v_
    Ac = SP.clr(Am); Av = s_(np.nan_to_num(np.array([Ac[gi[a], gi[b]] for a, b in U]))); r0h = 0.69 + (1 - 0.69) * min(max(h1, 0), 1); qfh = np.log(0.5 + (r0h - 0.5) * (2 * norm.cdf(np.abs(gam)) - 1) * np.sign(gam))
    FORMS = {"A (frozen + gated gamma)": w * s_(phi)[xs] + s_(PAIR) + kg * np.where(sig, qf, 0.0), "+ a s(A), a = h(t1)": w * s_(phi)[xs] + s_(PAIR + h1 * Av) + kg * np.where(sig, qf, 0.0), "+ a s(A), a = h(t1)^2": w * s_(phi)[xs] + s_(PAIR + h1 ** 2 * Av) + kg * np.where(sig, qf, 0.0), "+ a s(A), a = h(t1); r0 rises": w * s_(phi)[xs] + s_(PAIR + h1 * Av) + kg * np.where(sig, qfh, 0.0)}
    TS = np.nan_to_num(FORMS["A (frozen + gated gamma)"])
    Rk = int(y.sum()); h = tie_hits(TS, y, Rk); best = max(((aps(y, np.nan_to_num(c_)) / base, nm_, np.nan_to_num(c_)) for nm_, c_ in comp.items()), key=lambda t_: t_[0])
    out2 = {k_: round(aps(y, np.nan_to_num(v_)) / base, 2) for k_, v_ in FORMS.items()}; out2["h(t1)"] = round(h1, 3); out2["bar"] = round(best[0], 2); rows2.append(dict(panel=ds, days=win, band=band, **out2))
    rows.append(dict(panel=ds, days=win, band=band, calls=Rk, hits=round(h, 1), **{"prec %": round(100 * h / Rk, 2), "rec %": round(100 * h / Rk, 2)}, no=0, AUROC=round(auc(y, TS), 3), AUPRC=round(aps(y, TS), 3), **{"x base precision": "%.2fx (AUPRC %.2fx)" % ((h / Rk) / base, aps(y, TS) / base), "best competitor": "%s, AUPRC %.3f (%.2fx), x base precision %.2f" % (best[1], aps(y, best[2]), best[0], (tie_hits(best[2], y, Rk) / Rk) / base), "w / g / v / kappa_gamma": "%.2f / %.2f / %.2f / %.2f" % (w, gg, v, kg), "genes": len(G)}))
pd.set_option("display.width", 360); pd.set_option("display.max_colwidth", 70); print("TwinScore + gated gamma  filters: none; larger sample = t2; gamma bounded (r0 = 0.69), weight kappa_gamma, cross-significant pairs only"); print(pd.DataFrame(rows).to_string(index=False)); print("\nAUPRC x base, the cross-time co-variation term added with weight a = h(t1):"); print(pd.DataFrame(rows2).to_string(index=False))
