#!/usr/bin/env python3
"""Venn diagrams of the three Stage I top-N lists per dataset, one boxed example list per region joined to its region by a leader line.

Lists (all-TF Stage I run, TFs detected in >= 20 % of clonal cells; AEBP1, CENPA, HMGA1, YBX1 removed; listed pairs = Stage I after the gene filter):
  z_rho    : co-expression, ranked by |z_rho|
  z_rho_s  : cross-twin (shared part), ranked by |z_C|  (column 'h' of the TwinScore table)
  z_rho_D  : twin-difference (unshared part), ranked by |z_U|
Sets = TF-target pairs in the top N (default 100) of each list. Examples of a region are ordered by their average rank over the lists that contain the pair
(one list: the rank there; two lists: mean of the two ranks; three lists: mean of the three). Ranks are over all listed pairs of the dataset (taken from the top1000 files).
Outputs: analysis_data/fm_meta_programs_comparison/venn_top<N>/<fm>_venn_top<N>.pdf (Illustrator-editable: Arial, fonttype 42, no clip paths, transparent, boxes without fill), a .png preview and <fm>_venn_top<N>_regions.csv.
A layout check (all text inside its box, no overlapping boxes, nothing outside the page) is run on every figure.   usage: stage1_top100_venn.py [N=100]"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, Rectangle

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 100; PROP = len(sys.argv) > 3 and sys.argv[3] == "prop"   # 4th argument "prop": circle areas proportional to list sizes and overlap areas fitted to the overlap sizes, bigger fonts
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"   # "yscher": the PDF setup of yscher (Ys) - Stage I pairs of the ChIP-Atlas candidate edges, co-expression ranked by |rho|, gene filter applied, ranks recomputed inside the filtered set (FM01); "all": gene-filtered lists (top1000 files); "robust": gene filter AND within-sample robust pairs (pooled + every replicate, same sign), ranks recomputed inside that set
OUT = os.environ.get("VENN_OUT") or f"{ROOT}/analysis_data/fm_meta_programs_comparison/venn_top{N}" + {"robust": "_robust", "yscher": "_yscher", "yscher_drop": "_yscher_nobackfill", "drop": "_nobackfill", "chip": "_chip", "chip_drop": "_chip_nobackfill"}.get(MODE, "") + ("_proportional" if PROP else ""); os.makedirs(OUT, exist_ok=True)
fm.fontManager.addfont(f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts/Arial.ttf")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "cm"})
LISTS = [("z_rho", "$z_{\\rho}$", "#91bfdb"), ("z_C", "$z_{\\hat\\rho_s}$", "#fc8d59"), ("z_U", "$z_{\\hat\\rho_\\Delta}$", "#ffffbf")]   # user-chosen palette (RdYlBu members)
if MODE in ("yscher", "yscher_drop", "chip", "chip_drop"): LISTS[0] = ("z_rho", "$\\rho$", LISTS[0][2])   # her co-expression list is ranked by |rho|
W, H = 16.0, 13.6                                                     # page in inches = data units
VC = (8.0, 6.4); R = 1.65                                             # Venn centre, circle radius
CENT = {0: (VC[0] - 0.9, VC[1] + 0.52), 1: (VC[0] + 0.9, VC[1] + 0.52), 2: (VC[0], VC[1] - 1.04)}
ANCH = {"100": (VC[0] - 1.65, VC[1] + 0.75), "010": (VC[0] + 1.65, VC[1] + 0.75), "001": (VC[0], VC[1] - 1.95), "110": (VC[0], VC[1] + 1.45), "101": (VC[0] - 0.95, VC[1] - 0.55), "011": (VC[0] + 0.95, VC[1] - 0.55), "111": (VC[0], VC[1] + 0.1)}
BH_ = 2.6                                                             # box height (header + 4 examples)
BOX = {"110": (5.6, 9.95, 4.8, BH_), "100": (0.35, 7.55, 4.6, BH_), "010": (11.05, 7.55, 4.6, BH_), "101": (0.35, 3.85, 4.6, BH_), "011": (11.05, 3.85, 4.6, BH_), "001": (5.6, 0.5, 4.8, BH_), "111": (11.05, 0.5, 4.6, BH_)}
LEAD = {"110": (8.0, 9.95), "100": (4.95, 8.85), "010": (11.05, 8.85), "101": (4.95, 5.15), "011": (11.05, 5.15), "001": (8.0, 3.1), "111": (11.05, 3.1)}   # leader start on the box edge
NAME = {"100": "only $z_{\\rho}$", "010": "only $z_{\\hat\\rho_s}$", "001": "only $z_{\\hat\\rho_\\Delta}$", "110": "$z_{\\rho}$ and $z_{\\hat\\rho_s}$", "101": "$z_{\\rho}$ and $z_{\\hat\\rho_\\Delta}$", "011": "$z_{\\hat\\rho_s}$ and $z_{\\hat\\rho_\\Delta}$", "111": "all three"}
if MODE in ("yscher", "yscher_drop", "chip", "chip_drop"): NAME = {k: v.replace("$z_{\\rho}$", "$\\rho$") for k, v in NAME.items()}   # co-expression set is labelled rho in this mode
PLAIN = {"100": "only rho", "010": "only rho_s", "001": "only rho_D", "110": "rho and rho_s", "101": "rho and rho_D", "011": "rho_s and rho_D", "111": "all three"}
NEX = 5
SCEN = {"100": None, "010": 4, "001": 2, "110": 4, "101": 2, "011": 3, "111": 3}   # scenario from the (rho_s, rho_D) pair: top 100 of that list = 1 (!= 0), not = 0 (~ 0); (0,0)=1, (0,1)=2, (1,1)=3, (1,0)=4; the rho (co-expression) list is not used

def prop_layout(sizes, cnt, u=0.0962):
    """area-proportional 3-set layout: circle i has area sizes[i]*u; centres are fitted so that the 7 region areas approach cnt*u (least squares on a grid). Returns centres, radii, region anchor points, achieved region sizes (in pairs)."""
    from scipy.optimize import minimize
    r = np.sqrt(np.array(sizes, float) * u / np.pi); codes = ["100", "010", "001", "110", "101", "011", "111"]; tgt = np.array([cnt.get(c, 0) for c in codes], float)
    def lens(r1, r2, d):
        if d >= r1 + r2: return 0.0
        if d <= abs(r1 - r2): return np.pi * min(r1, r2) ** 2
        a1 = np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1)); a2 = np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
        return r1 * r1 * (a1 - np.sin(2 * a1) / 2) + r2 * r2 * (a2 - np.sin(2 * a2) / 2)
    def d_for(i, j, inter):
        if inter <= 0: return r[i] + r[j] + 0.25
        lo, hi = abs(r[i] - r[j]), r[i] + r[j]
        for _ in range(60):
            mid = (lo + hi) / 2
            if lens(r[i], r[j], mid) > inter * u: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    c = {k: cnt.get(k, 0) for k in codes}
    d0 = [d_for(0, 1, c["110"] + c["111"]), d_for(0, 2, c["101"] + c["111"]), d_for(1, 2, c["011"] + c["111"])]
    def centres(d):
        d12, d13, d23 = d; x3 = (d13 ** 2 - d23 ** 2 + d12 ** 2) / (2 * d12); y3 = -np.sqrt(max(d13 ** 2 - x3 ** 2, 0.0))
        return np.array([[0.0, 0.0], [d12, 0.0], [x3, y3]])
    g = np.arange(-5.0, 5.0, 0.03); X, Y = np.meshgrid(g, g); cell = 0.03 ** 2
    def masks(C):
        m = [((X - C[i, 0]) ** 2 + (Y - C[i, 1]) ** 2) <= r[i] ** 2 for i in range(3)]
        return {"100": m[0] & ~m[1] & ~m[2], "010": m[1] & ~m[0] & ~m[2], "001": m[2] & ~m[0] & ~m[1], "110": m[0] & m[1] & ~m[2], "101": m[0] & m[2] & ~m[1], "011": m[1] & m[2] & ~m[0], "111": m[0] & m[1] & m[2]}
    def err(d):
        d = np.maximum(d, 0.05)
        if d[0] + 1e-6 < abs(d[1] - d[2]) or d[0] > d[1] + d[2]: return 1e6
        M = masks(centres(d)); return float(sum((M[k].sum() * cell / u - t) ** 2 for k, t in zip(codes, tgt)))
    best = minimize(err, d0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
    C = centres(np.maximum(best.x, 0.05)); M = masks(C); ach = {k: float(M[k].sum() * cell / u) for k in codes}
    from scipy.ndimage import distance_transform_edt
    def inner(mask, xs, ys):
        d = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]; j = np.unravel_index(np.argmax(d), d.shape); return (float(xs[j]), float(ys[j]))
    anch = {}
    gf = np.arange(-5.0, 5.0, 0.004)
    for k in codes:
        mem = [i for i, ch in enumerate(k) if ch == "1"]; non = [i for i, ch in enumerate(k) if ch == "0"]
        if M[k].sum() >= 3: anch[k] = inner(M[k], X, Y); continue      # the region has area on the grid: its innermost point
        if len(mem) >= 2:   # tiny region: search a fine grid inside the member circles' overlap, outside the others
            x0_ = max(C[i, 0] - r[i] for i in mem); x1_ = min(C[i, 0] + r[i] for i in mem); y0_ = max(C[i, 1] - r[i] for i in mem); y1_ = min(C[i, 1] + r[i] for i in mem)
            if x0_ < x1_ and y0_ < y1_:
                fx, fy = np.meshgrid(gf[(gf >= x0_) & (gf <= x1_)], gf[(gf >= y0_) & (gf <= y1_)])
                ins = np.ones(fx.shape, bool)
                for i in mem: ins &= ((fx - C[i, 0]) ** 2 + (fy - C[i, 1]) ** 2) <= r[i] ** 2
                if ins.any():
                    out = ins.copy()
                    for i in non: out &= ((fx - C[i, 0]) ** 2 + (fy - C[i, 1]) ** 2) > r[i] ** 2
                    if out.any(): anch[k] = inner(out, fx, fy); continue
                    # the overlap lies inside the other circle(s): the most exposed point of the overlap (farthest beyond the non-member circles' edges)
                    sc = np.min([np.hypot(fx - C[i, 0], fy - C[i, 1]) - r[i] for i in non], axis=0) if non else np.zeros(fx.shape); sc = np.where(ins, sc, -np.inf); j = np.unravel_index(np.argmax(sc), sc.shape)
                    anch[k] = (float(fx[j]), float(fy[j])); continue
            cen = C[mem].mean(0); anch[k] = (float(cen[0]), float(cen[1])); continue
        v = C[mem[0]] - C.mean(0); cen = C[mem[0]] + r[mem[0]] * 0.6 * v / np.hypot(*v); anch[k] = (float(cen[0]), float(cen[1]))
    return C, r, anch, ach


def robust_lists(ds):
    """listed pairs (Stage I, gene filter, 4 regulators removed) that are also robust (pooled and every replicate, same sign); ranked by |z| inside that set"""
    import glob
    from scipy.stats import norm
    D = f"{ROOT}/analysis_data/{ds.lower()}/data/stage1_tf_ge20"; cols = ["TF", "gene", "rho", "z_rho", "h", "z_U", "filtered_out"]
    f = f"{D}/pdf_procedure_all_candidate_pairs.csv"
    if os.path.exists(f): A = pd.read_csv(f, usecols=cols + ["stage1"], dtype={"TF": "category", "gene": "category"})
    else:
        A = pd.concat([pd.read_csv(g, usecols=cols, dtype={"TF": "category", "gene": "category"}) for g in sorted(glob.glob(f"{D}/part*/pdf_procedure_all_candidate_pairs.csv"))], ignore_index=True)
        pv = 2 * norm.sf(np.abs(A.z_rho.to_numpy(np.float64))); o = np.argsort(pv); q = pv[o] * len(pv) / (np.arange(len(pv)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); A["stage1"] = qq <= 0.05
    A["TF"] = A.TF.astype(str); A["gene"] = A.gene.astype(str)
    K = pd.read_csv(f"{D}/stage1_pooled_and_all_samples_zrho.csv.gz", usecols=["TF", "gene"]); K["robust"] = True
    L = A[A.stage1 & ~A.filtered_out & ~A.TF.isin({"AEBP1", "CENPA", "HMGA1", "YBX1"})].merge(K, on=["TF", "gene"], how="inner").rename(columns={"h": "z_C"})
    print(ds, f"Stage I {int(A.stage1.sum()):,} -> gene-filtered {int((A.stage1 & ~A.filtered_out).sum()):,} -> also robust {len(L):,}", flush=True)
    for z in ("z_rho", "z_C", "z_U"): L["rank_" + z] = L[z].abs().rank(ascending=False, method="first").astype(int)
    top, rk = {}, {}
    for z in ("z_rho", "z_C", "z_U"):
        T = L.nsmallest(N, "rank_" + z).copy(); T["rank"] = T["rank_" + z]; T["pair"] = T.TF + "|" + T.gene; top[z] = T[["rank", "TF", "gene", "rho", "rank_z_rho", "rank_z_C", "rank_z_U", "pair"]]; rk[z] = top[z].set_index("pair")
    return top, rk, len(L)


def yscher_lists(ds, gene_filter=True):
    """yscher's setup (the PDF fm01_nofilter.pdf): Stage I pairs of the candidate edges (TF x ChIP-Atlas-bound target), gene filter applied; co-expression ranked by |rho|, shared by |z_C| (column h), unshared by |z_U|; ranks inside the filtered set"""
    B = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "rho", "h", "z_U", "stage1", "filtered_out"])
    L = (B[B.stage1 & ~B.filtered_out] if gene_filter else B[B.stage1]).rename(columns={"h": "z_C"}).copy()   # chip mode: ChIP-Atlas candidate edges only, NO gene filter
    print(ds, f"ChIP-Atlas edges: candidate {len(B):,} -> Stage I {int(B.stage1.sum()):,}" + (f" -> gene-filtered {len(L):,}" if gene_filter else " (no gene filter)"), flush=True)
    for key, col in (("z_rho", "rho"), ("z_C", "z_C"), ("z_U", "z_U")): L["rank_" + key] = L[col].abs().rank(ascending=False, method="first").astype(int)
    top, rk = {}, {}
    for z in ("z_rho", "z_C", "z_U"):
        T = L.nsmallest(N, "rank_" + z).copy(); T["rank"] = T["rank_" + z]; T["pair"] = T.TF + "|" + T.gene; top[z] = T[["rank", "TF", "gene", "rho", "rank_z_rho", "rank_z_C", "rank_z_U", "pair"]]; rk[z] = top[z].set_index("pair")
    return top, rk, len(L)


def nobackfill_lists(ds, yscher):
    """top-N of each list in the UNFILTERED Stage I set, then the pairs flagged by the gene filter (and, for the all-TF universe, the 4 removed regulators) are DROPPED with no replacements: lists end up shorter and of different sizes.
    Ranks stay the unfiltered ranks. yscher=True: her edges (FM01 candidate table, co-expression ranked by |rho|); False: all-TF run (co-expression ranked by |z_rho|)."""
    import glob
    from scipy.stats import norm
    if yscher:
        S = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "rho", "h", "z_U", "stage1", "filtered_out"]); S = S[S.stage1].copy(); S["bad"] = S.filtered_out.astype(bool); cr = "rho"
    else:
        D = f"{ROOT}/analysis_data/{ds.lower()}/data/stage1_tf_ge20"; cols = ["TF", "gene", "rho", "z_rho", "h", "z_U", "filtered_out"]; f = f"{D}/pdf_procedure_all_candidate_pairs.csv"
        if os.path.exists(f): A = pd.read_csv(f, usecols=cols + ["stage1"], dtype={"TF": "category", "gene": "category"})
        else:
            A = pd.concat([pd.read_csv(g, usecols=cols, dtype={"TF": "category", "gene": "category"}) for g in sorted(glob.glob(f"{D}/part*/pdf_procedure_all_candidate_pairs.csv"))], ignore_index=True)
            pv = 2 * norm.sf(np.abs(A.z_rho.to_numpy(np.float64))); o = np.argsort(pv); q = pv[o] * len(pv) / (np.arange(len(pv)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); A["stage1"] = qq <= 0.05
        S = A[A.stage1].copy(); S["TF"] = S.TF.astype(str); S["gene"] = S.gene.astype(str); S["bad"] = S.filtered_out.astype(bool) | S.TF.isin({"AEBP1", "CENPA", "HMGA1", "YBX1"}); cr = "z_rho"; del A
    S = S.rename(columns={"h": "z_C"}).reset_index(drop=True)
    for key, col in (("z_rho", cr), ("z_C", "z_C"), ("z_U", "z_U")): S["rank_" + key] = S[col].abs().rank(ascending=False, method="first").astype(int)
    top, rk = {}, {}
    for z in ("z_rho", "z_C", "z_U"):
        T = S.nsmallest(N, "rank_" + z); n0 = len(T); T = T[~T.bad].copy(); T["rank"] = T["rank_" + z]; T["pair"] = T.TF + "|" + T.gene
        top[z] = T[["rank", "TF", "gene", "rho", "rank_z_rho", "rank_z_C", "rank_z_U", "pair"]]; rk[z] = top[z].set_index("pair")
    print(ds, "flagged pairs dropped:", {z: N - len(top[z]) for z in top}, "-> list sizes", {z: len(top[z]) for z in top}, flush=True)
    return top, rk, len(S)


for ds in (["FM01"] if MODE in ("yscher", "yscher_drop") else ["FM01", "FM03", "FM05", "FM06"] if MODE in ("chip", "chip_drop") else ["FM01", "FM05", "FM06"] if MODE == "robust" else ["FM01", "FM03", "FM05", "FM06"] if MODE == "drop" else ["FM01", "FM03", "FM05", "FM06"]) if not os.environ.get("VENN_DS") else os.environ["VENN_DS"].split(","):   # VENN_DS=FM01,FM06 restricts the datasets
    top = {}; rk = {}; nuni = None
    if MODE == "robust": top, rk, nuni = robust_lists(ds)
    elif MODE == "yscher": top, rk, nuni = yscher_lists(ds)
    elif MODE == "chip": top, rk, nuni = yscher_lists(ds, gene_filter=False)
    elif MODE in ("yscher_drop", "drop", "chip_drop"): top, rk, nuni = nobackfill_lists(ds, MODE in ("yscher_drop", "chip_drop"))
    else:
        D = f"{ROOT}/analysis_data/{ds.lower()}/data/stage1_tf_ge20/top_lists_classic_TFs"
        for z, _, _ in LISTS:
            T = pd.read_csv(f"{D}/top1000_by_{z}.csv", usecols=["rank", "TF", "gene", "rho", "rank_z_rho", "rank_z_C", "rank_z_U"]); T["pair"] = T.TF.astype(str) + "|" + T.gene.astype(str)
            top[z] = T.head(N); rk[z] = T.set_index("pair")
    sets = {z: set(top[z].pair) for z, _, _ in LISTS}; allp = sets["z_rho"] | sets["z_C"] | sets["z_U"]; rows = []
    for p in allp:
        code = "".join("1" if p in sets[z] else "0" for z, _, _ in LISTS); src = next(rk[z] for z, _, _ in LISTS if p in sets[z]); ranks = {z: int(src.loc[p, "rank_" + z]) for z, _, _ in LISTS}
        mem = [z for (z, _, _), c in zip(LISTS, code) if c == "1"]
        rows.append(dict(region=code, region_name=PLAIN[code], in_rho=int(code[0]), in_rho_s=int(code[1]), in_rho_D=int(code[2]), scenario=SCEN[code], pair=p, TF=p.split("|")[0], gene=p.split("|")[1], rho=float(src.loc[p, "rho"]), rank_z_rho=ranks["z_rho"], rank_z_C=ranks["z_C"], rank_z_U=ranks["z_U"], avg_rank_in_member_lists=np.mean([ranks[z] for z in mem])))
    Rg = pd.DataFrame(rows).sort_values(["region", "avg_rank_in_member_lists"]); Rg.to_csv(f"{OUT}/{ds.lower()}_venn_top{N}_regions.csv", index=False); cnt = Rg.region.value_counts().to_dict()

    LBL = lambda i: f"top {N}" if MODE not in ("yscher_drop", "drop", "chip_drop") else f"{len(top[LISTS[i][0]])} of {N}"
    if PROP:   # larger fonts, wider boxes, area-proportional circles
        PW, PHH = 21.2, 15.3; CXp = PW / 2; bw, bwc, bh = 6.4, 6.4, 3.15
        BOXD = {"110": (CXp - bwc / 2, 11.0, bwc, bh), "100": (0.3, 7.6, bw, bh), "010": (PW - 0.3 - bw, 7.6, bw, bh), "101": (0.3, 4.0, bw, bh), "011": (PW - 0.3 - bw, 4.0, bw, bh), "001": (CXp - bwc / 2, 0.4, bwc, bh), "111": (PW - 0.3 - bw, 0.4, bw, bh)}
        LEADD = {"110": (CXp, 11.0), "100": (0.3 + bw, 7.6 + bh / 2), "010": (PW - 0.3 - bw, 7.6 + bh / 2), "101": (0.3 + bw, 4.0 + bh / 2), "011": (PW - 0.3 - bw, 4.0 + bh / 2), "001": (CXp, 0.4 + bh), "111": (PW - 0.3 - bw, 0.4 + bh)}
        sizes = [len(top[z]) for z, _, _ in LISTS]; Cc, Rr, ANC, ach = prop_layout(sizes, cnt); ACH = ach
        lo = (Cc - Rr[:, None]).min(0); hi = (Cc + Rr[:, None]).max(0); shift = np.array([CXp, 7.35]) - (lo + hi) / 2
        Cc = Cc + shift; ANC = {k: (v[0] + shift[0], v[1] + shift[1]) for k, v in ANC.items()}; CEN = {i: tuple(Cc[i]) for i in range(3)}; RAD = {i: float(Rr[i]) for i in range(3)}
        FS = dict(count=26, setlab=26, hdr=18, ex=14, scen=16, title=24)
        err_pairs = max(abs(ach[k] - cnt.get(k, 0)) for k in ach); print(ds, "area fit (pairs): max |achieved - actual| =", round(err_pairs, 1), {k: (cnt.get(k, 0), round(ach[k], 1)) for k in ach}, flush=True)
        row, hdr_dy, ex_y0, sw = 0.44, 0.14, 0.82, 0.30
    else:
        PW, PHH = W, H; BOXD, LEADD, ANC, CEN, RAD = BOX, LEAD, ANCH, CENT, {0: R, 1: R, 2: R}; ACH = None
        FS = dict(count=17, setlab=14, hdr=11.5, ex=11, scen=11.5, title=18); row, hdr_dy, ex_y0, sw = 0.36, 0.12, 0.66, 0.26
    if PROP:
        def _cross(c1, c2):
            (ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2) = LEADD[c1], ANC[c1], LEADD[c2], ANC[c2]; dn = (ax2 - ax1) * (by2 - by1) - (ay2 - ay1) * (bx2 - bx1)
            if abs(dn) < 1e-9: return False
            u_ = ((bx1 - ax1) * (by2 - by1) - (by1 - ay1) * (bx2 - bx1)) / dn; v_ = ((bx1 - ax1) * (ay2 - ay1) - (by1 - ay1) * (ax2 - ax1)) / dn
            return 0 < u_ < 1 and 0 < v_ < 1
        BOXD = dict(BOXD); LEADD = dict(LEADD)
        for c1 in list(BOXD):
            for c2 in list(BOXD):
                if c1 < c2 and _cross(c1, c2):
                    BOXD[c1], BOXD[c2] = BOXD[c2], BOXD[c1]; LEADD[c1], LEADD[c2] = LEADD[c2], LEADD[c1]   # the two boxes trade places so the leaders no longer cross
    fig = plt.figure(figsize=(PW, PHH)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, PW); ax.set_ylim(0, PHH); ax.axis("off"); ax.set_facecolor("none")
    for i, (z, lab, col) in enumerate(LISTS): ax.add_patch(Circle(CEN[i], RAD[i], facecolor=col, alpha=0.62, edgecolor="#666666", lw=1.0))
    placed = []; inplace = set()   # counts are drawn largest region first; a count that would sit on top of one already drawn is skipped (its box header still gives n)
    for c in sorted(ANC, key=lambda k: -cnt.get(k, 0)):
        x, y = ANC[c]
        if cnt.get(c, 0) > 0 and (ACH is None or ACH.get(c, 0) >= 1.5) and all(np.hypot(x - px, y - py) >= 0.62 for px, py in placed):   # a sliver of fitted area < 1.5 pairs cannot hold a number
            placed.append((x, y)); inplace.add(c); ax.text(x, y, str(cnt.get(c, 0)), ha="center", va="center", fontsize=FS["count"], fontweight="bold", zorder=5, color="#111111")
        elif cnt.get(c, 0) > 0:   # region too small / crowded: its count is written beside the end of its leader line instead, so it stays visible
            v = np.array(LEADD[c]) - np.array((x, y)); v = v / np.hypot(*v); q = np.array([-v[1], v[0]]); q = q if q[1] >= 0 else -q   # perpendicular, pointing up
            t_ = 0.5   # walk out along the leader until the number is in the white space outside every circle
            while t_ < 3.0 and any(np.hypot(*(np.array((x, y)) + t_ * v + 0.27 * q - np.array(CEN[i]))) < RAD[i] + 0.28 for i in range(3)): t_ += 0.1
            lx, ly = np.array((x, y)) + t_ * v + 0.27 * q; ax.text(lx, ly, str(cnt.get(c, 0)), ha="center", va="center", fontsize=FS["count"], fontweight="bold", zorder=5, color="#111111")
    if PROP:   # set labels stacked: a large symbol above its list size, so they stay narrow and clear of the boxes
        for i, (lx, ly, up) in enumerate([(CEN[0][0] - 0.9 * RAD[0] / 1.65, CEN[0][1] + RAD[0] + 0.1, True), (CEN[1][0] + 0.9 * RAD[1] / 1.65, CEN[1][1] + RAD[1] + 0.1, True), (CEN[2][0] - 1.0, CEN[2][1] - RAD[2] - 0.05, False)]):
            if up:
                ax.text(lx, ly, f"({LBL(i)})", ha="center", va="bottom", fontsize=17, color="#222222"); ax.text(lx, ly + 0.36, LISTS[i][1], ha="center", va="bottom", fontsize=FS["setlab"], color="#222222", fontweight="bold")
            else:
                ax.text(lx, ly, LISTS[i][1], ha="center", va="top", fontsize=FS["setlab"], color="#222222", fontweight="bold"); ax.text(lx, ly - 0.62, f"({LBL(i)})", ha="center", va="top", fontsize=17, color="#222222")
    else:
        ax.text(CEN[0][0] - 0.9 * RAD[0] / 1.65, CEN[0][1] + RAD[0] + 0.1, f"{LISTS[0][1]}  ({LBL(0)})", ha="center", va="bottom", fontsize=FS["setlab"], color="#222222", fontweight="bold")
        ax.text(CEN[1][0] + 0.9 * RAD[1] / 1.65, CEN[1][1] + RAD[1] + 0.1, f"{LISTS[1][1]}  ({LBL(1)})", ha="center", va="bottom", fontsize=FS["setlab"], color="#222222", fontweight="bold")
        ax.text(CEN[2][0] - (1.55 if not PROP else 1.0), CEN[2][1] - RAD[2] + (0.15 if not PROP else -0.05), f"{LISTS[2][1]}  ({LBL(2)})", ha="right", va="top", fontsize=FS["setlab"], color="#222222", fontweight="bold")
    if not os.environ.get("VENN_NOTITLE"):   # VENN_NOTITLE=1: figure without the title line
        ax.text(0.35, PHH - 0.3, f"{ds}: TF\u2013target pairs in the top {N} of the three Stage I lists" + (f"  (gene-filtered, within-sample robust pairs; n = {nuni:,})" if MODE == "robust" else f"  (yscher's edges, gene-filtered; n = {nuni:,})" if MODE == "yscher" else f"  (ChIP-Atlas edges only, no gene filter; n = {nuni:,})" if MODE == "chip" else "  (yscher's edges, flagged pairs dropped, no replacements)" if MODE == "yscher_drop" else "  (ChIP-Atlas edges, flagged pairs dropped, no replacements)" if MODE == "chip_drop" else "  (flagged pairs dropped, no replacements)" if MODE == "drop" else ""), fontsize=FS["title"], fontweight="bold", va="top")
    texts = {}; SW = {"100": [0], "010": [1], "001": [2], "110": [0, 1], "101": [0, 2], "011": [1, 2], "111": [0, 1, 2]}
    for c, (x0, y0, w, h) in BOXD.items():
        v = np.array(LEADD[c]) - np.array(ANC[c]); end = np.array(ANC[c]) + ((0.24 if c in inplace else 0.0) if PROP else 0.34) * v / np.hypot(*v)   # leader stops just short of the count
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="#555555", lw=1.0)); ax.plot([LEADD[c][0], end[0]], [LEADD[c][1], end[1]], color="#555555", lw=0.9, zorder=1)
        ax.plot(*end, marker="o", ms=3.5, color="#555555", zorder=6)
        for k, li in enumerate(SW[c]): ax.add_patch(Rectangle((x0 + 0.15 + (sw + 0.04) * k, y0 + h - 0.12 - sw - 0.1), sw, sw, facecolor=LISTS[li][2], edgecolor="#666666", lw=0.8))
        hx = x0 + 0.15 + (sw + 0.04) * len(SW[c]) + 0.08
        t = [ax.text(hx, y0 + h - hdr_dy, f"{NAME[c]}  (n = {cnt.get(c, 0)})", fontsize=FS["hdr"], fontweight="bold", va="top", color="#222222")]
        t.append(ax.text(x0 + w - 0.15, y0 + h - hdr_dy, (f"scenario {SCEN[c]}" if SCEN[c] else "no scenario"), fontsize=FS["scen"], va="top", ha="right", color="#444444"))
        d = Rg[Rg.region == c].head(NEX); yy = y0 + h - ex_y0
        if len(d) == 0: t.append(ax.text(x0 + 0.25, yy, "none", fontsize=FS["ex"], va="top", color="#555555"))
        for r in d.itertuples():
            t.append(ax.text(x0 + 0.25, yy, f"{r.TF}\u2013{r.gene}", fontsize=FS["ex"], va="top")); yy -= row   # rank text removed (ranks stay in the regions csv)
        texts[c] = t
    note = None   # list definitions / filter notes removed
    for a in fig.findobj():
        if hasattr(a, "set_clip_on"): a.set_clip_on(False)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = ax.transData.inverted(); problems = []
    def ext(a):
        b = a.get_window_extent(rend); (x0, y0), (x1, y1) = inv.transform([(b.x0, b.y0), (b.x1, b.y1)]); return x0, y0, x1, y1
    def seg_hits(p, q, box):   # does segment p-q cross the rectangle (strictly inside)?
        x0, y0, w, h = box; n = 200
        return any(x0 + 1e-3 < p[0] + (q[0] - p[0]) * k / n < x0 + w - 1e-3 and y0 + 1e-3 < p[1] + (q[1] - p[1]) * k / n < y0 + h - 1e-3 for k in range(n + 1))
    for c, (x0, y0, w, h) in BOXD.items():
        for t in texts[c]:
            a0, b0, a1, b1 = ext(t)
            if a0 < x0 - 1e-6 or a1 > x0 + w + 1e-6 or b0 < y0 - 1e-6 or b1 > y0 + h + 1e-6: problems.append(f"text outside box {c}: '{t.get_text()[:30]}'")
        for c2, box2 in BOXD.items():
            if c2 > c and not (x0 + w < box2[0] or box2[0] + box2[2] < x0 or y0 + h < box2[1] or box2[1] + box2[3] < y0): problems.append(f"boxes {c} and {c2} overlap")
            if c2 != c and seg_hits(LEADD[c], ANC[c], box2): problems.append(f"leader of {c} crosses box {c2}")
        # leader vs the other leaders
        for c2 in BOXD:
            if c2 > c:
                (ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2) = LEADD[c], ANC[c], LEADD[c2], ANC[c2]
                d = (ax2 - ax1) * (by2 - by1) - (ay2 - ay1) * (bx2 - bx1)
                if abs(d) > 1e-9:
                    u = ((bx1 - ax1) * (by2 - by1) - (by1 - ay1) * (bx2 - bx1)) / d; v = ((bx1 - ax1) * (ay2 - ay1) - (by1 - ay1) * (ax2 - ax1)) / d
                    if 0 < u < 1 and 0 < v < 1: problems.append(f"leaders {c} and {c2} cross")
    boxes_all = list(BOXD.values())
    BOXTEXT_IDS = {id(t) for lst in texts.values() for t in lst}
    for t in ax.texts:
        a0, b0, a1, b1 = ext(t)
        if a0 < 0 or b0 < 0 or a1 > PW or b1 > PHH: problems.append(f"text outside page: '{t.get_text()[:30]}'")
        if id(t) not in BOXTEXT_IDS:
            for c, (x0, y0, w, h) in BOXD.items():
                if not (a1 < x0 or x0 + w < a0 or b1 < y0 or y0 + h < b0): problems.append(f"'{t.get_text()[:14]}' overlaps box {c}")
    print(ds, {PLAIN[c]: cnt.get(c, 0) for c in PLAIN}, "| layout problems:", problems or "none", flush=True)
    fig.savefig(f"{OUT}/{ds.lower()}_venn_top{N}.pdf", format="pdf", transparent=True, facecolor="none", edgecolor="none"); fig.savefig(f"{OUT}/{ds.lower()}_venn_top{N}.png", dpi=100, facecolor="white"); plt.close(fig)
