"""2026-09-23: LARRY correlation_high -- TwinScore_supplement rescored with the CURRENT (v3) formula
from fatemap_pipeline/apply_twinscore_supplement_fatemap_allpairs.py, then a term ablation across
candidate-regulator ("TF") universes.

What changes vs the 2026-09-19 LARRY run (apply_twinscore_supplement_larry.py):
  * phi: no significance gate and no panel-median placeholder. phi = rho_dagger_gg / sqrt(h1*h2),
    each h floored individually at its OWN clone-label-permutation null SD (bootstrap_null_sd_diag),
    clipped to +/-10, NaN only when h1<=0 or h2<=0 (v3 default, not the neg-floor-only variant).
  * w_rel: same bootstrap_phi_noise_floor, with thr = median over genes of 2 * per-gene null SD.
  * gate_g: sd_twin_t2 from clone-label permutation of off-diagonal C (was analytic null_sd).
  * sd_reg: bootstrap too, but R = CLR(zreg) is invariant to a global rescale of zreg, so R is
    unchanged -- asserted below rather than assumed.
Everything else (S/zhet/z_dagger from yscher's real permutation run, D=PIDC, Wz, direction term)
does not depend on the above and is reused from the saved correlation_high_pair_terms.csv.

Input: resources/twinfer_input_yscher_actual_cp10k/correlation_high.csv (same log1p(CP10k) table
load_raw builds from the full mtx; checked: h_t1 and rho_dagger_gg reproduce the saved gene terms
to 1e-16). Avoids loading the 570 MB mtx, which OOMs on the login node.

Universes (pairs a->b, a = candidate regulator):
  ALL          every ordered pair (as scored)
  TF_like      a in CollecTRI sources U AnimalTFDB3 (the 2026-09-20 handoff's TF-sourced universe)
  AnimalTFDB   a in AnimalTFDB3 only (reference-independent TF list)
  CollecTRI_src a is a CollecTRI source (CIRCULAR: taken from the reference itself -- upper bound only)
Scoring modes: "subset" = scored on the full universe, then filtered (post-hoc, as on 09-20);
"restd" = s() re-standardised within the restricted universe (gates/w/CLR kept from the full run).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from paper_analysis.larry_hematopoiesis_validation import apply_twinscore_supplement_larry as m

GS = "correlation_high"
SEED = m.SEED
OUT_PAIR = f"{HERE}/{GS}_v3_pair_terms.csv"
OUT_GENE = f"{HERE}/{GS}_v3_gene_terms.csv"
OUT_ABL = f"{HERE}/{GS}_v3_tf_universe_ablation.csv"
log = m.log
s = m.s


# ------------------------------------------------------------------ v3 noise floors (copied from
# fatemap_pipeline/apply_twinscore_supplement_fatemap_allpairs.py, bound to the LARRY module's
# sister_matrix_from_frame / same_cell_matrix instead of the fatemap module's)
def bootstrap_null_sd_diag(raw_t, genes, seed, batch=30, min_batches=3, max_perm=210, tol=0.05):
    rng = np.random.default_rng(seed)
    draws = {g: [] for g in genes}
    prev = None
    n_done = 0
    while n_done < max_perm:
        for _ in range(batch):
            shuffled = raw_t.copy()
            shuffled["clone_id"] = rng.permutation(shuffled["clone_id"].to_numpy())
            try:
                C_perm, _ = m.sister_matrix_from_frame(shuffled, genes)
            except ValueError:
                continue
            for g in genes:
                v = C_perm.loc[g, g]
                if np.isfinite(v):
                    draws[g].append(v)
        n_done += batch
        cur = {g: (np.std(draws[g], ddof=1) if len(draws[g]) > 5 else np.nan) for g in genes}
        if n_done >= min_batches * batch and prev is not None:
            finite_pairs = [(cur[g], prev[g]) for g in genes if np.isfinite(cur[g]) and np.isfinite(prev[g])]
            if finite_pairs:
                rel = np.median([abs(c - p) / p for c, p in finite_pairs if p > 0])
                if rel < tol:
                    prev = cur
                    break
        prev = cur
    return prev, n_done


def bootstrap_null_sd_offdiag(raw_t, genes, seed, kind, extra=None, batch=8, min_batches=2, max_perm=48, tol=0.03):
    rng = np.random.default_rng(seed)
    off_mask = ~np.eye(len(genes), dtype=bool)
    vals = []
    n_done = 0
    prev_sd = None
    while n_done < max_perm:
        for _ in range(batch):
            shuffled = raw_t.copy()
            shuffled["clone_id"] = rng.permutation(shuffled["clone_id"].to_numpy())
            try:
                C_perm, _ = m.sister_matrix_from_frame(shuffled, genes)
                if kind == "C":
                    v = C_perm.to_numpy()[off_mask]
                else:
                    S_perm = m.same_cell_matrix(shuffled, genes)
                    v = (S_perm.to_numpy() - extra.to_numpy() * C_perm.to_numpy())[off_mask]
            except ValueError:
                continue
            vals.extend(v[np.isfinite(v)].tolist())
        n_done += batch
        cur_sd = np.std(vals, ddof=1) if len(vals) > 5 else np.nan
        if n_done >= min_batches * batch and prev_sd is not None and np.isfinite(cur_sd) and prev_sd > 0:
            if abs(cur_sd - prev_sd) / prev_sd < tol:
                prev_sd = cur_sd
                break
        prev_sd = cur_sd
    return prev_sd, n_done


def persistence_v3(h1, h2, rdg, genes, nf1, nf2, t1_raw, t2_raw):
    """v3 default branch of persistence_no_gate + _finish_phi."""
    PHI_CLIP = 10.0
    phi = {}
    for g in genes:
        ok = np.isfinite(h1[g]) and np.isfinite(h2[g]) and h1[g] > 0 and h2[g] > 0
        if not ok:
            phi[g] = np.nan
            continue
        denom = np.sqrt(max(h1[g], nf1[g]) * max(h2[g], nf2[g]))
        raw = rdg[g] / denom if denom > 0 else np.nan
        phi[g] = float(np.clip(raw, -PHI_CLIP, PHI_CLIP)) if np.isfinite(raw) else np.nan
    vals = np.array([v for v in phi.values() if np.isfinite(v)])
    var_phi = vals.var(ddof=1) if len(vals) > 1 else np.nan
    thr1 = float(np.median([2.0 * nf1[g] for g in genes]))
    thr2 = float(np.median([2.0 * nf2[g] for g in genes]))
    noise_var = m.bootstrap_phi_noise_floor(t1_raw, t2_raw, genes, thr1, thr2)
    floor = float(np.median(list(noise_var.values()))) if noise_var else 0.0
    w = max(0.0, 1.0 - floor / var_phi) if (np.isfinite(var_phi) and var_phi > 0) else 0.0
    return phi, w, floor, var_phi


# ------------------------------------------------------------------ metrics
def report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    base = float(y.mean())
    k = int(y.sum())
    top = np.argsort(-sc, kind="stable")[:k]
    auprc, ap = auc(rec, prec), average_precision_score(y, sc)
    return dict(n=len(y), n_true=k, base=base, hits_at_k=int(y[top].sum()),
                auroc=roc_auc_score(y, sc), auprc=auprc, auprc_x=auprc / base,
                ap=ap, ap_x=ap / base)


def main():
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] raw = pd.read_csv(f"{HERE}/resources/twinfer_input_yscher_actual_cp10k/{GS}.csv")
    raw = pd.read_csv(f"{RES_HERE}/resources/twinfer_input_yscher_actual_cp10k/{GS}.csv")
    genes = m.yscher_genes(GS)
    t1_raw = raw[raw.time_step == m.T1].reset_index(drop=True)
    t2_raw = raw[raw.time_step == m.T2].reset_index(drop=True)
    old = pd.read_csv(f"{HERE}/{GS}_pair_terms.csv")
    old_gene = pd.read_csv(f"{HERE}/{GS}_gene_terms.csv").set_index("gene")

    log(f"[{GS}] C, h, rho_dagger (recomputed; checked against saved gene terms)")
    C_t1, me1 = m.sister_matrix_from_frame(t1_raw, genes)
    C_t2, me2 = m.sister_matrix_from_frame(t2_raw, genes)
    X, _ = m.cross_matrix(t1_raw, t2_raw, genes)
    h1 = {g: C_t1.loc[g, g] for g in genes}
    h2 = {g: C_t2.loc[g, g] for g in genes}
    rdg = {g: X.loc[g, g] for g in genes}
    for col, d in [("h_t1", h1), ("h_t2", h2), ("rho_dagger_gg", rdg)]:
        diff = np.nanmax(np.abs(np.array([d[g] for g in genes]) - old_gene.loc[genes, col].to_numpy()))
        assert diff < 1e-9, (col, diff)

    log(f"[{GS}] per-gene h null SD (clone-label permutation)")
    b1, n1 = bootstrap_null_sd_diag(t1_raw, genes, seed=SEED)
    b2, n2 = bootstrap_null_sd_diag(t2_raw, genes, seed=SEED + 1)
    nf1 = {g: b1[g] if np.isfinite(b1[g]) else m.null_sd(me1) for g in genes}
    nf2 = {g: b2[g] if np.isfinite(b2[g]) else m.null_sd(me2) for g in genes}
    log(f"    {n1}/{n2} perms; median null SD t1={np.median(list(nf1.values())):.4f} "
        f"(analytic {m.null_sd(me1):.4f}), t2={np.median(list(nf2.values())):.4f} (analytic {m.null_sd(me2):.4f})")

    phi, w_rel, phi_floor, var_phi = persistence_v3(h1, h2, rdg, genes, nf1, nf2, t1_raw, t2_raw)
    n_nan = sum(not np.isfinite(v) for v in phi.values())
    log(f"    phi v3: {n_nan}/{len(genes)} NaN, w_rel={w_rel:.3f} (noise floor {phi_floor:.4f}, var_phi {var_phi:.4f})")

    log(f"[{GS}] gate_g / sd_reg from permutation null")
    U = list(zip(old.gene_1, old.gene_2))
    lam = pd.DataFrame(np.nan, index=genes, columns=genes)
    for (a, b), l in zip(U, old["lambda"]):
        lam.loc[a, b] = l
    np.fill_diagonal(lam.values, 0.0)
    sd_twin_t2, nb = bootstrap_null_sd_offdiag(t2_raw, genes, seed=SEED + 3, kind="C")
    zC = C_t2 / sd_twin_t2
    gate_g = m.signal_share(zC.to_numpy()[~np.eye(len(genes), dtype=bool)])
    sd_reg, nr = bootstrap_null_sd_offdiag(t2_raw, genes, seed=SEED + 2, kind="reg", extra=lam)
    zreg_old = pd.DataFrame(np.nan, index=genes, columns=genes)
    for (a, b), z in zip(U, old.zreg):
        zreg_old.loc[a, b] = z
    zreg_new = zreg_old * (m.DEFAULT_SD["het_t1"] / sd_reg)
    R_fn = m.clr_calibrate(zreg_new, genes)
    R_new = np.array([R_fn(a, b) for a, b in U])
    assert np.allclose(R_new, old.R.to_numpy(), atol=1e-9), "R should be invariant to sd_reg"
    log(f"    sd_twin_t2={sd_twin_t2:.4f} ({nb} perms, analytic {m.null_sd(me2):.4f}) -> gate_g "
        f"{old.gate_g.iloc[0]:.3f} -> {gate_g:.3f};  sd_reg={sd_reg:.4f} ({nr} perms, was "
        f"{m.DEFAULT_SD['het_t1']:.4f}); R unchanged (asserted)")

    # ---------------------------------------------------------- rescore
    v_gate = float(old.gate_v.iloc[0])
    phi_x = np.array([phi[a] for a, b in U])
    new = old.drop(columns=["scenario"]).copy()
    new["zreg"] = [zreg_new.loc[a, b] for a, b in U]
    new["gate_g"] = gate_g
    new["phi_x"] = phi_x
    new["w_rel"] = w_rel
    new["PAIR"] = s(new.D) + gate_g * s(new.R) + gate_g * v_gate * s(new.Wz)
    new["TwinScore"] = w_rel * s(phi_x) + s(new.PAIR) + new.direction_term
    new["TwinScore_v1_0919"] = old.TwinScore
    new.to_csv(OUT_PAIR, index=False)
    pd.DataFrame({"gene": genes, "h_t1": [h1[g] for g in genes], "h_t2": [h2[g] for g in genes],
                  "rho_dagger_gg": [rdg[g] for g in genes], "nf_h1": [nf1[g] for g in genes],
                  "nf_h2": [nf2[g] for g in genes], "phi_v3": [phi[g] for g in genes],
                  "phi_v1_0919": old_gene.loc[genes, "phi"].to_numpy()}).to_csv(OUT_GENE, index=False)
    log(f"wrote {OUT_PAIR}\nwrote {OUT_GENE}")

    # ---------------------------------------------------------- universes
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ct = pd.read_csv(f"{HERE}/resources/collectri_mouse.tsv", sep="\t")
    ct = pd.read_csv(f"{RES_HERE}/resources/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    assert (new.collectri_edge.to_numpy() == np.array([int(p in CE) for p in U])).all()
    ctf = set(ct.source_genesymbol)
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] atf = set(x.strip() for x in open(f"{HERE}/resources/mouse_TF_AnimalTFDB3.txt"))
    atf = set(x.strip() for x in open(f"{RES_HERE}/resources/mouse_TF_AnimalTFDB3.txt"))
    universes = {"ALL": set(genes), "TF_like": set(genes) & (ctf | atf),
                 "AnimalTFDB": set(genes) & atf, "CollecTRI_src": set(genes) & ctf}
    for k, v in universes.items():
        log(f"    {k}: {len(v)} source genes: {sorted(v) if len(v) < len(genes) else 'all'}")

    # ---------------------------------------------------------- score variants (term ablation)
    def variants(df):
        """Every s() is taken over the rows of `df` (full universe for "subset", restricted for "restd")."""
        sD, sR, sW = s(df.D), s(df.R), s(df.Wz)
        sP = s(sD + gate_g * sR + gate_g * v_gate * sW)
        wphi = w_rel * s(df.phi_x)
        dirt = df.direction_term.to_numpy()
        return {
            "TwinScore (v3 full)": wphi + sP + dirt,
            "- phi term": sP + dirt,
            "- direction term": wphi + sP,
            "- R (PAIR = D only)": wphi + s(sD + gate_g * v_gate * sW) + dirt,
            "- D (PAIR = gR only)": wphi + s(gate_g * sR + gate_g * v_gate * sW) + dirt,
            "PAIR alone": sP,
            "D alone (PIDC)": sD,
            "R alone": sR,
            "phi_x alone": s(df.phi_x),
            "direction alone": dirt,
            "TwinScore v1 (09-19 run)": df.TwinScore_v1_0919.to_numpy(),
        }

    full_var = variants(new)
    assert np.allclose(full_var["TwinScore (v3 full)"], new.TwinScore.to_numpy())
    rows = []
    for uname, src in universes.items():
        df = new[new.gene_1.isin(src)]
        y = df.collectri_edge.to_numpy()
        sel = df.index.to_numpy()
        for vname, sc in full_var.items():
            rows.append(dict(universe=uname, mode="subset", score=vname, **report(sc[sel], y)))
        if uname != "ALL":
            for vname, sc in variants(df.reset_index(drop=True)).items():
                rows.append(dict(universe=uname, mode="restd", score=vname, **report(sc, y)))

    # ---------------------------------------------------------- competitors
    U_full = U
    idx_full = {p: i for i, p in enumerate(U_full)}
    comp = {}
    for name in ["rho", "ppcor", "pidc", "genie3", "grnboost2"]:
        sc = m.load_competitor_allgenes(name, GS, U_full, genes)
        if sc is not None:
            comp[name] = sc
    for name in ["genie3", "grnboost2"]:
        f = f"{HERE}/{name}_{GS}_tfonly.csv"
        d = pd.read_csv(f)
        mm = {(a, b): v for a, b, v in d.itertuples(index=False)}
        comp[f"{name} (TF-only rerun)"] = np.array([mm.get(p, 0.0) for p in U_full])
    tf_like_rerun_src = set(pd.read_csv(f"{HERE}/genie3_{GS}_tfonly.csv").TF)
    for uname, src in universes.items():
        sel = new.gene_1.isin(src).to_numpy()
        y = new.collectri_edge.to_numpy()[sel]
        for name, sc in comp.items():
            if "TF-only" in name and not src <= tf_like_rerun_src:
                continue  # rerun only has TF_like regulators; meaningless on a wider universe
            rows.append(dict(universe=uname, mode="subset", score=f"[comp] {name}", **report(sc[sel], y)))

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_ABL, index=False)
    log(f"wrote {OUT_ABL}")
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.3f}".format):
        for (u, md), g in tab.groupby(["universe", "mode"], sort=False):
            print(f"\n=== universe={u}  mode={md}  (n={g.n.iloc[0]}, n_true={g.n_true.iloc[0]}, base={g.base.iloc[0]:.4f}) ===")
            print(g[["score", "hits_at_k", "auroc", "auprc", "auprc_x", "ap", "ap_x"]].to_string(index=False))


if __name__ == "__main__":
    main()
