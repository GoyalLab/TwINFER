# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""The TwinScore as calculate_twin_score defines it, evaluated on the scoring panels.

    TwinScore(x->y) = z(|rho(t1)|) + z(|rho(t2)|) - z(|rho(t2)-rho(t1)|) + pi
                      - |z_div| * I(z_het < -2.326)
                      - z_het   * I(z_d_het > 0)
                      + 1/2 * sign(gamma) * I(|z(gamma)| >= 1)

Every piece is the module's own: the three u_abs_* values are _null_unit(., use_absolute=True)
against the Step-I clone-derangement draws and are panel-standardised here by _panel_z over the
DIRECTED panel, exactly as calculate_twin_score does; z_div and z_het are the t1 within-time
scores from differentiate_single_state_reg_and_multiple_states and enter unstandardised;
z_d_het is (d_obs - mean)/sd of the observed twin change against the difference-shuffle draws;
gamma is signed and z(gamma) is _panel_z(u_gamma). No filter, no threshold, nothing added.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
ROOT = "/gpfs/projects/b1255/yscher/Transcriptomic Distance"
YCOPY = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports"  # [2026-10-01 added: copy of yscher exports (twinscore_script, twinscore_gated, panels/tf_target_panel, _probe_tmp gene_flags)]
# [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] TS = f"{ROOT}/exports/twinscore_script"
TS = f"{YCOPY}/twinscore_script"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] Cm = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_mouse.tsv", sep="\t")
Cm = pd.read_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_mouse.tsv", sep="\t")
_Ch = pd.read_csv(f"{ROOT}/resources/reference/collectri_human.csv")
def curated_for(BN):
    """CollecTRI by species of the run: hPSC and michaels are human, LARRY and CellTag mouse. Runs on the Perturb-seq panels (ps24_top100,
    ps46_top100; helpers/build_perturbseq_panel.py) are scored against that panel's Perturb-seq truth list instead."""
    for _nm in ("ps24_top100", "ps46_top100"):
        if _nm in BN:
            # [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] _t = pd.read_csv(f"{ROOT}/exports/panels/tf_target_panel/{_nm}_truth.csv"); return set(zip(_t.regulator, _t.target))
            _t = pd.read_csv(f"{YCOPY}/panels/tf_target_panel/{_nm}_truth.csv"); return set(zip(_t.regulator, _t.target))
    if "hpsc" in BN or "michaels" in BN or "mich_" in BN or "t26a01" in BN or "t26r12a01" in BN or "mic5a01" in BN:
        return set(zip(_Ch["source"], _Ch["target"]))
    return set(zip(Cm.source_genesymbol, Cm.target_genesymbol))

def panel_z(v):
    """_panel_z in correlation_functions.py: mean/std over the finite values, ddof=0."""
    v = np.asarray(v, float); r = np.full(v.shape, np.nan); f = np.isfinite(v)
    if not f.any(): return r
    x = v[f]; mu = float(x.mean()); sd = float(x.std(ddof=0))
    r[f] = 0.0 if (not np.isfinite(sd) or sd == 0) else (x - mu) / sd
    return r

def panel(BN):
    _ckp = f"{ROOT}/exports/runs/{BN}/stage_inputs_checkpoint.pkl"
    for _root in ("perm4", "celltag_twinfer"):
        if not os.path.exists(_ckp): _ckp = f"{ROOT}/exports/{_root}/{BN}/stage_inputs_checkpoint.pkl"
    ck = pickle.load(open(_ckp, "rb"))
    genes = sorted(ck["curr_gene_list"]); gs = set(genes)
    cur = {(a, b) for a, b in curated_for(BN) if a in gs and b in gs and a != b}
    univ = [(a, b) for a in genes for b in genes if a != b]
    y = np.array([1.0 if p in cur else 0.0 for p in univ])
    R = int(y.sum()); base = R / len(univ)

    def tab(name, key=None):
        return pd.read_csv(f"{TS}/{name}").set_index(["gene_1", "gene_2"])
    def get(d, c, P):
        return np.array([d[c].loc[p] if p in d.index else
                         (d[c].loc[(p[1], p[0])] if (p[1], p[0]) in d.index else np.nan)
                         for p in P], float)
    ZD = tab(f"ZDDAG_{BN}.csv"); DH = tab(f"DHET_{BN}.csv"); DV = tab(f"ZDIV_{BN}_t1.csv")
    DV2 = tab(f"ZDIV_{BN}_t2.csv")
    P = [tuple(p) for p in json.load(open(f"{TS}/{BN}.json"))["pairs"]]
    src = "STAGE1U" if os.path.exists(f"{TS}/STAGE1U_{BN}.csv") else "zrho-standin"
    if src == "STAGE1U":
        S1 = tab(f"STAGE1U_{BN}.csv")
        u1, u2, uc = (get(S1, c, P) for c in
                      ("u_abs_rho_t1", "u_abs_rho_t2", "u_abs_rho_change"))
        rho1 = get(S1, "rho_t1", P); rho2 = get(S1, "rho_t2", P)
    else:
        J = json.load(open(f"{TS}/{BN}.json")); assert "gzu5140" in J["src"]
        u1 = np.abs(np.array(J["z_rho_t1"], float)); u2 = np.abs(np.array(J["z_rho_t2"], float))
        uc = get(tab(f"CHANGE_{BN}.csv"), "z_change", P)
        rho1 = np.full(len(P), np.nan); rho2 = np.full(len(P), np.nan)
    gam = get(ZD, "gamma_obs", P); ug = get(ZD, "z_gamma", P); zdd = get(ZD, "z_Ddag", P)
    J = json.load(open(f"{TS}/{BN}.json")); assert "gzu5140" in J["src"]
    zr1 = np.abs(np.array(J["z_rho_t1"], float)); zr2 = np.abs(np.array(J["z_rho_t2"], float))
    sym = np.minimum(np.abs(np.array(J["z_fwd"], float)), np.abs(np.array(J["z_rev"], float)))
    zfwd = np.array(J["z_fwd"], float); zrev = np.array(J["z_rev"], float)
    dmax = np.maximum(np.abs(np.array(J["z_fwd"], float)), np.abs(np.array(J["z_rev"], float)))
    drift = np.abs(get(DH, "ref_change_mean", P)) / get(DH, "ref_change_sd", P)
    G = {}
    for tp in ("t1", "t2"):
        f_ = f"{TS}/GZUGATED_{BN}_{tp}.csv"
        v_ = get(tab(f"GZUGATED_{BN}_{tp}.csv"), "z_reg_gated", P) \
            if os.path.exists(f_) else np.full(len(P), np.nan)
        G["g" + tp] = np.abs(v_); G["r" + tp] = v_
    zdiv = get(DV, "z_div", P); zhet = get(DV, "z_het", P)
    zdiv2 = get(DV2, "z_div", P); zhet2 = get(DV2, "z_het", P)
    zdhet = ((get(DH, "rd_t2", P) - get(DH, "rd_t1", P)) - get(DH, "ref_change_mean", P)) \
            / get(DH, "ref_change_sd", P)
    return dict(BN=BN, genes=genes, univ=univ, y=y, R=R, base=base, src=src,
                idx={p: i for i, p in enumerate(P)}, P=P,
                u1=u1, u2=u2, uc=uc, gam=gam, ug=ug, zdiv=zdiv, zhet=zhet, zdhet=zdhet,
                rho1=rho1, rho2=rho2, zfwd=zfwd, zrev=zrev, ZD=zdd, zr1=zr1, zr2=zr2, sym=sym, dmax=dmax, drift=drift, zdiv2=zdiv2, zhet2=zhet2, **G)

def s_(v):
    """panel_z with the missing values pushed one unit below the minimum, so nothing abstains."""
    v = np.asarray(v, float); f = np.isfinite(v); z = panel_z(v)
    return np.where(f, z, (np.nanmin(z[f]) - 1.0) if f.any() else 0.0)

def module_score(d, terms=("rho", "change", "div", "het", "gamma"), return_keep=False):
    """Assemble on the DIRECTED universe, then panel_z as calculate_twin_score does."""
    n = len(d["univ"]); D = {}
    for k in ("u1", "u2", "uc", "gam", "ug", "zdiv", "zhet", "zdhet",
              "rho1", "rho2", "ZD", "zr1", "zr2", "sym", "dmax", "drift", "gt1", "gt2", "zdiv2", "zhet2", "rt1", "rt2"):
        v = np.empty(n); v[:] = np.nan
        for u, (x, w) in enumerate(d["univ"]):
            i = d["idx"].get((x, w)); sg = 1.0
            if i is None: i = d["idx"].get((w, x)); sg = -1.0
            if i is not None:
                v[u] = d[k][i] * (sg if k in ("gam", "ug", "ZD") else 1.0)
        D[k] = v
    z1, z2, zc, zg = (panel_z(D[k]) for k in ("u1", "u2", "uc", "ug"))
    sc = np.full(n, np.pi)
    # calculate_twin_score panel-standardises the u_* values and takes z_div, z_het, z_d_het in
    # their own null units. "raw_units" is the other convention applied throughout: every term
    # in null units, so a term sitting at its own null contributes about one unit of noise while
    # one detecting structure contributes proportionally more. No weight is chosen by hand.
    if "raw_units" in terms:
        R1, R2, RC, RG = (np.nan_to_num(D[k]) for k in ("u1", "u2", "uc", "ug"))
    else:
        R1, R2, RC, RG = (np.nan_to_num(v) for v in (z1, z2, zc, zg))
    if "rho" in terms:    sc = sc + R1 + R2
    if "change" in terms: sc = sc - RC
    if "div" in terms:
        gate = np.isfinite(D["zhet"]) & (D["zhet"] < -2.326) & np.isfinite(D["zdiv"])
        sc = sc - np.where(gate, np.abs(D["zdiv"]), 0.0)
    # the same statistic with the gate and the sign varied, to see which reading the data backs
    if "div_lo_plus" in terms:
        g_ = np.isfinite(D["zhet"]) & (D["zhet"] < -2.326) & np.isfinite(D["zdiv"])
        sc = sc + np.where(g_, np.abs(D["zdiv"]), 0.0)
    if "div_hi_minus" in terms:
        g_ = np.isfinite(D["zhet"]) & (D["zhet"] > 2.326) & np.isfinite(D["zdiv"])
        sc = sc - np.where(g_, np.abs(D["zdiv"]), 0.0)
    if "div_hi_plus" in terms:
        g_ = np.isfinite(D["zhet"]) & (D["zhet"] > 2.326) & np.isfinite(D["zdiv"])
        sc = sc + np.where(g_, np.abs(D["zdiv"]), 0.0)
    # figure, 2-state columns: z_d_het is -20 without regulation and -10 with it, the largest
    # regulation contrast of any z in the panel. It enters positively.
    # The six scenarios are a CASE analysis, not a sum: each detector only carries regulation
    # information inside its own regime, so it is applied only there and standardised there.
    #   regime from the state detectors:  drift    z_|drho| > 2.326
    #                                     2 state  z_d_het  < -2.326
    #                                     single   neither
    #   regulation inside the regime:     2 state  + z_d_het   (-10 with reg vs -20 without)
    #                                     drift    - z_|drho|  (2 with reg vs 5.5 without)
    if "regime" in terms:
        drift_r = np.nan_to_num(np.abs(D["uc"]) > 2.326, nan=False).astype(bool)
        two_r = np.nan_to_num(D["zdhet"] < -2.326, nan=False).astype(bool) & ~drift_r
        def within(mask, v, sign):
            out = np.zeros(len(sc))
            if mask.sum() > 2:
                x = np.asarray(v, float)[mask]; f = np.isfinite(x)
                if f.sum() > 2:
                    z = np.zeros(mask.sum())
                    z[f] = (x[f] - x[f].mean()) / max(x[f].std(ddof=1), 1e-12)
                    out[mask] = sign * z
            return out
        sc = sc + within(two_r, D["zdhet"], +1.0)
        sc = sc + within(drift_r, np.abs(D["uc"]), -1.0)
    if "dhet_neg" in terms: sc = sc - s_(D["zdhet"])
    # STABLE, the paper's preliminary step: the correlation must hold its SIGN across the
    # timepoints. A pair whose rho flips sign is a different relationship at t1 and t2.
    stable = np.nan_to_num(np.sign(D["rho1"]) == np.sign(D["rho2"]), nan=False).astype(bool)
    stable &= np.isfinite(D["rho1"]) & np.isfinite(D["rho2"])
    if "stab_term" in terms: sc = sc + stable.astype(float)
    if "F_stable" in terms: keep_stable = stable
    else: keep_stable = None
    # z_het entered symmetrically in the timepoints, negative sign, three combiners
    # the hinge applied to the min / max of z_het over the two timepoints -- symmetric
    for key, comb in (("hinge_min", np.minimum), ("hinge_max", np.maximum)):
        if key in terms:
            g2 = comb(np.nan_to_num(D["zhet"], nan=0.0), np.nan_to_num(D["zhet2"], nan=0.0))
            sc = sc + np.where(g2 < -2.326, g2, 0.0)
    if "het_sum" in terms: sc = sc - s_(D["zhet"]) - s_(D["zhet2"])
    if "het_min" in terms: sc = sc - np.minimum(s_(D["zhet"]), s_(D["zhet2"]))
    if "het_max" in terms: sc = sc - np.maximum(s_(D["zhet"]), s_(D["zhet2"]))
    if "hinge_sum" in terms:
        for k_ in ("zhet", "zhet2"):
            g_ = np.nan_to_num(D[k_] < -2.326, nan=False)
            sc = sc + np.where(g_, np.nan_to_num(D[k_]), 0.0)
    if "het_pos" in terms: sc = sc + s_(D["zhet"])
    if "het_neg" in terms: sc = sc - s_(D["zhet"])
    if "het_hinge_plus" in terms:      # + I(z_het < -2.326) * z_het
        g_ = np.nan_to_num(D["zhet"] < -2.326, nan=False)
        sc = sc + np.where(g_, np.nan_to_num(D["zhet"]), 0.0)
    if "het_hinge_minus" in terms:     # - I(z_het < -2.326) * z_het
        g_ = np.nan_to_num(D["zhet"] < -2.326, nan=False)
        sc = sc - np.where(g_, np.nan_to_num(D["zhet"]), 0.0)
    if "het_gate_only" in terms:
        sc = sc - np.nan_to_num((D["zhet"] < -2.326), nan=False).astype(float)
    if "div_two_minus" in terms:
        g_ = np.isfinite(D["zhet"]) & (np.abs(D["zhet"]) > 2.326) & np.isfinite(D["zdiv"])
        sc = sc - np.where(g_, np.abs(D["zdiv"]), 0.0)
    if "het" in terms:
        gate = np.isfinite(D["zdhet"]) & (D["zdhet"] > 0.0) & np.isfinite(D["zhet"])
        sc = sc - np.where(gate, D["zhet"], 0.0)
    if "gamma" in terms:
        b = 0.5 * np.sign(np.nan_to_num(D["gam"])) * (np.abs(np.nan_to_num(zg)) >= 1.0)
        sc = sc + b
    # alpha is the one free choice the module leaves open, 0.05 or 0.01; both are reported
    te = 1.960 if "alpha05_exist" in terms else 2.576      # Phi^-1(1-alpha/2)
    tr = 2.326 if "alpha01_reg" in terms else 1.645        # norm.ppf(1-alpha)
    if "reg_a001" in terms: tr = 3.090                     # alpha = 0.001
    if "reg_bonf" in terms:                                # alpha = 0.01 / (panel's pairs)
        from scipy import stats as _st
        tr = float(_st.norm.ppf(1.0 - 0.01 / max(1, len(d["P"]))))
    keep_fan = None
    flux_local = None
    cnt_raw_local = None
    # the three beige scenarios, each killed by its own state detector ONLY where the
    # regulation detectors are silent. z_reg and z_div are the two uniformly-positive
    # regulation detectors in the figure; z_drift flags drift, z_d_het flags 2 state.
    if "kill_noreg" in terms:
        regmax = np.maximum(np.nan_to_num(D["rt1"], nan=-1e9), np.nan_to_num(D["rt2"], nan=-1e9))
        divmax = np.maximum(np.nan_to_num(D["zdiv"], nan=-1e9), np.nan_to_num(D["zdiv2"], nan=-1e9))
        silent = (regmax <= 2.326) & (divmax <= 2.326)
        drift_f = np.nan_to_num(D["drift"] > 2.326, nan=False).astype(bool)
        two_f = np.nan_to_num(D["zdhet"] < -2.326, nan=False).astype(bool)
        keep_noreg = ~silent          # every regime: both regulation detectors silent -> NO
        del drift_f, two_f
    else:
        keep_noreg = None
    if "F_regdiv" in terms:      # called if EITHER regulation detector fires
        regmax = np.maximum(np.nan_to_num(D["rt1"], nan=-1e9), np.nan_to_num(D["rt2"], nan=-1e9))
        divmax = np.maximum(np.nan_to_num(D["zdiv"], nan=-1e9), np.nan_to_num(D["zdiv2"], nan=-1e9))
        keep_regdiv = (regmax > tr) | (divmax > tr)
    else:
        keep_regdiv = None
    def divpen(zh, zd):
        """the module's own penalty: |z_div| where the pair is divergent, z_het < -2.326."""
        g = np.isfinite(zh) & (zh < -2.326) & np.isfinite(zd)
        return np.where(g, np.abs(np.nan_to_num(zd)), 0.0)
    # the module applies the penalty at t1; naming a timepoint is a choice, so the symmetric
    # form takes the stronger of the two timepoints' penalties -- same quantity, same gate.
    if "div_sym" in terms:
        sc = sc - np.maximum(divpen(D["zhet"], D["zdiv"]), divpen(D["zhet2"], D["zdiv2"]))
    if "div_t2" in terms: sc = sc - divpen(D["zhet2"], D["zdiv2"])
    # separate_fan_outs_from_mutual_regulation: "a pair whose undirected correlation is far
    # outside the random-reference null (|z| large) is behaving like it is driven by a shared
    # hidden driver rather than a direct link between A and B". The module applies it as a
    # removal rule at |z|>8 on pairs with a common called regulator; the same statement read
    # on every pair is a graded penalty on |z_het|.
    if "fanout" in terms: sc = sc - s_(np.abs(D["zhet"]))
    if "fanout_sym" in terms:
        sc = sc - np.maximum(s_(np.abs(D["zhet"])), s_(np.abs(D["zhet2"])))
    if "F_fanout" in terms: keep_fan = np.nan_to_num(np.abs(D["zhet"]) <= 8.0, nan=False)
    # z_div is the fixed-Delta test: the observed rho_Delta(t) against the distribution obtained
    # by re-pairing the complete Delta_Y rows across clones, every Delta kept exactly as measured.
    # Significantly POSITIVE means the observed Delta_X-to-Delta_Y matching itself carries the
    # coupling, which is what a direct edge produces. One unit per timepoint, as the module's
    # existence term is written.
    if "divpos_sum" in terms: sc = sc + s_(D["zdiv"]) + s_(D["zdiv2"])
    if "divpos_t1" in terms: sc = sc + s_(D["zdiv"])
    # separate_fan_outs_from_mutual_regulation builds regulators_of from the edges called so
    # far and treats a pair with a COMMON upstream regulator as co-driven rather than directly
    # linked. Here the called set is the pairs that passed both declared filters, each oriented
    # by the sign of gamma -- the module's own direction quantity. The penalty is the number of
    # genes that regulate both members of the pair.
    if ("common_reg" in terms) or ("deg_target" in terms) or ("deg_sum" in terms) \
            or ("cr_score" in terms) \
            or ("cr_max" in terms) or ("cr_all" in terms):
        ok = np.ones(len(sc), bool)
        # the called set is exactly the pairs the DECLARED filters keep, same thresholds
        ok &= np.nan_to_num((D["zr1"] > te) | (D["zr2"] > te), nan=False).astype(bool)
        ok &= np.nan_to_num((D["rt1"] > tr) | (D["rt2"] > tr), nan=False).astype(bool)
        gcut = 1.0 if "cr_strict" in terms else 0.0
        rng = np.random.default_rng(20260907) if "cr_random" in terms else None
        called = set()
        for u, (x, w) in enumerate(d["univ"]):
            if not (ok[u] and np.isfinite(D["gam"][u])): continue
            if rng is not None:
                if rng.random() < 0.5: called.add((x, w))
            elif D["gam"][u] > 0 and abs(D["ug"][u]) >= gcut:
                called.add((x, w))
        upstream = {}
        for (c, t) in called: upstream.setdefault(t, set()).add(c)
        cnt = np.array([len((upstream.get(x, set()) & upstream.get(w, set())) - {x, w})
                        for (x, w) in d["univ"]], float)
        cnt_raw = cnt.copy()
        # the fan-out explanation is only as strong as its weaker arm: for each shared
        # regulator C, min(evidence C->x, evidence C->y), summed over the shared regulators
        if "cr_score" in terms:
            # the arm's strength is the score's OWN evidence for C->x, everything but this term
            g_ = module_score(d, tuple(t_ for t_ in terms
                                       if t_ not in ("common_reg", "cr_strength", "cr_score")))
            g_ = np.nan_to_num(g_, nan=-1e9)
            ev = {p_: g_[u_] for u_, p_ in enumerate(d["univ"])}
            cnt = np.array([sum(min(ev.get((c, x), 0.0), ev.get((c, w), 0.0))
                                for c in (upstream.get(x, set()) & upstream.get(w, set())) - {x, w})
                            for (x, w) in d["univ"]], float)
        elif "cr_max" in terms or "cr_all" in terms:
            # a fan-out needs ONE third gene that explains both arms, not many weak ones:
            # max over C of min(evidence C-x, evidence C-y). cr_all drops the requirement that
            # C be a called regulator of both, asking only that a third gene couple to both.
            g_ = np.maximum(np.nan_to_num(D["rt1"], nan=-1e9), np.nan_to_num(D["rt2"], nan=-1e9))
            ev = {}
            for u_, (a_, b_) in enumerate(d["univ"]): ev[(a_, b_)] = g_[u_]
            allg = set(d["genes"])
            cnt = np.array([max([min(ev.get((c, x), -1e9), ev.get((c, w), -1e9))
                                 for c in ((allg if "cr_all" in terms
                                            else (upstream.get(x, set()) & upstream.get(w, set())))
                                           - {x, w})] or [0.0])
                            for (x, w) in d["univ"]], float)
            cnt = np.where(cnt < -1e8, 0.0, cnt)
        elif "cr_strength" in terms:
            g_ = np.maximum(np.nan_to_num(D["rt1"], nan=-1e9), np.nan_to_num(D["rt2"], nan=-1e9))
            ev = {p_: g_[u_] for u_, p_ in enumerate(d["univ"])}
            cnt = np.array([sum(min(ev.get((c, x), 0.0), ev.get((c, w), 0.0))
                                for c in (upstream.get(x, set()) & upstream.get(w, set())) - {x, w})
                            for (x, w) in d["univ"]], float)
        # y's OTHER regulators are what can explain the x-y correlation without an x->y edge;
        # x's regulators cannot. deg_in(y) alone is that statement, and it is directional.
        din = np.array([len(upstream.get(w, set()) - {x}) for (x, w) in d["univ"]], float)
        dsum = np.array([len(upstream.get(x, set())) + len(upstream.get(w, set()))
                         for (x, w) in d["univ"]], float)
        # a gene that regulates many genes in the called set is acting as a regulator; the
        # difference of out-degrees is antisymmetric, so it orients the pair
        if "outdeg" in terms:
            od = {}
            for c, t_ in called: od[c] = od.get(c, 0) + 1
            v_ = np.array([od.get(x, 0) - od.get(w, 0) for (x, w) in d["univ"]], float)
            sc = sc + s_(v_)
        # A shared driver can only PRODUCE correlation, so co-driving is a live alternative
        # only where the pair is strongly co-expressed. The fan-out penalty is applied on that
        # condition; z_flux, which asks which gene is the regulator, applies everywhere.
        # A pair with NO shared regulator has nothing for the fan-out penalty to say, so the
        # term is neutral there and z_flux carries the pair alone. Standardising a column that
        # is mostly zeros otherwise pushes those pairs down by a constant that means nothing.
        if "cr_pos" in terms:
            has = cnt > 0
            zz = np.zeros(len(cnt))
            if has.sum() > 2:
                x_ = cnt[has]; zz[has] = (x_ - x_.mean()) / max(x_.std(ddof=1), 1e-12)
            cnt = zz; _cr_neutral = True
        if "cr_hi" in terms:
            zmax = np.maximum(np.nan_to_num(D["zr1"], nan=0.0), np.nan_to_num(D["zr2"], nan=0.0))
            cut = 3.090 if "cr_hi001" in terms else (
                  float(np.nanmedian(zmax)) if "cr_himed" in terms else 2.576)
            cnt = np.where(zmax > cut, cnt, 0.0)
        if "deg_target" in terms: sc = sc - s_(din)
        elif "deg_sum" in terms: sc = sc - s_(dsum)
        elif "cr_pos" in terms: sc = sc - cnt
        else: sc = sc - s_(cnt)
        cnt_raw_local = cnt_raw
        if "common_reg_only" in terms: sc = -s_(cnt)
    # z_flux(t1,t2) = reg(x) - reg(y). reg(g) is gene g's mean over its partners of the
    # antisymmetric quantity named below, standardised across the panel's genes; the difference
    # is antisymmetric, so it is added to x->y and subtracted from y->x.
    # z_flux(t1,t2) = reg(x) - reg(y) with reg(g) the gene's NET regulatory outflow: over the
    # pairs that pass the declared filters, the strength of the regulation evidence signed by
    # the direction, sum_w z_reg_gated(g,w) * sign(gamma(g,w)). Standardised across the panel's
    # genes; the difference is antisymmetric, added to x->y and subtracted from y->x.
    if "flux_net" in terms:
        okf = np.nan_to_num((D["zr1"] > te) | (D["zr2"] > te), nan=False).astype(bool)
        okf &= np.nan_to_num((D["rt1"] > tr) | (D["rt2"] > tr), nan=False).astype(bool)
        if "flux_all" in terms: okf = np.ones(len(sc), bool)       # reg over ALL partners
        gmax = np.maximum(np.nan_to_num(D["rt1"], nan=0.0), np.nan_to_num(D["rt2"], nan=0.0))
        if "flux_sumtp" in terms:                                  # z_reg summed over timepoints
            gmax = np.nan_to_num(D["rt1"], nan=0.0) + np.nan_to_num(D["rt2"], nan=0.0)
        # regulation is rho_reg significantly POSITIVE; a negative statistic is absence of
        # regulation, not negative regulation, so it contributes nothing to reg(g)
        if "flux_pos" in terms: gmax = np.maximum(gmax, 0.0)
        gsign = np.sign(D["ug"]) if "flux_zgsign" in terms else np.sign(D["gam"])
        net = {}; cnt_ = {}
        for u_, (a_, b_) in enumerate(d["univ"]):
            if okf[u_] and np.isfinite(D["gam"][u_]):
                net[a_] = net.get(a_, 0.0) + gmax[u_] * gsign[u_]
                cnt_[a_] = cnt_.get(a_, 0) + 1
        if "flux_mean" in terms:
            net = {k_: v_ / max(cnt_.get(k_, 1), 1) for k_, v_ in net.items()}
        arr = np.array([net.get(gn, 0.0) for gn in d["genes"]], float)
        mu_, sd_ = arr.mean(), max(arr.std(ddof=1), 1e-12)
        RGn = {gn: (net.get(gn, 0.0) - mu_) / sd_ for gn in d["genes"]}
        flux = np.array([RGn[x] - RGn[w] for (x, w) in d["univ"]], float)
        flux_local = flux
        sc = sc + s_(flux)          # unit weight across the panel's pairs, like every other term
        # |z_flux| is symmetric: it is large when one gene of the pair is a source and the
        # other a target, and small when both are targets -- which is what a fan-out pair is.
        if "flux_abs" in terms: sc = sc + s_(np.abs(flux))
        # a real edge needs its SOURCE to be a regulator; a fan-out pair is two targets, so
        # both reg values are low. reg(x) alone both orients and scores; max(reg) asks only
        # that at least one member of the pair is a regulator.
        if "reg_src" in terms:
            sc = sc + s_(np.array([RGn[x] for (x, w) in d["univ"]], float))
        if "reg_max" in terms:
            sc = sc + s_(np.array([max(RGn[x], RGn[w]) for (x, w) in d["univ"]], float))
        # exclusive cases: a pair WITH a shared regulator is judged by the fan-out penalty,
        # a pair WITHOUT one has no co-driving explanation and is judged by z_flux alone.
        if "cases" in terms and cnt_raw_local is not None:
            has = cnt_raw_local > 0
            zz = np.zeros(len(has))
            if has.sum() > 2:
                x_ = cnt_raw_local[has]
                zz[has] = (x_ - x_.mean()) / max(x_.std(ddof=1), 1e-12)
            sc = sc - zz - np.where(has, flux, 0.0)
    for key, src in (("flux_gamma", "ug"),):
        if key not in terms: continue
        v_ = D["ug"]
        per = {}
        for u_, (a_, b_) in enumerate(d["univ"]):
            if np.isfinite(v_[u_]): per.setdefault(a_, []).append(v_[u_])
        reg = {gname: float(np.mean(vals)) for gname, vals in per.items()}
        arr = np.array(list(reg.values()), float)
        mu_, sd_ = arr.mean(), max(arr.std(ddof=1), 1e-12)
        RGg = {gname: (val - mu_) / sd_ for gname, val in reg.items()}
        sc = sc + np.array([RGg.get(x, 0.0) - RGg.get(w, 0.0) for (x, w) in d["univ"]], float)
    # z_flux = reg(x) - reg(y), reg(g) = mean outward flux - mean inward flux of gene g, the
    # directed flux being the cross-difference correlation rho_dagger in its null units:
    #   outward(g) = mean over partners w of z_rho_dagger(g->w)
    #   inward(g)  = mean over partners w of z_rho_dagger(w->g)
    if "flux_dagger" in terms:
        out_, in_ = {}, {}
        for (a_, b_), zf, zr in zip(d["P"], d["zfwd"], d["zrev"]):
            if np.isfinite(zf):
                out_.setdefault(a_, []).append(zf); in_.setdefault(b_, []).append(zf)
            if np.isfinite(zr):
                out_.setdefault(b_, []).append(zr); in_.setdefault(a_, []).append(zr)
        regd = {g: (np.mean(out_.get(g, [0.0])) - np.mean(in_.get(g, [0.0]))) for g in d["genes"]}
        fl = np.array([regd[x] - regd[w] for (x, w) in d["univ"]], float)
        sc = sc + s_(fl)
    if "gamma_graded" in terms:
        sc = sc + (RG if "raw_units" in terms else s_(D["ug"]))
    if "gamma_neg" in terms: sc = sc - s_(D["ug"])
    # the same count with the called set left UNDIRECTED: genes that pass the filters with both
    # members of the pair, i.e. common neighbours rather than common regulators
    if "common_nb" in terms:
        ok = np.ones(len(sc), bool)
        # the called set is exactly the pairs the DECLARED filters keep, same thresholds
        ok &= np.nan_to_num((D["zr1"] > te) | (D["zr2"] > te), nan=False).astype(bool)
        ok &= np.nan_to_num((D["rt1"] > tr) | (D["rt2"] > tr), nan=False).astype(bool)
        nb = {}
        for u, (x, w) in enumerate(d["univ"]):
            if ok[u]: nb.setdefault(x, set()).add(w); nb.setdefault(w, set()).add(x)
        cnt = np.array([len((nb.get(x, set()) & nb.get(w, set())) - {x, w})
                        for (x, w) in d["univ"]], float)
        sc = sc - s_(cnt)
    if "rho_max" in terms: sc = sc + np.maximum(s_(D["u1"]), s_(D["u2"]))
    if "rhoZ_sum" in terms: sc = sc + s_(D["zr1"]) + s_(D["zr2"])
    if "rhoZ_max" in terms: sc = sc + np.maximum(s_(D["zr1"]), s_(D["zr2"]))
    if "rhoZ_min" in terms: sc = sc + np.minimum(s_(D["zr1"]), s_(D["zr2"]))
    if "rho_min" in terms: sc = sc + np.minimum(s_(D["u1"]), s_(D["u2"]))
    # every term enters at unit weight after standardisation; a quantity that exists at both
    # timepoints enters ONCE, as the symmetric mean, not as a sum that doubles its weight
    if "rho_mean" in terms: sc = sc + 0.5 * (panel_z(D["u1"]) + panel_z(D["u2"]))
    if "div_s" in terms:
        g = np.isfinite(D["zhet"]) & (D["zhet"] < -2.326) & np.isfinite(D["zdiv"])
        sc = sc - s_(np.where(g, np.abs(np.nan_to_num(D["zdiv"])), 0.0))
    if "reg" in terms: sc = sc + np.maximum(s_(D["gt1"]), s_(D["gt2"]))
    # signed, for the same reason the filter is signed: regulation is rho_reg(lambda) positive
    if "reg_signed" in terms: sc = sc + np.maximum(s_(D["rt1"]), s_(D["rt2"]))
    if "reg_signed_sum" in terms: sc = sc + 0.5 * (s_(D["rt1"]) + s_(D["rt2"]))
    # one unit per timepoint, the same shape the module gives the existence term
    if "reg_sum" in terms: sc = sc + s_(D["rt1"]) + s_(D["rt2"])
    # max(z_reg_gated(t1), z_reg_gated(t2)) on the raw z, then standardised once
    if "reg_maxraw" in terms:
        sc = sc + s_(np.maximum(np.nan_to_num(D["rt1"], nan=-1e9),
                                np.nan_to_num(D["rt2"], nan=-1e9)))
    if "drift" in terms: sc = sc - s_(D["drift"])
    # z_drift is high for drift WITHOUT regulation and lower for drift WITH it, so the penalty
    # is scaled down where the regulation evidence is strong instead of applied flat.
    # ---- the six-scenario score, derived, nothing searched -------------------------------
    # regime from the two state detectors, read off the figure:
    #   drift    z_|drho| > 2.326      (5.5 without regulation, 2 with; ~0 in every other cell)
    #   2 state  z_d_het  < -2.326     (-20 without regulation, -10 with; ~0 elsewhere)
    #   single   neither
    # regulation contrast inside each regime, blue minus beige:
    #   every regime  + z_reg_gated (+4.2 / +1.6 / +4.1)  and  + z_div (+2.5 / +1.5 / +3.0)
    #   2 state       + z_d_het   (-10 against -20, the largest contrast in the figure)
    #   drift         - z_|drho|  (2 against 5.5)
    # Every term applied ONLY inside the regime where the model says it discriminates, and
    # standardised within that regime so the three blocks are on one scale.
    # One panel-wide standardisation per term; the regime indicator only switches a term on or
    # off, so the scales stay comparable across regimes.
    # Only the two terms the real panels show to be regime-specific are conditioned:
    #   z_div     separates only inside the single-state regime (p<0.001 vs 0.054 / 0.128)
    #   z_d_het   separates only inside the drift regime (p=0.026)
    # Everything else discriminates in every regime and stays unconditional.
    if "cond3" in terms:
        drift_r = np.nan_to_num(D["drift"] > 2.326, nan=False).astype(bool)
        two_r = np.nan_to_num(D["zdhet"] < -2.326, nan=False).astype(bool) & ~drift_r
        single_r = ~drift_r & ~two_r
        def gated(mask, v, sign=1.0):
            z = np.nan_to_num(s_(v))
            mu = z[mask].mean() if mask.sum() else 0.0
            return sign * np.where(mask, z, mu)
        sc = sc + gated(single_r, D["zdiv"])
        sc = sc + gated(drift_r, D["zdhet"])
    if "cond2" in terms:
        # drift is the movement of the unrelated-pair REFERENCE, generate_difference_shuffle,
        # not the Step-I co-expression change.
        drift_r = np.nan_to_num(D["drift"] > 2.326, nan=False).astype(bool)
        two_r = np.nan_to_num(D["zdhet"] < -2.326, nan=False).astype(bool) & ~drift_r
        single_r = ~drift_r & ~two_r
        rmx = np.maximum(np.nan_to_num(D["rt1"], nan=-9e9), np.nan_to_num(D["rt2"], nan=-9e9))
        sc = sc + np.nan_to_num(panel_z(D["u1"])) + np.nan_to_num(panel_z(D["u2"]))
        sc = sc + s_(rmx)
        # switching a term off must be NEUTRAL: an out-of-regime pair sits at the in-regime
        # MEAN of that term, not at zero, so the two blocks are not offset against each other.
        def gated(mask, v, sign=1.0):
            z = np.nan_to_num(s_(v))
            mu = z[mask].mean() if mask.sum() else 0.0
            return sign * np.where(mask, z, mu)
        sc = sc + gated(single_r, D["zdiv"])
        sc = sc + gated(two_r, D["zdhet"])
        sc = sc + gated(drift_r, np.abs(D["uc"]), -1.0)
        sc = sc + gated(drift_r, D["drift"], -1.0)
    if "cond" in terms:
        drift_r = np.nan_to_num(np.abs(D["uc"]) > 2.326, nan=False).astype(bool)
        two_r = np.nan_to_num(D["zdhet"] < -2.326, nan=False).astype(bool) & ~drift_r
        single_r = ~drift_r & ~two_r
        def zin(mask, v, sign=1.0):
            out = np.zeros(len(sc)); x = np.asarray(v, float)
            f = mask & np.isfinite(x)
            if f.sum() > 2:
                out[f] = sign * (x[f] - x[f].mean()) / max(x[f].std(ddof=1), 1e-12)
            return out
        rmx = np.maximum(np.nan_to_num(D["rt1"], nan=-9e9), np.nan_to_num(D["rt2"], nan=-9e9))
        rho_ = np.maximum(np.nan_to_num(D["u1"], nan=-9e9), np.nan_to_num(D["u2"], nan=-9e9))
        for m_ in (single_r, two_r, drift_r):            # every regime: existence + regulation
            sc = sc + zin(m_, rho_) + zin(m_, rmx)
        sc = sc + zin(single_r, D["zdiv"])               # single: the fixed-Delta test
        sc = sc + zin(two_r, D["zdhet"])                 # 2 state: the change against the het null
        sc = sc + zin(drift_r, np.abs(D["uc"]), -1.0)    # drift: the co-expression change
        sc = sc + zin(drift_r, D["drift"], -1.0)         # drift: the reference movement
    if "six" in terms:
        drift_r = np.nan_to_num(np.abs(D["uc"]) > 2.326, nan=False).astype(bool)
        two_r = np.nan_to_num(D["zdhet"] < -2.326, nan=False).astype(bool)
        def zin(mask, v, sign):
            out = np.zeros(len(sc)); x = np.asarray(v, float)
            f = mask & np.isfinite(x)
            if f.sum() > 2:
                out[f] = sign * (x[f] - x[f].mean()) / max(x[f].std(ddof=1), 1e-12)
            return out
        sc = sc + s_(D["rt1"]) + s_(D["rt2"])            # z_reg, every regime
        sc = sc + s_(D["zdiv"]) + s_(D["zdiv2"])         # z_div, every regime
        sc = sc + zin(two_r, D["zdhet"], +1.0)           # 2 state only
        sc = sc + zin(drift_r, np.abs(D["uc"]), -1.0)    # drift only
    if "drift_med" in terms:      # penalise drift only where the regulation evidence is weak
        rmx = np.maximum(np.nan_to_num(D["rt1"], nan=0.0), np.nan_to_num(D["rt2"], nan=0.0))
        sc = sc - np.where(rmx < np.nanmedian(rmx), s_(D["drift"]), 0.0)
    if "drift_hi" in terms:
        rmx = np.maximum(np.nan_to_num(D["rt1"], nan=0.0), np.nan_to_num(D["rt2"], nan=0.0))
        sc = sc - np.where(rmx < 4.652, s_(D["drift"]), 0.0)
    if "drift_cond" in terms:
        rmx = np.maximum(np.nan_to_num(D["rt1"], nan=0.0), np.nan_to_num(D["rt2"], nan=0.0))
        w_ = 1.0 - np.clip(rmx / 4.652, 0.0, 1.0)      # 0 at twice the 0.01 critical value
        sc = sc - w_ * s_(D["drift"])
    if "drift2" in terms: sc = sc - 2.0 * s_(D["drift"])
    if "drift_plus" in terms: sc = sc + s_(D["drift"])
    # the change axis and the inherited channel as HINGES: a quantity that is ~0 outside its
    # regime fires only where it is significant, and enters in its own null units there
    # the same hinge on the existence axis: |rho| in its null units fires only where it clears
    # the Step-I critical value, 2.576, and enters unstandardised there
    if "rho_hinge_sum" in terms:
        for k_ in ("u1", "u2"):
            v_ = np.nan_to_num(D[k_], nan=0.0); sc = sc + np.where(v_ > 2.576, v_, 0.0)
    if "rho_hinge_max" in terms:
        v_ = np.maximum(np.nan_to_num(D["u1"], nan=0.0), np.nan_to_num(D["u2"], nan=0.0))
        sc = sc + np.where(v_ > 2.576, v_, 0.0)
    # hinge thresholds as alpha choices for the two new hinges: 1.645 (0.05), 2.326 (0.01), 2.576
    for key_, thr_ in (("stable_hinge_t1645", 1.645), ("stable_hinge_t2576", 2.576)):
        if key_ in terms:
            v_ = np.nan_to_num(D["uc"], nan=0.0); sc = sc - np.where(v_ > thr_, v_, 0.0)
    for key_, thr_ in (("sym_hinge_t1645", 1.645), ("sym_hinge_t2576", 2.576)):
        if key_ in terms:
            v_ = np.nan_to_num(D["sym"], nan=0.0); sc = sc - np.where(v_ > thr_, v_, 0.0)
    # the divergence test as a positive hinge on the weaker timepoint: fires only when z_div
    # clears 2.326 at BOTH timepoints, and enters as that weaker value
    if "div_hinge_min" in terms:
        g2 = np.minimum(np.nan_to_num(D["zdiv"], nan=0.0), np.nan_to_num(D["zdiv2"], nan=0.0))
        sc = sc + np.where(g2 > 2.326, g2, 0.0)
    if "div_hinge_max" in terms:
        g2 = np.maximum(np.nan_to_num(D["zdiv"], nan=0.0), np.nan_to_num(D["zdiv2"], nan=0.0))
        sc = sc + np.where(g2 > 2.326, g2, 0.0)
    # z_fanout(x,y) = max over C of min(z_rho(C,x), z_rho(C,y)): the strongest third gene coupled
    # to BOTH members, z_rho in its null units at a timepoint. separate_fan_outs_from_mutual_
    # regulation's statement read on Step-I co-expression alone. Entered symmetrically over the
    # two timepoints.
    if "zfanout_sum" in terms or "zfanout_max" in terms or "zfanout_t2" in terms or "zfanout_hinge" in terms or "zfanout_hinge_min" in terms:
        genes = d["genes"]; gi = {g: i for i, g in enumerate(genes)}; ng = len(genes)
        def proxy(zvec):
            Z = np.zeros((ng, ng))
            for (a_, b_), v_ in zip(d["univ"], np.nan_to_num(zvec, nan=0.0)): Z[gi[a_], gi[b_]] = v_
            out = np.empty(len(d["univ"]))
            for u_, (a_, b_) in enumerate(d["univ"]):
                ia, ib = gi[a_], gi[b_]
                m_ = np.minimum(Z[:, ia], Z[:, ib]); m_[ia] = -np.inf; m_[ib] = -np.inf
                out[u_] = m_.max()
            return out
        p1, p2 = proxy(D["u1"]), proxy(D["u2"])
        if "zfanout_sum" in terms: sc = sc - s_(p1) - s_(p2)
        if "zfanout_max" in terms: sc = sc - s_(np.maximum(p1, p2))
        if "zfanout_t2" in terms:  sc = sc - s_(p2)
        # the same construction as the other regime terms: fires only where a third gene is
        # significantly coupled to both members at the Step-I critical value, in null units
        if "zfanout_hinge" in terms:
            g2 = np.maximum(p1, p2); sc = sc - np.where(g2 > 2.576, g2, 0.0)
        if "zfanout_hinge_min" in terms:
            g2 = np.minimum(p1, p2); sc = sc - np.where(g2 > 2.576, g2, 0.0)
    if "stable_hinge" in terms:
        v_ = np.nan_to_num(D["uc"], nan=0.0); sc = sc - np.where(v_ > 2.326, v_, 0.0)
    # the s(z(.)) convention: every term is the panel-standardised z, the indicator on the z itself
    if "stable_hinge_s" in terms:
        v_ = np.nan_to_num(D["uc"], nan=0.0); sc = sc - np.where(v_ > 2.326, s_(v_), 0.0)
    if "hinge_max_s" in terms:
        g2 = np.maximum(np.nan_to_num(D["zhet"], nan=0.0), np.nan_to_num(D["zhet2"], nan=0.0))
        sc = sc + np.where(g2 < -2.326, s_(g2), 0.0)
    if "sym_hinge_s" in terms:
        v_ = np.nan_to_num(D["sym"], nan=0.0); sc = sc - np.where(v_ > 2.326, s_(v_), 0.0)
    if "drift_hinge" in terms:
        v_ = np.nan_to_num(D["drift"], nan=0.0); sc = sc - np.where(v_ > 2.326, v_, 0.0)
    if "sym_hinge" in terms:
        v_ = np.nan_to_num(D["sym"], nan=0.0); sc = sc - np.where(v_ > 2.326, v_, 0.0)
    if "stable_plus" in terms: sc = sc + panel_z(D["uc"])
    if "F_drift" in terms:            # drift called: |mean ref change| significant vs its own sd
        keep_drift = np.nan_to_num(D["drift"] < 2.326, nan=False).astype(bool)
    else:
        keep_drift = None
    if "change2" in terms: sc = sc - 2.0 * panel_z(D["uc"])
    if "sym" in terms: sc = sc - s_(D["sym"])
    # rho_dagger is the cross-difference correlation; a strong one in the winning direction is
    # a directional coupling existing at all. max over the two directions, symmetric in x/y.
    if "dagger_max" in terms: sc = sc + s_(D["dmax"])
    # declared filters: a pair that fails is CALLED NO and ranked last, never left unscored
    keep = np.ones(len(sc), bool)
    if keep_drift is not None: keep &= keep_drift
    if keep_noreg is not None: keep &= keep_noreg
    if keep_stable is not None: keep &= keep_stable
    if keep_regdiv is not None: keep &= keep_regdiv
    if keep_fan is not None: keep &= keep_fan.astype(bool)
    if "F_exist" in terms:
        keep &= np.nan_to_num((D["zr1"] > te) | (D["zr2"] > te), nan=False).astype(bool)
    if "F_reg" in terms:
        keep &= np.nan_to_num((D["gt1"] > tr) | (D["gt2"] > tr), nan=False).astype(bool)
    # rho_reg(lambda) = rho_same - lambda*rho_cross; regulation is the statistic significantly
    # POSITIVE. The module's own critical value, norm.ppf(1-alpha), is one-sided and is applied
    # signed (z_d_het > z_critical), so the filter is signed here too, not two-sided on |.|.
    # identify_actual_directed_edges calls a direction only where |z_gamma| exceeds its own
    # z_score_threshold, default 2.5. As a declared filter that is the module's Stage-IV call.
    # identify_reg_if_multiple_states calls "regulation" where z_d_het > norm.ppf(1-alpha) on
    # the observed twin change against the fresh-random-pair difference null. That is the
    # module's own Stage-III call, an alternative to the gated statistic used above.
    if "F_dhet" in terms:
        keep &= np.nan_to_num(D["zdhet"] > tr, nan=False).astype(bool)
    if "F_gamma" in terms:
        keep &= np.nan_to_num(np.abs(D["ug"]) > 2.5, nan=False).astype(bool)
    if "F_reg_signed" in terms:
        keep &= np.nan_to_num((D["rt1"] > tr) | (D["rt2"] > tr), nan=False).astype(bool)
    # A pair that fails a filter is CALLED NO and ranks below every pair that passed. It is
    # not left unscored: shifting the block down keeps every called-NO pair last while
    # preserving the score's own order inside it. A flat tie would discard that order.
    if not keep.all():
        span = np.nanmax(sc) - np.nanmin(sc)
        sc = np.where(keep, sc, sc - (span + 1.0))
    return (sc, keep) if return_keep else sc

def score_panel(d, terms=("rho", "change", "div", "het", "gamma")):
    sc = module_score(d, terms)
    o = np.argsort(-sc, kind="stable"); sv, yv = sc[o], d["y"][o]
    e = np.empty_like(yv); i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]: j += 1
        e[i:j] = yv[i:j].mean(); i = j
    return (average_precision_score(d["y"], sc) / d["base"],
            (e[:d["R"]].sum() / d["R"]) / d["base"])
