#!/usr/bin/env python
"""
Predict, WITHOUT running the Gillespie sims, how many distinct steady states each
real-network TwINFER simulation should produce -- and cross-check against the
Boolean attractor count of the same network.

For each network it reports:

  * mean-field fixed points of the deterministic (promoter-QSS) ODE limit, with
    linear-stability classification  -> # stable FPs = predicted # states
  * which fixed point the empty initial state (all promoters I, 0 mRNA/protein --
    exactly how the sim starts) relaxes to  -> whether the sim will be unimodal
  * feedback-loop gains at that FP  -> how close the network is to a bifurcation,
    and which knob (n, k_add) moves it
  * telegraph / burst dimensionless numbers  -> whether any bimodality would be
    genuine multistability vs promoter-switching noise
  * Boolean attractor count:
      - threshold-Boolean derived from the signed matrix + the SAME k_add/k_off
      - the published BoolODE rule file, where it matches the sim network
        (GSD, HSC, mCAD, VSC; EMT's rule file is a different, larger model)
  * an (n, k_add-scale) bifurcation grid: # stable FPs across the plane

Model (matches gillespie_simulations.py):
  promoter:  k_on_eff_i = max(0, k_on + reg_i(p)),   a_i = k_on_eff_i/(k_on_eff_i+k_off)
  reg_i    : 1 reg / same-sign multi -> sum of  sign*k_add*H(p_r; K, n),  H = p^n/(K^n+p^n)
             same-sign pair (additive) -> sign*k_add*(h1+h2+2 h1 h2)/((1+h1)(1+h2)),  h=(p/K)^n
             opposite-sign pair / >2   -> per-edge additive sum of signed Hills
  mRNA/prot: m_i* = kpm*a_i/kdm,   p_i* = kpp*m_i*/kdp
  K_{i->j} = protein level of i computed with every regulator at target_hill=0.5
             (generate_K_from_steady_state_calc)

Usage:
  python predict_multistability.py                # all networks -> PDF + CSV
  python predict_multistability.py --networks VSC GSD
  python predict_multistability.py --cle          # also run a cheap Langevin check
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import itertools
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

try:
    import networkx as nx
except Exception:
    nx = None

ROOT = f'{TWINFER_PROJECT_ROOT}'
MATRIX_DIR = f"{ROOT}/input_data/real_world_networks"
PARAM_CSV = f"{ROOT}/input_data/network_sweep/parameters.csv"
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] BOOLODE_DIR = f"{ROOT}/code/BoolODE/data"
BOOLODE_DIR = f"{ROOT}/clean_data/benchmarks/boolode/model_inputs"
OUT_DIR = f"{ROOT}/analysis_data/paper_analysis/real_networks"
OUT_PDF = f"{OUT_DIR}/real_networks_multistability_prediction.pdf"
OUT_CSV = f"{OUT_DIR}/real_networks_multistability_prediction.csv"

# sim network  ->  (signed matrix file, BoolODE rule file or None if it doesn't match)
NETWORKS = {
    "Circadian_cycle":   ("circadian.txt", None),
    "mCAD":              ("mCAD.txt",      "mCAD.txt"),
    "VSC":              ("VSC.txt",        "VSC.txt"),
    "B_cell_activation": ("B_cell.txt",    None),
    "HSC_balanced":      ("HSC.txt",       "HSC.txt"),
    "EMT":              ("EMT.txt",        None),   # BoolODE EMT.txt is a ~60-gene model, not this 17-gene one
    "GSD":              ("GSD.txt",        "GSD.txt"),
    "Pluripotent":      ("Pluripotent.txt", None),
}

# what the final-timepoint clustering of the actual sims found (rep 0), for the report
OBSERVED_CLUSTERS = {
    "Circadian_cycle": "k=2 (71/29%)  [oscillatory, not fixed-point]",
    "mCAD": "k=2 (97/3%)  [tiny outlier tail]",
    "VSC": "k=2 (57/43%)  [genuine split]",
    "B_cell_activation": "k=1",
    "HSC_balanced": "k=1",
    "EMT": "k=1",
    "GSD": "k=2 (90/10%)  [minority subpop]",
    "Pluripotent": "k=1",
}

TARGET_HILL = 0.5
KADD_POS, KADD_NEG = 6.0, 0.8       # resolve_all_k_add default_pos / default_neg
N_HILL = 2.0                        # n_matrix default
N_MULTISTART = 200
PICARD_ITERS = 1500
DEDUP_RTOL = 1e-2


# --------------------------------------------------------------------------- model
def load_matrix(fname):
    M = np.loadtxt(f"{MATRIX_DIR}/{fname}", dtype=float, delimiter=",")
    if M.ndim == 0:
        M = M.reshape(1, 1)
    return M  # M[i, j]: regulator i -> target j ; sign = activation/repression


def load_params():
    df = pd.read_csv(PARAM_CSV, index_col=0)
    r = df.loc[0]
    kdm = np.log(2) / r["mrna_half_life"]
    kdp = np.log(2) / r["protein_half_life"]
    return dict(
        k_on=float(r["k_on"]), k_off=float(r["k_off"]),
        kpm=float(r["k_prod_mRNA"]), kpp=float(r["k_prod_protein"]),
        kdm=float(kdm), kdp=float(kdp),
    )


def _sign(x):
    return int(np.sign(x))


def all_states(n):
    """(2**n, n) array of every binary state, bit j of row k -> column j."""
    k = np.arange(2 ** n, dtype=np.int64)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.int8)


class MeanField:
    """Deterministic promoter-QSS limit of the TwINFER Gillespie model."""

    def __init__(self, M, P, kadd_scale=1.0, n_hill=N_HILL):
        self.M = M
        self.n = M.shape[0]
        self.P = P
        self.n_hill = n_hill
        self.c = P["kpm"] * P["kpp"] / (P["kdm"] * P["kdp"])   # protein per unit promoter activity
        # per-target regulator lists / signs
        self.regs = [np.where(M[:, j] != 0)[0] for j in range(self.n)]
        self.signs = [[_sign(M[i, j]) for i in self.regs[j]] for j in range(self.n)]
        # per-edge k_add (sign-based default), scaled
        self.kadd = np.zeros((self.n, self.n))
        for j in range(self.n):
            for i in self.regs[j]:
                self.kadd[i, j] = (KADD_POS if M[i, j] > 0 else KADD_NEG) * kadd_scale
        self.K = self._calibrate_K()

    # ---- K calibration: generate_K_from_steady_state_calc ------------------
    def _calibrate_K(self):
        P = self.P
        prot = np.zeros(self.n)
        for j in range(self.n):
            R, S = self.regs[j], self.signs[j]
            if len(R) == 0:
                kon_eff = P["k_on"]
            elif len(R) == 1 or (len(R) == 2 and S[0] == S[1]):
                kadd = self.kadd[R[0], j]      # per-gene path uses first edge's value (same-sign)
                kon_eff = P["k_on"] + S[0] * kadd * TARGET_HILL
            else:
                kon_eff = P["k_on"] + sum(
                    S[k] * self.kadd[R[k], j] * TARGET_HILL for k in range(len(R))
                )
            kon_eff = max(kon_eff, 0.0)
            burst = kon_eff / (kon_eff + P["k_off"]) if (kon_eff + P["k_off"]) > 0 else 0.0
            m = P["kpm"] * burst / max(P["kdm"], 1e-12)
            prot[j] = max(m * P["kpp"] / max(P["kdp"], 1e-12), 0.1)
        K = np.ones((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.M[i, j] != 0:
                    K[i, j] = prot[i]
        return K

    # ---- regulatory input to promoter ON-rate for gene j ------------------
    def _reg(self, p, j):
        R, S = self.regs[j], self.signs[j]
        if len(R) == 0:
            return 0.0
        n = self.n_hill
        if len(R) == 1:
            r = R[0]
            h = (max(p[r], 0.0)) ** n
            Kn = self.K[r, j] ** n
            return S[0] * self.kadd[r, j] * h / (Kn + h)
        if len(R) == 2 and S[0] == S[1]:
            r1, r2 = R
            h1 = (max(p[r1], 0.0) / self.K[r1, j]) ** n
            h2 = (max(p[r2], 0.0) / self.K[r2, j]) ** n
            num = h1 + h2 + 2.0 * h1 * h2
            den = 1.0 + h1 + h2 + h1 * h2
            return S[0] * self.kadd[r1, j] * num / den
        # opposite-sign pair or >2 regulators: per-edge additive signed Hills
        tot = 0.0
        for k, r in enumerate(R):
            h = (max(p[r], 0.0)) ** n
            Kn = self.K[r, j] ** n
            tot += S[k] * self.kadd[r, j] * h / (Kn + h)
        return tot

    def a(self, p):
        """promoter active fraction per gene at protein vector p"""
        P = self.P
        out = np.empty(self.n)
        for j in range(self.n):
            kon_eff = max(P["k_on"] + self._reg(p, j), 0.0)
            out[j] = kon_eff / (kon_eff + P["k_off"])
        return out

    def G(self, p):
        """fixed-point map: p_next = c * a(p)"""
        return self.c * self.a(np.asarray(p, float))

    # ---- full 2N ODE for rigorous linear stability -----------------------
    def rhs(self, t, y):
        P = self.P
        m, p = y[: self.n], y[self.n:]
        av = self.a(p)
        dm = P["kpm"] * av - P["kdm"] * m
        dp = P["kpp"] * m - P["kdp"] * p
        return np.concatenate([dm, dp])

    def jac_full(self, m, p, eps=1e-4):
        y = np.concatenate([m, p])
        f0 = self.rhs(0, y)
        J = np.empty((2 * self.n, 2 * self.n))
        for k in range(2 * self.n):
            dy = y.copy()
            h = eps * max(abs(y[k]), 1.0)
            dy[k] += h
            J[:, k] = (self.rhs(0, dy) - f0) / h
        return J

    def fixed_points(self, n_starts=N_MULTISTART, bool_seeds=None):
        P = self.P
        c = self.c
        a_hi = 0.9 * c
        a_basal = c * P["k_on"] / (P["k_on"] + P["k_off"])
        starts = [np.zeros(self.n), np.full(self.n, a_basal), np.full(self.n, a_hi)]
        rng = np.random.default_rng(0)
        starts += [10 ** rng.uniform(np.log10(0.1), np.log10(c), self.n)
                   for _ in range(n_starts)]
        for x in (bool_seeds or [])[:16]:
            starts.append(np.where(np.array(x) > 0, 0.6 * c, 0.02 * c))

        roots = []

        def add(sol):
            sol = np.clip(sol, 0, None)
            if np.max(np.abs(self.G(sol) - sol)) > 1e-4 * (1 + np.max(sol)):
                return
            if not any(np.allclose(sol, r, rtol=DEDUP_RTOL, atol=1e-6 * c) for r in roots):
                roots.append(sol)

        # damped fixed-point iteration q <- (1-w) q + w G(q): unlike plain Picard
        # this does not spuriously oscillate, so a converged point is a true FP
        # (its ODE stability is decided separately by the full Jacobian below)
        w = 0.4
        for s in starts:
            q = np.clip(s, 0, None)
            for _ in range(PICARD_ITERS):
                q2 = (1 - w) * q + w * self.G(q)
                if np.max(np.abs(q2 - q)) < 1e-7 * (1 + np.max(q2)):
                    break
                q = q2
            add(q)
        # root-find -> also catch unstable fixed points
        for s in starts[:120]:
            sol, _info, ier, _ = fsolve(
                lambda z: self.G(np.clip(z, 0, None)) - np.clip(z, 0, None),
                s, full_output=True, xtol=1e-11, maxfev=4000)
            if ier == 1:
                add(sol)

        out = []
        for p_star in roots:
            m_star = P["kpm"] * self.a(p_star) / P["kdm"]
            J = self.jac_full(m_star, p_star)
            ev = np.linalg.eigvals(J)
            stable = np.all(ev.real < 1e-6)
            dom = ev[np.argmax(ev.real)]
            oscillatory = abs(dom.imag) > abs(dom.real) and dom.real > -1e-3
            out.append(dict(p=p_star, a=self.a(p_star), max_real_eig=float(ev.real.max()),
                            stable=bool(stable), oscillatory=bool(oscillatory),
                            dom_imag=float(abs(dom.imag))))
        out.sort(key=lambda d: (-d["stable"], -np.mean(d["a"])))
        return out

    def relax_from_empty(self, T=None):
        P = self.P
        if T is None:
            T = 40.0 / min(P["kdp"], P["kdm"])
        y0 = np.zeros(2 * self.n)
        sol = solve_ivp(self.rhs, (0, T), y0, method="LSODA", rtol=1e-8, atol=1e-8,
                        t_eval=[T])
        return sol.y[self.n:, -1]

    # ---- threshold-Boolean from the signed matrix (vectorised) ----------
    def _threshold_boolean_attractors(self, max_exhaustive=20, n_sample=80_000):
        n = self.n
        P = self.P
        # signed weight matrix: W[r, j] = sign_r * k_add_{r->j}
        W = np.zeros((n, n))
        for j in range(n):
            for k, r in enumerate(self.regs[j]):
                W[r, j] = self.signs[j][k] * self.kadd[r, j]

        if n <= max_exhaustive:
            X = all_states(n)
            total = 2 ** n
        else:
            rng = np.random.default_rng(1)
            X = rng.integers(0, 2, size=(n_sample, n)).astype(np.int8)
            total = n_sample

        for _ in range(300):
            Xn = (np.maximum(P["k_on"] + X @ W, 0.0) > P["k_off"]).astype(np.int8)
            if np.array_equal(Xn, X):
                break
            X = Xn
        # after iterating, rows that are self-consistent are fixed points
        fixed = np.maximum(P["k_on"] + X @ W, 0.0) > P["k_off"]
        is_fp = np.all(fixed.astype(np.int8) == X, axis=1)
        fps = np.unique(X[is_fp], axis=0)
        n_cyclic = int((~is_fp).sum())
        return dict(n_fp=len(fps), n_cyclic_starts=n_cyclic, scanned=total), [tuple(r) for r in fps]


# --------------------------------------------------------------- BoolODE rule model
def boolode_attractors(rule_file, max_n_exhaustive=20):
    path = f"{BOOLODE_DIR}/{rule_file}"
    if not os.path.exists(path):
        return None
    genes, rules = [], {}
    with open(path) as fh:
        header = fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            g, rule = re.split(r"\t+|\s{2,}", line.strip(), maxsplit=1)
            genes.append(g.strip())
            rules[g.strip()] = rule.strip()
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    # compile each rule to a numpy expression over columns of X  (B, n) bool
    def compile_rule(expr):
        def repl(m):
            tok = m.group(0)
            if tok == "not":
                return "~"
            if tok == "and":
                return "&"
            if tok == "or":
                return "|"
            if tok in idx:
                return f"X[:,{idx[tok]}]"
            return tok

        e = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, f" {expr} ")
        return compile(f"({e.strip()})", "<rule>", "eval")

    crules = [compile_rule(rules[g]) for g in genes]

    if n <= max_n_exhaustive:
        X = all_states(n).astype(bool)
        total = 2 ** n
    else:
        rng = np.random.default_rng(2)
        X = rng.integers(0, 2, size=(60_000, n)).astype(bool)
        total = 60_000

    def step(X):
        return np.stack([eval(cr, {"__builtins__": {}}, {"X": X}) for cr in crules], axis=1)

    for _ in range(400):
        Xn = step(X)
        if np.array_equal(Xn, X):
            break
        X = Xn
    is_fp = np.all(step(X) == X, axis=1)
    fps = np.unique(X[is_fp], axis=0)
    return dict(genes=genes, n=n, n_fp=len(fps),
                n_cyclic_starts=int((~is_fp).sum()), scanned=total)


# --------------------------------------------------------------------- loop gains
def loop_gains(mf, p_star):
    if nx is None:
        return []
    G = nx.DiGraph()
    for i in range(mf.n):
        for j in range(mf.n):
            if mf.M[i, j] != 0:
                G.add_edge(i, j, sign=_sign(mf.M[i, j]))
    # numerical d a_j / d p_i  (through the promoter only)
    eps = 1e-3
    out = []
    cycles = itertools.islice(nx.simple_cycles(G), 20000)
    for cyc in cycles:
        if len(cyc) > 6:
            continue
        gain = 1.0
        sign = 1
        for a_i, b_i in zip(cyc, cyc[1:] + cyc[:1]):
            p2 = p_star.copy()
            h = eps * max(p_star[a_i], 1.0)
            p2[a_i] += h
            dGb = (mf.G(p2)[b_i] - mf.G(p_star)[b_i]) / h
            gain *= dGb
            sign *= G[a_i][b_i]["sign"]
        out.append(dict(cycle=cyc, len=len(cyc), sign=sign, gain=float(gain),
                        abs_gain=abs(float(gain))))
    out.sort(key=lambda d: -d["abs_gain"])
    return out


# ------------------------------------------------------------------ bifurcation grid
def bif_grid(M, P, n_vals, s_vals, bool_seeds=None):
    Z = np.zeros((len(n_vals), len(s_vals)), dtype=int)
    for ii, nv in enumerate(n_vals):
        for jj, sv in enumerate(s_vals):
            mf = MeanField(M, P, kadd_scale=sv, n_hill=nv)
            fps = mf.fixed_points(n_starts=90, bool_seeds=bool_seeds)
            Z[ii, jj] = sum(f["stable"] for f in fps)
    return Z


# --------------------------------------------------------------------- optional CLE
def cle_check(mf, n_traj=150, T=None, seed=0):
    """Cheap chemical-Langevin surrogate for the Gillespie sim: N SDE runs from
    the empty state; report #modes in the final protein vector (via GMM-BIC 1..4)."""
    from sklearn.mixture import GaussianMixture

    P = mf.P
    if T is None:
        T = 6.0 / P["kdp"]          # ~6 protein lifetimes: enough to commit to a basin
    dt = 0.5 / P["k_off"]           # resolve the fastest (promoter) timescale
    steps = int(T / dt)
    rng = np.random.default_rng(seed)
    n = mf.n
    finals = np.zeros((n_traj, n))
    for t in range(n_traj):
        m = np.zeros(n)
        p = np.zeros(n)
        for _ in range(steps):
            av = mf.a(p)
            # reaction rates (per gene): mRNA prod ~ kpm*a, mRNA deg ~ kdm*m, prot prod ~ kpp*m, prot deg ~ kdp*p
            rp_m = P["kpm"] * av
            rd_m = P["kdm"] * np.maximum(m, 0)
            rp_p = P["kpp"] * np.maximum(m, 0)
            rd_p = P["kdp"] * np.maximum(p, 0)
            m += (rp_m - rd_m) * dt + np.sqrt(np.maximum((rp_m + rd_m) * dt, 0)) * rng.standard_normal(n)
            p += (rp_p - rd_p) * dt + np.sqrt(np.maximum((rp_p + rd_p) * dt, 0)) * rng.standard_normal(n)
            np.clip(m, 0, None, out=m)
            np.clip(p, 0, None, out=p)
        finals[t] = p
    X = np.log1p(finals)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    bics = []
    for k in range(1, 5):
        gm = GaussianMixture(k, covariance_type="full", n_init=3, reg_covar=1e-4,
                             random_state=0).fit(X)
        bics.append(gm.bic(X))
    return dict(k=int(np.argmin(bics) + 1), bics=bics, finals=finals)


# ----------------------------------------------------------------------- reporting
def analyse(net, do_cle=False):
    mfile, rfile = NETWORKS[net]
    M = load_matrix(mfile)
    P = load_params()
    mf = MeanField(M, P)

    binfo, bfps = mf._threshold_boolean_attractors()
    fps = mf.fixed_points(bool_seeds=bfps)
    n_stable = sum(f["stable"] for f in fps)
    p_empty = mf.relax_from_empty()
    # which FP does the empty start land on
    basin = min(range(len(fps)),
                key=lambda i: np.linalg.norm(np.log1p(fps[i]["p"]) - np.log1p(p_empty))) if fps else -1

    bo = boolode_attractors(rfile) if rfile else None
    gains = loop_gains(mf, fps[basin]["p"] if fps else p_empty)

    P_ = P
    burst = dict(
        kon_over_kdm=P_["k_on"] / P_["kdm"], koff_over_kdm=P_["k_off"] / P_["kdm"],
        kon_over_kdp=P_["k_on"] / P_["kdp"], koff_over_kdp=P_["k_off"] / P_["kdp"],
        burst_size=P_["kpm"] / P_["k_off"],
    )
    telegraph_bimodal = burst["kon_over_kdm"] < 1.0 and burst["koff_over_kdm"] < 3.0

    cle = cle_check(mf) if do_cle else None

    n_vals = [1.0, 2.0, 3.0, 4.0, 6.0]
    s_vals = [0.5, 1.0, 2.0, 3.0, 4.0]
    bif_Z = bif_grid(M, P, n_vals, s_vals, bool_seeds=bfps)

    return dict(
        net=net, n_genes=mf.n, M=M, mf=mf, fps=fps, n_stable=n_stable,
        p_empty=p_empty, basin=basin, bif_n_vals=n_vals, bif_s_vals=s_vals, bif_Z=bif_Z,
        bool_thresh=binfo, bool_thresh_fps=bfps, boolode=bo,
        gains=gains, burst=burst, telegraph_bimodal=telegraph_bimodal, cle=cle,
    )


def page(pdf, res):
    net = res["net"]
    mf = res["mf"]
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.35, wspace=0.25)

    # --- text panel ---
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    L = []
    L.append(f"$\\bf{{{net}}}$   ({res['n_genes']} genes)")
    L.append("")
    L.append(f"Mean-field ODE limit:  {len(res['fps'])} fixed point(s), "
             f"{res['n_stable']} STABLE  ->  predicted # states = {res['n_stable']}")
    for i, f in enumerate(res["fps"]):
        tag = "stable  " if f["stable"] else ("OSCILLATORY" if f["oscillatory"] else "unstable")
        star = "  <-- empty-state relaxes here" if i == res["basin"] else ""
        L.append(f"    FP{i}: {tag}  mean promoter-ON = {np.mean(f['a']):.3f}   "
                 f"max Re(eig) = {f['max_real_eig']:+.3e}{star}")
    L.append("")
    bt = res["bool_thresh"]
    L.append(f"Threshold-Boolean (signed matrix + same k_add/k_off):  "
             f"{bt['n_fp']} fixed-point attractor(s), "
             f"{bt['n_cyclic_starts']}/{bt['scanned']} starts -> cycles")
    if res["boolode"]:
        bo = res["boolode"]
        L.append(f"BoolODE rule file ({bo['n']} genes):  {bo['n_fp']} fixed-point attractor(s), "
                 f"{bo['n_cyclic_starts']}/{bo['scanned']} starts -> cycles")
    else:
        L.append("BoolODE rule file:  n/a (no matching published rule model for this sim network)")
    L.append("")
    L.append(f"Observed in the actual sims (final-timepoint clustering, rep 0):  "
             f"{OBSERVED_CLUSTERS.get(net, '?')}")
    L.append("")
    if res["gains"]:
        g = res["gains"][0]
        L.append(f"Strongest feedback loop:  len {g['len']}, "
                 f"{'positive' if g['sign'] > 0 else 'negative'}, |loop gain| = {g['abs_gain']:.3f}  "
                 f"({'>1 -> can support ' + ('bistability' if g['sign'] > 0 else 'oscillation') if g['abs_gain'] > 1 else '<1 -> no bifurcation from this loop'})")
    b = res["burst"]
    L.append(f"Burst numbers:  k_on/kdeg_mRNA={b['kon_over_kdm']:.2f}, "
             f"k_off/kdeg_mRNA={b['koff_over_kdm']:.2f}, "
             f"k_on/kdeg_prot={b['kon_over_kdp']:.1f}, burst size={b['burst_size']:.2f}   "
             f"-> telegraph bimodality: {'YES' if res['telegraph_bimodal'] else 'no'}")
    if res["cle"]:
        L.append(f"Langevin surrogate (200 SDE runs from empty state):  ~{res['cle']['k']} mode(s)")
    ax.text(0, 1, "\n".join(L), va="top", ha="left", family="monospace", fontsize=8.5,
            transform=ax.transAxes)

    # --- bifurcation grid ---
    ax = fig.add_subplot(gs[1, 0])
    n_vals, s_vals, Z = res["bif_n_vals"], res["bif_s_vals"], res["bif_Z"]
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, len(s_vals), 0, len(n_vals)], vmin=1)
    ax.set_xticks(np.arange(len(s_vals)) + 0.5); ax.set_xticklabels(s_vals)
    ax.set_yticks(np.arange(len(n_vals)) + 0.5); ax.set_yticklabels(n_vals)
    ax.set_xlabel("k_add scale (1.0 = sim default)"); ax.set_ylabel("Hill n (2.0 = sim default)")
    ax.set_title("# stable mean-field fixed points")
    for ii in range(len(n_vals)):
        for jj in range(len(s_vals)):
            ax.text(jj + 0.5, ii + 0.5, str(Z[ii, jj]), ha="center", va="center",
                    color="w" if Z[ii, jj] < Z.max() else "k", fontsize=8)
    # sim operating point: Hill n = 2.0, k_add scale = 1.0 (cell centres at index+0.5)
    ax.plot([s_vals.index(1.0) + 0.5], [n_vals.index(2.0) + 0.5], "r*", ms=16,
            markeredgecolor="k", label="sim default (n=2, k_add 6.0/0.8)")
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # --- fixed-point protein levels ---
    ax = fig.add_subplot(gs[1, 1])
    for i, f in enumerate(res["fps"]):
        ax.plot(range(mf.n), np.log10(f["p"] + 1), "o-" if f["stable"] else "x--",
                label=f"FP{i} ({'stab' if f['stable'] else 'uns'})", alpha=0.8)
    ax.plot(range(mf.n), np.log10(res["p_empty"] + 1), "k.:", lw=1, label="empty-start")
    ax.set_xlabel("gene index"); ax.set_ylabel("log10(protein + 1)")
    ax.set_title("fixed-point expression states"); ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Pre-simulation multistability prediction: {net}", fontsize=13)
    pdf.savefig(fig); plt.close(fig)


def summary_page(pdf, rows):
    fig, ax = plt.subplots(figsize=(13.5, 6)); ax.axis("off")
    cols = ["network", "genes", "mean-field\nstable FP", "threshold-\nBoolean FP",
            "BoolODE-rule\nattractors", "max |loop\ngain|", "observed in sims\n(final-t clustering)"]
    tab = [[r["net"], r["n_genes"], r["n_stable"], r["bool_thresh"]["n_fp"],
            (r["boolode"]["n_fp"] if r["boolode"] else "n/a"),
            (f"{r['gains'][0]['abs_gain']:.2f}" if r["gains"] else "-"),
            OBSERVED_CLUSTERS.get(r["net"], "?")] for r in rows]
    t = ax.table(cellText=tab, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=[0.13, 0.05, 0.10, 0.10, 0.11, 0.08, 0.30])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 2.2)
    ax.text(0.5, 0.90,
            "Predicted vs observed number of steady states   (sim defaults: Hill n=2, k_add 6.0 activ / 0.8 repress)",
            ha="center", fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.12,
            "mean-field stable FP = states the deterministic limit of the TwINFER model supports.\n"
            "threshold-Boolean = steep-n limit of the same ADDITIVE regulation (what the sim actually uses).\n"
            "BoolODE-rule attractors = the published Boolean model's AND/OR logic (the biological ceiling).\n"
            "The BoolODE >> mean-field gap (HSC 6->1, VSC 5->5 but only 2 reached from p0=0) is why the sims look single/low-state:\n"
            "additive-on-k_on regulation + a single empty start populate far fewer attractors than the network can hold.",
            ha="center", va="top", fontsize=8.5, family="monospace", transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def main():
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=list(NETWORKS))
    ap.add_argument("--cle", action="store_true", help="also run the Langevin surrogate check")
    ap.add_argument("--replot", action="store_true",
                    help="reuse cached analysis, only redraw the PDF/CSV")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, "multistability_cache")
    os.makedirs(cache_dir, exist_ok=True)

    rows = []
    with PdfPages(OUT_PDF) as pdf:
        for net in args.networks:
            cache = os.path.join(cache_dir, f"{net}.pkl")
            if args.replot and os.path.exists(cache):
                with open(cache, "rb") as fh:
                    res = pickle.load(fh)
                print(f"[{net}] (cached)", flush=True)
            else:
                print(f"[{net}] analysing ...", flush=True)
                res = analyse(net, do_cle=args.cle)
                with open(cache, "wb") as fh:
                    pickle.dump(res, fh)
            print(f"    MF stable FP = {res['n_stable']}, thresh-Bool FP = {res['bool_thresh']['n_fp']}, "
                  f"BoolODE FP = {res['boolode']['n_fp'] if res['boolode'] else '-'}", flush=True)
            page(pdf, res)
            rows.append(res)
        summary_page(pdf, rows)

    pd.DataFrame([{
        "network": r["net"], "n_genes": r["n_genes"],
        "meanfield_stable_fp": r["n_stable"],
        "meanfield_total_fp": len(r["fps"]),
        "threshold_boolean_fp": r["bool_thresh"]["n_fp"],
        "boolode_rule_fp": (r["boolode"]["n_fp"] if r["boolode"] else None),
        "empty_start_basin": r["basin"],
        "max_abs_loop_gain": (r["gains"][0]["abs_gain"] if r["gains"] else None),
        "telegraph_bimodal": r["telegraph_bimodal"],
        "observed_sim_clustering": OBSERVED_CLUSTERS.get(r["net"]),
    } for r in rows]).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_PDF}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
