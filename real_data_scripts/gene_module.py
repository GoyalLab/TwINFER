"""
module_networks.py

Module networks (Segal, Shapira, Regev, Pe'er, Botstein, Koller & Friedman,
Nature Genetics 34:166-176, 2003), implemented from the paper's Methods section.

Procedure
---------
1. Each candidate regulator is discretised across arrays into down / no change /
   up with k-means (k = 3).
2. Genes are initialised into K modules by hierarchical agglomerative clustering.
3. EM-style loop
     M-step: for each module learn a regulation program = regression tree.
             Decision nodes are queries "is regulator X up/down-regulated?";
             each leaf (a "context") holds Normal(mean, var) over the module's
             genes in the arrays that reach the leaf. Trees are grown root ->
             leaves; at each node the query with the best Bayesian-score gain
             is taken, until no split improves the score.
     E-step: each gene is moved to the module whose program gives its expression
             profile the highest probability (product over arrays). A regulator
             is never assigned to a module in which it is a regulatory input.
   Parameter-only M-steps alternate with E-steps until the partition is stable;
   only then are tree structures re-learned. Stop when no structure changes.
4. Post-processing: annotation enrichment (hypergeometric + Bonferroni) and
   regulator <-> condition association (binomial).

Not in the paper's text (so chosen here, see comments marked [choice]):
   the exact Bayesian score / priors (paper defers to a technical report),
   the initial clustering algorithm (paper used PCluster), min leaf size.

Known failure mode (the paper's Discussion describes it): if a true regulator gene is
clustered into its own target module, the "a member cannot regulate its module" rule
stops it from ever being selected, and the tree falls back on weaker proxy regulators.
Default behaviour follows the paper. Pass init_without_regulators=True (CLI:
--init-without-regulators) to cluster only non-regulator genes at initialisation and let
the first E-step place the regulators; on simulated data this recovered the true
regulators in every setting tried, versus 65-88% with the paper-faithful start.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.special import gammaln
from scipy.stats import binomtest, hypergeom

LOG2PI = np.log(2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Bayesian score of a Gaussian leaf  [choice: Normal-Gamma conjugate prior]
# --------------------------------------------------------------------------- #
@dataclass
class Prior:
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0


def _posterior(n, s1, s2, p: Prior):
    n = np.asarray(n, float)
    nn = np.maximum(n, 1.0)
    mean = np.asarray(s1, float) / nn
    ssd = np.maximum(np.asarray(s2, float) - n * mean ** 2, 0.0)
    kn = p.kappa0 + n
    an = p.alpha0 + n / 2.0
    bn = p.beta0 + 0.5 * ssd + p.kappa0 * n * (mean - p.mu0) ** 2 / (2.0 * kn)
    return kn, an, bn


def leaf_logml(n, s1, s2, p: Prior):
    """Log marginal likelihood of n values with sum s1 and sum of squares s2."""
    kn, an, bn = _posterior(n, s1, s2, p)
    n = np.asarray(n, float)
    return (gammaln(an) - gammaln(p.alpha0) + p.alpha0 * np.log(p.beta0)
            - an * np.log(bn) + 0.5 * (np.log(p.kappa0) - np.log(kn))
            - 0.5 * n * LOG2PI)


def leaf_params(n, s1, s2, p: Prior):
    """Posterior-mean mean and plug-in variance for a leaf."""
    kn, an, bn = _posterior(n, s1, s2, p)
    mu = (p.kappa0 * p.mu0 + np.asarray(s1, float)) / kn
    var = bn / an
    return mu, var


# --------------------------------------------------------------------------- #
# Regulator discretisation (paper: k-means into down / no change / up)
# --------------------------------------------------------------------------- #
def discretise_regulator(v: np.ndarray) -> np.ndarray:
    """Return int8 states in {-1, 0, +1}; NaN arrays are set to 0."""
    s = np.zeros(v.shape, dtype=np.int8)
    ok = ~np.isnan(v)
    x = v[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return s
    # Lloyd's k-means is initialisation-sensitive on skewed data (most arrays "no change"),
    # so run several starts and keep the lowest within-cluster squared error.
    inits = [np.linspace(x.min(), x.max(), 3),
             np.quantile(x, [1 / 6, 0.5, 5 / 6]),
             np.quantile(x, [0.1, 0.5, 0.9]),
             np.quantile(x, [0.02, 0.5, 0.98])]
    best = None
    for c in inits:
        c = np.asarray(c, float)
        for _ in range(100):
            lab = np.abs(x[:, None] - c[None, :]).argmin(1)
            new = np.array([x[lab == j].mean() if np.any(lab == j) else c[j]
                            for j in range(3)])
            if np.allclose(new, c):
                break
            c = new
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        sse = float(((x - c[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c, lab)
    _, c, lab = best
    rank = np.empty(3, dtype=int)
    rank[np.argsort(c)] = [-1, 0, 1]      # lowest centre = down, highest = up
    s[ok] = rank[lab]
    return s


# --------------------------------------------------------------------------- #
# Regulation program = regression tree
# --------------------------------------------------------------------------- #
class RegTree:
    def __init__(self, structure, leaf_masks, nodes):
        self.structure = structure        # nested tuple (cand, false_child, true_child) | None
        self.leaf_masks = leaf_masks      # contexts: one bool mask over arrays per leaf
        self.nodes = nodes                # decision nodes (dicts), pre-order
        self.used_regs = {nd["reg"] for nd in nodes}
        self.mu = np.zeros(len(leaf_masks))
        self.var = np.ones(len(leaf_masks))

    def fit_params(self, n_a, A1, A2, prior):
        for i, m in enumerate(self.leaf_masks):
            self.mu[i], self.var[i] = leaf_params(n_a[m].sum(), A1[m].sum(), A2[m].sum(), prior)

    def score(self, n_a, A1, A2, prior):
        return float(sum(leaf_logml(n_a[m].sum(), A1[m].sum(), A2[m].sum(), prior)
                         for m in self.leaf_masks))

    def array_params(self):
        n = len(self.leaf_masks[0])
        mu_a, var_a = np.empty(n), np.empty(n)
        for i, m in enumerate(self.leaf_masks):
            mu_a[m] = self.mu[i]
            var_a[m] = self.var[i]
        return mu_a, var_a


def learn_tree(n_a, A1, A2, B, allowed, prior, min_arrays, max_depth, min_gain, Bf=None):
    """
    Grow a regression tree root -> leaves.

    n_a, A1, A2 : per-array count, sum, sum of squares over the module's genes
    B           : (n_candidates x n_arrays) bool, candidate query answers
                  (row 2r = "regulator r up", row 2r+1 = "regulator r down")
    allowed     : (n_candidates,) bool, queries this module may use
    """
    n_arrays = B.shape[1]
    if Bf is None:
        Bf = B.astype(float)                              # precomputed once by the caller when possible
    V = np.stack([n_a, A1, A2], axis=1)                   # (n_arrays x 3): one matmul instead of three
    leaf_masks, nodes = [], []

    def grow(mask, depth):
        # [speed] work only on the arrays inside this context (columns idx), not all arrays
        idx = np.flatnonzero(mask)
        n_p, s1_p, s2_p = (float(x) for x in V[idx].sum(0))
        if depth < max_depth and idx.size >= 2 * min_arrays:
            parent = leaf_logml(n_p, s1_p, s2_p, prior)
            Bt = Bf[:, idx]                               # (true answer) & (in this context)
            k_t = Bt.sum(1)
            k_f = idx.size - k_t
            T = Bt @ V[idx]
            n_t, s1_t, s2_t = T[:, 0], T[:, 1], T[:, 2]
            # --- replaced (kept for reference): full-width mask product, three matmuls per node
            # mf = mask.astype(float)
            # Bt = Bf * mf
            # k_t = Bt.sum(1); k_f = mf.sum() - k_t
            # n_t, s1_t, s2_t = Bt @ n_a, Bt @ A1, Bt @ A2
            gain = (leaf_logml(n_t, s1_t, s2_t, prior)
                    + leaf_logml(n_p - n_t, s1_p - s1_t, s2_p - s2_t, prior) - parent)
            ok = allowed & (k_t >= min_arrays) & (k_f >= min_arrays)
            gain = np.where(ok, gain, -np.inf)
            c = int(np.argmax(gain))
            if gain[c] > min_gain:
                t_mask, f_mask = mask & B[c], mask & ~B[c]
                nodes.append(dict(cand=c, reg=c // 2, query=1 if c % 2 == 0 else -1,
                                  mask=mask.copy(), true=t_mask.copy(),
                                  depth=depth, gain=float(gain[c])))
                s_false = grow(f_mask, depth + 1)         # left  = answer "false"
                s_true = grow(t_mask, depth + 1)          # right = answer "true"
                return (c, s_false, s_true)
        leaf_masks.append(mask)
        return None

    structure = grow(np.ones(n_arrays, bool), 0)
    return RegTree(structure, leaf_masks, nodes)


_SHARED = {}       # set by ModuleNetwork before forking; workers read it copy-on-write


def _learn_worker(members):
    """Learn (and fit) one module's tree from the arrays shared via fork."""
    mn = _SHARED["mn"]
    return mn._learn(members)


# --------------------------------------------------------------------------- #
# The module network learner
# --------------------------------------------------------------------------- #
class ModuleNetwork:
    def __init__(self, n_modules=50, min_arrays=5, max_depth=20, min_gain=0.0,
                 max_iter=100, max_inner=30, prior: Prior | None = None,
                 linkage_method="ward", init_without_regulators=False, verbose=True,
                 min_arrays_frac=None, n_jobs=1):
        self.init_without_regulators = init_without_regulators
        self.n_modules = n_modules
        self.min_arrays = min_arrays      # [choice] smallest context allowed
        self.min_arrays_frac = min_arrays_frac   # if set: min_arrays = frac * n_arrays
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.max_iter = max_iter
        self.max_inner = max_inner
        self.prior = prior
        self.linkage_method = linkage_method
        self.verbose = verbose

    # ---- helpers ---------------------------------------------------------- #
    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _stats(self, members):
        n_a = self._M[members].sum(0).astype(float)
        A1 = self._X0[members].sum(0)
        A2 = self._X0sq[members].sum(0)
        return n_a, A1, A2

    def _learn(self, members):
        n_a, A1, A2 = self._stats(members)
        allowed = ~np.isin(self._cand_gene, members)   # a gene can't regulate its own module
        tree = learn_tree(n_a, A1, A2, self._B, allowed, self.prior,
                          self.min_arrays, self.max_depth, self.min_gain, Bf=self._Bf)
        tree.fit_params(n_a, A1, A2, self.prior)
        tree.sig = np.asarray(members, dtype=np.int64).tobytes()   # module membership it was learned from
        return tree

    def _learn_many(self, member_list, old_trees=None):
        """
        Learn trees for several modules. A module whose membership is identical to the one its
        current tree was learned from keeps that tree (the result would be identical); the rest
        are learned in parallel (fork) when n_jobs > 1.
        """
        out = [None] * len(member_list)
        todo = []
        for k, m in enumerate(member_list):
            sig = np.asarray(m, dtype=np.int64).tobytes()
            if old_trees is not None and getattr(old_trees[k], "sig", None) == sig:
                out[k] = old_trees[k]
            else:
                todo.append(k)
        if self.n_jobs > 1 and len(todo) > 1:
            _SHARED["mn"] = self
            with mp.get_context("fork").Pool(min(self.n_jobs, len(todo))) as pool:
                res = pool.map(_learn_worker, [member_list[k] for k in todo], chunksize=1)
        else:
            res = [self._learn(member_list[k]) for k in todo]
        for k, t in zip(todo, res):
            out[k] = t
        return out

    def _refit(self, trees, assign):
        for k, t in enumerate(trees):
            n_a, A1, A2 = self._stats(np.flatnonzero(assign == k))
            t.fit_params(n_a, A1, A2, self.prior)

    def _loglik(self, trees):
        """(genes x modules) log-probability of each gene's profile under each program."""
        G = self._X0.shape[0]
        Mf = self._M.astype(float)
        nobs = Mf.sum(1)
        L = np.empty((G, len(trees)))
        for k, t in enumerate(trees):
            mu, var = t.array_params()
            iv = 1.0 / var
            L[:, k] = -0.5 * (LOG2PI * nobs + Mf @ np.log(var) + self._X0sq @ iv
                              - 2.0 * self._X0 @ (mu * iv) + Mf @ (mu ** 2 * iv))
            for r in t.used_regs:                      # regulator can't join a module it regulates
                L[self._reg_gene[r], k] = -np.inf
        return L

    @staticmethod
    def _compact(assign, trees):
        keep = [k for k in range(len(trees)) if np.any(assign == k)]
        remap = {old: new for new, old in enumerate(keep)}
        return np.array([remap[a] for a in assign]), [trees[k] for k in keep]

    def _init_partition(self):
        # [choice] the paper used PCluster (hierarchical agglomerative); any balanced
        # clustering into K groups works. Ward on the raw profiles gives balanced clusters;
        # other methods ("average", "complete") use correlation distance.
        # init_without_regulators=False reproduces the paper (all genes, regulators included).
        # True clusters only non-regulator genes; regulators get -1 and are placed by the
        # first E-step. This avoids the trap where a regulator is clustered into its own
        # target module and can then never be used as that module's regulator.
        n = self._X0.shape[0]
        use = np.arange(n)
        if self.init_without_regulators:
            use = np.setdiff1d(use, self._reg_gene)
        Xc = self._X0[use]
        if self.linkage_method == "ward":
            Z = linkage(Xc, method="ward")
        else:
            D = np.nan_to_num(pdist(Xc, "correlation"), nan=1.0)
            Z = linkage(D, method=self.linkage_method)
        lab = np.unique(fcluster(Z, self.n_modules, "maxclust") - 1, return_inverse=True)[1]
        assign = np.full(n, -1, dtype=int)
        assign[use] = lab
        return assign

    # ---- main ------------------------------------------------------------- #
    def fit(self, expr: pd.DataFrame, regulators):
        """expr: genes x arrays (log ratios, NaN allowed). regulators: iterable of gene names."""
        self.genes_ = list(expr.index)
        self.arrays_ = list(expr.columns)
        X = expr.to_numpy(float)
        self._M = ~np.isnan(X)
        self._X0 = np.where(self._M, X, 0.0)
        self._X0sq = self._X0 ** 2

        gidx = {g: i for i, g in enumerate(self.genes_)}
        self.regs_ = [r for r in dict.fromkeys(regulators) if r in gidx]
        if not self.regs_:
            raise ValueError("none of the candidate regulators are in the expression matrix")
        self._reg_gene = np.array([gidx[r] for r in self.regs_])
        self._cand_gene = np.repeat(self._reg_gene, 2)

        S = np.vstack([discretise_regulator(X[g]) for g in self._reg_gene])
        self.reg_states_ = S
        self._B = np.empty((2 * len(self.regs_), X.shape[1]), bool)
        self._B[0::2] = S == 1
        self._B[1::2] = S == -1
        self._Bf = self._B.astype(float)               # built once; reused by every tree search
        if self.min_arrays_frac is not None:
            self.min_arrays = max(1, int(round(self.min_arrays_frac * X.shape[1])))
            self._log(f"min_arrays = {self.min_arrays} ({self.min_arrays_frac:.1%} of {X.shape[1]} arrays)")

        if self.prior is None:                         # [choice] weak data-scaled prior
            v = float(np.var(self._X0[self._M]))
            self.prior = Prior(mu0=float(self._X0[self._M].mean()), kappa0=1.0,
                               alpha0=1.0, beta0=max(v, 1e-6))

        assign = self._init_partition()
        trees = self._learn_many([np.flatnonzero(assign == k) for k in range(assign.max() + 1)])
        self._log(f"init: {len(trees)} modules, {len(self.regs_)} candidate regulators")

        for outer in range(self.max_iter):
            e_steps, moved = 0, 0
            for _ in range(self.max_inner):            # E-steps + parameter-only M-steps
                L = self._loglik(trees)
                new = L.argmax(1)
                bad = ~np.isfinite(L).any(1)
                new[bad] = np.maximum(assign[bad], 0)
                moved = int((new != assign).sum())
                assign = new
                e_steps += 1
                assign, trees = self._compact(assign, trees)
                self._refit(trees, assign)
                if moved == 0:
                    break

            # --- replaced: new_trees = [self._learn(np.flatnonzero(assign == k)) for k in range(len(trees))]
            new_trees = self._learn_many([np.flatnonzero(assign == k) for k in range(len(trees))], trees)
            changed = sum(a.structure != b.structure for a, b in zip(new_trees, trees))
            trees = new_trees
            score = sum(t.score(*self._stats(np.flatnonzero(assign == k)), self.prior)
                        for k, t in enumerate(trees))
            self._log(f"iter {outer:2d}: modules={len(trees)} E-steps={e_steps} "
                      f"last-moved={moved} trees-changed={changed} score={score:.1f}")
            if changed == 0:
                break

        self.assign_, self.trees_ = assign, trees
        return self

    # ---- outputs ---------------------------------------------------------- #
    def modules(self) -> pd.Series:
        return pd.Series(self.assign_ + 1, index=self.genes_, name="module")

    def programs(self) -> pd.DataFrame:
        """One row per decision node: which regulator / query, where in the tree, how much it helps."""
        rows = []
        for k, t in enumerate(self.trees_):
            for nd in t.nodes:
                rows.append(dict(module=k + 1, regulator=self.regs_[nd["reg"]],
                                 query="up" if nd["query"] == 1 else "down",
                                 depth=nd["depth"], n_arrays=int(nd["mask"].sum()),
                                 score_gain=nd["gain"]))
        return pd.DataFrame(rows)

    def contexts(self) -> pd.DataFrame:
        """One row per leaf: arrays in the context and the module's mean / variance there."""
        rows = []
        for k, t in enumerate(self.trees_):
            for i, m in enumerate(t.leaf_masks):
                rows.append(dict(module=k + 1, leaf=i, n_arrays=int(m.sum()),
                                 mean=t.mu[i], var=t.var[i],
                                 arrays=",".join(np.array(self.arrays_)[m])))
        return pd.DataFrame(rows)

    def regulator_targets(self) -> dict:
        """Targets of a regulator = union of genes of all modules whose tree uses it."""
        tg = {}
        for k, t in enumerate(self.trees_):
            genes = set(np.array(self.genes_)[self.assign_ == k])
            for r in t.used_regs:
                tg.setdefault(self.regs_[r], set()).update(genes)
        return tg

    def regulator_conditions(self, labels, alpha=0.05) -> pd.DataFrame:
        """
        Association of each regulator (per decision node) with experimental conditions.
        labels: one condition label per array. For a condition, the arrays that reach the node
        are split by the node's true/false branch and a binomial test is applied against the
        overall true-branch fraction at that node.   [choice: exact null not given in the paper]
        """
        labels = pd.Series(np.asarray(labels), index=self.arrays_)
        rows = []
        for k, t in enumerate(self.trees_):
            for nd in t.nodes:
                p0 = nd["true"].sum() / nd["mask"].sum()
                for cond in labels.unique():
                    sel = nd["mask"] & (labels == cond).to_numpy()
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    k_true = int((sel & nd["true"]).sum())
                    p = binomtest(k_true, n, p0).pvalue
                    if p < alpha:
                        rows.append(dict(module=k + 1, regulator=self.regs_[nd["reg"]],
                                         query="up" if nd["query"] == 1 else "down",
                                         condition=cond, n=n, n_true_branch=k_true, p=p,
                                         branch="true" if k_true / n > p0 else "false"))
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Annotation enrichment (paper: hypergeometric, Bonferroni P < 0.05 / n)
# --------------------------------------------------------------------------- #
def annotation_enrichment(groups: dict, annotations: dict, universe, alpha=0.05, min_genes=5):
    """
    groups: name -> set of genes (modules, or regulator target sets)
    annotations: name -> set of genes (GO / MIPS / KEGG categories ...)
    """
    universe = set(universe)
    ann = {a: g & universe for a, g in annotations.items()}
    ann = {a: g for a, g in ann.items() if len(g) >= min_genes}
    n_tests, N = max(len(ann), 1), len(universe)
    rows = []
    for name, genes in groups.items():
        genes = set(genes) & universe
        for a, gs in ann.items():
            k = len(genes & gs)
            if k == 0:
                continue
            p = hypergeom.sf(k - 1, N, len(gs), len(genes))
            if p < alpha / n_tests:
                rows.append(dict(group=name, annotation=a, overlap=k, group_size=len(genes),
                                 annot_size=len(gs), frac_of_group=k / len(genes), p=p))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Simulated data with known regulation programs
# --------------------------------------------------------------------------- #
def simulate(seed=0, n_arrays=150, n_regs=25, genes_per_module=60, n_noise=150, reg_noise=0.3):
    rng = np.random.default_rng(seed)
    st = rng.choice([-1, 0, 1], size=(n_regs, n_arrays), p=[0.2, 0.6, 0.2])
    reg = st * 1.5 + rng.normal(0, reg_noise, st.shape)
    up = lambda r: (st[r] == 1).astype(float)
    dn = lambda r: (st[r] == -1).astype(float)
    levels = [2.0 * up(0) - 2.0 * up(1),               # M1: reg0 activates, reg1 represses
              2.0 * up(2),                             # M2: reg2 activates
              -2.0 * dn(3),                            # M3: reg3 down -> module down
              1.5 * up(4) + 1.5 * up(5),               # M4: two activators
              2.0 * up(6) * (1 - up(7))]               # M5: reg6 up AND reg7 not up
    truth = {0: "REG00 +, REG01 -", 1: "REG02", 2: "REG03 (down)", 3: "REG04, REG05",
             4: "REG06 and not REG07"}
    rows, names, label = [], [], []
    for r in range(n_regs):
        rows.append(reg[r]); names.append(f"REG{r:02d}"); label.append(-1)
    for j, lv in enumerate(levels):
        for i in range(genes_per_module):
            rows.append(lv + rng.normal(0, 0.5, n_arrays)); names.append(f"M{j + 1}_g{i}"); label.append(j)
    for i in range(n_noise):
        rows.append(rng.normal(0, 0.8, n_arrays)); names.append(f"N_g{i}"); label.append(-1)
    expr = pd.DataFrame(np.vstack(rows), index=names, columns=[f"a{i}" for i in range(n_arrays)])
    return expr, [f"REG{r:02d}" for r in range(n_regs)], np.array(label), truth


def demo(init_without_regulators=False):
    expr, regs, label, truth = simulate()
    mn = ModuleNetwork(n_modules=8, init_without_regulators=init_without_regulators).fit(expr, regs)
    mods = mn.modules()
    print("\n=== recovery of simulated modules ===")
    for j in range(5):
        idx = np.flatnonzero(label == j)
        found = mods.iloc[idx].value_counts()
        k = found.index[0]
        prog = mn.programs()
        prog = prog[prog.module == k]
        print(f"true M{j + 1} (regulators: {truth[j]}): {found.iloc[0]}/{len(idx)} genes in found "
              f"module {k}; found regulators: "
              f"{', '.join(prog.regulator + ':' + prog['query']) or 'none'}")
    print("\n=== all programs ===")
    print(mn.programs().to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Module networks (Segal et al. 2003)")
    ap.add_argument("--demo", action="store_true", help="run on simulated data")
    ap.add_argument("--expr", help="CSV, genes x arrays, first column = gene name")
    ap.add_argument("--regulators", help="text file, one candidate regulator per line")
    ap.add_argument("--modules", type=int, default=50)
    ap.add_argument("--min-arrays", type=int, default=5)
    ap.add_argument("--min-arrays-frac", type=float, default=None,
                    help="smallest context as a fraction of arrays (overrides --min-arrays), e.g. 0.05")
    ap.add_argument("--n-jobs", type=int, default=1, help="processes for per-module tree learning")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--init-without-regulators", action="store_true",
                    help="cluster only non-regulator genes at initialisation (see module docstring)")
    a = ap.parse_args()
    if a.demo:
        demo(a.init_without_regulators)
        return
    expr = pd.read_csv(a.expr, index_col=0)
    regs = [l.strip() for l in open(a.regulators) if l.strip()]
    mn = ModuleNetwork(n_modules=a.modules, min_arrays=a.min_arrays, min_arrays_frac=a.min_arrays_frac,
                       n_jobs=a.n_jobs,
                       init_without_regulators=a.init_without_regulators).fit(expr, regs)
    mn.modules().to_csv(f"{a.out}/modules.csv")
    mn.programs().to_csv(f"{a.out}/programs.csv", index=False)
    mn.contexts().to_csv(f"{a.out}/contexts.csv", index=False)


if __name__ == "__main__":
    main()