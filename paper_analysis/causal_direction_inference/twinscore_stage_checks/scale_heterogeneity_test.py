"""Does a state-dependent count scale inflate the difference-first z_U (raw twin differences -> rank -> Pearson) but not the rank-first z_U (ranks over 1,500 twins + 3,000 singletons -> twin differences -> Pearson)?
Part 1 (synthetic, null): 2 states (50/50), genes independent, counts ~ Poisson(mean) with mean low=0.5 and high=0.5*ratio, twins independent given the state (so rho_hat_Delta = 0 truly), 1,500 twin pairs + 3,000 singletons,
 2,000 replicates per ratio; report SD and false-positive rate (|z|>1.96) of z_U = rho_hat_Delta*sqrt(1500) for: rank-first | difference-first | difference-first on scaled differences (x_a-x_b)/sqrt(x_a+x_b+1) | difference of log1p counts (no ranking).
Part 2 (real simulations, baseline design, mRNA): same four estimators on A_B two states / A_B drift / A_B single state (null) and A_to_B two states (edge)."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
_argv = sys.argv; sys.argv = [sys.argv[0]]
import rank_order_singletons as RS
sys.argv = _argv
F = RS.F; N = 1500; NS_ = 3000
def pear(a, b): a = a - a.mean(); b = b - b.mean(); return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
def estimators(Xt, Yt, xs, ys):
    n = len(Xt); w = np.concatenate([np.full(2 * n, 0.5), np.ones(len(xs))])
    rx = F.wrank(np.concatenate([Xt.ravel(), xs]), w)[:2 * n].reshape(n, 2); ry = F.wrank(np.concatenate([Yt.ravel(), ys]), w)[:2 * n].reshape(n, 2)
    out = {"rank-first": pear(rx[:, 0] - rx[:, 1], ry[:, 0] - ry[:, 1]), "difference-first (infer.py)": pear(rankdata(Xt[:, 0] - Xt[:, 1]), rankdata(Yt[:, 0] - Yt[:, 1]))}
    sc = lambda T: (T[:, 0] - T[:, 1]) / np.sqrt(T[:, 0] + T[:, 1] + 1)
    out["difference-first, scaled by sqrt(xa+xb+1)"] = pear(rankdata(sc(Xt)), rankdata(sc(Yt)))
    out["log1p then difference (no ranks)"] = pear(np.log1p(Xt[:, 0]) - np.log1p(Xt[:, 1]), np.log1p(Yt[:, 0]) - np.log1p(Yt[:, 1]))
    return {k: v * np.sqrt(n) for k, v in out.items()}
if __name__ == "__main__":
    rng = np.random.default_rng(7); rows = []
    for ratio in (1, 3, 10, 30):
        z = []
        for _ in range(2000):
            st = rng.random(N + NS_) < 0.5; mu = np.where(st, 0.5 * ratio, 0.5)
            X = rng.poisson(mu[:N, None], (N, 2)).astype(float); Y = rng.poisson(mu[:N, None], (N, 2)).astype(float)   # twins independent given state, genes independent given state
            xs = rng.poisson(mu[N:]).astype(float); ys = rng.poisson(mu[N:]).astype(float)
            z.append(estimators(X, Y, xs, ys))
        z = pd.DataFrame(z)
        for k in z: rows.append(dict(ratio_high_to_low=ratio, estimator=k, SD=z[k].std(), FP=(z[k].abs() > 1.96).mean()))
    t = pd.DataFrame(rows); print("PART 1 synthetic null (state-dependent scale, independent genes): z_U SD (ideal 1.00) / false-positive rate (ideal 0.05)")
    print(t.pivot(index="estimator", columns="ratio_high_to_low", values="SD").round(2).to_string()); print(t.pivot(index="estimator", columns="ratio_high_to_low", values="FP").round(3).to_string()); t.to_csv("scale_heterogeneity_synthetic.csv", index=False)
