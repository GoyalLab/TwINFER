"""Exact decomposition on the z_rho scale (twin-only designs N = 1500 and 3000, 20 h):
   z_rho = A_s + A_u,   A_s = C / sigma_rho (shared part in z_rho units),  A_u = U / sigma_rho = c * z_U,   c = sqrt((1-h_X)(1-h_Y)) * sigma_Delta / sigma_rho,
   sigma_rho^2 = (0.5 + 0.5 max(h_X,0) max(h_Y,0)) / N,  sigma_Delta = 1/sqrt(N),  h = sister correlation C(x,x) of the twin-centred ranks,  rho = S (twin cells only), U = S - C.
Computes these for every gene pair of every network simulation and for the two-gene scenarios; caches <O>/twin_decomp_networks.csv and <O>/twin_decomp_twogene.csv.   usage: python twin_decomposition.py [--jobs 8]"""
import argparse, os, re, zlib, glob, numpy as np, pandas as pd
from motif_shared_driver import rk_mid
import twin_design_zscores as T
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
def twin_stats(X, Y):
    n = len(X); rx = rk_mid(X.ravel()).reshape(n, 2); ry = rk_mid(Y.ravel()).reshape(n, 2); rx = rx - rx.mean(); ry = ry - ry.mean()
    M = lambda a, b: ((a * b).sum(1) / 2).mean(); nrm = np.sqrt(M(rx, rx) * M(ry, ry))
    S = ((rx[:, 0] * ry[:, 0] + rx[:, 1] * ry[:, 1]) / 2).mean() / nrm; C = ((rx[:, 0] * ry[:, 1] + rx[:, 1] * ry[:, 0]) / 2).mean() / nrm
    hxu = (rx[:, 0] * rx[:, 1]).mean() / M(rx, rx); hyu = (ry[:, 0] * ry[:, 1]).mean() / M(ry, ry); hx, hy = max(0., hxu), max(0., hyu)
    sr = np.sqrt((0.5 + 0.5 * hx * hy) / n); sd = 1 / np.sqrt(n); dx = rx[:, 0] - rx[:, 1]; dy = ry[:, 0] - ry[:, 1]; rd = (dx * dy).sum() / np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    z_rho = S / sr; z_U = rd / sd; c = np.sqrt(max((1 - hxu) * (1 - hyu), 0.)) * sd / sr; z_C = np.sign(S) * C / np.sqrt((1 + S * S + hx * hy) / (2 * n))
    return dict(S=S, C=C, U=S - C, hx=hxu, hy=hyu, sig_rho=sr, z_rho=z_rho, z_U=z_U, z_C=z_C, c=c, A_s=C / sr, A_u=(S - C) / sr, rd=rd)
def net_file(a):
    raw, sim, f = a; cols = pd.read_csv(f, nrows=0).columns; genes = sorted(int(re.match(r"gene_(\d+)_mRNA", c).group(1)) for c in cols if re.match(r"gene_\d+_mRNA$", c))
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + [f"gene_{g}_mRNA" for g in genes]); d = d[d.time_step == 20].sort_values(["clone_id", "replicate"]); ncl = d.clone_id.nunique()
    X = {g: d[f"gene_{g}_mRNA"].to_numpy(float).reshape(ncl, 2) for g in genes}; perm = np.random.RandomState(zlib.crc32(f"{raw}|{sim}".encode()) % 2**31).permutation(ncl); rows = []
    for N in (1500, 3000):
        c = perm[:N]
        for i in genes:
            for j in genes:
                if j <= i or X[i][c].std() == 0 or X[j][c].std() == 0: continue
                o = twin_stats(X[i][c], X[j][c]); o.update(raw=raw, sim=sim, N=N, x=i, y=j); rows.append(o)
    return rows
def twogene_rep(rep):
    import plot_new_metrics_boxplots as P
    rows = []; jobs = [(name, cfgs, None, si) for si, (name, cfgs) in enumerate(P.SCEN)] + [(f"{n} (drift)", None, n, 7) for n in ("A_B", "A_to_B")]
    for name, cfgs, net, si in jobs:
        rng = np.random.RandomState(rep * 10 + si)
        if net: Pd = P.load20(glob.glob(f"{P.DRIFT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs) == 1: Pd = P.load20(P.find(cfgs[0], rep)[0])
        else:
            S_ = [P.load20(P.find(c_, rep)[0]) for c_ in cfgs]; sa, sb = rng.choice(P.NCL, 3000, replace=False), rng.choice(P.NCL, 3000, replace=False); Pd = {g: np.vstack([S_[0][g][sa], S_[1][g][sb]]) for g in P.G}
        cc = np.random.RandomState(rep * 100 + si + 12345).permutation(P.NCL)
        for N in (1500, 3000):
            o = twin_stats(Pd["gene_1"][cc[:N]], Pd["gene_2"][cc[:N]]); o.update(scenario=name, rep=rep, N=N); rows.append(o)
    return rows
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    if not os.path.exists(O + "twin_decomp_networks.csv"):
        with mp.Pool(a.jobs) as p: res = p.map(net_file, T.files(), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + "twin_decomp_networks.csv", index=False); print("networks done", flush=True)
    if not os.path.exists(O + "twin_decomp_twogene.csv"):
        with mp.Pool(a.jobs) as p: res = p.map(twogene_rep, range(100), chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(O + "twin_decomp_twogene.csv", index=False); print("twogene done", flush=True)
