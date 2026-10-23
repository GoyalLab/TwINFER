"""Item 4: can an observable per-gene statistic flag autoregulated genes, and does a per-pair C-penalty switch built on it help?
Stage 1  per gene of every mixed simulation file (sims 0-2): features from a realistic twin sample at t = 20 h (3,000 twin clones): h (sister Spearman), Fano, zero fraction, CV, skew; and
         'memory' = Spearman(x_sister_a(1 h), x_sister_b(20 h)) over the other 3,000 clones (cross-time clones, as in infer.py).  Target: gene has a self-loop (any / + / -).
Stage 2  pair score |z_rho| - b|z_C| + |z_U| with b chosen per pair: fixed (0, 0.5, 1), oracle (b = 1 if no self-loop gene in the pair else 0), flag (b = 1 if neither gene is flagged, else 0), soft flag (b = 1 - max pct-rank).
         Flag = top half of the panel (one panel = one simulation) by the chosen feature.  AUPRC x base on the existing design-X pair table, mixed networks, mean over networks."""
import glob, os, re, zlib, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, skew, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"; B = A + "code/Beeline/inputs/mixed_network_sweep/"
def files():
    best = {}
    for f in glob.glob(A + "simulation_data/mixed_network_sweep/df_grn_*.csv"):
        m = re.search(r"df_(.+?_rep\d+)_rep(\d+)_\d{8}_", os.path.basename(f))
        if m and int(m.group(2)) <= 2:
            k = (m.group(1), int(m.group(2)))
            if k not in best or os.path.getmtime(f) > os.path.getmtime(best[k]): best[k] = f
    return sorted((k[0], k[1], f) for k, f in best.items())
def one(a):
    net, sim, f = a; cols = pd.read_csv(f, nrows=0).columns; genes = sorted(int(re.match(r"gene_(\d+)_mRNA", c).group(1)) for c in cols if re.match(r"gene_\d+_mRNA$", c))
    d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + [f"gene_{g}_mRNA" for g in genes]); d = d[d.time_step.isin([1, 20])].sort_values(["time_step", "clone_id", "replicate"])
    t20 = d[d.time_step == 20]; t1 = d[d.time_step == 1]; ncl = t20.clone_id.nunique(); perm = np.random.RandomState(zlib.crc32(f"{net}|{sim}".encode()) % 2**31).permutation(ncl)
    twin, cross = perm[:3000], perm[3000:]; gs = pd.read_csv(B + net + "/GroundTruthNetwork.csv"); sl = {int(a.split("_")[1]): t for a, t, b in zip(gs.Gene1, gs.Type, gs.Gene2) if a == b}; rows = []
    for g in genes:
        X = t20[f"gene_{g}_mRNA"].to_numpy(float).reshape(ncl, 2); X1 = t1[f"gene_{g}_mRNA"].to_numpy(float).reshape(ncl, 2); xt = X[twin]; v = xt.ravel()
        h = spearmanr(xt[:, 0], xt[:, 1])[0]; mem = spearmanr(X1[cross, 0], X[cross, 1])[0]
        rows.append(dict(net=net, sim=sim, gene=g, selfloop=int(g in sl), sign=sl.get(g, "0"), h=h, mem=mem, fano=v.var() / max(v.mean(), 1e-9), zero=(v == 0).mean(), cv=v.std() / max(v.mean(), 1e-9), skew=skew(v), mean=v.mean()))
    return rows
if __name__ == "__main__":
    out = O + "autoreg_gene_features.csv"
    if not os.path.exists(out):
        with Pool(8) as p: G = pd.DataFrame([r for rr in p.map(one, files(), chunksize=1) for r in rr])
        G.to_csv(out, index=False)
    G = pd.read_csv(out); feats = ["h", "mem", "fano", "zero", "cv", "skew", "mean"]
    # within-panel percent rank so that features are comparable across networks
    for f in feats: G[f + "_r"] = G.groupby(["net", "sim"])[f].rank(pct=True).fillna(0.5)     # constant genes (NaN feature) -> neutral rank
    pd.set_option("display.width", 220); res = []
    for tgt, mask in (("any self-loop", None), ("+ self-loop (vs none)", G.sign.isin(["+", "0"])), ("- self-loop (vs none)", G.sign.isin(["-", "0"]))):
        x = G if mask is None else G[mask]; y = x.selfloop if mask is None else (x.sign != "0").astype(int)
        res.append(dict(target=tgt, n=len(x), pos=int(y.sum()), **{f: round(roc_auc_score(y, x[f + "_r"]), 2) for f in feats}))
    print("Gene-level AUROC of each feature (within-panel percent rank) for having a self-loop (0.5 = no information):"); print(pd.DataFrame(res).to_string(index=False))
    # ---- stage 2
    exec(open("motif_pair_zU_networks.py").read().split("d[\"azU\"]")[0]); m = d[d.fam == "mixed"].dropna(subset=["z_rho", "z_C", "z_U"]).copy(); m = m[m.sim <= 2]
    m["edge"] = m.rel.str.startswith("edge").astype(int); m["azr"] = m.z_rho.abs(); m["azC"] = m.z_C.abs(); m["azU"] = m.z_U.abs()
    SL = {(r.net, r.sim, r.gene): r.selfloop for r in G.itertuples()}; RK = {f: {(r.net, r.sim, r.gene): getattr(r, f + "_r") for r in G.itertuples()} for f in feats}
    m["sl"] = [max(SL.get((n, s, int(x)), 0), SL.get((n, s, int(y)), 0)) for n, s, x, y in zip(m.raw, m.sim, m.x, m.y)]
    def pr(f): return np.array([max(RK[f].get((n, s, int(x)), 0.5), RK[f].get((n, s, int(y)), 0.5)) for n, s, x, y in zip(m.raw, m.sim, m.x, m.y)])
    best = max(["h", "mem", "fano", "zero", "cv", "skew", "mean"], key=lambda f: abs(roc_auc_score(G.selfloop, G[f + "_r"]) - 0.5)); print("\nstrongest single feature for any self-loop:", best)
    S = {"|z_rho| (b=0, g=0)": m.azr, "fixed b=0": m.azr + m.azU, "fixed b=0.5": m.azr - 0.5 * m.azC + m.azU, "fixed b=1": m.azr - m.azC + m.azU, "ORACLE b=1 (no self-loop gene) else 0": m.azr - np.where(m.sl == 1, 0, 1) * m.azC + m.azU}
    for f in ["h", "mem", "fano", "zero", "cv", "skew", "mean"]:
        for sgn, lab in ((1, "high"), (-1, "low")):
            r = pr(f) if sgn == 1 else 1 - pr(f); S[f"flag {f} {lab}: b=1 unless pair has a top-half gene else 0"] = m.azr - np.where(r > 0.5, 0, 1) * m.azC + m.azU
    rows = []
    for (net, sim), x in m.groupby(["raw", "sim"]):
        if x.edge.sum() in (0, len(x)): continue
        r = dict(auto="autoreg" if re.search(r"_a[1-9]\d*_pos", net) else "no autoreg", n=re.match(r"grn_n(\d+)", net).group(1))
        for k, v in S.items(): r[k] = average_precision_score(x.edge, v.loc[x.index]) / x.edge.mean()
        rows.append(r)
    R = pd.DataFrame(rows); T = R.groupby(["auto"]).mean(numeric_only=True).T; T["overall"] = R.mean(numeric_only=True); print("\nAUPRC x base (mixed, sims 0-2, mean over networks):"); print(T.round(3).sort_values("overall", ascending=False).to_string())
