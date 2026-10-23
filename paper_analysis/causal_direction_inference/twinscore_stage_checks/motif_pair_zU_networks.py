"""z_U / z_C / z_rho by motif relation of the gene pair, inside the network_sweep_final and mixed_network_sweep networks (pair z-scores already computed, Figure-2 design, B = 20 h).
Relation from the ground-truth topology (off-diagonal edges, self-loops ignored), exclusive, in this priority:
 edges:     mutual (a<->b) | FF bypass edge (a->b and a->k->b exists) | edge with shared parent (p->a and p->b, plus the edge a--b) | plain direct edge
 non-edges: fan-out pair (common parent, no edge between) | cascade (path >=2 either way, no common parent) | fan-in pair (common child only) | unrelated
Output: <O>/motif_pair_zU_by_class.csv"""
import numpy as np, pandas as pd
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; Z = A + "analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/"; B = A + "code/Beeline/inputs/"
O = A + "analysis_data/paper_analysis/causal_direction_inference/zrho_pdf_network_auprc/"
sets = {"grn_n6_e5_pos100_density": ("network_sweep_final_e5_pairs.csv", "network_sweep_final", "grn_n6_e5_pos100_density_{r}"),
        "grn_n6_e9_pos0_sign_ratio": ("network_sweep_final_e9_pos0_sign_ratio_pairs.csv", "network_sweep_final", "grn_n6_e9_pos0_sign_ratio_{r}"),
        "grn_n6_e9_pos50_sign_ratio": ("network_sweep_final_e9_pos50_sign_ratio_pairs.csv", "network_sweep_final", "grn_n6_e9_pos50_sign_ratio_{r}"),
        "grn_n6_e9_pos100_center": ("network_sweep_final_e9_pos100_center_pairs.csv", "network_sweep_final", "grn_n6_e9_pos100_center_{r}"),
        "n6_e13_pos100": ("network_sweep_final_e13_pos100_pairs_both_designs.csv", "e13_pos100", "n6_e13_pos100_{r}"),
        "grn_n6_e17_pos100_density": ("network_sweep_final_e17_pos100_density_pairs.csv", "network_sweep_final", "grn_n6_e17_pos100_density_{r}")}
fr = []
for g, (f, fo, pat) in sets.items():
    d = pd.read_csv(Z + f); d["raw"] = d.network.map(lambda r: pat.format(r=r)); d["gtdir"] = B + fo + "/"; d["fam"] = "sweep"; fr.append(d)
m = pd.read_csv(A + "analysis_data/mixed_network_sweep/twinscore_pairs_SCU.csv"); m["raw"] = m.net; m["gtdir"] = B + "mixed_network_sweep/"; m["fam"] = "mixed"; fr.append(m)
d = pd.concat(fr, ignore_index=True).dropna(subset=["z_U", "z_C", "z_rho"]); cache = {}   # [2026-10-16 was subset incl. "U": silently dropped the e13_pos100 file, which has no U column]
if "U" not in d: d["U"] = np.nan
def topo(raw, gtdir):
    if raw not in cache:
        g = pd.read_csv(gtdir + raw + "/GroundTruthNetwork.csv"); g = g[g.Gene1 != g.Gene2]; n = max(max(int(a.split("_")[1]), int(b.split("_")[1])) for a, b in zip(g.Gene1, g.Gene2))
        M = np.zeros((n + 1, n + 1), int)
        for a, b in zip(g.Gene1, g.Gene2): M[int(a.split("_")[1]), int(b.split("_")[1])] = 1
        R = M.copy()
        for _ in range(n): R = ((R + R @ M) > 0).astype(int)
        cache[raw] = (M, R)
    return cache[raw]
def rel(raw, gtdir, a, b):
    M, R = topo(raw, gtdir); n = M.shape[0]
    f, r = M[a, b], M[b, a]
    if f and r: return "edge: mutual"
    if f or r:
        s, t = (a, b) if f else (b, a)
        if any(M[s, k] and M[k, t] for k in range(n) if k not in (s, t)): return "edge: feed-forward bypass (s->k->t also)"
        if any(M[p, s] and M[p, t] for p in range(n) if p not in (s, t)): return "edge: shares a parent (fan-out + edge)"
        return "edge: plain direct"
    if any(M[p, a] and M[p, b] for p in range(n) if p not in (a, b)): return "non-edge: fan-out pair (common parent)"
    if R[a, b] or R[b, a]: return "non-edge: cascade (indirect path)"
    if any(M[a, c] and M[b, c] for c in range(n) if c not in (a, b)): return "non-edge: fan-in pair (common child)"
    return "non-edge: unrelated"
d["rel"] = [rel(r, g, int(x), int(y)) for r, g, x, y in zip(d.raw, d.gtdir, d.x, d.y)]
d["azU"] = d.z_U.abs(); d["azC"] = d.z_C.abs(); d["azr"] = d.z_rho.abs()
def tab(x):
    return x.groupby("rel").agg(n=("z_U", "size"), med_abs_zU=("azU", "median"), pct_zU_sig=("azU", lambda v: 100 * (v > 1.96).mean()), med_abs_zC=("azC", "median"),
                                pct_zC_sig=("azC", lambda v: 100 * (v > 1.96).mean()), med_abs_zrho=("azr", "median"), frac_U_neg=("U", lambda v: (v.dropna() < 0).mean())).round(2)
T = tab(d); T.to_csv(O + "motif_pair_zU_by_class.csv"); pd.set_option("display.width", 250)
print("ALL"); print(T.to_string()); print("\nsweep"); print(tab(d[d.fam == "sweep"]).to_string()); print("\nmixed"); print(tab(d[d.fam == "mixed"]).to_string())
