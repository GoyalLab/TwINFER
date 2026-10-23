"""Out-of-sample symmetry check for the regulated-mutual motif (figure 4): z_U of pair 1-3 minus pair 1-2 at 6000 twin pairs on replicates 100-399 (not used in the figure, which uses 0-99). Exact symmetry => expected difference 0."""
import numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats import wilcoxon
import twin_size_scan_fig3_fig4 as S
def one(a):
    m, r, f = a; d = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate", "gene_1_mRNA", "gene_2_mRNA", "gene_3_mRNA"]); d = d[d.time_step == 20]
    X = {g: d.pivot(index="clone_id", columns="replicate", values=f"gene_{g}_mRNA").values for g in (1, 2, 3)}
    return r, S.twin_stats(X[1], X[2])["z_U"], S.twin_stats(X[1], X[3])["z_U"]
if __name__ == "__main__":
    F = S.files4(); reps = [r for r in sorted({r for (m, r) in F if m == "Mutual_regulation"}) if 100 <= r < 400]; print(len(reps), "replicates", flush=True)
    with mp.Pool(10) as p: res = p.map(one, [("Mutual_regulation", r, F[("Mutual_regulation", r)]) for r in reps], chunksize=1)
    d = pd.DataFrame(res, columns=["rep", "z12", "z13"]); x = d.z13 - d.z12
    print(f"n={len(d)} mean z12={d.z12.mean():.2f} z13={d.z13.mean():.2f}; diff mean={x.mean():.2f} (SE {x.std()/np.sqrt(len(x)):.2f}); frac 1-3>1-2={np.mean(x>0):.2f}; Wilcoxon p={wilcoxon(x).pvalue:.3f}")
