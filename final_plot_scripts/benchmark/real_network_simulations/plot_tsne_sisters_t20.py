"""Sanity figure for the all-cells t=20 real-network benchmark: per network (replicate = first run), 12,000 cells.
Row 1: t-SNE (log1p counts, standardized, PCA<=20) coloured by KMeans cluster (k by silhouette, 2-6).
Row 2: same t-SNE, 150 random clones with a line joining the two sisters.
Row 3: sister distance vs random-pair distance (Euclidean over all PCs, up to 50), histogram.
Prints median sister/random distance ratio and fraction of sisters in the same cluster. Illustrator-editable PDF (guide recipe).
"""
import os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
D = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts"
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]: fm.fontManager.addfont(os.path.join(D, f))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif", "svg.fonttype": "none",
                     "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
IN = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/Beeline/inputs/real_networks_t20_all"
OUT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/real_network_simulations"
NETS = ["B_cell_activation", "EMT", "GSD", "HSC", "Pluripotent", "VSC", "mCAD"]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(3, 7, figsize=(30, 12))
for c, net in enumerate(NETS):
    run = sorted(glob.glob(f"{IN}/{net}/simrep*_all_t20"))[0]
    X = np.log1p(pd.read_csv(f"{run}/ExpressionData.csv", index_col=0).to_numpy(float).T)   # 12000 x genes, sisters = adjacent rows
    X = X[:, X.std(0) > 0]; X = (X - X.mean(0)) / X.std(0)
    P = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X); P20 = P[:, :20]
    Z = TSNE(n_components=2, perplexity=30, random_state=0, n_jobs=int(os.environ.get("OMP_NUM_THREADS", 8)), init="pca").fit_transform(P20)
    best = max(range(2, 7), key=lambda k: silhouette_score(P20[::4], KMeans(k, n_init=3, random_state=0).fit_predict(P20[::4])))
    lab = KMeans(best, n_init=5, random_state=0).fit_predict(P20)
    a, b = np.arange(0, len(P), 2), np.arange(1, len(P), 2)
    ds = np.linalg.norm(P[a] - P[b], axis=1); dr = np.linalg.norm(P[a] - P[rng.permutation(b)], axis=1)
    print(f"{net:18s} genes={X.shape[1]:2d} k={best} median sister/random dist = {np.median(ds)/np.median(dr):.2f}  sisters same cluster = {np.mean(lab[a]==lab[b]):.2f}", flush=True)
    ax[0, c].scatter(Z[:, 0], Z[:, 1], c=[PAL[l] for l in lab], s=1.5, linewidths=0); ax[0, c].set_title(net.replace("_", " "))
    ax[1, c].scatter(Z[:, 0], Z[:, 1], c="#d0d0d0", s=1.5, linewidths=0)
    for i in rng.choice(len(a), 150, replace=False): ax[1, c].plot(Z[[a[i], b[i]], 0], Z[[a[i], b[i]], 1], color="#eb6834", lw=0.8)
    h = np.linspace(0, np.percentile(dr, 99.5), 50)
    ax[2, c].hist(dr, bins=h, color="#d0d0d0", label="random pairs"); ax[2, c].hist(ds, bins=h, histtype="step", color="#2a78d6", lw=2, label="sisters")
    for r in (0, 1): ax[r, c].set_xlabel("t-SNE 1"); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, c].set_ylabel("t-SNE 2" if c == 0 else ""); ax[1, c].set_ylabel("t-SNE 2" if c == 0 else "")
    ax[2, c].set_xlabel("distance (PCs)"); ax[2, c].set_ylabel("cells" if c == 0 else "")
    if c == 0: ax[2, c].legend(frameon=False)
for x in ax.ravel():
    x.set_facecolor("none"); [s.set_visible(False) for k, s in x.spines.items() if k in ("top", "right")]
plt.tight_layout()
for art in fig.findobj():
    if hasattr(art, "set_clip_on"): art.set_clip_on(False)
f = f"{OUT}/sanity_tsne_sister_t20_real_networks.pdf"
fig.savefig(f, format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
fig.savefig("/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/68faf95e-4522-47c6-8548-955d44c09184/scratchpad/sanity_tsne.png", dpi=55, bbox_inches="tight", facecolor="white")
print("wrote", f)
