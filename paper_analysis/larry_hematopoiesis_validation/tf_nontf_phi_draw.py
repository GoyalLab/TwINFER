"""One independent draw of the TF-vs-detection-matched-non-TF phi comparison. SEED env var picks
the draw. Writes tf_nontf_phi_draw_{SEED}.csv (gene, is_tf, detection, phi -- point estimate only,
no bootstrap reliability weight, since only the per-gene phi values are needed to pool across many
independent draws into one well-powered TF-vs-non-TF test)."""
import os
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(__file__))
from paper_analysis.larry_hematopoiesis_validation import apply_twinscore_supplement_larry as m
import pandas as pd
import numpy as np
import scipy.io as sio

SEED = int(os.environ.get("SEED", "0"))

ct = pd.read_csv("resources/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
collectri_tfs = set(ct.source_genesymbol.unique())
animaltfdb = set(x.strip() for x in open("resources/mouse_TF_AnimalTFDB3.txt"))
all_tf_like = collectri_tfs | animaltfdb

X = sio.mmread(f"{m.SOURCE}/larry_qc_counts.mtx").tocsr()
genes_full = np.array(open(f"{m.SOURCE}/genes.txt").read().split())
obs = pd.read_csv(f"{m.SOURCE}/obs_metadata.csv", index_col=0)
day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int).to_numpy()
days = sorted(set(day))
per_day_frac = np.stack([np.asarray((X[day == d] > 0).sum(axis=0)).ravel() / (day == d).sum() for d in days])
frac_expr = per_day_frac.min(axis=0)
det = pd.Series(frac_expr, index=genes_full)

MIN_DET = 0.05
tf_pool = det[(det.index.isin(all_tf_like)) & (det >= MIN_DET)]
nontf_pool = det[(~det.index.isin(all_tf_like)) & (det >= MIN_DET)]

rng = np.random.default_rng(SEED)
tf_genes = sorted(rng.choice(tf_pool.index, size=100, replace=False).tolist())
nontf_remaining = nontf_pool.copy()
matched_nontf = []
for g in tf_genes:
    target = tf_pool[g]
    idx = (nontf_remaining - target).abs().idxmin()
    matched_nontf.append(idx)
    nontf_remaining = nontf_remaining.drop(idx)
matched_nontf = sorted(matched_nontf)
genes = sorted(set(tf_genes) | set(matched_nontf))

gidx = {g: i for i, g in enumerate(genes_full)}
col = [gidx[g] for g in genes]
cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
mat = np.log1p(X[:, col].toarray().astype(float) / cell_total[:, None] * 1e4)
df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
df.insert(0, "time_step", day)
df.insert(0, "cell_id", obs.index.to_numpy())
df.insert(0, "clone_id", obs["larry_clone_singletcode"].to_numpy())
t1_raw = df[df.time_step == 2].reset_index(drop=True)
t2_raw = df[df.time_step == 4].reset_index(drop=True)

C1, meff_t1 = m.sister_matrix_from_frame(t1_raw, genes)
C2, meff_t2 = m.sister_matrix_from_frame(t2_raw, genes)
h1 = {g: C1.loc[g, g] for g in genes}
h2 = {g: C2.loc[g, g] for g in genes}
rho_dag, meff_cross = m.cross_matrix(t1_raw, t2_raw, genes)
rho_dag_diag = {g: rho_dag.loc[g, g] for g in genes}

var_h1, var_h2 = m.null_sd(meff_t1) ** 2, m.null_sd(meff_t2) ** 2
thr1, thr2 = 2.0 * np.sqrt(var_h1), 2.0 * np.sqrt(var_h2)
phi_raw = {}
for g in genes:
    h1g, h2g, rdg = h1[g], h2[g], rho_dag_diag[g]
    ok = np.isfinite(h1g) and np.isfinite(h2g) and h1g > thr1 and h2g > thr2
    denom = np.sqrt(h1g * h2g) if ok and h1g > 0 and h2g > 0 else np.nan
    phi_raw[g] = rdg / denom if ok and np.isfinite(denom) and denom > 0 else np.nan

out = pd.DataFrame({"gene": genes, "seed": SEED, "is_tf": [g in set(tf_genes) for g in genes],
                     "detection": [det[g] for g in genes], "phi": [phi_raw[g] for g in genes]})
out.to_csv(f"tf_nontf_phi_draw_{SEED}.csv", index=False)
print(f"seed={SEED} done: n_tf_defined={out[out.is_tf].phi.notna().sum()} "
      f"n_nontf_defined={out[~out.is_tf].phi.notna().sum()}", flush=True)
