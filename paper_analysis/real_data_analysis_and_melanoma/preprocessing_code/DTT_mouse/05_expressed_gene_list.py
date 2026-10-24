"""Step 5: genes detected (count>=1) in >=5% of the cells used for the same-type sister analysis, keyed by Ensembl ID.
Cells: both members of every same-type strict sister pair (twins) + the random pool cells (singletons) from inputs/wanted_cells.csv.
Gene universe: authors' df_gene.rds order (matrix_column_to_ensembl.csv); flag in_GEO_df_gene kept.
Outputs: <DTT_OUT>/expressed_genes_5pct.csv ; <DTT_OUT>/counts_samecell_analysis.npz (+ cells csv)"""
import glob, os, sys, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import DTT_GEO, DTT_REPO, FINAL, INPUT, REBUILD      # noqa: E402
DTT_OUT = os.environ.get("DTT_OUT", os.path.join(REBUILD, "DTT_mouse", "F"))      # rebuilt files; final_data is never overwritten
F = DTT_OUT + '/'
st = pd.read_csv(F + 'sister_pairs_strict.csv'); st = st[st.same_type]
w = pd.read_csv(INPUT + '/wanted_cells.csv'); single = w[~w.in_pair].cell_id.tolist()
twins = pd.concat([st.cell_A, st.cell_B]).tolist(); print(len(st), 'same-type pairs;', len(twins), 'twin cells;', len(single), 'singleton cells')
need = {c: i for i, c in enumerate(twins + single)}; order = twins + single; nt = len(twins)
parts, got = [], []
for f in sorted(glob.glob(F + 'counts_wanted/*.npz')):
    m = sp.load_npz(f).tocsr(); cid = open(f.replace('.npz', '_cells.txt')).read().split()
    sel = [i for i, c in enumerate(cid) if c in need]; parts.append(m[sel]); got += [cid[i] for i in sel]; del m
X = sp.vstack(parts, format='csr'); del parts
X = X[[got.index(c) if False else i for i, c in sorted(((i, c) for i, c in enumerate(got)), key=lambda z: need[z[1]])]]
got = sorted(got, key=lambda c: need[c]); assert got == order, (len(got), len(order))
print(X.shape, 'cells x genes', X.nnz, 'nnz')
tot = np.asarray(X.sum(1)).ravel(); det = np.bincount(X.indices, minlength=X.shape[1]) / X.shape[0]
g = pd.read_csv(FINAL + '/DTT_mouse_data/matrix_column_to_ensembl.csv')      # column -> gene table of the authors' df_gene (input, not rebuilt); g['frac_cells_detected'] = det; g['mean_cp10k'] = np.bincount(X.indices, weights=(X.data / np.repeat(tot, np.diff(X.indptr))) * 1e4, minlength=X.shape[1]) / X.shape[0]
e = g[(g.frac_cells_detected >= 0.05)].sort_values('frac_cells_detected', ascending=False)
e.to_csv(F + 'expressed_genes_5pct.csv', index=False)
print('genes detected in >=5% of cells:', len(e), '| in GEO df_gene:', int(e.in_GEO_df_gene.sum())); print(e.gene_type.value_counts().to_dict())
sp.save_npz(F + 'counts_samecell_analysis.npz', X); pd.DataFrame({'cell_id': order, 'role': ['twin'] * nt + ['singleton'] * len(single), 'total_counts': tot}).to_csv(F + 'counts_samecell_analysis_cells.csv', index=False)
