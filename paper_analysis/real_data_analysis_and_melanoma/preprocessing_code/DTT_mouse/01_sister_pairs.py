"""Step 1: strict sister pairs in the DTT full placed tree (+ funnel counts, cousins, tape ties).

Sister pair = internal node with exactly two children, both sequenced cells (leaves).
Funnel: all pairs -> parent dated >= E12.5 -> both cells also in the 655,701-tip dated backbone.
Outputs (<DTT_OUT> = <REBUILD>/DTT_mouse/F): sister_pairs_strict.csv, cousin_cells.csv
"""
import sys, os, gzip, csv, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import DTT_GEO, DTT_REPO, FINAL, INPUT, REBUILD      # noqa: E402
DTT_OUT = os.environ.get("DTT_OUT", os.path.join(REBUILD, "DTT_mouse", "F"))      # rebuilt files; final_data is never overwritten
REPO = DTT_REPO                                   # the authors' dtt-mouse repository (raw_data link): dated tape tree, support data
sys.path.insert(0, REPO + '/tools'); from tree_io import load_tree
OUT = DTT_OUT
os.makedirs(OUT, exist_ok=True)
S = REPO + '/support_data/'
log = lambda *a: print(*a, flush=True)

z = load_tree(S + 'merged_full_placed.nwk')
par, tm, isleaf, names = z['parent'], z['time'], z['is_leaf'], z['names']
zb = load_tree(S + 'merged_minB2h_lineage_constrained.nwk')
bb = set(zb['names'][zb['is_leaf']]); log('backbone tips', len(bb), 'full tips', int(isleaf.sum()), 'tip time max', tm[isleaf].max())

meta = pd.read_csv(S + 'cell_metadata.v8.txt.gz', sep='\t', usecols=['cell_id', 'celltype']).set_index('cell_id').celltype
nchild = np.bincount(par[par >= 0], minlength=len(par))
nleafch = np.bincount(par[(par >= 0) & isleaf], minlength=len(par))
cher = np.flatnonzero((nchild == 2) & (nleafch == 2))
log('cherries (strict sister pairs):', len(cher))
leaves = np.flatnonzero(isleaf & (par >= 0)); o = np.argsort(par[leaves], kind='stable'); L = leaves[o]; P = par[L]
first = np.searchsorted(P, cher)
a, b = L[first], L[first + 1]
df = pd.DataFrame({'node': cher, 'age': tm[cher], 'cell_A': names[a], 'cell_B': names[b]})
df['celltype_A'] = df.cell_A.map(meta); df['celltype_B'] = df.cell_B.map(meta)
df['in_backbone'] = df.cell_A.isin(bb) & df.cell_B.isin(bb)
log('total', len(df), '| age>=12.5', int((df.age >= 12.5).sum()), '| and both in backbone', int(((df.age >= 12.5) & df.in_backbone).sum()))
st = df[(df.age >= 12.5) & df.in_backbone].copy()
st['same_type'] = st.celltype_A == st.celltype_B
log('strict:', len(st), 'same type', int(st.same_type.sum()), 'diff', int((~st.same_type).sum()), 'types', len(set(st.celltype_A) | set(st.celltype_B)))

# edit-distance evidence (differing called sites) for Figure 1
g = np.load(S + 'newcode_v8_all.npz', allow_pickle=True); code = g['code']; grow = {c: i for i, c in enumerate(g['cell_ids'])}
ga = st.cell_A.map(grow).values; gb = st.cell_B.map(grow).values
ok = ~(pd.isna(ga) | pd.isna(gb)); disc = np.full(len(st), -1)
A = code[ga[ok].astype(int)].reshape(ok.sum(), -1); B = code[gb[ok].astype(int)].reshape(ok.sum(), -1)
disc[ok] = ((A >= 0) & (B >= 0) & (A != B)).sum(1); st['n_discordant_sites'] = disc
# also for every cherry dated >= 12.5 (Figure 1 uses all dated pairs)
d2 = df.copy(); ga = d2.cell_A.map(grow); gb = d2.cell_B.map(grow); ok = ~(ga.isna() | gb.isna()); dd = np.full(len(d2), -1)
A = code[ga[ok].astype(int).values].reshape(ok.sum(), -1); B = code[gb[ok].astype(int).values].reshape(ok.sum(), -1)
dd[ok.values] = ((A >= 0) & (B >= 0) & (A != B)).sum(1); d2['n_discordant_sites'] = dd
d2[['node', 'age', 'in_backbone', 'n_discordant_sites']].to_csv(OUT + '/all_cherries_age_edits.csv', index=False)

# tied third cell: some other cell with identical genotype on commonly called sites to both sisters (tape-identical to the pair)
# (computed in 03_robustness.py; here only cousins)
# cousins: leaf children of uncle internal nodes (other internal children of the grandparent)
children = {}
for c in np.flatnonzero(par >= 0): children.setdefault(par[c], []).append(c)
rows = []
for n_, gp in zip(st.node.values, par[st.node.values]):
    for u in children.get(gp, []):
        if u == n_ or isleaf[u]: continue
        for c in children[u]:
            if isleaf[c]: rows.append((n_, names[c]))
cz = pd.DataFrame(rows, columns=['node', 'cousin_cell']); cz['celltype'] = cz.cousin_cell.map(meta)
st.to_csv(OUT + '/sister_pairs_strict.csv', index=False); cz.to_csv(OUT + '/cousin_cells.csv', index=False)
log('pairs with >=1 cousin cell (any):', cz.node.nunique(), 'cousin cells', cz.cousin_cell.nunique())
log('age>=13.0 share', float((st.age >= 13).mean()))
