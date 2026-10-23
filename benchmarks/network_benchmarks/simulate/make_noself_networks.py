"""Self-loop-free copies of the autoregulated mixed networks (control experiment for the autoregulation analysis).
For every input_data/mixed_network_sweep/grn_*_a<k>_*.txt with k > 0: same matrix with the diagonal set to 0 -> input_data/mixed_network_noself/<same file name>.
Everything else (edges, signs, parameters) is unchanged, so each noself network pairs 1:1 with its original by name.  Never overwrites an existing file."""
import glob, os, re, numpy as np
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"; SRC = ROOT + "input_data/mixed_network_sweep/"; DST = ROOT + "input_data/mixed_network_noself/"
os.makedirs(DST, exist_ok=True); n = 0
for f in sorted(glob.glob(SRC + "grn_*.txt")):
    if not re.search(r"_a[1-9]\d*_pos", os.path.basename(f)): continue
    out = DST + os.path.basename(f)
    if os.path.exists(out): continue
    M = np.loadtxt(f, delimiter=",", dtype=int); n_self = int(np.count_nonzero(np.diag(M))); M2 = M.copy(); np.fill_diagonal(M2, 0)
    assert np.count_nonzero(M2) == np.count_nonzero(M) - n_self
    np.savetxt(out, M2, fmt="%d", delimiter=","); n += 1
print(n, "networks written to", DST, "| total", len(glob.glob(DST + "*.txt")))
