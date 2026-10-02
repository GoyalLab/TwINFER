# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""For each real network, find (Hill n, k_add scale) that maximises the number of
WELL-SEPARATED stable mean-field fixed points, and report basin fractions from
random inits + from the empty (sim) start."""
import sys, numpy as np
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code")
from simulations.network_analysis.predict_multistability import MeanField, load_matrix, load_params, NETWORKS

P = load_params()
NETS = ["VSC", "mCAD", "GSD", "HSC_balanced", "EMT", "B_cell_activation", "Pluripotent"]
N_VALS = [2.0, 3.0, 4.0]
S_VALS = [1.0, 2.0, 3.0]

def basins(mf, fps, n_rand=300):
    """fraction of random log-uniform inits (and the empty start) landing in each stable FP"""
    stab = [f["p"] for f in fps if f["stable"]]
    if len(stab) < 2:
        return None
    c = mf.c
    rng = np.random.default_rng(0)
    starts = [10**rng.uniform(np.log10(0.1), np.log10(c), mf.n) for _ in range(n_rand)]
    counts = np.zeros(len(stab))
    for s in starts:
        q = np.clip(s, 0, None)
        for _ in range(1500):
            q2 = 0.6*q + 0.4*mf.G(q)
            if np.max(np.abs(q2-q)) < 1e-7*(1+np.max(q2)): break
            q = q2
        counts[int(np.argmin([np.linalg.norm(np.log1p(q)-np.log1p(t)) for t in stab]))] += 1
    # empty start
    q = np.zeros(mf.n)
    for _ in range(3000):
        q2 = 0.6*q + 0.4*mf.G(q); q = q2
    empty_fp = int(np.argmin([np.linalg.norm(np.log1p(q)-np.log1p(t)) for t in stab]))
    return counts/counts.sum(), empty_fp

for net in NETS:
    mfile = NETWORKS[net][0]
    M = load_matrix(mfile)
    print(f"\n===== {net} ({M.shape[0]} genes) =====")
    best = None
    for nv in N_VALS:
        row = []
        for sv in S_VALS:
            mf = MeanField(M, P, kadd_scale=sv, n_hill=nv)
            fps = mf.fixed_points(n_starts=150)
            ns = sum(f["stable"] for f in fps)
            b = basins(mf, fps) if ns >= 2 else None
            minfrac = b[0].min() if b else 0.0
            row.append(f"{ns}({minfrac:.2f})")
            score = (ns, minfrac)
            if b and (best is None or score > best[0]):
                best = (score, nv, sv, b[0], b[1])
        print(f"  n={nv}: " + "  ".join(f"s{S_VALS[i]}={row[i]}" for i in range(len(S_VALS))))
    if best:
        (ns, mf_), nv, sv, frac, empty = best
        print(f"  -> BEST: n={nv}, k_add x{sv}  |  {ns} stable FPs, basin fractions {np.round(frac,2).tolist()}, "
              f"empty-start -> FP{empty}")
    else:
        print("  -> no (n<=4, scale<=3) point gives >=2 balanced stable FPs")
