"""Design X (1,500 twins + 3,000 singletons) built on the SAME clone permutation as the twin-only designs, so that designs share cells:
   q = permutation of the 6,000 clones (same seed as the twin-only designs);  twin-only N=1500 -> q[:1500];  N=3000 -> q[:3000];  X: twins = q[:1500] (identical to N=1500), singletons = one random sister of each clone in q[3000:].
analyse() draws its own partition internally; FixedRng makes it return p = [q[1500:3000], q[:1500], q[3000:]] so that its B twins (perm[1500:3000]) = q[:1500] and its singletons (perm[3000:]) = q[3000:].
Only the 20 h B-sample outputs (z_rho, z_C, z_U, S, C, U) are used; cross-sample outputs are ignored (P1 = P2)."""
import numpy as np
from figure2_1000_zscores import analyse, NCL
class FixedRng:
    def __init__(self, perm, pick, seed): self.perm, self.pick, self.rs = perm, pick, np.random.RandomState(seed)
    def permutation(self, n): return self.perm.copy() if n == len(self.perm) else self.rs.permutation(n)
    def randint(self, lo, hi, size=None): return self.pick
def x_stats(g1, g2, q, pick):
    P = {"gene_1": g1, "gene_2": g2}; p = np.concatenate([q[1500:3000], q[:1500], q[3000:]]); o = analyse(P, P, FixedRng(p, pick, 7), 2)
    return dict(z_rho=o["z_rho"], z_C=o["z_C"], z_U=o["z_U"], S=o["S"], C=o["C"], U=o["U"], rho=o["rho"])
