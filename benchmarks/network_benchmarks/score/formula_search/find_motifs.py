"""Find Fan-out and Feed-forward triads embedded in a larger connectivity matrix, matching the
figure_4 canonical patterns as EXACT induced subgraphs on the 3 genes (ignoring their other
connections to genes outside the triad):

  Fan_out(A;B,C):      A->B, A->C exist; B->A, C->A, B->C, C->B all absent.
  Feed_forward(A,B,C): A->B, A->C, B->C exist; B->A, C->A, C->B all absent.

Returns physical gene-index triples with roles labeled, for every permutation of 3 genes that
matches (checking all orderings since the roles aren't given).
"""
import numpy as np


def find_motifs(M):
    n = M.shape[0]
    B = (M != 0).astype(int)
    fan_out, feed_forward = [], []
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                # Fan_out: a->b, a->c ; b->a,c->a,b->c,c->b absent.  (b<c to avoid double count of the pair)
                if b < c and B[a, b] and B[a, c] and not B[b, a] and not B[c, a] and not B[b, c] and not B[c, b]:
                    fan_out.append((a, b, c))
                # Feed_forward: a->b, a->c, b->c ; b->a,c->a,c->b absent.
                if B[a, b] and B[a, c] and B[b, c] and not B[b, a] and not B[c, a] and not B[c, b]:
                    feed_forward.append((a, b, c))
    return fan_out, feed_forward
