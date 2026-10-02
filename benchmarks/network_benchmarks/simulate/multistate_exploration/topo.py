from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, networkx as nx, itertools

MDIR = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks'
NETS = {"Circadian":"circadian.txt","mCAD":"mCAD.txt","VSC":"VSC.txt",
        "B_cell":"B_cell.txt","HSC":"HSC.txt","EMT":"EMT.txt","GSD":"GSD.txt",
        "Pluripotent":"Pluripotent.txt"}

for name,f in NETS.items():
    M = np.loadtxt(f"{MDIR}/{f}", delimiter=",", dtype=float)
    n = M.shape[0]
    S = np.sign(M)
    edges = [(i,j,int(S[i,j])) for i in range(n) for j in range(n) if S[i,j]!=0]
    nE = len(edges)
    nPos = sum(1 for *_,s in edges if s>0); nNeg = nE-nPos
    indeg = (S!=0).sum(0)
    selfl = [(i,int(S[i,i])) for i in range(n) if S[i,i]!=0]
    self_pos = sum(1 for _,s in selfl if s>0)
    # 2-cycles
    two = []
    for i in range(n):
        for j in range(i+1,n):
            if S[i,j]!=0 and S[j,i]!=0:
                two.append((int(S[i,j]),int(S[j,i])))
    toggles = sum(1 for a,b in two if a<0 and b<0)
    mutact  = sum(1 for a,b in two if a>0 and b>0)
    mixed   = sum(1 for a,b in two if a*b<0)
    # feedback loops up to len 5, sign
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i,j,s in edges: G.add_edge(i,j,s=s)
    posL=negL=0; short_pos=0
    for cyc in itertools.islice(nx.simple_cycles(G), 200000):
        if len(cyc)>5: continue
        sgn=1
        for a,b in zip(cyc,cyc[1:]+cyc[:1]): sgn*=G[a][b]["s"]
        if sgn>0:
            posL+=1
            if len(cyc)<=3: short_pos+=1
        else: negL+=1
    sccs = sorted((len(c) for c in nx.strongly_connected_components(G)), reverse=True)
    in_cycle = sum(s for s in sccs if s>1)
    print(f"{name:11s} n={n:2d} E={nE:3d}  act%={100*nPos/nE:3.0f} rep%={100*nNeg/nE:3.0f}  "
          f"indeg mean={indeg.mean():.1f} max={indeg.max():2d}  "
          f"self+={self_pos}/{len(selfl)}  toggles={toggles} mutact={mutact} mixed2={mixed}  "
          f"posLoops={posL:4d}(short{short_pos}) negLoops={negL:4d}  "
          f"SCC sizes={sccs[:4]}  genes_in_cycle={in_cycle}/{n}")
