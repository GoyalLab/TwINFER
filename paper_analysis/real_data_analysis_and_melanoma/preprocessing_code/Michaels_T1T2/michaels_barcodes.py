"""Lineage barcodes from the Michaels 2024 TLB amplicon reads, by the paper's own rule.

STAR Methods, "Extracting lineage barcode reads from FASTQ files" and "Clone construction and
processing". Nothing here is invented: the construct, the search pattern, every filter and every
threshold are the ones the paper states.

  construct   a 16-mer randomised barcode upstream of the poly(A) in an EGFP transcript, written
              AT NNNN CT NNNN AC NNNN TC NNNN GT -- four 4-mers separated by fixed dinucleotides
  read 1      10x v3.1: 16 bp cell barcode, then a 12 bp UMI
  read 2      carries the construct; the pattern is searched for and the four N-blocks concatenated

  filters, in the paper's order
    1. same UMI, barcodes within Levenshtein 1  -> keep the one with the most reads
    2. same UMI, barcodes further apart         -> discard both
    3. a barcode seen once that is 1 nt from a barcode seen more than once -> merged into it;
       a barcode seen once and nothing else     -> discarded
    4. the cell barcode must be in the filtered scRNA-seq matrix
    5. a barcode seen with only one cell        -> discarded
    6. a cell-barcode/lineage-barcode pair needs >= 2 counts, on different UMIs
    7. within a cell, barcodes under 50% of that cell's top barcode count -> discarded
"""
import os,sys,re,gzip,glob,json,collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import MICHAELS_GEO,MICHAELS_TLB_FASTQ,REBUILD      # noqa: E402
IN=MICHAELS_TLB_FASTQ; OUT=os.path.join(REBUILD,"Michaels_T1T2","clones")      # TLB amplicon FASTQs (raw_data link); the clone table is written to <REBUILD>, not into yscher's folder
os.makedirs(OUT,exist_ok=True)
PAT=re.compile(r"AT([ACGT]{4})CT([ACGT]{4})AC([ACGT]{4})TC([ACGT]{4})GT")
CB_LEN,UMI_LEN=16,12
def cell_whitelist():
    wl={}
    for s in ("T1","T2"):
        f=glob.glob(f"{MICHAELS_GEO}/*_%s_barcodes.tsv.gz"%s)
        if not f: continue
        with gzip.open(f[0],"rt") as fh:
            for line in fh: wl[line.strip().split("-")[0]]=s
    return wl
WL=cell_whitelist()
print("cell barcodes in the two 10x matrices: %d"%len(WL),flush=True)
counts=collections.Counter()      # (cb, umi, lb) -> reads
nread=nhit=0
for r1 in sorted(glob.glob(os.path.join(IN,"*_1.fastq.gz"))):
    r2=r1.replace("_1.fastq.gz","_2.fastq.gz")
    with gzip.open(r1,"rt") as f1, gzip.open(r2,"rt") as f2:
        while True:
            h1=f1.readline()
            if not h1: break
            s1=f1.readline().strip(); f1.readline(); f1.readline()
            f2.readline(); s2=f2.readline().strip(); f2.readline(); f2.readline()
            nread+=1
            if len(s1)<CB_LEN+UMI_LEN: continue
            m=PAT.search(s2)
            if not m: continue
            cb=s1[:CB_LEN]; umi=s1[CB_LEN:CB_LEN+UMI_LEN]
            if "N" in cb or "N" in umi: continue
            counts[(cb,umi,"".join(m.groups()))]+=1; nhit+=1
    print("  %s: %d reads, %d with the construct"%(os.path.basename(r1),nread,nhit),flush=True)
print("reads: %d; carrying the construct: %d (%.1f%%)"%(nread,nhit,100*nhit/max(nread,1)),flush=True)
def lev1(a,b):
    if a==b: return True
    if abs(len(a)-len(b))>1: return False
    if len(a)==len(b): return sum(x!=y for x,y in zip(a,b))<=1
    if len(a)>len(b): a,b=b,a
    for i in range(len(b)):
        if a==b[:i]+b[i+1:]: return True
    return False
# 1,2: resolve barcodes sharing a UMI within a cell
byumi=collections.defaultdict(list)
for (cb,umi,lb),n in counts.items(): byumi[(cb,umi)].append((lb,n))
kept={}
dropped_umi=0
for k,v in byumi.items():
    if len(v)==1: kept[k]=v[0]; continue
    v.sort(key=lambda t:-t[1])
    if all(lev1(v[0][0],lb) for lb,_ in v[1:]): kept[k]=(v[0][0],sum(n for _,n in v))
    else: dropped_umi+=1
print("UMIs: %d; dropped for incompatible barcodes: %d"%(len(byumi),dropped_umi),flush=True)
# 3: consensus barcodes
lbcount=collections.Counter()
for (cb,umi),(lb,n) in kept.items(): lbcount[lb]+=1
consensus=[lb for lb,c in lbcount.items() if c>1]
cset=set(consensus)
byfirst=collections.defaultdict(list)
for lb in consensus: byfirst[lb[:6]].append(lb)
def rescue(lb):
    for i in range(0,len(lb),4):
        for c in "ACGT":
            alt=lb[:i]+c+lb[i+1:]
            if alt in cset: return alt
    for cand in consensus:
        if lev1(lb,cand): return cand
    return None
final=collections.Counter()   # (cb, lb) -> UMIs
for (cb,umi),(lb,n) in kept.items():
    if lb not in cset:
        lb2=rescue(lb)
        if lb2 is None: continue
        lb=lb2
    final[(cb,lb)]+=1
print("distinct barcodes: %d; consensus (seen >1): %d"%(len(lbcount),len(consensus)),flush=True)
# 4: cell must be in the matrices
final={k:v for k,v in final.items() if k[0] in WL}
# 6: >= 2 UMIs per cell-barcode pair
final={k:v for k,v in final.items() if v>=2}
# 5: a barcode seen with only one cell
percell=collections.defaultdict(list)
ncells=collections.Counter()
for (cb,lb),v in final.items(): ncells[lb]+=1
final={k:v for k,v in final.items() if ncells[k[1]]>1}
# 7: within a cell, drop barcodes below half the top count
for (cb,lb),v in final.items(): percell[cb].append((lb,v))
edges=collections.defaultdict(set)
for cb,v in percell.items():
    top=max(n for _,n in v)
    for lb,n in v:
        if n>=0.5*top: edges[lb].add(cb)
print("cells with a barcode: %d; barcodes used: %d"%(len({c for s in edges.values() for c in s}),len(edges)),flush=True)
import networkx as nx
G=nx.Graph()
for lb,cells in edges.items():
    cs=sorted(cells)
    for i in range(len(cs)):
        G.add_node(cs[i])
        for j in range(i+1,len(cs)): G.add_edge(cs[i],cs[j])
raw=[c for c in nx.connected_components(G)]
print("raw clones (connected components): %d"%len(raw),flush=True)
def degree_filter(H):
    if H.number_of_nodes()<2: return H
    mx=max(d for _,d in H.degree())
    keep=[n for n,d in H.degree() if d>0.55*mx]
    return H.subgraph(keep).copy()
def try_cut(H):
    if H.number_of_nodes()<4: return None
    n0=H.number_of_nodes()
    for cutter in (nx.minimum_edge_cut,nx.minimum_node_cut):
        try: cut=cutter(H)
        except Exception: continue
        K=H.copy()
        if cutter is nx.minimum_edge_cut: K.remove_edges_from(cut)
        else: K.remove_nodes_from(cut)
        parts=[K.subgraph(c).copy() for c in nx.connected_components(K)]
        if len(parts)>1 and all(p.number_of_nodes()>=2 and p.number_of_nodes()>=0.25*n0 for p in parts):
            return parts
    return None
out=[]; stack=[G.subgraph(c).copy() for c in raw]
while stack:
    H=stack.pop()
    p=try_cut(H)
    if p: stack.extend(p); continue
    H=degree_filter(H)
    if H.number_of_nodes()<2: continue
    comps=list(nx.connected_components(H))
    if len(comps)>1: stack.extend(H.subgraph(c).copy() for c in comps); continue
    p=try_cut(H)
    if p: stack.extend(p); continue
    out.append(sorted(H.nodes()))
print("clones after the cut/degree processing: %d"%len(out),flush=True)
sizes=collections.Counter(len(c) for c in out)
print("multi-cell clones: %d; cells in a clone: %d"%(sum(1 for c in out if len(c)>1),sum(len(c) for c in out)),flush=True)
print("clone size distribution: %s"%dict(sorted(sizes.items())[:12]),flush=True)
rows=[]
for i,c in enumerate(out):
    for cb in c: rows.append((cb,WL.get(cb,""),"clone_%d"%i))
import csv
with open(os.path.join(OUT,"cell_to_clone.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["cell_barcode","sample","clone_id"]); w.writerows(rows)
mt=sum(1 for c in out if len({WL.get(cb,"") for cb in c})>1)
print("multi-time clones (cells from both T1 and T2): %d"%mt,flush=True)
print("wrote %s (%d rows)"%(os.path.join(OUT,"cell_to_clone.csv"),len(rows)),flush=True)
