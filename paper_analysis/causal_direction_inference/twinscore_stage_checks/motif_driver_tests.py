"""PDF Sec. 9 (shared driver, from C) and Sec. 10 (unshared driver, from U) on the figure_4 motif replicates (twin pairs at 20 h, N clones).
For pair (y,z) the candidate TF x is the third gene. Same files/replicates/sampling as motif_replicates_zscores.py.
 Shared  (needs inherited state z_C>2.326): C(x,y), C(x,z) significant (|C|/sigma_C > 2.326, sigma_C^2=(1+S^2+hx*hy)/(2N)); C(x,x)/sigma_xx > 2.326
   (sigma_xx^2=(2+hx^2)/(2N)); sign(C(x,y)C(x,z))=sign(C(y,z)); f_x = C(x,y)C(x,z)/(C(x,x)C(y,z)); driver if |f_x-1|<2 sd (delete-group jackknife, 50 groups).
 Unshared (pair called, z_U>1.96): |rho_delta(x,y)|/sigma_D, |rho_delta(x,z)|/sigma_D > 2.326 (sigma_D=1/sqrt(N)), consistent signs;
   g_x = U(x,y)U(x,z)/(U(x,x)U(y,z)), U(x,x)=1-h_x; unshared fan-out via x if |g_x-1|<2 sd.   [assumption: 'U/sigma_D' read as rho_delta/sigma_D = z_U-style]"""
import argparse, os, numpy as np, pandas as pd, glob, re
from u_time_pool import rk as rkm
from motif_replicates_zscores import files, load, MOT, T2
from motif_shared_driver import PAIRS, TRUTH
G=50
NS=(3000,6000)
def mats(R,idx=None):
    """R: dict gene -> (n,2) centred ranks. Returns S,C,h matrices over genes 1..3 on clones idx."""
    gs=[1,2,3]; S=np.zeros((3,3)); C=np.zeros((3,3)); M={}
    for g in gs:
        r=R[g] if idx is None else R[g][idx]; M[g]=((r**2).sum(1)/2).mean()
    for a in gs:
        for b in gs:
            ra=R[a] if idx is None else R[a][idx]; rb=R[b] if idx is None else R[b][idx]; n=np.sqrt(M[a]*M[b])
            S[a-1,b-1]=((ra[:,0]*rb[:,0]+ra[:,1]*rb[:,1])/2).mean()/n; C[a-1,b-1]=((ra[:,0]*rb[:,1]+ra[:,1]*rb[:,0])/2).mean()/n
    return S,C
def stat(S,C,x,y,z):
    h=lambda g:C[g-1,g-1]; Cx=lambda a,b:C[a-1,b-1]; U=lambda a,b:S[a-1,b-1]-C[a-1,b-1]
    f=Cx(x,y)*Cx(x,z)/(Cx(x,x)*Cx(y,z)); Ux=lambda a,b:(1-h(a)) if a==b else U(a,b)
    g=U(x,y)*U(x,z)/(Ux(x,x)*U(y,z)); return f,g
def jack(R,n,x,y,z,rng):
    grp=rng.permutation(n)%G; fs=[];gs=[]
    for k in range(G):
        S,C=mats(R,np.where(grp!=k)[0]); f,g=stat(S,C,x,y,z); fs.append(f); gs.append(g)
    sd=lambda v:np.sqrt((G-1)/G*((np.array(v)-np.mean(v))**2).sum()); return sd(fs),sd(gs)
def run(a):
    m,r,fn,out=a; dest=f"{out}/{m}_rep_{r:04d}.csv"
    if os.path.exists(dest): return
    D=load(fn); rng=np.random.RandomState(r); NCL=D[T2][1].shape[0]; perm=rng.permutation(NCL); rows=[]
    for N in NS:   # [2026-10-04 was (3000,6000); now selectable via --ns, default unchanged]
        c=perm[:N]; R={}
        for g in (1,2,3):
            x=rkm(D[T2][g][c].ravel()).reshape(N,2); R[g]=x-x.mean()
        S,C=mats(R); sD=1/np.sqrt(N)
        for (y,z) in PAIRS:
            x=({1,2,3}-{y,z}).pop(); 
            h=lambda g:max(0.,C[g-1,g-1]); sC=lambda a,b:np.sqrt((1+S[a-1,b-1]**2+h(a)*h(b))/(2*N))
            zC=lambda a,b:np.sign(S[a-1,b-1])*C[a-1,b-1]/sC(a,b); U=lambda a,b:S[a-1,b-1]-C[a-1,b-1]
            rd=lambda a,b:U(a,b)/np.sqrt((1-h(a))*(1-h(b)))
            inherited=zC(y,z)>2.326; zU=rd(y,z)/sD; called=zU>1.96
            f,g=stat(S,C,x,y,z); sdf,sdg=jack(R,N,x,y,z,rng)
            sh_ok=inherited and abs(C[x-1,y-1])/sC(x,y)>2.326 and abs(C[x-1,z-1])/sC(x,z)>2.326 and C[x-1,x-1]/np.sqrt((2+h(x)**2)/(2*N))>2.326 and np.sign(C[x-1,y-1]*C[x-1,z-1])==np.sign(C[y-1,z-1])
            un_ok=called and abs(rd(x,y))/sD>2.326 and abs(rd(x,z))/sD>2.326 and np.sign(U(x,y)*U(x,z))==np.sign(U(y,z))
            rows.append(dict(motif=m,rep=r,pair=f"{y}-{z}",truth=TRUTH[m][(y,z)],N=N,x=x,inherited=inherited,called=called,f_x=f,sd_f=sdf,shared_ok=sh_ok,shared_driver=bool(sh_ok and abs(f-1)<2*sdf),
                            g_x=g,sd_g=sdg,unshared_ok=un_ok,unshared_fanout=bool(un_ok and abs(g-1)<2*sdg)))
    pd.DataFrame(rows).to_csv(dest,index=False)
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--nreps",type=int,default=50);ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);ap.add_argument("--ns",type=int,nargs="*",default=[3000,6000]);a=ap.parse_args()
    NS=tuple(a.ns)  # (set before the pool forks)
    F=files(); common=sorted(set.intersection(*[{r for (mm,r) in F if mm==m} for m in MOT]))[:a.nreps]; os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p: p.map(run,[(m,r,F[(m,r)],a.out) for m in MOT for r in common])
    print("done",len(common))
