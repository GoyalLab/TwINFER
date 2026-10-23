"""Run replicates of simulate_1k.py (3 motifs x 1000 reps) as parallel subprocesses inside ONE multi-core job.
Global replicate index g = 0..2999 (motif = g//1000, rep = g%1000). Slice k of K takes g % K == k, so 10 jobs cover everything.
A replicate is skipped if its df_rows_*_{Motif}_{motif}_{rep}_*.csv already exists (resumable)."""
import argparse, glob, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
MOT=["Fan_out","Feed_forward","Mutual_regulation"]
HERE=os.path.dirname(os.path.abspath(__file__))
ap=argparse.ArgumentParser(); ap.add_argument("--slice",type=int,required=True); ap.add_argument("--n-slices",type=int,default=10)
ap.add_argument("--jobs",type=int,default=12); ap.add_argument("--cores-per",type=int,default=4); ap.add_argument("--out",required=True)
ap.add_argument("--burnin-hours",type=int,default=None); ap.add_argument("--limit",type=int,default=None,help="test: only the first N of this slice")
ap.add_argument("--g",type=int,nargs="*",default=None,help="test: explicit global indices")
# [2026-10-04 added] --g-file: fixed list of global indices (one per line); slice k takes list[k::n_slices] so a resubmission is balanced.
ap.add_argument("--g-file",default=None); ap.add_argument("--dry-run",action="store_true")
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); os.makedirs(f"{a.out}/logs_multicore",exist_ok=True)
if a.g_file: _all=[int(x) for x in open(a.g_file).read().split()]; gs=_all[a.slice::a.n_slices]
else: gs=a.g if a.g is not None else [g for g in range(3000) if g%a.n_slices==a.slice]
# [2026-10-04] original line kept above in the else branch (g%n_slices==slice)
if a.limit: gs=gs[:a.limit]
def run(g):
    c,r=g//1000,g%1000
    if glob.glob(f"{a.out}/df_rows_*_ncells_6000_{MOT[c]}_{c}_{r}_*.csv"): return g,"exists"
    cmd=[sys.executable,f"{HERE}/simulate_1k.py","--config_index",str(c),"--n_reps","1","--rep_offset",str(r),"--cores",str(a.cores_per),
         "--output_folder",a.out,"--save_before_division","0"]+(["--burnin_hours",str(a.burnin_hours)] if a.burnin_hours else [])
    with open(f"{a.out}/logs_multicore/g{g:04d}.log","w") as lf: rc=subprocess.call(cmd,stdout=lf,stderr=subprocess.STDOUT,env=dict(os.environ,NUMBA_NUM_THREADS=str(a.cores_per)))
    return g,("ok" if rc==0 else f"FAILED rc={rc}")
if a.dry_run: print(a.slice,len(gs),gs); sys.exit(0)
print(f"slice {a.slice}/{a.n_slices}: {len(gs)} replicates, {a.jobs} workers x {a.cores_per} cores",flush=True)
with ThreadPoolExecutor(a.jobs) as ex:
    for g,st in ex.map(run,gs): print(g,st,flush=True)
