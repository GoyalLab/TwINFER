"""Mutual_regulation test simulations: one replicate of each parameter variant, run in parallel inside one node.
Variants (all genes use row 0 of the parameter table; K is recomputed from the steady state by the simulator):
  control | n3 (Hill n=3 on all edges) | n4 | kprod2x (k_prod_mRNA x2) | mrnahl2x (mRNA half-life x2) | kadd12 | koff17 | combo (n=4, k_off=17.2, k_add=12)
Writes the variant parameter tables to <out>/param_tables/ and the simulations to <out>/<variant>/."""
import argparse, os, subprocess, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.abspath(os.path.join(HERE,"../../simulation_example_input_data/median_parameter.csv"))
def variants():
    nn=[c for c in pd.read_csv(SRC,index_col=0).columns if c.startswith("n_")]; ka=[c for c in pd.read_csv(SRC,index_col=0).columns if c.startswith("k_add_")]
    def mk(**kw):
        d=pd.read_csv(SRC,index_col=0)
        for k,v in kw.items():
            if k=="n": d[nn]=v
            elif k=="k_add": d[ka]=v
            elif k=="kprod_x": d["k_prod_mRNA"]*=v
            elif k=="mrnahl_x": d["mrna_half_life"]*=v
            else: d[k]=v
        return d
    return {"control":mk(),"n3":mk(n=3),"n4":mk(n=4),"kprod2x":mk(kprod_x=2),"mrnahl2x":mk(mrnahl_x=2),"kadd12":mk(k_add=12),"koff17":mk(k_off=17.2),"combo":mk(n=4,k_off=17.2,k_add=12)}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--rep",type=int,required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--cores-per",type=int,default=6); ap.add_argument("--burnin-hours",type=int,default=None); ap.add_argument("--only",nargs="*",default=None)
    a=ap.parse_args(); V=variants(); os.makedirs(f"{a.out}/param_tables",exist_ok=True)
    names=a.only or list(V)
    for k in names: V[k].to_csv(f"{a.out}/param_tables/{k}.csv")
    def run(k):
        o=f"{a.out}/{k}"; os.makedirs(f"{o}/logs",exist_ok=True)
        cmd=[sys.executable,f"{HERE}/simulate_1k.py","--config_index","2","--n_reps","1","--rep_offset",str(a.rep),"--cores",str(a.cores_per),
             "--output_folder",o,"--save_before_division","0","--param_csv",f"{a.out}/param_tables/{k}.csv"]+(["--burnin_hours",str(a.burnin_hours)] if a.burnin_hours else [])
        with open(f"{o}/logs/rep{a.rep}.log","w") as lf: rc=subprocess.call(cmd,stdout=lf,stderr=subprocess.STDOUT,env=dict(os.environ,NUMBA_NUM_THREADS=str(a.cores_per)))
        return k,rc
    print(f"rep {a.rep}: {len(names)} variants x {a.cores_per} cores",flush=True)
    with ThreadPoolExecutor(len(names)) as ex:
        for k,rc in ex.map(run,names): print(k,"ok" if rc==0 else f"FAILED rc={rc}",flush=True)
