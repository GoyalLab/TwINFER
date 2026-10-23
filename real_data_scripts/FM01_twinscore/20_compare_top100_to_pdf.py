import sys,pandas as pd
import importlib.util, os
_s=importlib.util.spec_from_file_location("m19",os.path.join(os.path.dirname(os.path.abspath(__file__)),"19_pdf_fm01_top100_lists.py")); m19=importlib.util.module_from_spec(_s); _s.loader.exec_module(m19); lists=m19.lists
D={'unnorm':'twinscore_pdf_candidates','all (cc+cx+rep+mt)':'twinscore_pdf_candidates_strat/all','cell cycle only':'twinscore_pdf_candidates_strat/cc','complexity only':'twinscore_pdf_candidates_strat/cx'}
rows=[]
for k,p in D.items():
    d=pd.read_csv(p+'/pdf_procedure_listed_pairs.csv'); 
    t={'unshared |z_U|':d.assign(a=d.z_U.abs()).sort_values('a',ascending=False),'shared h':d.assign(a=d.h).sort_values('a',ascending=False),'co-expr |rho|':d.assign(a=d.rho.abs()).sort_values('a',ascending=False)}
    key={'unshared |z_U|':'unshared','shared h':'shared','co-expr |rho|':'coexpression'}
    for n,x in t.items():
        top=list(zip(x.TF,x.gene))[:100]; ov=len(set(top)&set(lists[key[n]]))
        from collections import Counter; c=Counter(a for a,b in top).most_common(3)
        rows.append((k,n,ov,len(set(a for a,b in top)),c,'; '.join(f'{a} {b}' for a,b in top[:5])))
        if k!='unnorm': pd.DataFrame(top,columns=['TF','gene']).to_csv(f"{p}/top100_{key[n]}.csv",index=False)
for r in rows: print(*r,sep=' | ')
