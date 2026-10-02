from twinfer.utils.paths import get_repo_root as _twinfer_get_repo_root  # [2026-09-30 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] src=open(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline/run_infer_fatemap_ab_split.py').read()
src=open(f'{_twinfer_get_repo_root()}/paper_analysis/fatemap_pipeline/run_infer_fatemap_ab_split.py').read()
i=src.index('def subsample_per_side'); j=src.index('\ndef ',i+5) if '\ndef ' in src[i+5:] else len(src)
ns={'pd':pd,'SUBSAMPLE_SEED':101010}; exec(src[i:j],ns); f=ns['subsample_per_side']
rng=np.random.default_rng(1)
n=3000; df=pd.DataFrame({'cell_id':[f'c{i}' for i in range(n)],'clone_id':rng.choice(['x','y','z','w'],n,p=[.6,.25,.1,.05]),'time_step':rng.integers(0,2,n)})
a=f(df,100); b=f(df.sample(frac=1,random_state=5).reset_index(drop=True),100)
side=a.groupby(['clone_id','time_step']).size()
print('max per side',side.max(),'| same cells after shuffling input rows:',set(a.cell_id)==set(b.cell_id),'| n',len(a))
exp=df.groupby(['clone_id','time_step']).size().clip(upper=100).sum(); print('expected total',exp,'got',len(a),'| small sides untouched:',
  bool(all(set(df[(df.clone_id==c)&(df.time_step==t)].cell_id)==set(a[(a.clone_id==c)&(a.time_step==t)].cell_id) for (c,t),v in df.groupby(['clone_id','time_step']).size().items() if v<=100)))
