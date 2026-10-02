from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, os, collections
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,'/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline')
import pandas as pd, numpy as np
from paper_analysis.fatemap_pipeline.fatemap_qc_utils import build_sample_sheet, run_singlet_calling
from paper_analysis.fatemap_pipeline import build_fm06_qc_matrix as b
ss = build_sample_sheet(b.STARCODE_PATH, b.SAMPLE_NUM_TO_REPLICATE)
res = {}
for cut in (3, 2):
    singlets, sc, resc = run_singlet_calling(ss, dataset_name='FM06', min_umi_cutoff=cut)
    sc = dict(sc.items()); res[cut] = (sc, set(resc))
    print('cutoff', cut, 'singlet cells', len(sc), 'distinct clones', len(set(sc.values())), 'rescued', len(set(resc)), flush=True)
c3, c2 = res[3][0], res[2][0]
obs = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/finalized_data/FM06_data/qc_filtered/obs_metadata.csv', index_col=0)
kept = set(obs.index)  # 'A:cellbarcode'
key = lambda k: f'{k[0]}:{k[1]}'
same = [k for k in c3 if k in c2 and c2[k] == c3[k]]
changed = [k for k in c3 if k in c2 and c2[k] != c3[k]]
lost = [k for k in c3 if k not in c2]
gained = [k for k in c2 if k not in c3]
print('\n== singletCode level (all cells, before mito/gene QC) ==')
print('unchanged clone assignment', len(same), '| barcode reassigned', len(changed), '| lost (singlet at 3 not at 2)', len(lost), '| gained (singlet at 2 only)', len(gained))
print('\n== among the 12,863 cells currently kept ==')
kc = lambda L: [k for k in L if key(k) in kept]
print('kept cells with reassigned barcode', len(kc(changed)), '| kept cells that lose singlet status', len(kc(lost)))
print('gained cells (need mito+gene QC still):', len(gained))
# clone level
cl3 = collections.Counter(c3.values()); cl2 = collections.Counter(c2.values())
new_clones = set(cl2) - set(cl3); gone = set(cl3) - set(cl2); grown = [c for c in cl3 if c in cl2 and cl2[c] > cl3[c]]
print('\n== clone (barcode) level, all cells ==')
print('clones at 3:', len(cl3), '| at 2:', len(cl2), '| new clones only at 2:', len(new_clones), '| clones lost:', len(gone), '| existing clones that gain cells:', len(grown))
print('clone size dist at 3: ', pd.Series(list(cl3.values())).describe()[['mean','50%','max']].round(2).to_dict())
print('clone size dist at 2: ', pd.Series(list(cl2.values())).describe()[['mean','50%','max']].round(2).to_dict())
print('singleton clones (size 1) at 3:', sum(v==1 for v in cl3.values()), ' at 2:', sum(v==1 for v in cl2.values()))
# A/B spanning clones
def span(c):
    d = collections.defaultdict(set)
    for (s_, cid), bc in c.items(): d[bc].add(s_)
    return sum(1 for v in d.values() if len(v) == 2), d
s3, d3 = span(c3); s2, d2 = span(c2)
print('clones spanning both replicates at 3:', s3, ' at 2:', s2)
gained_bc = collections.Counter(c2[k] for k in gained)
big = gained_bc.most_common(8); print('barcodes gaining most cells at 2:', big)
print('gained cells landing in a clone that already existed at 3:', sum(1 for k in gained if c2[k] in cl3), '/', len(gained))
