from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import json

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OURS = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/gene_sets.json'
OURS = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/gene_sets.json'
# [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] YDIR = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/panels/tf_target_panel"
YDIR = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports/panels/tf_target_panel"

ours = json.load(open(OURS))
pairs = [
    ("correlation_high", "p4a01_corrhigh"),
    ("correlation_mid",  "p4a01_corrmid"),
    ("correlation_low",  "p4a01_corrlow"),
    ("variability_high", "p4a01_hvg_q67100"),
    ("variability_mid",  "p4a01_hvg_q3367"),
    ("variability_low",  "p4a01_hvg_q0033"),
    ("detection_high",   "p4a01_detect_q90100"),
    ("detection_mid",    "p4a01_detect_q7590"),
    ("detection_low",    "p4a01_detect_q5075"),
]

for oname, yname in pairs:
    o = set(ours[oname])
    y = set(json.load(open(f"{YDIR}/{yname}_panel.json"))["panel"])
    inter = o & y
    j = len(inter) / len(o | y)
    print(f"\n=== {oname}  vs  {yname} ===")
    print(f"  ours {len(o):3d}   yscher {len(y):3d}   shared {len(inter):3d}   "
          f"Jaccard {j:.2f}   (ours-only {len(o-y)}, yscher-only {len(y-o)})")
    print(f"  shared     : {','.join(sorted(inter))}")
    print(f"  ours only  : {','.join(sorted(o - y))}")
    print(f"  yscher only: {','.join(sorted(y - o))}")
