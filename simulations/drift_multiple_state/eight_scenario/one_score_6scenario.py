"""Test S = s(|z_gamma|) + s(|rho_cross_xy|) + s(|rho_t2|) -- the single combined formula that
won the network_sweep_final t1=1h/t1=10h trade-off (see analysis_data/network_sweep_final/
one_score_search.py) -- on the 6 drift/regulation 2-gene scenarios, as a regulation-PRESENCE
detector (regulated vs unregulated), same framing as the earlier D = s(|z_gamma|)+s(|z_reg_gated|)
comparison. Loaded/standardized separately per t1 (no pooling, per prior instruction).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario'
SCEN = ["no_regulation", "A_to_B", "multistate_A_to_B", "multistate_A_B", "kramp_A_to_B", "kramp_A_B"]


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


def load(sub):
    rows = []
    for jf in sorted(glob.glob(f"{ROOT}/{sub}/*_rep_*.json")):
        d = json.load(open(jf))
        scen = d.get("scenario") or os.path.basename(jf).rsplit("_rep_", 1)[0]
        if scen not in SCEN:
            continue
        rho_t2 = d["gene_t2_gene_1_gene_2"]
        rxy0 = d["step4_rho_cross_1to2"]; ryx0 = d["step4_rho_cross_2to1"]
        for dirn, rxy, zg in (("g1→g2", rxy0, d["z_gamma_1to2"]),
                              ("g2→g1", ryx0, d["z_gamma_2to1"])):
            rows.append(dict(scen=scen, dirn=dirn, rep=os.path.basename(jf),
                              rho_t2=rho_t2, rxy=rxy, z_gamma=zg,
                              y=0 if scen == "no_regulation" else 1))
    return pd.DataFrame(rows)


for t1, sub in [(1, "t1_1_t2_20"), (10, "t1_10_t2_20")]:
    df = load(sub)
    S = s(np.abs(df.z_gamma.to_numpy())) + s(np.abs(df.rxy.to_numpy())) + s(np.abs(df.rho_t2.to_numpy()))
    df["S"] = S
    auroc = roc_auc_score(df.y, df.S)
    auprc = average_precision_score(df.y, df.S)
    print(f"\n=== t1={t1}h  (n={len(df)}, {df.rep.nunique()} reps x 6 scen x 2 dir) ===")
    print(f"S = s(|z_gamma|)+s(|rho_cross_xy|)+s(|rho_t2|)   AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
    print(df.groupby("scen").S.median().reindex(SCEN).round(2).to_string())

    # component-only baselines for reference
    for name, comp in [("z_gamma alone", np.abs(df.z_gamma)), ("rho_cross_xy alone", np.abs(df.rxy)),
                        ("rho_t2 alone", np.abs(df.rho_t2))]:
        print(f"  {name:20s} AUROC={roc_auc_score(df.y, comp):.3f}  AUPRC={average_precision_score(df.y, comp):.3f}")
