#!/usr/bin/env python3
"""Candidate pairs for the all-TF Stage I run: every Lambert TF detected in >= 20 % of the cells that carry a clone barcode (the cells the TwinScore script scores)
x every gene detected in >= 5 % of those cells (the pipeline's gene floor). Excluded: the TF paired with itself, and the technical genes MALAT1, NEAT1, XIST, FIRRE and MT- genes as targets (PDF, section 1).
Writes analysis_data/<fm>/data/stage1_tf_ge20/{candidates.tsv (TF, gene; no header), tf_list.csv (TF, detection_pct, mean_log2_cp10k)}.   usage: stage1_tf_ge20_prepare.py FM01 [FM03 ...]"""
import os, sys
import numpy as np, pandas as pd, scipy.io as sio
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
TF = {l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()}
for d in sys.argv[1:]:
    Q = f"{ROOT}/finalized_data/{d}_data/qc_filtered"; g = np.array(open(f"{Q}/genes.txt").read().split()); o = pd.read_csv(f"{Q}/obs_metadata.csv", index_col=0)
    ok = o.fatemap_clone_singletcode.notna().to_numpy(); X = sio.mmread(f"{Q}/{d.lower()}_qc_counts.mtx").tocsr()[np.where(ok)[0]]
    det = np.asarray((X > 0).mean(0)).ravel(); tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    isTF = np.array([x in TF for x in g]); tfi = np.where(isTF & (det >= 0.20))[0]; gi = np.where(det >= 0.05)[0]
    tech = np.array([x in {"MALAT1", "NEAT1", "XIST", "FIRRE"} or x.startswith("MT-") for x in g])
    mean_expr = [float(np.log2(X[:, i].toarray().ravel() / tot * 1e4 / 10 + 1).mean()) for i in tfi]
    out = f"{ROOT}/analysis_data/{d.lower()}/data/stage1_tf_ge20"; os.makedirs(out, exist_ok=True)
    pd.DataFrame({"TF": g[tfi], "detection_pct": (100 * det[tfi]).round(2), "mean_log2_cp10k": np.round(mean_expr, 4)}).sort_values("detection_pct", ascending=False).to_csv(f"{out}/tf_list.csv", index=False)
    tg = [x for x, i in zip(g[gi], gi) if not tech[i]]
    with open(f"{out}/candidates.tsv", "w") as f:
        for t in g[tfi]:
            for x in tg:
                if x != t: f.write(f"{t}\t{x}\n")
    print(d, "clone cells", int(ok.sum()), "| TFs >=20%:", len(tfi), "| target genes:", len(tg), "| pairs:", f"{len(tfi) * len(tg):,}", flush=True)
