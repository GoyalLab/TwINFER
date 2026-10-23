#!/usr/bin/env python3
"""Stage the inputs for the all-TF Stage I run (twinscore_pdf_candidates.py) of LARRY, CellTag3 and Michaels, one unit per timepoint.
Counts are raw integer counts. LARRY: finalized_data/LARRY_data/filtered mtx; CellTag3: rebuild/.../celltag_clones_repaired_nopad_counts.h5ad (step 7 of the authors' chain; same 54,657 cells as the shipped normalized file),
Michaels: reconstructed counts of the 3,948 lineage singlets (human). LARRY and CellTag3 are mouse: gene symbols are upper-cased to match the (human) Lambert TF list and the exclusion list; mismatches are written to
analysis_data/stage1_inputs/symbol_mismatches.csv.  Clone = fatemap_clone_singletcode column (LARRY larry_clone_singletcode, CellTag3 repaired_clone, Michaels clone_id).
Per unit: candidates = TFs detected in >=20% of the clonal cells of that unit x genes detected in >=5% (as stage1_tf_ge20_prepare.py), written to analysis_data/<unit>/data/stage1_tf_ge20/{candidates.tsv,tf_list.csv}.
usage: stage1_mouse_prepare.py [larry] [celltag3] [michaels]   (default all; CellTag3 loads a 2.2 GB h5ad and must run as a batch job, the login node kills it)"""
import sys
WHICH = sys.argv[1:] or ["larry", "celltag3", "michaels"]
import os, re, json
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sps
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; ST = f"{ROOT}/analysis_data/stage1_inputs"
TF = {l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()}
EX = set(json.load(open(f"{ROOT}/analysis_data/fm01/data/twinscore_pdf_candidates_validation_20261014/exclusion_list_B.json"))["_union_all"])
mism = []


def stage(name, X, genes, obs, mouse):
    d = f"{ST}/{name}"; os.makedirs(d, exist_ok=True)
    genes = np.array([g.upper() for g in genes]) if mouse else np.array(genes)
    sps.save_npz(f"{d}/counts.npz", sps.csr_matrix(X).astype(np.float32)); open(f"{d}/genes.txt", "w").write("\n".join(genes) + "\n"); obs.to_csv(f"{d}/obs_metadata.csv")
    gs = set(genes); mism.append(dict(dataset=name, mouse=mouse, genes=len(genes), Lambert_TFs_in_data=len(TF & gs), Lambert_TFs_total=len(TF), exclusion_genes_in_data=len(EX & gs), exclusion_total=len(EX)))
    print(name, X.shape, "TFs matched", len(TF & gs), "/", len(TF), "excl matched", len(EX & gs), "/", len(EX), flush=True)
    return d, genes


def units(name, d, X, genes, obs, ucol, labels):
    ok = obs.fatemap_clone_singletcode.notna().to_numpy()
    for u, lab in labels.items():
        m = ok & (obs[ucol].astype(str) == str(u)).to_numpy(); Xu = sps.csr_matrix(X)[np.where(m)[0]]
        det = np.asarray((Xu > 0).mean(0)).ravel(); tot = np.asarray(Xu.sum(1)).ravel(); tot[tot == 0] = 1
        tfi = np.where(np.array([g in TF for g in genes]) & (det >= 0.20))[0]; gi = np.where(det >= 0.05)[0]
        tech = np.array([g in {"MALAT1", "NEAT1", "XIST", "FIRRE"} or g.startswith("MT-") for g in genes])
        mean_expr = [float(np.log2(Xu[:, i].toarray().ravel() / tot * 1e4 / 10 + 1).mean()) for i in tfi]
        out = f"{ROOT}/analysis_data/{lab}/data/stage1_tf_ge20"; os.makedirs(out, exist_ok=True)
        pd.DataFrame({"TF": genes[tfi], "detection_pct": (100 * det[tfi]).round(2), "mean_log2_cp10k": np.round(mean_expr, 4)}).sort_values("detection_pct", ascending=False).to_csv(f"{out}/tf_list.csv", index=False)
        tg = [genes[i] for i in gi if not tech[i]]
        with open(f"{out}/candidates.tsv", "w") as f:
            for t in genes[tfi]:
                for x in tg:
                    if x != t: f.write(f"{t}\t{x}\n")
        print(f"  {lab}: clonal cells {int(m.sum())} | TFs>=20%: {len(tfi)} | targets: {len(tg)} | pairs: {len(tfi) * len(tg):,}", flush=True)


def run_larry():
    Q = f"{ROOT}/finalized_data/LARRY_data/filtered"; obs = pd.read_csv(f"{Q}/obs_metadata.csv", index_col=0)
    obs["fatemap_clone_singletcode"] = obs.larry_clone_singletcode; obs["unit"] = obs.library.str.extract(r"_d(\d+)_")[0]
    X = sio.mmread(f"{Q}/larry_qc_counts.mtx").tocsr(); genes = open(f"{Q}/genes.txt").read().split()
    d, g = stage("larry", X, genes, obs, True); units("larry", d, X, g, obs, "unit", {"2": "larry_d2", "4": "larry_d4", "6": "larry_d6"})


def run_celltag3():
    import anndata as ad
    A = ad.read_h5ad(f"{ROOT}/analysis_data/dataset_qc_verification/rebuild/CellTag3/mirror/datasets/celltag/celltag_clones_repaired_nopad_counts.h5ad")
    obs = A.obs[["Time point", "experimental_day", "repaired_clone"]].copy(); obs["fatemap_clone_singletcode"] = obs.repaired_clone.astype(object).where(obs.repaired_clone.notna(), np.nan); obs["unit"] = obs["Time point"].astype(str)
    print("CellTag3 time points", obs.unit.value_counts().to_dict())
    X = sps.csr_matrix(A.X); genes = list(A.var_names); del A
    d, g = stage("celltag3", X, genes, obs, True); units("celltag3", d, X, g, obs, "unit", {"2": "celltag3_t2", "4": "celltag3_t4"})


def run_michaels():
    Q = f"{ROOT}/finalized_data/Michaels_T1T2_data"; obs = pd.read_csv(f"{Q}/lineage_singlets_reconstructed_obs.csv", index_col=0)
    obs["fatemap_clone_singletcode"] = obs.clone_id.astype(str); obs["unit"] = obs.batch
    X = sps.load_npz(f"{Q}/lineage_singlets_reconstructed_counts.npz"); genes = open(f"{Q}/lineage_singlets_reconstructed_genes.txt").read().split()
    d, g = stage("michaels", X, genes, obs, False); units("michaels", d, X, g, obs, "unit", {"T1": "michaels_t1", "T2": "michaels_t2"})


for w in WHICH:
    {"larry": run_larry, "celltag3": run_celltag3, "michaels": run_michaels}[w]()
    pd.DataFrame(mism).to_csv(f"{ST}/symbol_mismatches_{w}.csv", index=False)
