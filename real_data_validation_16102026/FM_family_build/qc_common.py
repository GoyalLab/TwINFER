#!/usr/bin/env python3
"""Shared registry + loaders for the dataset QC verification (qc_verify.py) and the 5k-HVG PCA (pca5k.py).

Nothing here trusts the build scripts' summaries: every number is recomputed from the stored count
matrices and, where raw Cell Ranger output exists, from the raw matrices.

REGISTRY[name] fields
  final:      how to load the final count matrix (cells x genes, raw integer counts) + obs
  raw:        {sample: loader of the raw Cell Ranger matrix} (None when no raw is available here)
  stated:     {sample: (mito_pct_max, min_genes, max_genes)} as hard-coded in the build scripts
  mito_prefix: 'MT-' (human) or 'mt-' (mouse); matched case-insensitively
  summary:    preprocessing_summary.json (claims to check) or None
"""
import json
import os
import re

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
# [2026-10-16 replaced: FIN = f"{ROOT}/finalized_data"] QCV_FIN / QCV_OUT redirect the verification to the full-sheet / corrected-rescue rebuild
FIN = os.environ.get("QCV_FIN", f"{ROOT}/finalized_data")
RAWFM = f"{ROOT}/real_data/Fatemap_barcodes"
# [2026-10-16 replaced: OUT = f"{ROOT}/analysis_data/dataset_qc_verification"]
OUT = os.environ.get("QCV_OUT", f"{ROOT}/analysis_data/dataset_qc_verification")


# ----------------------------------------------------------------------------- metrics
def qc_metrics(X, var_names, prefix="MT-"):
    """total counts, genes with count>0, % counts in genes starting with prefix (case-insensitive)."""
    X = X.tocsr()
    mito = np.array([g.upper().startswith(prefix.upper()) for g in var_names])
    assert mito.sum() > 0, "no mito genes found"
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mt = np.asarray(X[:, mito].sum(axis=1)).ravel().astype(float)
    pct = np.divide(100.0 * mt, tot, out=np.zeros_like(tot), where=tot > 0)
    ng = np.asarray((X > 0).sum(axis=1)).ravel()
    return pd.DataFrame({"total_counts": tot, "n_genes": ng, "pct_counts_mt": pct}), int(mito.sum())


# ----------------------------------------------------------------------------- loaders
def read_h5ad_csr(path, group="X"):
    """CSR group of an h5ad -> scipy csr (no anndata/pandas dependence on obs)."""
    with h5py.File(path, "r") as f:
        g = f[group]
        shape = tuple(g.attrs["shape"])
        return sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


def read_h5ad_obs_var(path):
    import anndata as ad
    with h5py.File(path, "r") as f:
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
    return obs, var


def load_qc_filtered(qc_dir, mtx_name, transpose=False):
    """finalized_data/<ds>/qc_filtered: counts mtx + cells.txt + genes.txt + obs_metadata.csv."""
    X = sio.mmread(f"{qc_dir}/{mtx_name}").tocsr()
    if transpose:
        X = X.T.tocsr()
    cells = [l.rstrip("\n") for l in open(f"{qc_dir}/cells.txt")]
    genes = [l.rstrip("\n") for l in open(f"{qc_dir}/genes.txt")]
    assert X.shape == (len(cells), len(genes)), (X.shape, len(cells), len(genes))
    obs = pd.read_csv(f"{qc_dir}/obs_metadata.csv", index_col=0)
    assert list(obs.index) == cells, "obs_metadata index != cells.txt"
    return X, cells, genes, obs


def load_10x_triplet(prefix, strip=r"-1$"):
    """Cell Ranger <prefix>_{matrix.mtx,features.tsv,barcodes.tsv}.gz -> (cells x genes csr, symbols, barcodes)."""
    X = sio.mmread(f"{prefix}_matrix.mtx.gz").T.tocsr()
    feat = pd.read_csv(f"{prefix}_features.tsv.gz", sep="\t", header=None)
    bc = pd.read_csv(f"{prefix}_barcodes.tsv.gz", sep="\t", header=None)[0].str.replace(strip, "", regex=True)
    assert X.shape == (len(bc), len(feat))
    return X, feat[1].to_numpy(), bc.to_numpy()


def load_10x_dir(d, strip=r"-1$"):
    X = sio.mmread(f"{d}/matrix.mtx.gz").T.tocsr()
    feat = pd.read_csv(f"{d}/features.tsv.gz", sep="\t", header=None)
    bc = pd.read_csv(f"{d}/barcodes.tsv.gz", sep="\t", header=None)[0].str.replace(strip, "", regex=True)
    assert X.shape == (len(bc), len(feat))
    return X, feat[1].to_numpy(), bc.to_numpy()


def load_10x_h5_genes_by_cells(path):
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        mat = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        bc = g["barcodes"][:].astype(str)
        genes = g["features"]["name"][:].astype(str)
    return mat.T.tocsr(), genes, np.array([b.replace("-1", "") for b in bc])


# ----------------------------------------------------------------------------- registry
def _fm(ds, reps, mito, gmin, gmax, sub=""):
    return dict(
        qc_dir=f"{FIN}/{ds}_data/qc_filtered",
        mtx=f"{ds.lower()}_qc_counts.mtx",
        h5ad=f"{FIN}/{ds}_data/{ds}_integrated.h5ad" if ds in ("FM01", "FM06", "FM08") else None,
        raw={r: (lambda p=p: load_10x_triplet(f"{RAWFM}/{sub}{p}")) for r, p in reps.items()},
        stated={r: (mito, gmin, gmax) for r in reps},
        mito_prefix="MT-",
    )


REGISTRY = {
    "FM01": _fm("FM01", {"A": "GSM7434407_FM01_A_1uMPLX", "B": "GSM7434408_FM01_B_1uMPLX"}, 26.0, 200, 7200),
    "FM06": _fm("FM06", {"A": "GSM7434419_FM06_A_WM989Naive", "B": "GSM7434420_FM06_B_WM989Naive"}, 21.0, 200, 5000),
    "FM08": _fm("FM08", {"A": "GSM7434423_FM08_A_PrimaryMelanocytes", "B": "GSM7434424_FM08_B_PrimaryMelanocytes"},
                26.0, 200, 7200),
    "FM03": _fm("FM03", {"A": "GSM7434413_FM03_A_DMSO_1uMPLX"}, 26.0, 200, 7200, "GSE233766_RAW_more_data_extracted/"),
    "FM05": _fm("FM05", {"A1": "GSM7434417_FM05_A1_250nMPLX", "A2": "GSM7434418_FM05_A2_250nMPLX"},
                26.0, 200, 7200, "GSE233766_RAW_more_data_extracted/"),
    "FM07": _fm("FM07", {"A": "GSM7434421_FM07_A_Continuous_1uMPLX", "B": "GSM7434422_FM07_B_Continuous_1uMPLX"},
                26.0, 200, 7200, "GSE233766_RAW_more_data_extracted/"),
}
for _k in ("FM03", "FM05", "FM07"):
    REGISTRY[_k]["mtx"] = f"{_k.lower()}_qc_counts.mtx"
    REGISTRY[_k]["h5ad"] = None

REGISTRY["HS054_Sot48h"] = dict(
    qc_dir=f"{FIN}/HS054_Sot48h_data/qc_filtered", mtx="hs054_qc_counts.mtx",
    h5ad=f"{FIN}/HS054_Sot48h_data/HS054_Sot48h_integrated.h5ad",
    raw={"HS054_Sot48h": lambda: load_10x_dir(f"{ROOT}/real_data/pancreatic_cancer/HS054_Sot48h/outs/filtered_feature_bc_matrix")},
    stated={"HS054_Sot48h": (25.0, 200, 7000)}, mito_prefix="MT-")

_WM = ["T47D-naive-1", "T47D-naive-2", "T47D-lag-1", "T47D-lag-2", "T47D-late-1", "T47D-late-2"]
_WM_MITO = {s: (20.0 if "naive" in s else 25.0) for s in _WM}
_WM_MAX = {s: (7500 if "naive" in s else 8000) for s in _WM}
REGISTRY["Watermelon"] = dict(
    qc_dir=f"{FIN}/Watermelon_data/qc_filtered", mtx="watermelon_qc_counts.mtx", h5ad=None,
    stage_h5ads=[f"{FIN}/Watermelon_{s}_data/Watermelon_{s}_integrated.h5ad" for s in ("naive", "lag", "late")],
    raw={s: (lambda s=s: load_10x_dir(f"{ROOT}/real_data/watermelon_data/{s}", strip=r"-\d+$")) for s in _WM},
    stated={s: (_WM_MITO[s], 2500, _WM_MAX[s]) for s in _WM}, mito_prefix="MT-")

_H5 = "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/%s/outs/filtered_feature_bc_matrix.h5"
REGISTRY["hPSC_20260927"] = dict(
    qc_dir=f"{FIN}/hPSC_20260927_data/qc_filtered", mtx="hPSC_20260927_qc_counts.mtx", transpose=True, h5ad=None,
    raw={s: (lambda s=s: load_10x_h5_genes_by_cells(_H5 % s)) for s in ("endo_T0", "endo_T1")},
    stated={s: (10.0, 1000, 10**9) for s in ("endo_T0", "endo_T1")}, mito_prefix="MT-")


def final_counts(name):
    """-> (X cells x genes counts csr, cell ids, gene names, obs DataFrame with column 'sample', mito symbols)."""
    c = REGISTRY[name]
    if "final" in c:
        X, cells, genes, obs, symbols = c["final"]()
    else:
        X, cells, genes, obs = load_qc_filtered(c["qc_dir"], c["mtx"], transpose=c.get("transpose", False))
        symbols = genes
    if "sample" not in obs.columns:
        for k in ("replicate", "library", "pool"):
            if k in obs.columns:
                obs["sample"] = obs[k]
                break
        else:
            obs["sample"] = name
    return X, cells, genes, obs, symbols


def stated(cfg, s):
    """stated cutoffs for sample s as a dict (defaults: no total-count limits)."""
    v = cfg["stated"][s]
    d = dict(mito=v[0], min_genes=v[1], max_genes=v[2], min_total=0.0, max_total=float("inf"))
    if len(v) > 3:
        d["min_total"] = v[3]
    if len(v) > 4:
        d["max_total"] = v[4]
    return d


def split_id(cell_id):
    s, b = cell_id.rsplit(":", 1)
    return s, b


def load_summary(name):
    p = REGISTRY[name]["qc_dir"] + "/preprocessing_summary.json"
    return json.load(open(p)) if os.path.exists(p) else None


# ----------------------------------------------------------------------------- LARRY (pilot: LSK_d2_1 only)
def load_larry_zip_library(lib):
    """Per-library count matrix inside LARRY_data/raw/<lib>.zip (inDrops quantification, cells x 25,289 genes);
    inDrops 'bcXXXX' cell hashes are mapped to the real ACGT-ACGT gel barcode with the zip's abundant_barcodes.pickle.
    No FASTQ / barcode-calling step is involved (counts.tsv.gz is read as shipped)."""
    import pickle
    import subprocess
    import zipfile
    zp = f"{FIN}/LARRY_data/raw/{lib}.zip"
    with zipfile.ZipFile(zp) as zf:
        with zf.open("abundant_barcodes.pickle") as f:
            d = pickle.load(f)
    hash2acgt = {bc: acgt for acgt, (bc, cnt) in d.items()}
    proc = subprocess.Popen(f"unzip -p {zp} {lib}.counts.tsv.gz | zcat", shell=True, stdout=subprocess.PIPE, text=True)
    header = proc.stdout.readline().rstrip("\n").split("\t")
    genes = header[1:]
    rows, cols, data, bcs = [], [], [], []
    for i, line in enumerate(proc.stdout):
        parts = line.rstrip("\n").split("\t")
        bcs.append(hash2acgt.get(parts[0], parts[0]))
        v = np.array(parts[1:], dtype=np.int32)
        nz = np.flatnonzero(v)
        rows.append(np.full(nz.shape, i, dtype=np.int32)); cols.append(nz.astype(np.int32)); data.append(v[nz])
    assert proc.wait() == 0
    X = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(len(bcs), len(genes)))
    # total counts threshold of the whitelist step as stated in the build summary (dip-test + Otsu, per library)
    thr = json.load(open(f"{FIN}/LARRY_data/filtered/preprocessing_summary.json"))["whitelist_per_library"][lib]["threshold"]
    tot = np.asarray(X.sum(axis=1)).ravel()
    return X, np.array(genes), np.array(bcs), {"whitelist: total counts > %.1f (stated Otsu threshold)" % thr: tot > thr}


REGISTRY["LARRY"] = dict(
    qc_dir=f"{FIN}/LARRY_data/filtered", mtx="larry_qc_counts.mtx", h5ad=None, mito_prefix="mt-",
    raw={"LSK_d2_1": lambda: load_larry_zip_library("LSK_d2_1")},  # pilot: one library only
    stated={"LSK_d2_1": (20.0, 200, 10**9)})


# ----------------------------------------------------------------------------- Michaels et al. (human, 3 hashtag pools)
def _michaels_raw(pool):
    prefix = {"PoolA": "GSM7916611_PoolA", "PoolC": "GSM7916612_PoolC", "PoolD": "GSM7916613_PoolD"}[pool]
    d = f"{ROOT}/real_data/Michaels_et_al"
    X = sio.mmread(f"{d}/{prefix}_matrix.mtx.gz").T.tocsr()
    bc = pd.read_csv(f"{d}/{prefix}_barcodes.tsv.gz", header=None, sep="\t")[0].to_numpy()
    feat = pd.read_csv(f"{d}/{prefix}_features.tsv.gz", header=None, sep="\t")
    hto = pd.read_csv(f"{d}/{prefix}_HTOclusters.csv.gz").set_index("Barcode")["Cluster"].reindex(bc)
    ok = (~hto.isin(["Doublet", "Negative"]) & hto.notna()).to_numpy()
    # barcodes keep their '-1' (final obs_names are '<barcode>-1-<pool>'); genes are Ensembl ids, symbols for mito
    return X, feat[0].to_numpy(), bc, {"HTO demultiplexing: singlet only": ok, "_symbols": feat[1].to_numpy()}


def _michaels_final():
    import anndata as ad
    f = f"{ROOT}/real_data/Michaels_et_al/timecourse_after_cell_gene_qc.h5ad"
    obs, var = read_h5ad_obs_var(f)
    X = read_h5ad_csr(f, "X")
    obs = obs.copy()
    obs["sample"] = obs["pool"].astype(str)
    return X, list(obs.index), list(var.index), obs, var["gene_symbol"].astype(str).to_numpy()


REGISTRY["Michaels_et_al"] = dict(
    final=_michaels_final, h5ad=None, mito_prefix="MT-",
    raw={p: (lambda p=p: _michaels_raw(p)) for p in ("PoolA", "PoolC", "PoolD")},
    stated={p: (20.0, 1000, 10**9, 0.0, 85000.0) for p in ("PoolA", "PoolC", "PoolD")},
    # keys of final cells are '<barcode>-1-<pool>'; raw key = '<pool>:<barcode>-1' -> see key_of_final
    key_of_final=lambda cid: (cid.rsplit("-", 1)[1] + ":" + cid.rsplit("-", 1)[0]))


# ----------------------------------------------------------------------------- datasets NOT in finalized_data: final-only (no raw matrix here)
def _spacebar_final():
    """5 tissue sections (cell_by_gene CSV: object_id + targeted gene panel + barcode-call columns), restricted to the genes
    common to all 5 sections. Values are used as shipped (integer test is done by the caller)."""
    frames, panels = [], []
    d = f"{ROOT}/real_data/SpaceBar_data"
    for i in range(1, 6):
        f = f"{d}/Section{i}_cell_by_gene_clustered.csv"
        hd = open(f).readline().rstrip("\n").split(",")
        j = [k for k, x in enumerate(hd) if x.startswith("bc_")][0]
        panels.append(hd[2:j])
    common = [g for g in panels[0] if all(g in p for p in panels)]
    mats, cells, lab = [], [], []
    for i in range(1, 6):
        df = pd.read_csv(f"{d}/Section{i}_cell_by_gene_clustered.csv", usecols=["object_id"] + common, index_col="object_id")
        mats.append(sp.csr_matrix(df[common].to_numpy(dtype=np.float32)))
        cells += [f"Section{i}:{c}" for c in df.index]; lab += [f"Section{i}"] * len(df)
    obs = pd.DataFrame({"sample": lab}, index=cells)
    return sp.vstack(mats).tocsr(), cells, common, obs, np.array(common)


def _celltag3_final():
    f = f"{ROOT}/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad"
    obs, var = read_h5ad_obs_var(f)
    obs = obs.copy()
    obs["sample"] = "day" + obs["Time point"].astype(str)
    return read_h5ad_csr(f, "X"), list(obs.index), list(var.index), obs, np.array(var.index)


def _petrace_final(tag):
    f = f"{ROOT}/real_data/PeTrace/{tag}.h5td"
    obs, var = read_h5ad_obs_var(f)
    obs = obs.copy()
    obs["sample"] = obs["capture"].astype(str)
    return read_h5ad_csr(f, "X"), list(obs.index), list(var.index), obs, var.index.astype(str).to_numpy()  # var index = gene_name


def _dtt_final():
    d = f"{FIN}/DTT_mouse_data"
    z = np.load(f"{d}/counts_samecell_analysis.npz", allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    cells = pd.read_csv(f"{d}/counts_samecell_analysis_cells.csv")
    m = pd.read_csv(f"{d}/matrix_column_to_ensembl.csv").sort_values("matrix_column_0based")
    assert X.shape == (len(cells), len(m))
    obs = cells.set_index("cell_id")
    obs["sample"] = [c.split("_")[0] for c in obs.index]
    return X, list(obs.index), m["rowname"].tolist(), obs, m["gene_short_name"].astype(str).to_numpy()


REGISTRY["SpaceBar"] = dict(final=_spacebar_final, raw=None, h5ad=None, mito_prefix="MT-")
REGISTRY["CellTag3"] = dict(final=_celltag3_final, raw=None, h5ad=None, mito_prefix="MT-")
REGISTRY["DTT_mouse"] = dict(final=_dtt_final, raw=None, h5ad=None, mito_prefix="mt-")
PETRACE = [f"{st}-R{r}" for st in ("E7.5", "E8.0", "E8.5", "E9.0", "E9.5") for r in (1, 2, 3)]
for _t in PETRACE:
    REGISTRY[f"PeTrace_{_t}"] = dict(final=(lambda t=_t: _petrace_final(t)), raw=None, h5ad=None, mito_prefix="mt-")


# ----------------------------------------------------------------------------- PeTrace merged per day (R1+R2+R3)
PETRACE_STAGES = ["E7.5", "E8.0", "E8.5", "E9.0", "E9.5"]


def _petrace_merged_final(stage):
    """all replicates (R1, R2, R3) of one stage stacked: cells x genes counts (same 78,258 genes in every file)."""
    mats, cells, obs_all, genes = [], [], [], None
    for r in (1, 2, 3):
        f = f"{ROOT}/real_data/PeTrace/{stage}-R{r}.h5td"
        obs, var = read_h5ad_obs_var(f)
        if genes is None:
            genes = list(var.index)
        assert list(var.index) == genes, f"{f}: gene list differs from R1"
        mats.append(read_h5ad_csr(f, "X"))
        cells += list(obs.index)
        o = obs.copy()
        o["sample"] = stage
        obs_all.append(o[["sample"]])
    assert len(set(cells)) == len(cells), "cell ids not unique across replicates"
    X = sp.vstack(mats, format="csr")   # >2^31 nonzeros for E9.5 -> scipy switches to int64 indices
    return X, cells, genes, pd.concat(obs_all), np.array(genes)


for _s in PETRACE_STAGES:
    REGISTRY[f"PeTrace_{_s}"] = dict(final=(lambda s=_s: _petrace_merged_final(s)), raw=None, h5ad=None, mito_prefix="mt-")


# ----------------------------------------------------------------------------- Michaels et al. T1 / T2 (clone labels from yscher's cell_to_clone.csv)
YM = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/datasets/michaels2024/processed"


def _michaels_t12_raw(s):
    g = {"T1": "GSM7916614_T1", "T2": "GSM7916615_T2"}[s]
    X, sym, bc = load_10x_triplet(f"{ROOT}/real_data/Michaels_et_al/{g}")
    return X, sym, bc


def _michaels_t12_final():
    """yscher's QC'd cells (mito <= 20 %, >= 200 genes), integer counts; ids 'T1|<barcode>-1'."""
    mats, cells, obs_all, genes = [], [], [], None
    for s in ("T1", "T2"):
        f = f"{YM}/michaels_{s}_qc.h5ad"
        obs, var = read_h5ad_obs_var(f)
        genes = genes or list(var.index)
        assert list(var.index) == genes
        mats.append(read_h5ad_csr(f, "X"))
        cells += list(obs.index)
        o = obs.copy()
        o["sample"] = s
        obs_all.append(o)
    return sp.vstack(mats, format="csr"), cells, genes, pd.concat(obs_all), np.array(genes)


REGISTRY["Michaels_T1T2"] = dict(
    final=_michaels_t12_final, h5ad=None, mito_prefix="MT-",
    raw={s: (lambda s=s: _michaels_t12_raw(s)) for s in ("T1", "T2")},
    stated={s: (20.0, 200, 10**9) for s in ("T1", "T2")},
    key_of_final=lambda cid: cid.replace("|", ":").removesuffix("-1"))     # 'T1|AAAC...-1' -> 'T1:AAAC...'
