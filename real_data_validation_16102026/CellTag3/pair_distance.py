#!/usr/bin/env python3
"""PCA distance between clonal pairs vs random pairs, per sample and per (sample, cell type).

usage: pair_distance.py <dataset> [<dataset> ...] | combine
Input : <OUT>/pca5k/<dataset>_pca5k.npz (from pca5k.py: top-5000-HVG PCA, 50 PCs) + clone / sample / cell-type labels (META below).
Output: <OUT>/pair_distance/<dataset>_pair_distance_summary.csv   one row per (sample, cell type or ALL)
        <OUT>/pair_distance/<dataset>_pair_distances.csv.gz       pair-level distances (capped per group)
        <OUT>/pair_distance/pair_distance_<dataset>_samples.png / _celltypes.png

Definitions
  * Euclidean distance in PCA space over ALL 50 PCs (same as compute_distances in LARRY_analysis.ipynb); the distance over the
    first K_elbow PCs (elbow of the variance curve) is kept as a secondary column (*_elbowK).
  * clonal pair  = two distinct cells of the same clone within the same sample. Only cells carrying a clone label are used.
  * random pair  = two distinct cells of the same sample drawn from DIFFERENT clones (same clone-labelled population, so the
    baseline has the same cell mix as the clonal pairs); as many random pairs as clonal pairs.
  * cell-type level: both cells must have that cell type. Clonal pairs whose two cells differ in cell type (heterotypic) are
    ignored; random pairs are drawn from the same cell type of the same sample.
  * normalized version: every distance of a group (sample / sample x cell type) is divided by that group's median clonal-pair
    distance (all PCs), so median clonal = 1; saved as column dist_norm and as *_normalized plots for merging across experiments.
  * at most CAP clonal pairs per group (random choice weighted by clone pair count); AUC = P(clonal distance < random distance).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_common import FIN, OUT, ROOT, REGISTRY, PETRACE, read_h5ad_obs_var

PCA_DIR = f"{OUT}/pca5k"
PD_DIR = f"{OUT}/pair_distance"
CAP = 50_000          # max clonal pairs per group
KEEP = 20_000         # pair-level rows saved per group and pair type
MIN_PAIRS = 20        # min homotypic clonal pairs for a (sample, cell type) group
MAX_PANELS = 24
EXCLUDE_FROM_COMBINED = {"hPSC_20260927",                       # removed from the analysis on request
                         "PeTrace_E7.5-R1", "PeTrace_E8.0-R2"}   # old per-replicate tree-level test runs       # cell-type panels per figure (the rest stay in the CSV)
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- labels
def _qc(ds_dir, extra=None):
    o = pd.read_csv(f"{FIN}/{ds_dir}/qc_filtered/obs_metadata.csv", index_col=0)
    return o


def meta(name):
    """DataFrame indexed by cell id with columns sample, clone, celltype (celltype may be NaN everywhere)."""
    if name in ("FM01", "FM03", "FM05", "FM06", "FM07", "FM08"):
        o = _qc(f"{name}_data")
        df = pd.DataFrame({"sample": name, "clone": o["fatemap_clone_singletcode"]}, index=o.index)  # replicates A/B = one sample
        df["celltype"] = np.nan
        if name in ("FM01", "FM06", "FM08"):
            ob, _ = read_h5ad_obs_var(f"{FIN}/{name}_data/{name}_integrated.h5ad")
            df["celltype"] = ob["cluster"].astype(str).reindex(df.index).to_numpy()   # Goyal et al. cluster labels
        return df
    if name == "HS054_Sot48h":
        o = _qc("HS054_Sot48h_data")
        return pd.DataFrame({"sample": name, "clone": o["fatemap_clone_singletcode"], "celltype": np.nan}, index=o.index)
    if name.startswith("Watermelon_"):
        o = _qc(f"{name}_data")
        return pd.DataFrame({"sample": o["sample"], "clone": o["fatemap_clone_singletcode"], "celltype": np.nan}, index=o.index)
    if name == "hPSC_20260927":
        o = _qc("hPSC_20260927_data")
        return pd.DataFrame({"sample": o["sample"], "clone": o["fatemap_clone_singletcode"], "celltype": np.nan}, index=o.index)
    if name in ("hPSC_endo_yscher", "hPSC_undifferentiated"):
        f = (f"{FIN}/hPSC_endo_yscher_data/endo_kbnac_spliced_unspliced.h5ad" if name == "hPSC_endo_yscher"
             else f"{FIN}/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad")
        o, _ = read_h5ad_obs_var(f)
        return pd.DataFrame({"sample": o["timepoint"].astype(str), "clone": o["clone_c5"].astype(object), "celltype": np.nan}, index=o.index)
    if name == "hPSC_Novo_Nordisk":
        fr = []
        for s in ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]:
            o = pd.read_csv(f"{FIN}/hPSC_Novo_Nordisk_data/qc_filtered/{s}/obs_metadata.csv", index_col=0)
            fr.append(pd.DataFrame({"sample": s, "clone": o["fatemap_clone_singletcode"].astype(str),
                                    "celltype": o["clust_annot_02"].astype(str)}, index=o.index))
        return pd.concat(fr)
    if name == "LARRY":
        o, _ = read_h5ad_obs_var(f"{FIN}/LARRY_data/annotated/filtered_annotated_k15.h5ad")
        ids = [f"{a}:{b[:8]}-{b[8:]}" for a, b in (i.split("|") for i in o.index)]
        # all libraries of a day are one sample (LSK_d2_1..3 -> day2, LSK_d4_* -> day4, LSK_d6_* -> day6); clone = barcode string, shared across libraries
        day = o["library"].astype(str).str.extract(r"_d(\d+)_")[0].map(lambda d: f"day{d}")
        return pd.DataFrame({"sample": day.to_numpy(), "clone": o["larry_clone_singletcode"].astype(str).to_numpy(),
                             "celltype": o["Cell type annotation"].astype(str).to_numpy()}, index=ids)
    if name == "Michaels_T1T2":
        c = pd.read_csv(f"{FIN}/Michaels_T1T2_data/cell_to_clone.csv")
        ids = [f"{s}|{b}-1" for s, b in zip(c["sample"], c["cell_barcode"])]
        lab = pd.Series(c["clone_id"].astype(str).to_numpy(), index=ids)
        z = np.load(f"{PCA_DIR}/Michaels_T1T2_pca5k.npz", allow_pickle=True)
        cells = pd.Index(z["cells"].astype(str))
        return pd.DataFrame({"sample": [x.split("|")[0] for x in cells], "clone": lab.reindex(cells).to_numpy(), "celltype": np.nan}, index=cells)
    if name == "CellTag3":
        o, _ = read_h5ad_obs_var(f"{ROOT}/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad")
        return pd.DataFrame({"sample": "day" + o["Time point"].astype(str), "clone": o["repaired_clone"].astype(float),
                             "celltype": o["Cell type annotation"].astype(str)}, index=o.index)
    if name == "Schaffer_naive":   # Naive1-3 merged into one sample; clone = singletCode clone (analysis_data/stage1_inputs/schaffer/obs_metadata.csv)
        o = pd.read_csv(f"{ROOT}/analysis_data/stage1_inputs/schaffer/obs_metadata.csv", index_col=0)
        return pd.DataFrame({"sample": "Schaffer_naive", "clone": o["fatemap_clone_singletcode"].astype(str), "celltype": np.nan}, index=o.index)
    if name == "SpaceBar":
        fr = []
        for i in range(1, 6):
            o = pd.read_csv(f"{ROOT}/real_data/SpaceBar_data/Section{i}_cell_by_gene_clustered.csv", usecols=["object_id", "bc_cluster"], index_col="object_id")
            fr.append(pd.DataFrame({"sample": f"Section{i}", "clone": o["bc_cluster"].astype(float).to_numpy(), "celltype": np.nan},
                                   index=[f"Section{i}:{c}" for c in o.index]))
        return pd.concat(fr)
    if name.startswith("PeTrace_"):
        tag = name[8:]
        tags = [f"{tag}-R{r}" for r in (1, 2, 3)] if "-R" not in tag else [tag]     # 'E9.5' = R1+R2+R3 merged
        fr = []
        for t in tags:
            o, _ = read_h5ad_obs_var(f"{ROOT}/real_data/PeTrace/{t}.h5td")
            fr.append(pd.DataFrame({"sample": tag, "clone": o["clone"].astype(str), "celltype": o["cell_type"].astype(str)}, index=o.index))
        m = pd.concat(fr)
        assert m.index.is_unique
        return m
    raise KeyError(name)


def dtt_pairs():
    """DTT: no clone ids; clonal pairs = strict sister pairs (cell_A, cell_B) with their cell types."""
    sp_ = pd.read_csv(f"{FIN}/DTT_mouse_data/sister_pairs_strict.csv")
    return sp_


# ----------------------------------------------------------------------------- pair sampling
def clonal_pairs(clones, cap):
    """clones: array of clone ids (one per cell, NaN/None = unlabeled) -> (i, j) index arrays of clonal pairs (capped)."""
    s = pd.Series(np.arange(len(clones)))
    g = [v.to_numpy() for _, v in s.groupby(pd.Series(clones).astype(str)) if len(v) >= 2]
    g = [x for x in g]
    if not g:
        return np.array([], int), np.array([], int), 0, 0
    npairs = np.array([len(x) * (len(x) - 1) // 2 for x in g], dtype=float)
    total = int(npairs.sum())
    if total <= cap:
        I, J = [], []
        for x in g:
            a, b = np.triu_indices(len(x), 1)
            I.append(x[a]); J.append(x[b])
        return np.concatenate(I), np.concatenate(J), total, len(g)
    pick = rng.choice(len(g), size=cap, p=npairs / npairs.sum())
    I, J = np.empty(cap, int), np.empty(cap, int)
    for k, c in enumerate(pick):
        a, b = rng.choice(len(g[c]), size=2, replace=False)
        I[k], J[k] = g[c][a], g[c][b]
    return I, J, total, len(g)


def random_pairs(clones, n):
    """n pairs of distinct cells from different clones."""
    clones = np.asarray(pd.Series(clones).astype(str))
    N = len(clones)
    I, J = [], []
    need = n
    for _ in range(50):
        a = rng.integers(0, N, size=need * 2)
        b = rng.integers(0, N, size=need * 2)
        ok = (a != b) & (clones[a] != clones[b])
        I.append(a[ok]); J.append(b[ok])
        if sum(len(x) for x in I) >= n:
            break
    I, J = np.concatenate(I)[:n], np.concatenate(J)[:n]
    return I, J


def auc_less(c, r):
    u = mannwhitneyu(c, r, alternative="less")
    return 1.0 - u.statistic / (len(c) * len(r)), u.pvalue    # P(clonal < random)


def group_stats(P, K, idx, clones, label):
    """idx: indices (into P) of the group's cells. -> (summary dict, pair-level DataFrame) or None."""
    cl = np.asarray(clones)[idx]
    I, J, total, nclones = clonal_pairs(cl, CAP)
    if len(I) < MIN_PAIRS:
        return None
    n = len(I)
    Ri, Rj = random_pairs(cl, n)
    if len(np.unique(np.asarray(pd.Series(cl).astype(str)))) < 2 or len(Ri) < MIN_PAIRS:
        return None            # no cells from other clones in this group -> no random baseline
    n = min(n, len(Ri))
    I, J = I[:n], J[:n]
    Ri, Rj = Ri[:n], Rj[:n]
    X = P[idx]
    dK = lambda a, b: np.linalg.norm(X[a] - X[b], axis=1)   # primary: all PCs
    dE = lambda a, b: np.linalg.norm(X[a][:, :KE] - X[b][:, :KE], axis=1)
    cK, rK, cE, rE = dK(I, J), dK(Ri, Rj), dE(I, J), dE(Ri, Rj)
    a, p = auc_less(cK, rK)
    a50, p50 = auc_less(cE, rE)
    s = dict(n_cells_with_clone=int(len(idx)), n_clones_ge2=int(nclones), n_clonal_pairs_available=int(total), n_pairs_used=int(n),
             K=int(K), K_elbow=int(KE), median_dist_clonal=float(np.median(cK)), median_dist_random=float(np.median(rK)),
             ratio_clonal_over_random=float(np.median(cK) / np.median(rK)), auc_clonal_closer=float(a), p_mwu_less=float(p),
             median_dist_clonal_elbowK=float(np.median(cE)), median_dist_random_elbowK=float(np.median(rE)),
             auc_clonal_closer_elbowK=float(a50), p_mwu_less_elbowK=float(p50))
    k = min(KEEP, n)
    sel = rng.choice(n, size=k, replace=False)
    mc = float(np.median(cK))
    s["norm_median_random"] = float(np.median(rK) / mc)     # median random distance when median clonal = 1
    pairs = pd.concat([pd.DataFrame({"pair_type": "clonal", "dist_K": cK[sel], "dist_elbowK": cE[sel], "dist_norm": cK[sel] / mc}),
                       pd.DataFrame({"pair_type": "random", "dist_K": rK[sel], "dist_elbowK": rE[sel], "dist_norm": rK[sel] / mc})])
    return s, pairs



# ----------------------------------------------------------------------------- PeTrace: sister pairs from the lineage trees
PETRACE_DAYS = ["E7.5", "E8.0", "E8.5", "E9.0", "E9.5"]


def petrace_sister_pairs(day, cells):
    """The `clone` column of the .h5td files is a whole lineage TREE (E9.5-R1: 382k cells in 3 trees), not a clone. The relationship between
    cells is inside the tree (obst): leaf cells that are children of the same parent node are sisters. Returns (a, b, group) index arrays into
    `cells` (all pairs among the leaf cells of each parent node, replicates R1-R3 of the day pooled) plus the number of sister groups."""
    import h5py
    pos = pd.Series(np.arange(len(cells)), index=cells)
    A, Bv, G = [], [], []
    gid = 0
    for r in (1, 2, 3):
        with h5py.File(f"{ROOT}/real_data/PeTrace/{day}-R{r}.h5td", "r") as f:
            trees = json.loads(f["obst"][()].decode())
        for tname, t in trees.items():
            for par, ch in t["edges"].items():
                lv = [pos[c] for c in ch if c in pos.index]
                if len(lv) < 2:
                    continue
                lv = np.asarray(lv)
                i, j = np.triu_indices(len(lv), 1)
                A.append(lv[i]); Bv.append(lv[j]); G.append(np.full(len(i), gid, dtype=np.int64))
                gid += 1
        del trees
    return np.concatenate(A), np.concatenate(Bv), np.concatenate(G), gid


def analyse_petrace_sisters(name):
    day = name[len("PeTrace_"):]
    global KE
    z = np.load(f"{PCA_DIR}/{name}_pca5k.npz", allow_pickle=True)
    cells, P = z["cells"].astype(str), z["pca"].astype(np.float32)
    KE = max(5, int(json.load(open(f"{PCA_DIR}/summary_{name}.json"))["elbow_pc"]))
    K = P.shape[1]
    ct = pdist_meta_celltype(name, cells)
    a, b, g, ngroups = petrace_sister_pairs(day, cells)
    print(f"[{name}] {len(a):,} sister pairs in {ngroups:,} sister groups; {len(np.unique(np.concatenate([a, b]))):,} cells in a sister group", flush=True)
    groups = [("ALL", np.arange(len(a)), np.arange(len(cells)))]
    homo = ct[a] == ct[b]
    for c in pd.unique(ct[np.concatenate([a[homo], b[homo]])]):
        sel = np.flatnonzero(homo & (ct[a] == c))
        groups.append((c, sel, np.flatnonzero(ct == c)))          # heterotypic sister pairs ignored; random pairs from all cells of that type
    rows, pair_frames = [], []
    for c, sel, pool in groups:
        if len(sel) < MIN_PAIRS or len(pool) < 2:
            continue
        navail = len(sel)
        used = sel if navail <= CAP else rng.choice(sel, size=CAP, replace=False)      # random subsample of ALL sister pairs
        n = len(used)
        ra, rb = rng.choice(pool, size=n * 2), rng.choice(pool, size=n * 2)
        ok = ra != rb
        ra, rb = ra[ok][:n], rb[ok][:n]
        n = min(n, len(ra))
        aa, bb = a[used][:n], b[used][:n]
        cK, rK = np.linalg.norm(P[aa] - P[bb], axis=1), np.linalg.norm(P[ra] - P[rb], axis=1)
        cE, rE = np.linalg.norm(P[aa, :KE] - P[bb, :KE], axis=1), np.linalg.norm(P[ra, :KE] - P[rb, :KE], axis=1)
        au, p = auc_less(cK, rK)
        a5, p5 = auc_less(cE, rE)
        mc = float(np.median(cK))
        rows.append(dict(dataset=name, sample=day, celltype=c, n_cells_with_clone=int(len(np.unique(np.concatenate([a[sel], b[sel]])))),
                         n_clones_ge2=int(len(np.unique(g[sel]))), n_clonal_pairs_available=int(navail), n_pairs_used=int(n), K=K, K_elbow=KE,
                         median_dist_clonal=mc, median_dist_random=float(np.median(rK)), ratio_clonal_over_random=float(mc / np.median(rK)),
                         auc_clonal_closer=float(au), p_mwu_less=float(p), median_dist_clonal_elbowK=float(np.median(cE)),
                         median_dist_random_elbowK=float(np.median(rE)), auc_clonal_closer_elbowK=float(a5), p_mwu_less_elbowK=float(p5),
                         note="clonal pair = sister cells (leaves under the same parent node of the lineage tree); n_clones_ge2 = number of sister groups",
                         norm_median_random=float(np.median(rK) / mc)))
        k = min(KEEP, n)
        pair_frames.append(pd.concat([pd.DataFrame({"pair_type": "clonal", "dist_K": cK[:k], "dist_elbowK": cE[:k], "dist_norm": cK[:k] / mc}),
                                      pd.DataFrame({"pair_type": "random", "dist_K": rK[:k], "dist_elbowK": rE[:k], "dist_norm": rK[:k] / mc})]).assign(dataset=name, sample=day, celltype=c))
    return rows, pair_frames


def pdist_meta_celltype(name, cells):
    """cell type of every cell of a (merged) PeTrace day, aligned to `cells`"""
    day = name[len("PeTrace_"):]
    fr = []
    for r in (1, 2, 3):
        o, _ = read_h5ad_obs_var(f"{ROOT}/real_data/PeTrace/{day}-R{r}.h5td")
        fr.append(o["cell_type"].astype(str))
    return pd.concat(fr).reindex(cells).to_numpy()


# ----------------------------------------------------------------------------- run
def analyse(name):
    z = np.load(f"{PCA_DIR}/{name}_pca5k.npz", allow_pickle=True)
    cells, P = z["cells"].astype(str), z["pca"].astype(np.float32)
    global KE
    KE = max(5, int(json.load(open(f"{PCA_DIR}/summary_{name}.json"))["elbow_pc"]))   # secondary distance only
    K = P.shape[1]                                                                      # primary: all PCs
    rows, pair_frames = [], []
    if name.startswith("PeTrace_") and name[len("PeTrace_"):] in PETRACE_DAYS:
        rows, pair_frames = analyse_petrace_sisters(name)
    elif name == "DTT_mouse":
        sis = dtt_pairs()
        pos = pd.Series(np.arange(len(cells)), index=cells)
        sis = sis[sis["cell_A"].isin(pos.index) & sis["cell_B"].isin(pos.index)].copy()
        sis["sample"] = sis["cell_A"].str.split("_").str[0]
        m = pd.DataFrame({"sample": [c.split("_")[0] for c in cells]}, index=cells)
        for s_, g in sis.groupby("sample"):
            idx_s = np.flatnonzero(m["sample"].to_numpy() == s_)
            groups = [("ALL", g)] + [(ct, gg) for ct, gg in g[g["same_type"]].groupby("celltype_A")]
            for ct, gg in groups:
                if len(gg) < MIN_PAIRS:
                    continue
                gg = gg.iloc[:CAP]
                a, b = pos[gg["cell_A"]].to_numpy(), pos[gg["cell_B"]].to_numpy()
                n = len(a)
                if ct == "ALL":
                    pool = idx_s
                else:   # random cells of the same cell type in that experiment: cell type of final cells from sister file
                    tmap = pd.concat([g["cell_A"].rename("c").to_frame().assign(t=g["celltype_A"]), g["cell_B"].rename("c").to_frame().assign(t=g["celltype_B"])])
                    ids = tmap[tmap["t"] == ct]["c"].unique()
                    pool = pos[ids].to_numpy()
                ra, rb = rng.choice(pool, size=n * 2), rng.choice(pool, size=n * 2)
                ok = ra != rb
                ra, rb = ra[ok][:n], rb[ok][:n]
                n = min(n, len(ra))
                cK, rK = np.linalg.norm(P[a[:n]] - P[b[:n]], axis=1), np.linalg.norm(P[ra[:n]] - P[rb[:n]], axis=1)
                cE, rE = np.linalg.norm(P[a[:n], :KE] - P[b[:n], :KE], axis=1), np.linalg.norm(P[ra[:n], :KE] - P[rb[:n], :KE], axis=1)
                au, p = auc_less(cK, rK)
                a5, p5 = auc_less(cE, rE)
                rows.append(dict(dataset=name, sample=s_, celltype=ct, n_cells_with_clone=len(pool), n_clones_ge2=n, n_clonal_pairs_available=len(gg),
                                 n_pairs_used=n, K=K, K_elbow=KE, median_dist_clonal=float(np.median(cK)), median_dist_random=float(np.median(rK)),
                                 ratio_clonal_over_random=float(np.median(cK) / np.median(rK)), auc_clonal_closer=float(au), p_mwu_less=float(p),
                                 median_dist_clonal_elbowK=float(np.median(cE)), median_dist_random_elbowK=float(np.median(rE)),
                                 auc_clonal_closer_elbowK=float(a5), p_mwu_less_elbowK=float(p5), note="clone = strict sister pair"))
                k = min(KEEP, n)
                mc = float(np.median(cK))
                rows[-1]["norm_median_random"] = float(np.median(rK) / mc)
                pair_frames.append(pd.concat([pd.DataFrame({"pair_type": "clonal", "dist_K": cK[:k], "dist_elbowK": cE[:k], "dist_norm": cK[:k] / mc}),
                                              pd.DataFrame({"pair_type": "random", "dist_K": rK[:k], "dist_elbowK": rE[:k], "dist_norm": rK[:k] / mc})]).assign(dataset=name, sample=s_, celltype=ct))
    else:
        m = meta(name).reindex(cells)
        assert m["sample"].notna().all(), f"{name}: {m['sample'].isna().sum()} PCA cells without metadata"
        has_ct = m["celltype"].notna().any()
        lab = m["clone"].astype(object).where(m["clone"].notna(), None)
        for s_ in pd.unique(m["sample"]):
            ins = np.flatnonzero((m["sample"] == s_).to_numpy() & lab.notna().to_numpy())
            if len(ins) < 2:
                continue
            res = group_stats(P, K, ins, lab.to_numpy(), "ALL")
            if res:
                rows.append(dict(dataset=name, sample=s_, celltype="ALL", **res[0], note=""))
                pair_frames.append(res[1].assign(dataset=name, sample=s_, celltype="ALL"))
            if has_ct:
                for ct in pd.unique(m["celltype"].iloc[ins]):
                    sub = ins[(m["celltype"].iloc[ins] == ct).to_numpy()]
                    res = group_stats(P, K, sub, lab.to_numpy(), ct)    # homotypic pairs only (both cells of this cell type)
                    if res:
                        rows.append(dict(dataset=name, sample=s_, celltype=ct, **res[0], note="homotypic pairs only"))
                        pair_frames.append(res[1].assign(dataset=name, sample=s_, celltype=ct))
    os.makedirs(PD_DIR, exist_ok=True)
    if not rows:
        print(f"[{name}] no group with >= {MIN_PAIRS} clonal pairs", flush=True)
        return
    S = pd.DataFrame(rows)
    S.to_csv(f"{PD_DIR}/.{name}_s.tmp.csv", index=False)
    os.replace(f"{PD_DIR}/.{name}_s.tmp.csv", f"{PD_DIR}/{name}_pair_distance_summary.csv")
    pd.concat(pair_frames).to_csv(f"{PD_DIR}/{name}_pair_distances.csv.gz", index=False)
    plot(name, S, pd.concat(pair_frames))
    print(f"[{name}] {len(S)} groups -> {PD_DIR}", flush=True)


# ----------------------------------------------------------------------------- plots (style of plot.ipynb, Figure 5b)
import matplotlib.font_manager as fm
import seaborn as sns

FONT_DIR = f"{ROOT}/code/TwINFER/package/twinfer/plotting/assets/fonts"
for _f in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"):
    try:
        fm.fontManager.addfont(f"{FONT_DIR}/{_f}")
    except Exception as e:
        print("font not loaded:", _f, e)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.sans-serif": ["Arial"], "font.family": "sans-serif",
                     "svg.fonttype": "none", "mathtext.fontset": "cm", "axes.labelsize": 18, "axes.titlesize": 16,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
YLAB = "Transcriptomic distance [a.u.]"
YLAB_NORM = "Transcriptomic distance\n(median clonal pair = 1)"
PALETTE = {"clonal pairs": "lightgray", "random pairs": "gray"}


def get_significance_symbol(pval):
    if pval > 0.05:
        return "ns"
    elif pval <= 0.0001:
        return "****"
    elif pval <= 0.001:
        return "***"
    elif pval <= 0.01:
        return "**"
    return "*"


def add_stat_annotation(ax, label, ymax, y_offset=0.05):
    """significance bracket above the two boxes (as in plot.ipynb)"""
    h = ymax * (1 + y_offset)
    ax.plot([0, 0, 1, 1], [h, h * 1.05, h * 1.05, h], lw=1.5, color="black", clip_on=False)
    ax.text(0.5, h * 1.07, label, ha="center", va="bottom", fontsize=12, clip_on=False)


def _panel(ax, d, title, ylim_top, col="dist_K", ylab=None):
    """d: DataFrame with columns dist, pair_type ('clonal'/'random')"""
    d = d.assign(pair_type=d["pair_type"].map({"clonal": "clonal pairs", "random": "random pairs"}), **{YLAB: d[col]})
    sns.boxplot(data=d, x="pair_type", y=YLAB, hue="pair_type", order=list(PALETTE), hue_order=list(PALETTE), palette=PALETTE,
                showfliers=False, ax=ax)     # outlier points hidden: they overlapped the significance bracket (stats use all pairs)
    sns.despine(right=True, top=True, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylab or YLAB)
    c, r = d[d["pair_type"] == "clonal pairs"][YLAB], d[d["pair_type"] == "random pairs"][YLAB]
    _, p = mannwhitneyu(c, r, alternative="two-sided")
    ax.set_ylim(0, ylim_top)
    add_stat_annotation(ax, get_significance_symbol(p), ylim_top / 1.25)
    ax.set_title(title, pad=30, fontsize=14)
    return p


def _save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", format=ext, bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    fig.savefig(f"{stem}.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _grid(n, ncol_max=4, w=4.6, h=6.0):
    ncol = min(n, ncol_max)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), sharey=True, squeeze=False)
    return fig, axes.ravel(), nrow


def _legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.8) for c in PALETTE.values()]
    fig.legend(handles, list(PALETTE), loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)


def _ytop(parts, col="dist_K"):
    """shared y limit: highest upper whisker (Q3 + 1.5 IQR, capped at the max) over all boxes, +25% headroom for the bracket."""
    top = 0.0
    for p in parts:
        for _, g in p.groupby("pair_type"):
            q1, q3 = np.percentile(g[col], [25, 75])
            top = max(top, min(q3 + 1.5 * (q3 - q1), g[col].max()))
    return float(top) * 1.25


def plot(name, S, pairs):
    """raw version (distance in PCA units) and a normalized version (distance / median clonal-pair distance of the group)."""
    for col, ylab, suf in (("dist_K", None, ""), ("dist_norm", YLAB_NORM, "_normalized")):
        samp = S[S["celltype"] == "ALL"]
        if len(samp):
            fig, axes, nrow = _grid(len(samp))
            parts = [pairs[(pairs["sample"] == r.sample) & (pairs["celltype"] == "ALL")] for r in samp.itertuples()]
            top = _ytop(parts, col)
            for ax, r, d in zip(axes, samp.itertuples(), parts):
                _panel(ax, d, f"{r.sample}\n{r.n_pairs_used:,} pairs", top, col, ylab)
            for ax in axes[len(samp):]:
                ax.axis("off")
            _legend(fig)
            plt.tight_layout(rect=[0, 0, 1, 0.9 if nrow == 1 else 0.94])
            _save(fig, f"{PD_DIR}/pair_distance_{name}_samples{suf}")
        ct = S[S["celltype"] != "ALL"]
        for s in pd.unique(ct["sample"]):
            sub = ct[ct["sample"] == s].sort_values("n_pairs_used", ascending=False)
            n_all = len(sub)
            sub = sub.head(MAX_PANELS)
            big = len(sub) > 12
            fig, axes, nrow = _grid(len(sub), ncol_max=6 if big else 4, w=3.4 if big else 4.2, h=5.0 if big else 6.0)
            parts = [pairs[(pairs["sample"] == s) & (pairs["celltype"] == r.celltype)] for r in sub.itertuples()]
            top = _ytop(parts, col)
            for ax, r, d in zip(axes, sub.itertuples(), parts):
                lab = r.celltype if len(r.celltype) < 26 else r.celltype[:24] + "…"
                _panel(ax, d, f"{lab}\n{r.n_pairs_used:,} pairs", top, col, ylab)
            for ax in axes[len(sub):]:
                ax.axis("off")
            _legend(fig)
            more = f" - {len(sub)} of {n_all} cell types with most pairs shown (all in CSV)" if n_all > MAX_PANELS else ""
            fig.suptitle(f"{name} / {s}: same-cell-type pairs (heterotypic clonal pairs ignored){more}", y=0.995, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.86 if nrow == 1 else 0.92])
            _save(fig, f"{PD_DIR}/pair_distance_{name}_{s}_celltypes{suf}".replace(" ", "_"))


def replot(name):
    """redraw the plots of one dataset from its saved CSVs (no recomputation)"""
    S = pd.read_csv(f"{PD_DIR}/{name}_pair_distance_summary.csv", keep_default_na=False)
    pairs = pd.read_csv(f"{PD_DIR}/{name}_pair_distances.csv.gz", keep_default_na=False)
    S["n_pairs_used"] = S["n_pairs_used"].astype(int)
    plot(name, S, pairs)
    print(f"[{name}] replotted", flush=True)


def combine():
    fr = [pd.read_csv(f"{PD_DIR}/{f}") for f in sorted(os.listdir(PD_DIR)) if f.endswith("_pair_distance_summary.csv") and not f.startswith("ALL") and f[:-len("_pair_distance_summary.csv")] not in EXCLUDE_FROM_COMBINED]
    if fr:
        pd.concat(fr).to_csv(f"{PD_DIR}/.ALL_tmp.csv", index=False)
        os.replace(f"{PD_DIR}/.ALL_tmp.csv", f"{PD_DIR}/ALL_datasets_pair_distance_summary.csv")
        print(f"combined {len(fr)} datasets", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    skip_combine = "--no-combine" in a          # parallel runs: combine once at the end
    a = [x for x in a if x != "--no-combine"]
    if a and a[0] == "replot":
        for n in a[1:]:
            replot(n)
    elif a == ["combine"]:
        combine()
    else:
        for n in a:
            analyse(n)
        if not skip_combine:
            combine()
