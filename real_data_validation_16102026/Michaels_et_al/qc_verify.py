#!/usr/bin/env python3
"""Verify a finalized dataset against its raw data and plot the QC metrics.

usage: qc_verify.py <dataset> [<dataset> ...]
Writes analysis_data/dataset_qc_verification/<dataset>/{verify.json, raw_cells_qc.csv.gz, qc_<dataset>.png, funnel_<dataset>.png}

Nothing is taken from the build scripts' summaries: raw metrics, final metrics, filter outcomes and every
stated cutoff are recomputed here, and the summaries' numbers are listed next to the recomputed ones.
Datasets without a raw matrix in REGISTRY (cfg['raw'] is None) get final-only metrics (see final_only()).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_common import (OUT, REGISTRY, final_counts, load_summary, qc_metrics, read_h5ad_csr,
                       read_h5ad_obs_var, stated)

rng = np.random.default_rng(0)


def check_integrated_h5ad(path, X_final, cells):
    """Does the integrated h5ad hold the same counts as the qc_filtered matrix, and is X plain log1p(CP10k)?"""
    import h5py
    out = {"file": path}
    obs, var = read_h5ad_obs_var(path)
    out["n_obs"], out["n_var"] = len(obs), len(var)
    out["obs_names_equal_cells_txt"] = bool(list(obs.index) == list(cells))
    C = read_h5ad_csr(path, "layers/counts")
    out["counts_layer_equals_qc_mtx"] = bool(C.shape == X_final.shape and (C != X_final).nnz == 0)
    X = read_h5ad_csr(path, "X")
    idx = rng.choice(X.shape[0], size=min(500, X.shape[0]), replace=False)
    sums = np.asarray(np.expm1(X[idx]).sum(axis=1)).ravel()
    out["X_expm1_rowsum_min_max_(expect_1e4)"] = [float(sums.min()), float(sums.max())]
    L = read_h5ad_csr(path, "layers/lognorm")
    out["X_equals_lognorm_layer"] = bool(abs(X[idx] - L[idx]).max() < 1e-5)
    sub = C[idx].astype(float)
    tot = np.asarray(sub.sum(axis=1)).ravel()
    ref = sub.multiply(1e4 / tot[:, None]).tocsr()
    ref.data = np.log1p(ref.data)
    out["X_equals_recomputed_log1p_CP10k_maxabs"] = float(abs(X[idx] - ref).max())
    out["n_hvg_flag"] = int(var["highly_variable"].sum()) if "highly_variable" in var else None
    with h5py.File(path, "r") as f:
        out["obsm"] = list(f["obsm"].keys())
        out["X_pca_shape"] = list(f["obsm/X_pca"].shape) if "X_pca" in f["obsm"] else None
        if "uns/pca/variance_ratio" in f:
            out["pca_n_pcs_stored_variance_ratio"] = int(f["uns/pca/variance_ratio"].shape[0])
    return out


def _row_check(Xr, rgenes, Xf, genes, rbc_keys, cells, fin_keys, sel):
    """Are the final rows identical to the raw rows (on the final genes)?"""
    rgenes, genes = np.asarray(rgenes), np.asarray(genes)
    if len(rgenes) == len(genes):
        dup = pd.Series(rgenes).duplicated(keep=False).to_numpy()   # pipeline renamed duplicated symbols
        if bool((rgenes == genes)[~dup].all()):
            gi, note = slice(None), f"same gene order ({int(dup.sum())} duplicated symbols renamed)"
        else:
            return None, "gene lists differ"
    else:
        ix = pd.Index(rgenes)
        if not ix.is_unique:
            return None, f"gene lists differ (raw {len(rgenes)} vs final {len(genes)}, raw symbols not unique)"
        gi = ix.get_indexer(genes)
        miss = int((gi < 0).sum())
        note = f"final genes are a subset of raw ({len(genes) - miss}/{len(rgenes)} raw genes kept; {miss} final genes absent from this raw matrix, compared on the rest)"
        fcols = np.flatnonzero(gi >= 0)
        gi = gi[fcols]
    kk = pd.Series(np.arange(len(rbc_keys)), index=rbc_keys)
    fpos = pd.Series(np.arange(len(cells)), index=cells)
    same = 0
    for i in sel:
        rr = Xr[kk[fin_keys[i][1]]]
        fr = Xf[fpos[fin_keys[i][0]]]
        if not isinstance(gi, slice):
            rr, fr = rr[:, gi], fr[:, fcols]
        same += int((rr != fr).nnz == 0)
    return same, note


def final_only(name):
    """No raw matrix available here: recompute the QC metrics from the final matrix only and plot them."""
    cfg = REGISTRY[name]
    od = f"{OUT}/{name}"
    os.makedirs(od, exist_ok=True)
    Xf, cells, genes, obs, symbols = final_counts(name)
    res = {"dataset": name, "mode": "final-only (no raw matrix available locally)", "n_cells": int(Xf.shape[0]),
           "n_genes_in_matrix": int(Xf.shape[1])}
    d = Xf.data[: min(len(Xf.data), 5_000_000)]
    res["values_are_integer_counts"] = bool(np.allclose(d, np.round(d)))
    res["min_value"] = float(Xf.data.min()); res["max_value"] = float(Xf.data.max())
    ng = np.asarray((Xf > 0).sum(axis=1)).ravel()
    m = pd.DataFrame({"n_genes": ng, "sample": obs["sample"].astype(str).to_numpy()}, index=cells)
    if res["values_are_integer_counts"]:
        mm, nmt = qc_metrics(Xf, symbols, cfg["mito_prefix"]) if any(
            str(g).upper().startswith(cfg["mito_prefix"].upper()) for g in symbols) else (None, 0)
        m["total_counts"] = np.asarray(Xf.sum(axis=1)).ravel()
        if mm is not None:
            m["pct_counts_mt"] = mm["pct_counts_mt"].to_numpy()
        res["n_mito_genes"] = nmt
    else:
        res["note"] = "matrix is not raw counts (normalized values): total counts and mito % cannot be recomputed"
    stored = [c for c in ("n_genes", "total_counts", "pct_counts_mt", "n_genes_by_counts") if c in obs.columns]
    res["stored_metric_columns_in_obs"] = stored
    res["stored_vs_recomputed_maxabs_diff"] = {c: float(np.abs(obs[c].astype(float).to_numpy() - m[c].to_numpy()).max())
                                               for c in ("n_genes", "total_counts", "pct_counts_mt") if c in obs.columns and c in m.columns}
    res["per_sample"] = {s: {c: [float(g[c].min()), float(g[c].median()), float(g[c].max())]
                              for c in m.columns if c != "sample"} | {"n_cells": int(len(g))}
                         for s, g in m.groupby("sample")}
    res["note_columns"] = "per-sample entries are [min, median, max]"
    json.dump(res, open(f"{od}/verify.json", "w"), indent=1, default=str)
    m.to_csv(f"{od}/final_cells_qc.csv.gz")
    cols = [c for c in ("total_counts", "n_genes", "pct_counts_mt") if c in m.columns]
    fig, ax = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.6), squeeze=False)
    for j, c in enumerate(cols):
        v = np.log10(np.clip(m[c], 1, None)) if c == "total_counts" else m[c]
        for s, g in m.groupby("sample"):
            vv = np.log10(np.clip(g[c], 1, None)) if c == "total_counts" else g[c]
            ax[0, j].hist(vv, bins=60, histtype="step", label=str(s) if m["sample"].nunique() <= 8 else None)
        ax[0, j].set_xlabel("log10 UMI per cell" if c == "total_counts" else c); ax[0, j].set_ylabel("cells")
        if m["sample"].nunique() <= 8:
            ax[0, j].legend(fontsize=6)
    fig.suptitle(f"{name}: final cells only (no raw data to compare), n={len(m):,}")
    fig.tight_layout(); fig.savefig(f"{od}/qc_{name}.png", dpi=110); plt.close(fig)
    print(f"[{name}] final-only done -> {od}", flush=True)


def run(name):
    cfg = REGISTRY[name]
    if cfg.get("raw") is None:
        return final_only(name)
    od = f"{OUT}/{name}"
    os.makedirs(od, exist_ok=True)
    res = {"dataset": name, "stated_cutoffs": {k: stated(cfg, k) for k in cfg["stated"]}}

    # ---------------------------------------------------------------- final cells, recomputed
    Xf, cells, genes, obs, fsymbols = final_counts(name)
    prefix = cfg["mito_prefix"]
    fm, n_mt_genes = qc_metrics(Xf, fsymbols, prefix)
    fm.index = cells
    res["n_mito_genes_final_matrix"] = n_mt_genes
    res["n_final_cells"], res["n_genes_in_matrix"] = int(Xf.shape[0]), int(Xf.shape[1])
    res["counts_are_integers"] = bool(np.allclose(Xf.data, np.round(Xf.data)))
    res["counts_min"] = float(Xf.data.min())
    scol = {"n_genes": "n_genes" if "n_genes" in obs else "n_genes_by_counts", "total_counts": "total_counts",
            "pct_counts_mt": "pct_counts_mt"}
    res["stored_vs_recomputed_maxabs_diff"] = {
        k: float(np.abs(obs[v].astype(float).to_numpy() - fm[k].to_numpy()).max()) for k, v in scol.items() if v in obs}
    fm["sample"] = obs["sample"].astype(str).to_numpy()
    kf = cfg.get("key_of_final", lambda c: c)         # final cell id -> '<sample>:<barcode>'
    fm["key"] = [kf(c) for c in cells]

    hs = [cfg["h5ad"]] if cfg.get("h5ad") else cfg.get("stage_h5ads", [])
    res["integrated_h5ad_checks"] = []
    for h in hs:
        if not os.path.exists(h):
            res["integrated_h5ad_checks"].append({"file": h, "missing": True})
        elif cfg.get("h5ad"):
            res["integrated_h5ad_checks"].append(check_integrated_h5ad(h, Xf, cells))
        else:  # stage files hold a subset of cells
            o, _ = read_h5ad_obs_var(h)
            pos = pd.Series(np.arange(len(cells)), index=cells).loc[o.index].to_numpy()
            res["integrated_h5ad_checks"].append(check_integrated_h5ad(h, Xf[pos], list(o.index)))

    # ---------------------------------------------------------------- raw
    frames, per_sample = [], {}
    final_keys = set(fm["key"])
    for s, loader in cfg["raw"].items():
        out = loader()
        Xr, rgenes, rbc = out[:3]
        extra = dict(out[3]) if len(out) > 3 else {}
        rsym = extra.pop("_symbols", rgenes)
        st = stated(cfg, s)
        m, _ = qc_metrics(Xr, rsym, prefix)
        m["sample"], m["barcode"] = s, rbc
        m["key"] = [f"{s}:{b}" for b in rbc]
        for lab, mask in extra.items():
            m["pre: " + lab] = np.asarray(mask, dtype=bool)
        pre_cols = [c for c in m.columns if c.startswith("pre: ")]
        m["pass_pre"] = m[pre_cols].all(axis=1) if pre_cols else True
        m["pass_mito"] = m["pct_counts_mt"] <= st["mito"]
        m["pass_genes"] = (m["n_genes"] >= st["min_genes"]) & (m["n_genes"] <= st["max_genes"]) & \
                          (m["total_counts"] >= st["min_total"]) & (m["total_counts"] <= st["max_total"])
        m["in_final"] = m["key"].isin(final_keys)
        m["fate"] = np.select(
            [m["in_final"], ~m["pass_pre"], ~m["pass_mito"], ~m["pass_genes"]],
            ["final", "removed: pre-filter (" + "; ".join(l[5:] for l in pre_cols) + ")" if pre_cols else "x",
             "removed: mito", "removed: gene/UMI range"], "removed: not a singlet (passes all RNA filters)")
        frames.append(m)
        fin = fm[fm["sample"] == s]
        fin_keys = list(zip(fin.index, [k.split(":", 1)[1] for k in fin["key"]]))
        sel = rng.choice(len(fin_keys), size=min(300, len(fin_keys)), replace=False) if len(fin_keys) else []
        found = m["in_final"].sum()
        same, note = _row_check(Xr, rgenes, Xf, genes, m["barcode"].to_numpy(), cells, fin_keys, sel) if found == len(fin_keys) else (None, "skipped: some final cells not found in raw")
        allp = m["pass_pre"] & m["pass_mito"] & m["pass_genes"]
        per_sample[s] = {
            "n_raw": int(len(m)),
            **{"n_pass_" + l: int(m[l].sum()) for l in pre_cols},
            "n_pass_mito": int(m["pass_mito"].sum()), "n_pass_gene_umi_range": int(m["pass_genes"].sum()),
            "n_pass_all_RNA_filters": int(allp.sum()), "n_final": int(len(fin)),
            "n_removed_by_pre_filter": int((~m["pass_pre"]).sum()),
            "n_removed_by_mito_after_pre": int((m["pass_pre"] & ~m["pass_mito"]).sum()),
            "n_removed_by_range_after_pre_mito": int((m["pass_pre"] & m["pass_mito"] & ~m["pass_genes"]).sum()),
            "n_removed_as_non_singlet": int((allp & ~m["in_final"]).sum()),
            "n_final_failing_stated_mito": int((fin["pct_counts_mt"] > st["mito"]).sum()),
            "n_final_failing_stated_gene_umi_range": int(((fin["n_genes"] < st["min_genes"]) | (fin["n_genes"] > st["max_genes"])
                                                          | (fin["total_counts"] < st["min_total"]) | (fin["total_counts"] > st["max_total"])).sum()),
            "n_final_not_found_in_raw": int(len(fin) - found),
            "final_pct_mt_max": float(fin["pct_counts_mt"].max()),
            "final_n_genes_min_max": [int(fin["n_genes"].min()), int(fin["n_genes"].max())],
            "final_total_counts_min_max": [float(fin["total_counts"].min()), float(fin["total_counts"].max())],
            "raw_pct_mt_median": float(m["pct_counts_mt"].median()), "raw_n_genes_median": float(m["n_genes"].median()),
            "raw_total_counts_median": float(m["total_counts"].median()),
            "final_total_counts_median": float(fin["total_counts"].median()),
            "final_rows_identical_to_raw_rows": (f"{same}/{len(sel)} sampled; {note}" if same is not None else note),
        }
        del Xr
    raw = pd.concat(frames, ignore_index=True)
    res["per_sample"] = per_sample
    res["total_raw"] = int(len(raw))
    res["total_final_in_raw"] = int(raw["in_final"].sum())
    res["summary_json_claims"] = load_summary(name) if "qc_dir" in cfg else None
    raw.drop(columns=["key"]).to_csv(f"{od}/raw_cells_qc.csv.gz", index=False)
    json.dump(res, open(f"{od}/verify.json", "w"), indent=1, default=str)

    # ---------------------------------------------------------------- plots
    S = list(cfg["raw"].keys())
    fig, ax = plt.subplots(len(S), 4, figsize=(17, 3.3 * len(S)), squeeze=False)
    cols = {"final": "#1f77b4", "removed: mito": "#d62728", "removed: gene/UMI range": "#ff7f0e",
            "removed: not a singlet (passes all RNA filters)": "#2ca02c"}
    for i, s in enumerate(S):
        m = raw[raw["sample"] == s]
        st = stated(cfg, s)
        f = m[m["in_final"]]
        for j, (col, lab, log) in enumerate([("total_counts", "UMI counts per cell (log10)", True),
                                             ("n_genes", "genes detected per cell", False),
                                             ("pct_counts_mt", "% mito counts", False)]):
            a = ax[i, j]
            v_all, v_f = m[col].to_numpy(), f[col].to_numpy()
            if log:
                v_all, v_f = np.log10(np.clip(v_all, 1, None)), np.log10(np.clip(v_f, 1, None))
            bins = np.linspace(v_all.min(), v_all.max(), 70)
            a.hist(v_all, bins=bins, color="#bbbbbb", label=f"raw (n={len(m)})")
            a.hist(v_f, bins=bins, color="#1f77b4", alpha=0.8, label=f"final (n={len(f)})")
            lines = {"n_genes": [st["min_genes"], st["max_genes"]], "pct_counts_mt": [st["mito"]],
                     "total_counts": [np.log10(x) for x in (st["min_total"], st["max_total"]) if 0 < x < 1e12]}[col]
            for x in lines:
                if x < 1e8:
                    a.axvline(x, color="r", ls="--", lw=1)
            a.set_xlabel(lab); a.set_ylabel("cells"); a.set_title(s, fontsize=9); a.legend(fontsize=7)
        a = ax[i, 3]
        sub = m.sample(min(len(m), 12000), random_state=0)
        fates = [k for k in m["fate"].unique() if k != "final"] + ["final"]
        palette = {k: v for k, v in cols.items()}
        for k in fates:
            palette.setdefault(k, "#9467bd")
        for k in fates:
            q = sub[sub["fate"] == k]
            a.scatter(q["n_genes"], q["pct_counts_mt"], s=2, c=palette[k], label=f"{k[:42]} ({(m['fate'] == k).sum()})", rasterized=True)
        a.axhline(st["mito"], color="r", ls="--", lw=1)
        for x in (st["min_genes"], st["max_genes"]):
            if x < 1e8:
                a.axvline(x, color="r", ls="--", lw=1)
        a.set_xlabel("genes detected"); a.set_ylabel("% mito"); a.legend(fontsize=6, markerscale=4)
    fig.suptitle(f"{name}: raw cells vs final cells (dashed = cutoffs stated in the build script)")
    fig.tight_layout(); fig.savefig(f"{od}/qc_{name}.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.8 * len(S) + 3, 4))
    w = 0.2
    xs = np.arange(len(S))
    for k, (key, lab, c) in enumerate([("n_raw", "raw", "#bbbbbb"), ("n_pass_mito", "pass mito", "#d62728"),
                                       ("n_pass_all_RNA_filters", "pass all RNA filters", "#ff7f0e"),
                                       ("n_final", "final (singlet)", "#1f77b4")]):
        ax.bar(xs + (k - 1.5) * w, [per_sample[s][key] for s in S], w, label=lab, color=c)
    ax.set_xticks(xs); ax.set_xticklabels(S, rotation=30, ha="right", fontsize=8); ax.set_ylabel("cells"); ax.legend(fontsize=8)
    ax.set_title(f"{name}: cells surviving each filter (recomputed from raw)")
    fig.tight_layout(); fig.savefig(f"{od}/funnel_{name}.png", dpi=110); plt.close(fig)
    print(f"[{name}] done -> {od}", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
