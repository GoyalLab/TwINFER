#!/usr/bin/env python3
"""Run the dip-test + Otsu whitelist (fully automatic, no external target -- see
build_diptest_otsu_whitelist.py) through the SAME full downstream chain (mito filter ->
singletCode multiplet removal -> 200-gene floor) as the current production knee(v2)
whitelist, so the comparison is on final cell counts, not just the whitelist stage.

Settles empirically whether the dip-test's "unimodal -> keep everyone" fallback (which
keeps low-but-not-doomed-below-200 counts, e.g. LSK_d2_2's 200-1000 count range) actually
survives singletCode's own confident-barcode requirement, rather than reasoning about it
in the abstract.
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio
import singletCode

PROCESSED = "/scratch/gzu5140/ka_twinfer/larry_dataset/processed"
K = 0.19349593495847234  # the shared multiplier chosen for the v2 (knee) whitelist


def knee_threshold(counts):
    c = np.sort(counts)[::-1]
    c = c[c > 0]
    x = np.log10(np.arange(1, len(c) + 1))
    y = np.log10(c)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = p2 - p1
    line_norm = line / np.linalg.norm(line)
    vecs = np.stack([x - p1[0], y - p1[1]], axis=1)
    proj_len = vecs @ line_norm
    proj = np.outer(proj_len, line_norm) + p1
    dist = np.linalg.norm(vecs - (proj - p1), axis=1)
    return float(c[np.argmax(dist)])


def run_downstream(strategy_name, whitelist_mask, obs, cells, umi_table_full, pass_mito, pass_genes):
    after_mito = whitelist_mask & pass_mito

    wl_lib = obs.loc[whitelist_mask, "library"].to_numpy()
    wl_bc = cells[whitelist_mask].str.split(":", n=1).str[1].to_numpy()
    whitelist_keys = set(zip(wl_lib, wl_bc))

    keep_row = [(s, c) in whitelist_keys for s, c in
                zip(umi_table_full["sample"].to_numpy(), umi_table_full["cellID"].to_numpy())]
    umi_table = umi_table_full[keep_row].reset_index(drop=True)
    print(f"[{strategy_name}] umi_table restricted to whitelist: {len(umi_table)} rows", flush=True)

    singletCode.check_sample_sheet(umi_table)
    good_data, singlet_stats = singletCode.get_singlets(umi_table, dataset_name=f"LARRY_{strategy_name}")
    singlets = good_data[good_data["label"] == "Singlet"]
    singlet_keys = set(zip(singlets["sample"].to_numpy(), singlets["cellID"].to_numpy()))

    obs_library = obs["library"].to_numpy()
    obs_barcode = cells.str.split(":", n=1).str[1].to_numpy()
    is_singlet = np.array([(lib, bc) in singlet_keys for lib, bc in zip(obs_library, obs_barcode)])

    after_singlet = after_mito & is_singlet
    after_genes = after_singlet & pass_genes

    return {
        "strategy": strategy_name,
        "n_whitelist": int(whitelist_mask.sum()),
        "n_after_mito": int(after_mito.sum()),
        "n_after_singletcode": int(after_singlet.sum()),
        "n_after_gene_floor": int(after_genes.sum()),
    }


def main():
    print("[compare] loading combined matrix", flush=True)
    X = sio.mmread(f"{PROCESSED}/larry_combined_counts.mtx").tocsr()
    genes = pd.Series(open(f"{PROCESSED}/genes.txt").read().split())
    cells = pd.Series(open(f"{PROCESSED}/cells.txt").read().split())
    obs = pd.read_csv(f"{PROCESSED}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes))
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)

    mito_mask = genes.str.lower().str.startswith("mt-").to_numpy()
    assert mito_mask.sum() > 0, "no mitochondrial genes found -- refusing a no-op filter"
    mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel().astype(float)
    pct_counts_mt = np.divide(100.0 * mito_counts, total_counts,
                               out=np.zeros_like(total_counts), where=total_counts > 0)
    pass_mito = pct_counts_mt <= 20.0

    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    pass_genes = n_genes_per_cell >= 200

    umi_table_full = pd.read_csv(f"{PROCESSED}/umi_table.csv")

    libs = sorted(obs["library"].unique())
    lib_masks = {lib: (obs["library"] == lib).to_numpy() for lib in libs}

    # knee (v2) whitelist mask, recomputed fresh (same as production)
    knee_mask = np.zeros(len(obs), dtype=bool)
    for lib in libs:
        m = lib_masks[lib]
        thr = knee_threshold(total_counts[m]) * K
        knee_mask |= m & (total_counts >= thr)

    # dip-test + Otsu whitelist, loaded from the already-computed CSV (library, barcode)
    dt_wl = pd.read_csv(f"{PROCESSED}/diptest_otsu_whitelist.csv")
    dt_keys = set(zip(dt_wl["library"], dt_wl["barcode"]))
    obs_barcode_all = cells.str.split(":", n=1).str[1].to_numpy()
    diptest_mask = np.array([(lib, bc) in dt_keys for lib, bc in zip(obs["library"].to_numpy(), obs_barcode_all)])

    print(f"[compare] whitelist totals: knee_v2={int(knee_mask.sum())}  diptest_otsu={int(diptest_mask.sum())}",
          flush=True)

    downstream = []
    for name, wl_mask in [("knee_v2", knee_mask), ("diptest_otsu", diptest_mask)]:
        res = run_downstream(name, wl_mask, obs, cells, umi_table_full, pass_mito, pass_genes)
        downstream.append(res)
        print(f"[compare] {json.dumps(res)}", flush=True)

    out = {
        "k_shared_multiplier_knee_v2": K,
        "whitelist_totals": {"knee_v2": int(knee_mask.sum()), "diptest_otsu": int(diptest_mask.sum())},
        "downstream": downstream,
    }
    with open(f"{PROCESSED}/diptest_vs_knee_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[compare] wrote {PROCESSED}/diptest_vs_knee_comparison.json", flush=True)


if __name__ == "__main__":
    main()
