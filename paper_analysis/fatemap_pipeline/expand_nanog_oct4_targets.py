#!/usr/bin/env python3
"""NANOG and POU5F1(OCT4) are the only 2 TFs in the panel with BOTH ChIP-Atlas binding
evidence AND orthogonal Perturb-seq perturbation evidence (GSE283614 / Table S2: ESC
Cluster1 DEGs = NANOG-knockdown-responsive genes, ESC Cluster2 DEGs = OCT4-knockdown
-responsive genes). Per user instruction, give these two more targets than the standard
top-4: rank each TF's ChIP-Atlas-bound candidates that ALSO appear in its Perturb-seq DEG
table by |avg_log2FC| (the perturbation effect size), take the top N_EXPANDED, and merge
into the top4 panel from pick_hpsc_endoderm_gene_panel.py (replacing their top4 entries).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json

import openpyxl
import pandas as pd

EDGES_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data/hpsc_endoderm_chipatlas_candidate_pairs.csv'
TOP4_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data/hpsc_endoderm_candidate_pairs_top4.csv'
S2_PATH = f'{TWINFER_PROJECT_ROOT}/real_data/humanTFs/hPSC_s2.xlsx'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
N_EXPANDED = 20

TF_TO_DEG_SHEET = {"NANOG": "ESC Cluster1 DEGs", "POU5F1": "ESC Cluster2 DEGs"}


def load_deg_sheet(sheet_name):
    wb = openpyxl.load_workbook(S2_PATH, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header_i = next(i for i, r in enumerate(rows) if r and r[0] == "gene")
    header = rows[header_i]
    df = pd.DataFrame(rows[header_i + 1:], columns=header)
    fc_col = [c for c in df.columns if c and c.startswith("avg_log2FC")][0]
    df = df.rename(columns={fc_col: "avg_log2FC"})
    df["avg_log2FC"] = pd.to_numeric(df["avg_log2FC"], errors="coerce")
    df["p_val_adj"] = pd.to_numeric(df["p_val_adj"], errors="coerce")
    return df[["gene", "avg_log2FC", "p_val_adj"]].dropna()


def main():
    edges = pd.read_csv(EDGES_PATH)
    top4 = pd.read_csv(TOP4_PATH)

    expanded_rows = []
    for tf, sheet in TF_TO_DEG_SHEET.items():
        deg = load_deg_sheet(sheet)
        candidates = set(edges.loc[edges.TF == tf, "target"]) - {tf}
        deg_in_candidates = deg[deg["gene"].isin(candidates)].copy()
        deg_in_candidates["abs_fc"] = deg_in_candidates["avg_log2FC"].abs()
        deg_in_candidates = deg_in_candidates.sort_values("abs_fc", ascending=False)
        top = deg_in_candidates.head(N_EXPANDED)
        print(f"{tf}: {len(candidates)} ChIP-bound candidates, {len(deg_in_candidates)} also in "
              f"Perturb-seq DEG table ({sheet}), keeping top {len(top)} by |avg_log2FC|", flush=True)
        for _, r in top.iterrows():
            expanded_rows.append({"TF": tf, "target": r["gene"], "avg_log2FC": r["avg_log2FC"],
                                   "p_val_adj": r["p_val_adj"]})

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.to_csv(f"{OUT_DIR}/hpsc_endoderm_nanog_oct4_expanded_targets.csv", index=False)

    # merge: drop NANOG/POU5F1's old top-4 rows, add the expanded rows
    merged = top4[~top4.TF.isin(TF_TO_DEG_SHEET)][["TF", "target"]].copy()
    merged = pd.concat([merged, expanded_df[["TF", "target"]]], ignore_index=True)
    merged.to_csv(f"{OUT_DIR}/hpsc_endoderm_candidate_pairs_final.csv", index=False)

    final_genes = sorted(set(merged.TF) | set(merged.target))
    json.dump(final_genes, open(f"{OUT_DIR}/hpsc_endoderm_final_gene_panel.json", "w"))

    print(f"\nmerged edges: {len(merged)}", flush=True)
    print(f"distinct TFs: {merged.TF.nunique()}", flush=True)
    print(f"distinct targets: {merged.target.nunique()}", flush=True)
    print(f"final gene panel size: {len(final_genes)}", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_candidate_pairs_final.csv", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_final_gene_panel.json", flush=True)


if __name__ == "__main__":
    main()
