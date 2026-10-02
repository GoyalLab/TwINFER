#!/usr/bin/env python3
"""FM06 melanoma-specific TF-target ground truth (ChIP-Atlas binding + KnockTF knockdown), for comparison
with CollecTRI as the positive-edge set used to score TwINFER / TwinScore_supplement / competitors.

Adapted from a melanoma TF-target network builder (Verfaillie et al. 2015, ncomms7683-style approach) to:
  - the actual cell line of FM06 (WM989 naive), added to the melanoma keyword list
  - the KnockTF schema returned by decoupler>=2.2 (`dc.ds.knocktf()`): obs = experiments (column 'source' = TF,
    'Tissue.Type', 'Biosample.Name', ...), X = genes x log2FC. KnockTF tags its melanoma cell-line experiments
    (A375, SK-MEL-5, Ma-Mel-15, NZM12, MZ7, ...) with Tissue.Type == 'Skin', which is used as the melanoma filter
    instead of matching each individual cell-line name.
  - restricting the output positive set to evidence == 'both' (ChIP binding AND KnockTF DE agree), the
    high-confidence tier the original script recommends for building a regulon.

Usage:
    /home/gzu5140/.conda/envs/twinfer-code/bin/python3 build_fm06_melanoma_ground_truth.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import io
import json
import time

import pandas as pd
import requests
from scipy.stats import hypergeom

BASE = f'{TWINFER_PROJECT_ROOT}'
OUT_DIR = f"{BASE}/analysis_data/fm06/data"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI = f"{BASE}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv"
COLLECTRI = f"{BASE}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv"
GENE_SETS_JSON = f"{OUT_DIR}/gene_sets_fm06.json"

TFS = ["MITF", "SOX10", "JUN", "JUND", "FOSL1", "FOSL2", "TEAD1", "TEAD2", "TEAD3", "TEAD4"]

GENOME = "hg38"
DISTANCE_CODE = "5"

MELANOMA_KEYWORDS = ["melanoma", "sk-mel", "skmel", "a375", "501mel", "colo829", "mewo", "wm989", "wm", "malme"]
LINEAGE_FALLBACK_KEYWORDS = ["melanocyte", "skin", "keratinocyte", "fibroblast", "neural crest", "sk-n-sh", "a549"]

CHIP_SCORE_MIN = 250
KD_LOG2FC_MIN = 1.0
BACKGROUND_N = 20000

CHIPATLAS_TARGET_URL = "https://chip-atlas.dbcls.jp/data/{genome}/target/{tf}.{dist}.tsv"


def get_knocktf_melanoma_targets(tfs, log2fc_min=KD_LOG2FC_MIN):
    import decoupler as dc

    print("Downloading/loading KnockTF (cached after first run)...")
    adata = dc.ds.knocktf(thr_fc=None, verbose=False)  # thr_fc=None: no pre-filtering, apply our own threshold
    meta = adata.obs.copy()

    is_melanoma = (meta["Tissue.Type"].astype(str) == "Skin") | (
        meta["Biosample.Name"].astype(str).str.lower().apply(lambda s: any(k in s for k in MELANOMA_KEYWORDS))
    )

    kd_targets = {}
    for tf in tfs:
        mask = is_melanoma & (meta["source"].astype(str).str.upper() == tf.upper())
        exp_ids = meta.index[mask]
        if len(exp_ids) == 0:
            print(f"  [KnockTF] no melanoma experiments found for {tf}")
            kd_targets[tf] = set()
            continue
        sub = adata[exp_ids, :].to_df()
        hit_mask = (sub.abs() >= log2fc_min).any(axis=0)
        genes = set(sub.columns[hit_mask])
        kd_targets[tf] = genes
        lines = sorted(meta.loc[exp_ids, "Biosample.Name"].unique())
        print(f"  [KnockTF] {tf}: {len(exp_ids)} melanoma experiment(s) ({', '.join(lines)}), "
              f"{len(genes)} DE genes at |log2FC|>={log2fc_min}")

    return kd_targets


def get_chipatlas_targets(tfs, score_min=CHIP_SCORE_MIN):
    chip_targets = {}
    for tf in tfs:
        url = CHIPATLAS_TARGET_URL.format(genome=GENOME, tf=tf, dist=DISTANCE_CODE)
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ChIP-Atlas] failed to fetch {tf}: {e}")
            chip_targets[tf] = (set(), "none")
            continue

        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        if df.empty or df.shape[1] < 2:
            print(f"  [ChIP-Atlas] empty/unexpected response for {tf}")
            chip_targets[tf] = (set(), "none")
            continue

        gene_col = df.columns[0]
        header_text = [str(c).lower() for c in df.columns[1:]]
        melanoma_cols = [c for c, h in zip(df.columns[1:], header_text) if any(k in h for k in MELANOMA_KEYWORDS)]
        lineage_cols = [c for c, h in zip(df.columns[1:], header_text) if any(k in h for k in LINEAGE_FALLBACK_KEYWORDS)]

        if melanoma_cols:
            use_cols, tier = melanoma_cols, "melanoma"
        elif lineage_cols:
            use_cols, tier = lineage_cols, "lineage_fallback"
        else:
            use_cols, tier = df.columns[1:], "any_celltype"

        scores = df[use_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        genes = set(df.loc[scores >= score_min, gene_col].astype(str))
        chip_targets[tf] = (genes, tier)
        print(f"  [ChIP-Atlas] {tf}: tier={tier}, {len(use_cols)} experiment(s), {len(genes)} genes at score>={score_min}")
        time.sleep(0.5)

    return chip_targets


def build_network(chip_targets, kd_targets, background_n=BACKGROUND_N):
    rows, summary = [], []
    for tf in chip_targets:
        chip_genes, tier = chip_targets[tf]
        kd_genes = kd_targets.get(tf, set())
        both = chip_genes & kd_genes
        if chip_genes and kd_genes:
            pval = hypergeom.sf(len(both) - 1, background_n, len(chip_genes), len(kd_genes))
        else:
            pval = float("nan")
        summary.append({"TF": tf, "chip_tier": tier, "n_chip_targets": len(chip_genes), "n_kd_degs": len(kd_genes),
                         "n_overlap": len(both), "overlap_pvalue": pval})
        for g in chip_genes | kd_genes:
            evidence = "both" if g in both else ("chip_only" if g in chip_genes else "kd_only")
            rows.append({"TF": tf, "target": g, "evidence": evidence, "chip_tier": tier})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def compare_with_collectri(network_df):
    """How the high-confidence (evidence=='both') melanoma edges relate to CollecTRI for the same TFs."""
    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    ct_pairs = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    ct_sources = set(ct.source_genesymbol)

    mel_high = network_df[network_df.evidence == "both"]
    mel_pairs = set(zip(mel_high.TF, mel_high.target))

    rows = []
    for tf in TFS:
        mel_tf_pairs = {(a, b) for a, b in mel_pairs if a == tf}
        ct_tf_pairs = {(a, b) for a, b in ct_pairs if a == tf}
        rows.append({
            "TF": tf,
            "in_collectri_as_source": tf in ct_sources,
            "n_melanoma_gt_targets": len(mel_tf_pairs),
            "n_collectri_targets": len(ct_tf_pairs),
            "n_overlap": len(mel_tf_pairs & ct_tf_pairs),
            "n_melanoma_only": len(mel_tf_pairs - ct_tf_pairs),
            "n_collectri_only": len(ct_tf_pairs - mel_tf_pairs),
            "jaccard": (len(mel_tf_pairs & ct_tf_pairs) / len(mel_tf_pairs | ct_tf_pairs)
                        if (mel_tf_pairs | ct_tf_pairs) else float("nan")),
        })
    overall_overlap = len(mel_pairs & ct_pairs)
    total = {"TF": "TOTAL", "in_collectri_as_source": None,
             "n_melanoma_gt_targets": len(mel_pairs), "n_collectri_targets": sum(r["n_collectri_targets"] for r in rows),
             "n_overlap": overall_overlap, "n_melanoma_only": len(mel_pairs) - overall_overlap,
             "n_collectri_only": sum(r["n_collectri_only"] for r in rows), "jaccard": float("nan")}
    rows.append(total)
    return pd.DataFrame(rows), mel_pairs, ct_pairs


def restrict_to_panels(network_df):
    """Also emit a version restricted to genes appearing anywhere in the FM06 panels, for direct use as an eval universe."""
    gene_sets = json.load(open(GENE_SETS_JSON))
    panel_genes = set()
    for genes in gene_sets.values():
        panel_genes.update(genes)
    sub = network_df[network_df.TF.isin(panel_genes) & network_df.target.isin(panel_genes)]
    return sub, panel_genes


if __name__ == "__main__":
    print("=== Step A: KnockTF (melanoma / Skin-tissue knockdown, FM06 TFs) ===")
    kd_targets = get_knocktf_melanoma_targets(TFS)

    print("\n=== Step B: ChIP-Atlas (melanoma / same-lineage binding) ===")
    chip_targets = get_chipatlas_targets(TFS)

    print("\n=== Step C: build network + enrichment ===")
    network_df, summary_df = build_network(chip_targets, kd_targets)

    network_df.to_csv(f"{OUT_DIR}/melanoma_tf_network_fm06.tsv", sep="\t", index=False)
    summary_df.to_csv(f"{OUT_DIR}/melanoma_tf_network_fm06_summary.tsv", sep="\t", index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\n=== Step D: compare high-confidence (evidence=='both') edges with CollecTRI ===")
    cmp_df, mel_pairs, ct_pairs = compare_with_collectri(network_df)
    cmp_df.to_csv(f"{OUT_DIR}/melanoma_tf_network_fm06_vs_collectri.tsv", sep="\t", index=False)
    print(cmp_df.to_string(index=False))

    print("\n=== Step E: restrict to FM06 panel genes (evaluation universe) ===")
    panel_sub, panel_genes = restrict_to_panels(network_df)
    panel_sub.to_csv(f"{OUT_DIR}/melanoma_tf_network_fm06_panel_restricted.tsv", sep="\t", index=False)
    both_panel = panel_sub[panel_sub.evidence == "both"]
    print(f"{len(panel_genes)} distinct panel genes across all 9 gene sets; "
          f"{len(panel_sub)} melanoma-GT edges (any evidence) touch panel genes on both ends; "
          f"{len(both_panel)} of those are evidence=='both' (high-confidence positives): "
          f"{sorted(set(zip(both_panel.TF, both_panel.target)))}")

    print(f"\nWrote:\n  {OUT_DIR}/melanoma_tf_network_fm06.tsv\n  {OUT_DIR}/melanoma_tf_network_fm06_summary.tsv"
          f"\n  {OUT_DIR}/melanoma_tf_network_fm06_vs_collectri.tsv\n  {OUT_DIR}/melanoma_tf_network_fm06_panel_restricted.tsv")
