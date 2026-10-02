"""Compare Bowness et al. perturb-seq hematopoiesis results (real_data/Bowness_et_al_perturbseq_
results_5x.csv -- CRISPR knockdown of 20 TFs, differential expression of every response gene per
target) against CollecTRI mouse as ground truth.

Two questions:
  1. Agreement: of the (TF, response_gene) pairs Bowness calls significant, how many are already
     documented CollecTRI edges (precision), and of CollecTRI's documented edges for these 20 TFs,
     how many does Bowness's screen recover (recall)? Reported overall and per TF.
  2. Ranking check: using |mean_log2FC| (perturbation effect size) to rank ALL candidate
     (TF, response_gene) pairs (not just the significant ones) and CollecTRI membership as the
     0/1 label, what's the AUPRC vs. random? This is the same full_report() metric the rest of
     this directory's scripts use for TwINFER's own z-scores, applied here to the raw perturb-seq
     effect size itself, as a sanity check on how much CollecTRI and perturb-seq agree before using
     Bowness as an independent ground truth (see build_gene_sets_bowness.py).

Run: python3 compare_bowness_collectri.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
R = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
BOWNESS_PATH = f'{TWINFER_PROJECT_ROOT}/real_data/Bowness_et_al_perturbseq_results_5x.csv'


def load_bowness():
    df = pd.read_csv(BOWNESS_PATH, low_memory=False)
    df["significant"] = df["significant"].astype(str).map({"True": True, "False": False})
    df = df[df["pass_qc"].astype(str) == "True"]
    df = df[df["target_gene"] != df["response_id"]]
    return df


def load_collectri():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    return set(zip(ct.source_genesymbol, ct.target_genesymbol))


def main():
    bow = load_bowness()
    CE = load_collectri()
    tfs = sorted(bow["target_gene"].unique())
    print(f"Bowness: {len(bow):,} (TF, response_gene) pairs tested, {bow['target_gene'].nunique()} TFs, "
          f"{bow['response_id'].nunique():,} distinct response genes")
    print(f"CollecTRI mouse: {len(CE):,} edges total\n")

    sig = bow[bow["significant"] == True]
    sig_edges = set(zip(sig["target_gene"], sig["response_id"]))
    ct_edges_for_tfs = {(a, b) for (a, b) in CE if a in tfs}
    tp = sig_edges & ct_edges_for_tfs
    precision = len(tp) / len(sig_edges) if sig_edges else float("nan")
    recall = len(tp) / len(ct_edges_for_tfs) if ct_edges_for_tfs else float("nan")
    print("=== Overall agreement: Bowness-significant edges vs. CollecTRI ===")
    print(f"Bowness significant edges:            {len(sig_edges):,}")
    print(f"CollecTRI edges for these 20 TFs:      {len(ct_edges_for_tfs):,}")
    print(f"overlap (TP):                          {len(tp):,}")
    print(f"precision (TP / Bowness-significant):  {precision:.4f}")
    print(f"recall    (TP / CollecTRI edges):       {recall:.4f}\n")

    print("=== Per-TF agreement ===")
    rows = []
    for tf in tfs:
        s_tf = {b for (a, b) in sig_edges if a == tf}
        c_tf = {b for (a, b) in ct_edges_for_tfs if a == tf}
        tp_tf = s_tf & c_tf
        rows.append(dict(
            tf=tf, n_significant=len(s_tf), n_collectri=len(c_tf), n_overlap=len(tp_tf),
            precision=len(tp_tf) / len(s_tf) if s_tf else float("nan"),
            recall=len(tp_tf) / len(c_tf) if c_tf else float("nan"),
        ))
    per_tf = pd.DataFrame(rows).sort_values("n_significant", ascending=False)
    print(per_tf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] per_tf.to_csv(f"{R}/../bowness_vs_collectri_per_tf.csv", index=False)
    per_tf.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/bowness_vs_collectri_per_tf.csv", index=False)

    print("\n=== Ranking check: |mean_log2FC| (all tested pairs) vs. CollecTRI membership ===")
    all_pairs = bow.dropna(subset=["mean_log2FC"])
    sc = all_pairs["mean_log2FC"].abs().to_numpy()
    y = np.array([1 if p in CE else 0 for p in zip(all_pairs["target_gene"], all_pairs["response_id"])])
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    print(f"n_pairs={len(y):,}  n_true={int(y.sum()):,}  auprc={auprc:.4f}  "
          f"auprc_random={rand:.4f}  auprc_x={auprc / rand:.3f}")

    # [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] print(f"\nwrote {R}/../bowness_vs_collectri_per_tf.csv")
    print(f"\nwrote {TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/bowness_vs_collectri_per_tf.csv")


if __name__ == "__main__":
    main()
