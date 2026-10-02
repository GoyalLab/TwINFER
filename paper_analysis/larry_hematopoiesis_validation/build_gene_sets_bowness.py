"""Build TwINFER benchmark gene panels from the Bowness et al. perturb-seq screen (20-TF CRISPR
knockdown in LARRY hematopoiesis, real_data/Bowness_et_al_perturbseq_results_5x.csv), using
perturbation effect size as the panel-selection criterion instead of CollecTRI |rho|
(pick_gene_sets.ipynb's correlation_high/mid/low recipe) -- same edge-banding /
top-TFs-plus-top-targets structure, just sourced from a different (and, per
compare_bowness_collectri.py, largely independent: auprc_x=1.68x vs CollecTRI, precision ~2%)
ground truth.

Recipe (mirrors pick_gene_sets.ipynb cell 12's correlation_high/mid/low):
  1. Restrict to Bowness pairs that pass_qc, are `significant`, and whose response_id is present
     in the LARRY filtered gene universe with per-day-minimum detection >= MIN_DETECTION_FRAC
     (matches pick_gene_sets.ipynb's own floor).
  2. Rank all surviving (TF, response_gene) edges by |mean_log2FC| (perturbation effect size,
     the direct analogue of |rho| for a correlation-based panel).
  3. bowness_high / _mid / _low = the top / middle / bottom N_BAND_EDGES edges of that ranking.
  4. Within each band, group by TF; keep every TF that has >=1 edge in the band (only 20 TFs
     exist in this screen at all, so no "top n_tfs by count" subselection like the CollecTRI
     recipe needs -- unlike CollecTRI's few-thousand-TF universe, all of Bowness's TFs are already
     hand-picked hematopoietic regulators). Each TF contributes up to N_TARGETS_PER_TF targets,
     its highest-|effect| ones in that band.

Output: resources/gene_sets_bowness.json ({panel_name: [gene, ...]}) -- same shape as
resources/gene_sets.json / gene_sets_yscher.json, so it plugs directly into
preprocessing/build_twinfer_inputs_per_geneset.py via GENE_SETS_JSON=resources/gene_sets_bowness.json,
and resources/gene_sets_detail_bowness.json (per-TF detail, for inspection).

The ground-truth edge set for downstream AUPRC scoring against these panels is NOT CollecTRI --
it's Bowness's own `significant` calls (see build_bowness_ground_truth_edges below), since the
whole point is an independent-of-CollecTRI benchmark.

Run: python3 build_gene_sets_bowness.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
BOWNESS_PATH = f'{TWINFER_PROJECT_ROOT}/real_data/Bowness_et_al_perturbseq_results_5x.csv'

MIN_DETECTION_FRAC = 0.05   # matches pick_gene_sets.ipynb / larry_raw_annotate.py's floor
N_BAND_EDGES = 50           # high/mid/low = top/middle/bottom 50 edges by |mean_log2FC|
N_TARGETS_PER_TF = 4        # matches pick_gene_sets.ipynb's correlation recipe


def per_day_min_detection():
    X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
    genes = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
    obs = pd.read_csv(os.path.join(SOURCE, "obs_metadata.csv"), index_col=0)
    assert X.shape == (len(obs), len(genes))
    day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False)
    assert day.notna().all()
    day = day.astype(int).to_numpy()
    days = sorted(set(day))
    per_day_frac = np.stack([
        np.asarray((X[day == d] > 0).sum(axis=0)).ravel() / (day == d).sum()
        for d in days
    ])
    frac_min = per_day_frac.min(axis=0)
    return pd.Series(frac_min, index=genes)


def load_bowness_significant():
    df = pd.read_csv(BOWNESS_PATH, low_memory=False)
    df["significant"] = df["significant"].astype(str).map({"True": True, "False": False})
    df = df[(df["pass_qc"].astype(str) == "True") & (df["significant"] == True)]
    df = df[df["target_gene"] != df["response_id"]]
    df = df.dropna(subset=["mean_log2FC"])
    return df


def edges_to_panel(edges, effect, n_targets=N_TARGETS_PER_TF):
    """edges: list of (tf, target). effect: dict[(tf,target)] -> |mean_log2FC|.
    Every TF with >=1 edge in `edges` is kept (Bowness's 20 TFs are all hand-picked already);
    each TF gets its top `n_targets` targets by effect size in this band."""
    by_tf = {}
    for tf, tgt in edges:
        by_tf.setdefault(tf, []).append(tgt)
    genes_out, detail = set(), []
    for tf in sorted(by_tf, key=lambda t: -np.median([effect[(t, x)] for x in by_tf[t]])):
        genes_out.add(tf)
        tgts = sorted(by_tf[tf], key=lambda x: effect[(tf, x)], reverse=True)[:n_targets]
        genes_out.update(tgts)
        detail.append((tf, [(t, round(effect[(tf, t)], 3)) for t in tgts]))
    return sorted(genes_out), detail


def main():
    print("computing per-day-minimum detection fraction over the LARRY filtered matrix...")
    frac_min = per_day_min_detection()
    detected = set(frac_min.index[frac_min >= MIN_DETECTION_FRAC])
    print(f"{len(detected):,} / {len(frac_min):,} genes pass detection >= {MIN_DETECTION_FRAC} "
          f"in every day\n")

    bow = load_bowness_significant()
    bow = bow[bow["response_id"].isin(detected) & bow["target_gene"].isin(detected)]
    bow["effect"] = bow["mean_log2FC"].abs()
    print(f"{len(bow):,} significant Bowness edges survive the detection floor "
          f"({bow['target_gene'].nunique()} TFs, {bow['response_id'].nunique():,} response genes)")

    pairs = list(zip(bow["target_gene"], bow["response_id"]))
    effect = dict(zip(pairs, bow["effect"]))
    ranked = sorted(pairs, key=lambda p: effect[p], reverse=True)
    mid = len(ranked) // 2
    h = N_BAND_EDGES
    bands = {
        "high": ranked[:h],
        "mid": ranked[mid - h // 2: mid - h // 2 + h],
        "low": ranked[-h:],
    }
    for lvl in ("high", "mid", "low"):
        eff = [effect[p] for p in bands[lvl]]
        print(f"bowness_{lvl:<4}: {len(eff)} edges, |mean_log2FC| in [{min(eff):.3f}, {max(eff):.3f}]")

    gene_sets, gene_sets_detail = {}, {}
    for lvl in ("high", "mid", "low"):
        genes_out, detail = edges_to_panel(bands[lvl], effect)
        gene_sets[f"bowness_{lvl}"] = genes_out
        gene_sets_detail[f"bowness_{lvl}"] = detail
        print(f"bowness_{lvl:<4}: {len(genes_out)} genes ({len(detail)} TFs + up to "
              f"{N_TARGETS_PER_TF} targets each)")
        print(f"  TFs: {[tf for tf, _ in detail]}")

    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] out_path = os.path.join(HERE, "resources", "gene_sets_bowness.json")
    out_path = os.path.join(RES_HERE, "resources", "gene_sets_bowness.json")
    json.dump(gene_sets, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")

    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] detail_path = os.path.join(HERE, "resources", "gene_sets_detail_bowness.json")
    detail_path = os.path.join(RES_HERE, "resources", "gene_sets_detail_bowness.json")
    json.dump(gene_sets_detail, open(detail_path, "w"), indent=2)
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
