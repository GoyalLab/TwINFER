"""2026-09-18: Score LARRY's 9 yscher gene panels with current_score (TODO4v2 old-variant, gated)
PLUS partial_corr_term (see handoff/2026-09-17_vsc_and_e9pos0_followup.md and
work_in_progress/benchmark/partial_corr_term.py) -- the conditional/multivariate term added this
session to address VSC's indirect-correlation confound on the simulated benchmarks. Checks
whether the same fix helps (or hurts) on real biological data.

Partial correlation is a population-level statistic (not twin-pair-specific) computed once per
gene panel from ALL cells, log1p(cp10k)-normalized (same convention as
preprocessing/build_twinfer_inputs_per_geneset.py's NORMALIZE=cp10k). It does NOT depend on the
unfiltered/annotfilter twin_def distinction (that only changes clone_id/twin-pairing downstream,
not the expression matrix itself -- see build_annotation_filtered_inputs.py's docstring) so the
same partial_corr_term is added to both twin_defs' current_score.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from paper_analysis.larry_hematopoiesis_validation.apply_todo4v2_allpairs_with_competitors import (
    R, GENE_SETS, TWIN_DEFS, METHODS, s, full_report, load_zreg_map, todo4v2_score,
    load_competitor_allgenes,
)
from benchmarks.network_benchmarks.score.partial_corr_term import DEFAULT_RIDGE, partial_corr_term

SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
GENE_SETS_YSCHER = os.path.join(os.path.dirname(__file__), "resources", "gene_sets_yscher.json")


def load_all_cells_cp10k():
    X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
    genes = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
    cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    return X, genes, cell_total


def panel_matrix_cp10k(X, genes, cell_total, gene_list):
    gene_index = {g: i for i, g in enumerate(genes)}
    col_idx = [gene_index[g] for g in gene_list]
    mat = X[:, col_idx].toarray().astype(float)
    return np.log1p(mat / cell_total[:, None] * 1e4)


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    gene_sets_yscher = json.load(open(GENE_SETS_YSCHER))
    X_all, genes_all, cell_total = load_all_cells_cp10k()

    pcorr_term_cache = {}

    rows = []
    competitor_cache = {}
    for twin_def, cfg in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
            zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
            if not os.path.exists(dpath) or not os.path.exists(zpath):
                print(f"[{twin_def}/{gs}] SKIP -- inputs not found yet")
                continue
            dd = pd.read_csv(dpath)
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
            z_reg_map = load_zreg_map(gs, cfg["infer_suffix"])

            U = list(zip(dd.gene_1, dd.gene_2))
            y = np.array([1 if p in CE else 0 for p in U])
            if y.sum() < 3:
                print(f"[{twin_def}/{gs}] SKIP -- only {y.sum()} true edges")
                continue

            TODO4V2, n_pass_gate = todo4v2_score(dd, U, z_reg_map)
            todo4v2_m = full_report(TODO4V2, y)
            del z_reg_map

            if gs not in pcorr_term_cache:
                gene_list = gene_sets_yscher[gs]
                Xp = panel_matrix_cp10k(X_all, genes_all, cell_total, gene_list)
                pcorr_term_cache[gs] = partial_corr_term(Xp, gene_list, U, ridge=DEFAULT_RIDGE)
            pterm = pcorr_term_cache[gs]

            TODO4V2_PC = np.where(np.isfinite(TODO4V2), TODO4V2 + pterm, TODO4V2)
            todo4v2_pc_m = full_report(TODO4V2_PC, y)

            if gs not in competitor_cache:
                best_name, best_m = None, None
                for m in METHODS:
                    try:
                        sc = load_competitor_allgenes(m, gs, U)
                    except FileNotFoundError:
                        continue
                    cm = full_report(s(sc), y)
                    if best_m is None or cm["auprc_x"] > best_m["auprc_x"]:
                        best_name, best_m = m, cm
                competitor_cache[gs] = (best_name, best_m)
            best_name, best_m = competitor_cache[gs]
            print(f"[{twin_def}/{gs}] done", flush=True)

            rows.append(dict(
                twin_def=twin_def, gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
                todo4v2_auprc_x=todo4v2_m["auprc_x"],
                todo4v2_pc_auprc_x=todo4v2_pc_m["auprc_x"],
                delta=todo4v2_pc_m["auprc_x"] - todo4v2_m["auprc_x"],
                best_competitor=best_name,
                comp_auprc_x=best_m["auprc_x"] if best_m else np.nan,
                todo4v2_pass_gate=n_pass_gate,
            ))

    df = pd.DataFrame(rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{os.path.dirname(__file__)}/todo4v2_partial_corr_larry_comparison.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/todo4v2_partial_corr_larry_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print()
    for twin_def in df.twin_def.unique():
        sub = df[df.twin_def == twin_def]
        print(f"[{twin_def}] mean todo4v2={sub.todo4v2_auprc_x.mean():.3f}x  "
              f"todo4v2+partial_corr={sub.todo4v2_pc_auprc_x.mean():.3f}x  "
              f"comp={sub.comp_auprc_x.mean():.3f}x  "
              f"wins-vs-comp: todo4v2={((sub.todo4v2_auprc_x>sub.comp_auprc_x).sum())}/{len(sub)}  "
              f"todo4v2+pc={((sub.todo4v2_pc_auprc_x>sub.comp_auprc_x).sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
