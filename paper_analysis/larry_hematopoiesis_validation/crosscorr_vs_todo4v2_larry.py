"""2026-09-17: Does cross-corr-only (abs(rho_cross_xy) alone) beat TODO4v2 on LARRY the way it
does on the 3 simulated benchmarks? User's claim is it does NOT -- check directly, and correlate
the crosscorr-vs-todo4v2 gap against panel density to find what predicts which one should win,
so the formula could eventually pick a regime adaptively instead of being one fixed sum."""
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(__file__))
from paper_analysis.larry_hematopoiesis_validation.apply_todo4v2_allpairs_with_competitors import (
    R, GENE_SETS, TWIN_DEFS, METHODS, s, full_report, load_zreg_map, todo4v2_score,
    load_competitor_allgenes,
)


from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: project root for the clean_data/ repointing, see REPOINT_LOG.tsv]
def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    rows = []
    for twin_def, cfg in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
            zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
            if not os.path.exists(dpath) or not os.path.exists(zpath):
                continue
            dd = pd.read_csv(dpath)
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
            z_reg_map = load_zreg_map(gs, cfg["infer_suffix"])

            U = list(zip(dd.gene_1, dd.gene_2))
            y = np.array([1 if p in CE else 0 for p in U])
            if y.sum() < 3:
                continue

            CC = dd.rho_cross_xy.abs().to_numpy()
            cc_m = full_report(CC, y)

            TODO4V2, _ = todo4v2_score(dd, U, z_reg_map)
            todo4v2_m = full_report(TODO4V2, y)

            best_name, best_m = None, None
            for m in METHODS:
                try:
                    sc = load_competitor_allgenes(m, gs, U)
                except FileNotFoundError:
                    continue
                cm = full_report(s(sc), y)
                if best_m is None or cm["auprc_x"] > best_m["auprc_x"]:
                    best_name, best_m = m, cm

            density = y.sum() / len(U)
            rows.append(dict(
                twin_def=twin_def, gene_set=gs, n_pairs=len(U), n_true=int(y.sum()), density=density,
                crosscorr_auprc_x=cc_m["auprc_x"], todo4v2_auprc_x=todo4v2_m["auprc_x"],
                best_competitor=best_name, comp_auprc_x=best_m["auprc_x"] if best_m else np.nan,
                crosscorr_beats_todo4v2=cc_m["auprc_x"] > todo4v2_m["auprc_x"],
            ))

    df = pd.DataFrame(rows)
    df["todo4v2_over_crosscorr"] = df.todo4v2_auprc_x / df.crosscorr_auprc_x
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out = f"{os.path.dirname(__file__)}/crosscorr_vs_todo4v2_larry.csv"
    out = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/crosscorr_vs_todo4v2_larry.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nmean crosscorr_auprc_x={df.crosscorr_auprc_x.mean():.3f}  "
          f"todo4v2_auprc_x={df.todo4v2_auprc_x.mean():.3f}  "
          f"crosscorr wins {df.crosscorr_beats_todo4v2.sum()}/{len(df)}")
    from scipy.stats import spearmanr
    rho, p = spearmanr(df.density, df.todo4v2_over_crosscorr)
    print(f"spearman(density, todo4v2/crosscorr) = {rho:.3f} (p={p:.3g})")
    rho2, p2 = spearmanr(df.n_pairs, df.todo4v2_over_crosscorr)
    print(f"spearman(n_pairs, todo4v2/crosscorr) = {rho2:.3f} (p={p2:.3g})")


if __name__ == "__main__":
    main()
