# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Shuffle-free ("analytic") TwINFER for the new LARRY gene sets.

Computes the clone-weighted Spearman correlations exactly as infer.py does (same helpers, no
permutation), turns every z-score into its analytic form
    signed 0-centred:   z = rho / (1/sqrt(m_eff-1))
    het (offset):       z = (rho_Delta - rho_Delta_random) / (1/sqrt(m_eff-1))
    d_het:              z = (d - d_random) / sqrt(1/(m_t1-1) + 1/(m_t2-1))
    |.| statistics:     z = (|rho| - s*sqrt(2/pi)) / (s*sqrt(1-2/pi)),   s = null SD
    z_gamma:            z = gamma / (s_cross * sqrt(2*(1-2/pi)))
m_eff = clone-weighted Kish effective sample size from the clone-size distribution.
Then calculate_twin_score -> analytic ranked_edges per gene set.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os, sys, time
import numpy as np, pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    calculate_twin_score,
)

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INP = f"{HERE}/resources/twinfer_input"
INP = f"{RES_HERE}/resources/twinfer_input"
OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/analytic_infer'
os.makedirs(OUT, exist_ok=True)
T1, T2, SEED = 2, 4, 0
S2PI, VAR = np.sqrt(2 / np.pi), (1 - 2 / np.pi)
N_RAND = 60  # random-pair reference draws to average for the het / d_het centre

# Permutation-null SDs are a property of the LARRY clone structure, not the gene set: measured
# once from the completed 5000-shuffle runs (correlation_high / detection_mid; CV < 4% across
# gene sets and pairs). Used in place of the structural Kish 1/sqrt(m_eff-1), which is ~2-9% off
# depending on the null's construction.
SD = dict(step1_t1=0.0172, step1_t2=0.0143, div_t1=0.0256,
          het_t1=0.0250, d=0.0300, change=0.0205, cross=0.0237)

GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]


def kish(sizes):
    sizes = np.asarray(sizes, float)
    return sizes.sum() ** 2 / (sizes ** 2).sum()


def m_effs(df):
    """Clone-weighted Kish effective sample sizes for one input table (gene-set independent)."""
    out = {}
    for tp, tag in ((T1, "t1"), (T2, "t2")):
        g = df[df.time_step == tp].groupby("clone_id").size()
        # step1: cell weight 1/n_c, clone weight 1 -> Sigma w = n_clone, Sigma w^2 = sum 1/n_c
        out[f"step1_{tag}"] = g.sum() ** 2 / g.sum() if False else len(g) ** 2 / (1.0 / g).sum()
        gm = g[g >= 2]; cn2 = gm * (gm - 1) / 2
        out[f"twin_{tag}"] = len(gm) ** 2 / (1.0 / cn2).sum()
    d2 = set(df[df.time_step == T1].clone_id); d4 = set(df[df.time_step == T2].clone_id)
    g2 = df[df.time_step == T1].groupby("clone_id").size(); g4 = df[df.time_step == T2].groupby("clone_id").size()
    both = d2 & d4
    out["cross"] = len(both) ** 2 / sum(1.0 / (g2[c] * g4[c]) for c in both)
    return out


def analytic_for_geneset(gs):
    df = pd.read_csv(f"{INP}/{gs}.csv")
    genes = [c[:-5] for c in df.columns if c.endswith("_mRNA")]
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    t1_tw = assign_twin_id(t1_raw).reset_index(drop=True)
    t2_tw = assign_twin_id(t2_raw).reset_index(drop=True)

    rho1 = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, genes, use_clone=True)
    rho2 = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, genes, use_clone=True)

    tw1, _ = calculate_twin_random_correlations(t1_raw, t1_tw, genes, random_state=SEED, unit="clone")
    tw2, _ = calculate_twin_random_correlations(t2_raw, t2_tw, genes, random_state=SEED, unit="clone")
    # average N_RAND fresh random-pair reference draws for a stable centre
    r1 = np.mean([calculate_twin_random_correlations(t1_raw, t1_tw, genes, random_state=SEED + 100 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    r2 = np.mean([calculate_twin_random_correlations(t2_raw, t2_tw, genes, random_state=SEED + 200 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    rand1 = pd.DataFrame(r1, index=genes, columns=genes)
    rand2 = pd.DataFrame(r2, index=genes, columns=genes)

    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in genes for b in genes if a != b]
    xcorr = get_cross_correlations(at1, at2, gene_pairs=ordered + [(g, g) for g in genes], unit="clone")

    M = m_effs(df)
    s1a, s1b = SD["step1_t1"], SD["step1_t2"]
    stw_a = SD["div_t1"]
    s_het = SD["het_t1"]
    s_d = SD["d"]
    s_change = SD["change"]
    s_x = SD["cross"]

    rows = []
    for a, b in ordered:
        rt1, rt2 = float(rho1.loc[a, b]), float(rho2.loc[a, b])
        rd1, rd2 = float(tw1.loc[a, b]), float(tw2.loc[a, b])
        rr1, rr2 = float(rand1.loc[a, b]), float(rand2.loc[a, b])
        rchg = rt2 - rt1
        d = rd2 - rd1
        rxy, ryx = float(xcorr.loc[a, b]), float(xcorr.loc[b, a])
        gamma = abs(rxy) - abs(ryx)
        rows.append(dict(
            gene_1=a, gene_2=b,
            rho_t1=rt1, rho_t2=rt2, rho_delta_t2=rd2, rho_change=rchg,
            z_abs_rho_t1=(abs(rt1) - s1a * S2PI) / (s1a * np.sqrt(VAR)),
            z_abs_rho_t2=(abs(rt2) - s1b * S2PI) / (s1b * np.sqrt(VAR)),
            z_abs_rho_change=(abs(rchg) - s_change * S2PI) / (s_change * np.sqrt(VAR)),
            z_rho_change=rchg / s_change,
            z_div=rd1 / stw_a,
            z_het=(rd1 - rr1) / s_het,
            z_d_het=(d - (rr2 - rr1)) / s_d,
            z_d=(d - (rr2 - rr1)) / s_d,
            rescued_from_no_regulation=False,
            rho_cross_xy=rxy, rho_cross_yx=ryx,
            gamma=gamma,
            z_gamma=gamma / (s_x * np.sqrt(2 * VAR)),
            is_final_directed_edge=False,
        ))
    tsi = pd.DataFrame(rows)
    ranked = calculate_twin_score({"twin_score_inputs": tsi})
    ranked = ranked.sort_values("twinScore", ascending=False).reset_index(drop=True)
    d_out = f"{OUT}/{gs}"; os.makedirs(d_out, exist_ok=True)
    tsi.to_csv(f"{d_out}/twin_score_inputs.csv", index=False)
    ranked.to_csv(f"{d_out}/ranked_edges.csv", index=False)
    json.dump({k: float(v) for k, v in M.items()}, open(f"{d_out}/m_eff.json", "w"), indent=2)
    return len(ranked), M


if __name__ == "__main__":
    only = sys.argv[1:] or GENE_SETS
    for gs in only:
        t = time.time()
        n, M = analytic_for_geneset(gs)
        print(f"{gs:18s}: {n:5d} directed edges, m_eff step1_t1={M['step1_t1']:.0f} "
              f"twin_t1={M['twin_t1']:.0f} twin_t2={M['twin_t2']:.0f} cross={M['cross']:.0f}  "
              f"({time.time()-t:.0f}s)", flush=True)
    print(f"\nwrote analytic ranked_edges to {OUT}/<gene_set>/")
