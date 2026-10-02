from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from tmp_2026-09-22_phi_v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, os, glob
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
import numpy as np
import pandas as pd
import apply_twinscore_supplement_fatemap_allpairs as m

null_sd = m.null_sd
bootstrap_phi_noise_floor = m.bootstrap_phi_noise_floor
log = m.log
s = m.s
PHI_CLIP = 10.0

def phi_fixed_v2(h1, h2, rho_dagger, genes, meff_twin_t1, meff_twin_t2):
    var_h1, var_h2 = null_sd(meff_twin_t1) ** 2, null_sd(meff_twin_t2) ** 2
    nf1, nf2 = np.sqrt(var_h1), np.sqrt(var_h2)
    phi = {}
    n_floored = 0
    for g in genes:
        h1g, h2g, rdg = h1[g], h2[g], rho_dagger[g]
        ok = np.isfinite(h1g) and np.isfinite(h2g) and h1g > 0 and h2g > 0
        if not ok:
            phi[g] = np.nan
            continue
        h1s, h2s = max(h1g, nf1), max(h2g, nf2)
        n_floored += int(h1g < nf1 or h2g < nf2)
        denom = np.sqrt(h1s * h2s)
        phi[g] = float(np.clip(rdg / denom, -PHI_CLIP, PHI_CLIP))
    return phi, n_floored, nf1, nf2

done = {("fm08","correlation_mid"), ("fm08","correlation_high")}
targets = []
for dataset in ["fm06", "fm08"]:
    for gs in ["variability_high","variability_mid","variability_low",
               "detection_high","detection_mid","detection_low",
               "correlation_high","correlation_mid","correlation_low"]:
        if (dataset, gs) not in done:
            targets.append((dataset, gs))

for dataset, gs in targets:
    gf = f"{TWINFER_PROJECT_ROOT}/analysis_data/{dataset}/data/twinscore_supplement_{dataset}_{gs}_absplit_allpairs_gene_terms.csv"
    pf = gf.replace("_gene_terms.csv", "_pair_terms.csv")
    if not os.path.exists(gf):
        log(f"[{dataset.upper()}/{gs}] MISSING gene_terms file, skipping")
        continue
    gd = pd.read_csv(gf)
    pdf = pd.read_csv(pf)
    genes = gd.gene.tolist()
    h1 = dict(zip(gd.gene, gd.h_t1))
    h2 = dict(zip(gd.gene, gd.h_t2))
    rho_dagger = dict(zip(gd.gene, gd.rho_dagger_gg))
    phi_old = dict(zip(gd.gene, gd.phi))

    DATASET = dataset.upper()
    full, genes_chk, qc_dir, genes_full, gidx = m.load_raw_ab(DATASET, gs)
    t1_raw = full[full.time_step == 0].reset_index(drop=True)
    t2_raw = full[full.time_step == 1].reset_index(drop=True)
    meff_twin_t1 = m.sister_matrix_from_frame(t1_raw, genes)[1]
    meff_twin_t2 = m.sister_matrix_from_frame(t2_raw, genes)[1]

    phi_new, n_floored, nf1, nf2 = phi_fixed_v2(h1, h2, rho_dagger, genes, meff_twin_t1, meff_twin_t2)
    n_changed = sum(1 for g in genes if not np.isclose(phi_old.get(g, np.nan), phi_new.get(g, np.nan), equal_nan=True))
    max_old = np.nanmax(np.abs(list(phi_old.values())))
    max_new = np.nanmax(np.abs(list(phi_new.values())))

    var_h1, var_h2 = null_sd(meff_twin_t1) ** 2, null_sd(meff_twin_t2) ** 2
    thr1, thr2 = 2.0 * np.sqrt(var_h1), 2.0 * np.sqrt(var_h2)
    phi_vals = np.array([v for v in phi_new.values() if np.isfinite(v)])
    var_phi = phi_vals.var(ddof=1) if len(phi_vals) > 1 else np.nan
    noise_var = bootstrap_phi_noise_floor(t1_raw, t2_raw, genes, thr1, thr2)
    floor = float(np.median(list(noise_var.values()))) if noise_var else 0.0
    w_rel_new = max(0.0, 1.0 - floor / var_phi) if (np.isfinite(var_phi) and var_phi > 0) else 0.0

    pdf["phi_x"] = pdf["gene_1"].map(phi_new)
    pdf["TwinScore"] = w_rel_new * s(pdf["phi_x"].to_numpy()) + s(pdf["PAIR"].to_numpy()) + pdf["direction_term"].to_numpy()
    gd["phi"] = gd["gene"].map(phi_new)
    pdf.to_csv(pf, index=False)
    gd.to_csv(gf, index=False)
    log(f"[{DATASET}/{gs}] {n_floored}/{len(genes)} floored, phi changed {n_changed}/{len(genes)}, "
        f"max|phi| {max_old:.2f}->{max_new:.2f}, w_rel={w_rel_new:.3f}")
