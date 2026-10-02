# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys
import numpy as np

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "code/TwINFER/paper_analysis/fatemap_pipeline")
import apply_twinscore_supplement_fatemap_allpairs as m

null_sd = m.null_sd
load_raw_ab = m.load_raw_ab
sister_matrix_from_frame = m.sister_matrix_from_frame

N_PERM = 200
SEED = 0

for dataset, gs in [("FM06", "correlation_high"), ("FM08", "correlation_mid")]:
    full, genes, qc_dir, genes_full, gidx = load_raw_ab(dataset, gs)
    t1_raw = full[full.time_step == 0].reset_index(drop=True)

    C_real, meff_t1 = sister_matrix_from_frame(t1_raw, genes)
    analytic_sd = null_sd(meff_t1)

    rng = np.random.default_rng(SEED)
    perm_h = {g: [] for g in genes}
    for p in range(N_PERM):
        shuffled = t1_raw.copy()
        shuffled["clone_id"] = rng.permutation(shuffled["clone_id"].to_numpy())
        try:
            C_perm, _ = sister_matrix_from_frame(shuffled, genes)
        except ValueError:
            continue
        for g in genes:
            perm_h[g].append(C_perm.loc[g, g])

    emp_sds = {g: np.std(perm_h[g], ddof=1) for g in genes if len(perm_h[g]) > 5}
    emp_means = {g: np.mean(perm_h[g]) for g in genes if len(perm_h[g]) > 5}

    sds = np.array(list(emp_sds.values()))
    means = np.array(list(emp_means.values()))
    print(f"=== {dataset}/{gs}  (n_genes={len(genes)}, meff_twin_t1={meff_t1:.1f}) ===")
    print(f"  analytic null_sd(meff_t1) = {analytic_sd:.4f}")
    print(f"  empirical permutation-null SD of h, across {len(sds)} genes: "
          f"median={np.median(sds):.4f}  mean={sds.mean():.4f}  "
          f"[min={sds.min():.4f}, max={sds.max():.4f}]")
    print(f"  empirical permutation-null MEAN of h (should be ~0 if unbiased): "
          f"median={np.median(means):.4f}  mean={means.mean():.4f}")
    print(f"  ratio analytic/empirical (median): {analytic_sd/np.median(sds):.3f}")
    print()
