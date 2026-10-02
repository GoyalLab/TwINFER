# Handoff — 2026-09-20: TwinScore supplement implementation, validation, and φ (persistence) investigation

## What was asked
Implement the continuous TwinScore formula from `TwinScore_supplement.pdf` (S, C, φ/persistence,
D, R, Wz, γ/κγ/q, final score) on real LARRY hematopoiesis data, using the gene panels/methodology
referenced in `apply_todo4v2_partial_corr_larry.py`.

## Main script
`apply_twinscore_supplement_larry.py` — implements the full formula. Key design decisions, all
disclosed in the script's own docstrings:
- **Gene panels**: NOT `resources/gene_sets_yscher.json` (TwINFER's own recreation, subtly wrong —
  see below). Uses `yscher_genes()` / `YSCHER_STEM`, which pulls the ACTUAL gene lists yscher used,
  from `/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/twinscore_script/
  larry_log1pPF_stable24_det5_p4a01_{corrhigh,corrmid,corrlow,hvg_q67100,hvg_q3367,hvg_q0033}_24.json`.
- **S, z_het, direction (z†/γ)**: reused verbatim from yscher's own real-permutation run (same
  JSONs above have `rho_t1/t2`, `z_het`, `fwd/rev`, `z_fwd/z_rev`) rather than recomputed
  analytically — the PDF states these ARE the manuscript's existing statistics.
- **C, heritability (h), persistence (φ)**: computed fresh via the package's own
  `assign_twin_id`/`split_twins`/`get_cross_correlations` (real weighted-Spearman, no permutation
  needed for these — they're just correlations).
- **D**: real PIDC (Bayesian-blocks discretization + PUC), adapted from
  `preprocessing/run_competitor_methods.py`.
- **Wz**: real Schäfer-Strimmer-style shrinkage partial correlation on split-half binomial-thinned
  counts, with a genuine 20-replicate bootstrap for both the shrinkage intensity and the SE (see
  bug #1 below).
- **R, zreg**: `sd_reg` uses the repo's existing calibrated proxy (`DEFAULT_SD["het_t1"]`) since
  yscher's own code doesn't compute this exact quantity either — no better source found.

## Two real bugs found and fixed (both same root cause: analytic noise-variance proxy → real bootstrap)

**Bug #1 — Wz's signal-share gate `v` was badly inflated.** Original `se` for the shrunk partial
correlation used an analytic formula (`1/sqrt(meff-1)`, later tried `1/sqrt(meff-k-3)` — didn't
help, k is negligible next to meff in the thousands). Checked directly against yscher's real
`twin_layers/real/{run}.pkl` output for this exact panel: her `v=0.00` (no real Wz signal on
LARRY), mine was `v=0.88`. **Fix**: genuine bootstrap — 20 independent binomial split-half draws,
shrinkage intensity and SE both computed from the empirical variance across draws, not a
closed-form formula. Result: `v` now matches her `0.00` exactly.

**Bug #2 — persistence reliability `w` collapsed to exactly 0.00 on every panel except the two
"high" ones.** Same root cause: the PDF's own delta-method formula for φ's noise floor blows up
when h1/h2 are small (every panel except correlation_high/variability_high has median h ~0.03-0.11
vs ~0.13-0.20). **Fix**: `bootstrap_phi_noise_floor()` — clone-level cluster bootstrap (resample
clones with replacement, rebuild h1/h2/ρ† from each resampled draw), same significance gate
(h>2·null_sd) applied *inside* each bootstrap replicate too (first version didn't do this, which
let near-zero-denominator replicates blow up the variance to as high as 146 on one gene — fixed by
adding the gate inside the loop). Result on `correlation_high`: w 0.82→0.76 (close to her 0.79).
On `correlation_mid`/`correlation_low`, still landed at exactly w=0.00 even after the fix — this
turned out to be a **third, separate, non-bug finding** (see below).

## Third finding: her own code deviates from the PDF text for `corrmid`/`corrlow`
Traced exactly why my post-bugfix w still didn't match her 0.49/0.73 for corrmid/corrlow. Her
`twinscore_mega_A.py`'s φ-numerator, for any panel not in her `B_pooled_rho_tw.pkl` cache (only
the 6 corrhigh lists are cached — corrmid/corrlow are NOT), falls back to `dg = mem[gene]` from
`regdet/realMem_{run}.pkl` — a DIFFERENT cached quantity (correlates with my ρ†_gg at only r=0.61,
not the same thing), not literally ρ†(g,g) as the PDF specifies. Independently confirmed her own
heritability inputs (h1/h2) DO match mine almost exactly (r=0.955-0.991) — so the heritability
estimator was never the issue, only the φ-numerator substitution for uncached panels.
**Resolution**: replicated her exact fallback formula using her own cached `mem`/`icc_nosplit`/`SD`
files for corrmid/corrlow specifically → w now matches her values to 3 decimals (0.491 vs 0.49,
0.728 vs 0.73). `correlation_high` and `variability_*` still use the PDF-literal ρ†_gg formula
(no cached fallback exists for variability_* since it's not one of her original panels).

## Final validated numbers (`twinscore_supplement_larry_summary.csv`)

| gene_set | calls | hits | AUROC | AUPRC ×base | w | gate_g | κγ |
|---|---|---|---|---|---|---|---|
| correlation_high | 107 | 21 | 0.717 | 3.04x | 0.759 | 0.973 | 0.803 |
| correlation_mid | 360 | 25 | 0.510 | 1.05x | **0.491** (her: 0.49) | 0.353 | 0.113 |
| correlation_low | 341 | 23 | 0.505 | 1.01x | **0.728** (her: 0.73) | 0.450 | 0.069 |
| variability_high | 164 | 16 | 0.637 | 1.50x | 0.579 | 0.965 | 0.736 |
| variability_mid | 112 | 3 | 0.511 | 1.04x | 0.000 | 0.000 | 0.042 |
| variability_low | 97 | 3 | 0.493 | 1.04x | 0.000 | 0.520 | 0.275 |

`correlation_high` vs yscher's mega table (`RESULTS_FOR_DESKTOP_2026-09-17.md`, LARRY day2-4
corrhigh row): calls 107=107 exact, hits 21 vs her 22 (close), AUPRC×base 3.04x vs her stated
4.15x (top-k-precision-based)/3.21x (AUPRC-based) — close on the AUPRC-based version.

## Panel-selection confound identified (important interpretive caveat)
`correlation_high/mid/low` panels were built by selecting genes FROM curated CollecTRI edges
ranked by |ρ| (top/mid/bottom band) — so comparing TwinScore's performance across these three
bands mostly re-measures how separable the reference edges are from noise at each |ρ| level, not
how good TwinScore is per se (confirmed: ALL competitor methods degrade identically on
corrmid/corrlow, e.g. best competitor drops from PIDC 2.36x on corrhigh to rho 1.22x on corrmid).
`variability_high/mid/low` (dispersion-selected, NOT correlation-based) are the fairer test of
TwinScore's non-correlation machinery — `variability_high` beats its own best competitor (1.50x vs
PIDC 1.18x), a real, non-circular win.

## Non-TF source-gating gap found and quantified
TwinScore's formula has NO concept of "is this gene a TF" anywhere — `w·s(φ_x)` and the full pair
universe treat every panel gene as a potential source. On `correlation_high`'s top-107 ranked
pairs: **72/107 (67%) have a non-TF source gene** and can therefore NEVER be a true hit (CollecTRI
never lists a non-TF as a source; confirmed 0/72 are true edges) — they just crowd out real
TF-sourced candidates. Restricting the ranking to TF-sourced-only pairs (690 of 2162 pairs, still
covers all 107 true edges): hits rise from 21→35, precision 19.6%→32.7%. BUT: AUROC stays flat
(0.717→0.705) and AUPRC×base actually DROPS (3.04x→2.41x) — because removing ~1470 near-certain
non-hits also raises the base/random rate (4.95%→15.51%) faster than the score improves. **AUROC
is the metric that's actually base-rate-insensitive here; AUPRC and AUPRC×base are NOT** (a random
classifier's expected AUPRC equals the base rate) — this was a real self-correction mid-session,
worth remembering. Net: TF-gating is a real precision fix, not a ranking-quality fix — TwinScore
was already ranking TF-sourced pairs about this well within the full universe.

Re-ran GENIE3/GRNBoost2 with candidate regulators restricted to the panel's actual TF-like genes
(script: `rerun_competitors_tf_only.py`, ran via SLURM `--account=b1042 --partition=genomics`)
and post-hoc filtered PIDC/ppcor/rho (symmetric methods, TF-restriction doesn't change their
ranking) to the same TF-sourced-only universe. **TwinScore still wins on every metric** (AUROC
0.705, AUPRC 0.374, ×base 2.41x) vs best competitor PIDC (2.15x) — a smaller, more defensible
margin than the original 3.04x-vs-2.36x headline, but real.

## φ (persistence) investigation — what it is and isn't
- **Formula**: φ_g = ρ†(g,g) / √(h_g(t1)·h_g(t2)) — an attenuation-correction (classical test
  theory): normalizes the raw cross-time self-correlation by how strong the heritable signal is at
  each timepoint, isolating "what fraction of whatever real signal exists survives the gap" from
  "how strong is the signal" (two genuinely independent axes — demonstrated with a table where
  Ybx1 has high signal/negative persistence, Junb has the lowest signal but high persistence).
- **Negative φ ≠ "no signal"** — it's a real, structured *inverted* pattern (day-2-high clones are
  reliably day-4-low). Distinct from φ≈0 (true noise, no cross-time relationship). Runx1 shows a
  significant negative φ (−0.34) — surprising given its centrality to hematopoiesis, unexplained.
- **NOT a TF-detection signal**: tested via random TF draws, top-100-detected TF vs non-TF
  (confounded by pool-size effects — TF pool is 1815 genes vs 23474 non-TF, so "top 100 most
  detected" reaches a more extreme percentile for non-TF purely from sample size, not a real
  expression-level difference — TFs actually have HIGHER median detection genome-wide, confirmed
  by direct histogram check). Properly detection-matched (100 TF + 100 nearest-neighbor-matched
  non-TF, repeated as 10 independent fresh draws, pooled n=180 TF/193 non-TF defined-φ genes):
  real but MODEST effect, TF median φ 0.101 vs non-TF 0.045, one-sided MWU p=0.027. Direction
  consistent in 9/10 independent draws, but effect size is small relative to within-group spread
  (sd ~0.33-0.36 vs median gap ~0.06) — TF-status is a weak, noisy predictor of φ, not a strong one.
- **CollecTRI-target enrichment**: high-φ non-TF genes ARE enriched for being CollecTRI targets,
  but the FIRST version of this test (z=20) was confounded by pooling in the CollecTRI-edge-selected
  panels (correlation_high/mid/low — their non-TF genes are targets BY CONSTRUCTION, not because
  of φ). Restricted to only detection/dispersion-selected sources: still significant but smaller,
  z=18, 109/152=71.7% vs 17.1% genome-wide base rate — a real, if partly self-fulfilling
  (CollecTRI targets tend to be well-studied/expressed genes), effect.
- **What φ actually identifies**: specific individual genes with a durable, division-surviving
  expression identity — NOT a category (not TF, not "regulator", not "lineage-relevant"). Top
  recurring hits across every independent check: **Gata1 (φ=0.99, highest of anything checked),
  Gata2 (0.86), Cebpa/b/e (0.56-0.69)** — all classic myeloid/erythroid fate-commitment master
  regulators known to use autoregulatory/bistable-switch mechanisms. Bottom: generic
  high-turnover housekeeping/RNA-binding machinery (Fus, Ybx1, Pin1, Npm1, Smarca5) — consistently
  negative across every panel checked.
- **Best-supported explanation for WHY φ works here**: this is a differentiation dataset, and
  differentiation IS the process of establishing a new heritable identity via exactly the
  autoregulatory mechanism φ detects. Consistent with yscher's own research log: φ actively HURTS
  performance on the held-out Perturb-seq (causal knock-down) reference lists — "φ(x) is a
  CollecTRI-TF-class prior... not regulator evidence." φ tracks fate-lock-in specifically, not
  regulatory activity in general.
- **Jun vs Junb, a cautionary tale on interpreting individual gene φ values**: Jun (φ=0.21) vs
  Junb (φ=0.68) — same AP-1 family. Initially attributed to Jun's detection nearly tripling
  day2→day4 (acutely inducible) vs Junb's stability — but a follow-up clone-level delta-decomposition
  test did NOT support this mechanism (if anything showed the opposite pattern for the
  concentration-of-increase metric) — **this speculative mechanism was retracted, not confirmed**.
  Separately, Junb's own φ=0.68 is itself suspect: its h1/h2 clear the significance gate by the
  THINNEST margins of anything checked (+0.008, +0.004 over threshold, vs Gata2's +0.27/+0.43) —
  likely an unstable, noise-dominated point estimate, not trustworthy evidence on its own.
- Checked real, well-known HSC/myeloid regulator gene lists (`all_hematopoiesis_regulators_phi.csv`,
  68 genes incl. yscher's actual "panel12" 11-gene canonical panel: Cebpa, Egr1, Fli1, Gata1,
  Gata2, Gfi1, Jun, Nab2, Spi1, Tal1, Zfpm1). Even hand-curated "important" panels span the full
  range (Gata1 0.99 to Zfpm1 −0.18). **Spi1/PU.1 — arguably THE most-cited myeloid master
  regulator — fails to clear the significance gate in EVERY check run (undefined φ each time)**,
  likely a real detection/technical limitation (dosage-sensitive, lowly expressed) rather than
  evidence against its importance. Restricting to genes plausibly relevant to MYELOID
  differentiation specifically (not generic HSC self-renewal, which showed no signal at all —
  Meis1/Hoxa9/Mecom/Hlf all undefined, plausibly because this assay window isn't sampling
  self-renewing HSCs) gave the highest, most reliable panel-wide `w=0.72` of the whole
  investigation — real granule/effector genes (Cpa3, Csf2rb, Itgam, Csf1r, Prtn3, Cybb, Ctsc,
  Elane) show real, substantial φ alongside the Cebp-family TFs.

## Open / unresolved at end of session
Was attempting to replicate the same 68-gene φ check on the CellTag dataset (local data found:
`/gpfs/projects/b1255/hzhang/TwINFER_KA/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad`,
clone_id=`repaired_clone`, `Time point` 2=day2.5/4=day5, X already log1pPF-normalized, all 68 genes
present) — but the computation was silently producing 0-byte output / no CSV across two attempts,
with a "completed, exit 0" status that doesn't match a script that should print several lines.
Was mid-diagnosis (isolating whether this is a background-execution/infra issue vs a script bug)
when this handoff was requested. **Next step**: re-run this interactively (foreground, small
timeout) rather than backgrounded, to see the actual error directly, before trying again in the
background.

## Key files
- `apply_twinscore_supplement_larry.py` — main implementation (all fixes above are in here, dated
  comments mark each one)
- `twinscore_supplement_larry_summary.csv` — final validated summary, all 6 panels
- `{gene_set}_pair_terms.csv` / `{gene_set}_gene_terms.csv` — full per-pair/per-gene term tables,
  6 panels
- `all_hematopoiesis_regulators_phi.csv` — 68-gene HSC/myeloid/panel12 φ table (LARRY)
- `rerun_competitors_tf_only.py` — TF-restricted GENIE3/GRNBoost2 + post-hoc-filtered PIDC/ppcor/rho
- `tf_nontf_phi_draw.py` + `tf_nontf_phi_draw_{0..9}.csv` — the 10-draw detection-matched TF vs
  non-TF replication
- SLURM: submit via `sbatch --account=p32655 --partition=short` for the main pipeline (32G mem,
  ~10-20 min per panel), `--account=b1042 --partition=genomics` for GENIE3/GRNBoost2 reruns
