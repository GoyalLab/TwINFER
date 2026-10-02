# Analytic (shuffle-free) TwINFER z-scores — derivation

`analytic_zscores.py` replaces each 5000-permutation null with a Gaussian whose mean and variance
are computed in closed form. This note derives those two moments, the three correction terms, and
states where the approximation breaks. Validation on the 6 drift/regulation scenarios:
`validate_analytic_zscores.py` (overall corr 0.9999, median |Δz| 0.016 across 7 z-types).

---

## 1. Every TwINFER z-score is a permutation test of a weighted Spearman correlation

Let there be `m` **exchangeable units** (biological clones, or twin pairs). Unit `i` carries a
weight `w_i` (clone mode: each clone's weights sum to 1) and a pair of rank values
`(a_i, b_i)` — for step 1 these are the weighted ranks of gene X and gene Y across cells; for the
Δ-statistics they are the weighted ranks of the twin differences ΔX, ΔY.

The observed statistic is a **weighted Pearson correlation of ranks**:

```
              Σ_i w_i · ã_i · b̃_i
  ρ̂  =  ────────────────────────────────── ,      ã_i = a_i − (Σ w a)/(Σ w),   b̃ likewise
        sqrt( Σ_i w_i ã_i²  ·  Σ_i w_i b̃_i² )
```

The permutation null re-labels one side: replace `b̃_i` by `b̃_{π(i)}` for a random permutation π
(step 1's clone-matching null and the direction null are of this form; z_div's fixed-Δ null and
z_het's fresh-random-pair null are close variants). The denominator is invariant under π, so the
**null statistic is linear in the permutation**:

```
  T(π) = c · Σ_i w_i ã_i b̃_{π(i)} ,      c = 1 / sqrt( Σ w ã² · Σ w b̃² )   (a constant).
```

Statistics of the form `Σ_i u_i v_{π(i)}` have exact permutation moments (Pitman 1937, Hoeffding
1952) — no resampling needed.

---

## 2. The two permutation moments

Write `u_i = w_i ã_i`, `v_i = b̃_i`. Over a uniform random permutation π of `{1..m}`:

**Mean.**
```
  E_π[ Σ u_i v_{π(i)} ]  =  (Σ_i u_i)(Σ_i v_i) / m .
```
Both `ã` and `b̃` are weighted-centred, so `Σ w ã = 0` ⇒ `Σ u_i = 0`. Hence **`E_π[T] = 0`** for
every "destroy the X–Y association" null (step 1, z_div, z_d_div, z_rho_cross, z_rho_change).

**Variance.** With `S_u^{(k)} = Σ_i u_i^k`, `S_v^{(k)} = Σ_i v_i^k`:
```
  Var_π[ Σ u_i v_{π(i)} ]
      =  (1/(m−1)) [ S_u^{(2)} S_v^{(2)}  −  (1/m) (S_u^{(1)})² (S_v^{(1)})² ]
       + (correction terms in S_u^{(1)}S_u^{(2)}, S_v^{(1)}S_v^{(2)}, ...).
```
With `S_u^{(1)} = S_v^{(1)} = 0` this collapses to
```
  Var_π[T]  =  c² · S_u^{(2)} S_v^{(2)} / (m − 1) .
```
Now `S_v^{(2)} = Σ b̃_i²` and, because `u_i = w_i ã_i`, `S_u^{(2)} = Σ w_i² ã_i²`. For **equal
weights** `w_i ≡ 1/m` this gives `c² S_u^{(2)} S_v^{(2)} = (Σ ã²·Σ b̃²)/m² · (1/(Σ ã²·Σ b̃²)) · … `
which reduces to the textbook result
```
  Var_π[ρ̂]  ≈  1 / (m − 1) .
```

So to leading order the null is **`N(0, 1/(m−1))`** and

```
  z  ≈  ( ρ̂_obs − 0 )  ·  sqrt(m − 1) .
```

---

## 3. Correction term 1 — `m` is the Kish effective sample size, not the raw count

`Var_π[T] = c² · (Σ w_i² ã_i²) · (Σ b̃_i²) / (m−1)`. With unequal weights the factor
`Σ w_i² ã_i²` is **not** `(Σ w_i ã_i²)/m`. Approximating `ã_i²` as roughly constant in `i`,

```
  Σ w_i² ã_i²        Σ w_i²             1
  ──────────────  ≈  ───────  =  ──────────────  ,      m_eff  =  (Σ w_i)² / Σ w_i²   (Kish).
  Σ w_i ã_i²         Σ w_i         m_eff / (Σ w_i)
```

So `sd(ρ̂_null) ≈ 1 / sqrt(m_eff − 1)` with `m_eff` the **clone-weighted Kish effective sample
size**. Concretely, from the clone-size distribution:

| statistic | exchangeable unit & weights | `m_eff` |
|---|---|---|
| step 1 (gene–gene ρ, `use_clone=True`) | cell weight `1/n_c`, clone total weight 1 | `n_clone² / Σ_c (1/n_c)` |
| z_het, z_div (twin ρ_Δ, `unit="clone"`) | clone weight 1 split over its `C(n_c,2)` enumerated twins | `n_twinclone² / Σ_c 1/C(n_c,2)` |
| z_rho_cross, z_gamma (cross-time ρ) | clone weight 1 split over its `n2·n4` cross pairs | `n_bothday² / Σ_c 1/(n2_c n4_c)` |
| z_d_het, z_d_div (`d = ρ_Δ(t2) − ρ_Δ(t1)`) | disjoint t1/t2 twin sets ⇒ Var adds | `1 / (1/m_twin_t1 + 1/m_twin_t2)` |

These are `m_eff_step1 / m_eff_twin / m_eff_cross / m_eff_d` in the module.

**Why it matters:** for LARRY day 2, `Σ C(n_c,2)` over sibling clones is 3,424 twin pairs, but
`m_eff_twin ≈ 1,538` — because the `C(n_c,2)` twins inside a clone are highly correlated, and the
raw sibling-clone count (1,076) is too low because those twins still carry information. Only the
Kish value fits the observed null SD.

---

## 4. Correction term 2 — calibrate the SD once against the null's actual construction

The Kish `1/sqrt(m_eff − 1)` is the leading term. It is **2–15 % off** depending on the null:

- the **clone-matching** step-1 null (derange clones, draw one target cell per source cell) carries
  slightly *more* variance than Kish predicts (`m_implied ≈ 0.9 × Kish`);
- the **fixed-Δ** z_div null (hold ΔX, re-pair ΔY across *other* clones) is a *restricted*
  permutation ⇒ variance `× (1 − 1/n_clone)`, negligible here;
- the **cell-time-label-shuffle** z_change null pools the t1 and t2 cells and re-splits, so the two
  pseudo-groups are *positively* correlated ⇒ `Var(Δρ_null)` is **smaller** than `1/(m_t1−1) +
  1/(m_t2−1)` (`m_implied ≈ 2,800` vs the naive ≈ 2,100 for LARRY).

Because the null SD is a function of the **clone-size distribution only** — not the gene set
(CV < 4 % across gene sets and across pairs) — it is measured **once** from any completed
5000-shuffle run (or one short shuffle job) and reused. The module ships the LARRY days-2/4
values as `DEFAULT_SD`:

```
SD_STEP1_T1 = 0.0172   SD_STEP1_T2 = 0.0143   SD_DIV_T1 = 0.0256   SD_DIV_T2 = 0.0154
SD_HET_T1   = 0.0250   SD_D        = 0.0300   SD_CHANGE = 0.0205   SD_CROSS  = 0.0237
```

In `validate_analytic_zscores.py` the sims' own reported `*_null_std` medians are used as the
calibrated SDs — the same "measure once" step. With them, every slope is 1.00 (§7); with the
uncalibrated Kish, step 1's slope is 0.65 (analytic z ~1.5× too big for the large-|z| multistate
pairs).

---

## 5. Correction term 3 — the non-zero centre for z_het / z_d_het

z_het compares the *twin* `ρ_Δ(t1)` to the `ρ_Δ` of **fresh unrelated cell pairs**. Random pairs
still share the population-level covariance of ΔX and ΔY, so

```
  E_π[ ρ_Δ,random ]  ≠  0 .
```

On LARRY this offset is ≈ 0.16 (≈ 6 SD); in the two-state *multistate* sims it reaches ≈ 0.45
(≈ 20 SD). So

```
  z_het   =  ( ρ_Δ(t1) − E[ρ_Δ,random(t1)] ) / SD_HET_T1 ,
  z_d_het =  ( d − ( E[ρ_Δ,random(t2)] − E[ρ_Δ,random(t1)] ) ) / SD_D .
```

**Estimating the centre.** `infer.py` stores one draw as `random_delta_reference_matrix`; a single
draw has SE ≈ `SD_HET_T1` ≈ 1 z-unit, which is the whole error (`z_het_naive` in §7: median
|Δz| 0.61). Average **~30–60 fresh re-pairings** (`calculate_twin_random_correlations` with
different `random_state`) — SE drops to `SD_HET_T1 / sqrt(50)` ≈ 0.14 z-units — or use
`corr_random_delta_t1/t2.csv` if it already aggregates. This is still ~100× cheaper than the
5000-draw null and yields median |Δz| 0.004 (§7).

z_div, z_d_div, z_rho_cross, z_rho_change keep centre 0.

---

## 6. The `|·|` statistics and z_gamma

`z_abs_rho_t1`, `z_abs_rho_t2`, `z_abs_rho_change` compare `|ρ̂_obs|` to `|null draw|`. If the
null draw is `X ~ N(0, s)` then `|X|` is **half-normal**:

```
  E|X| = s·sqrt(2/π)  ≈  0.7979 s ,      SD|X| = s·sqrt(1 − 2/π)  ≈  0.6028 s .
```

Hence

```
  z_abs  =  ( |ρ̂_obs| − s·sqrt(2/π) )  /  ( s·sqrt(1 − 2/π) ) .          # z_abs() in the module
```

`z_gamma` scores `γ = |ρ_cross(x→y)| − |ρ_cross(y→x)|`. Its null is `|A| − |B|` with
`A, B ~ N(0, s_cross)` from the two directional cross-correlations. Treating them as
approximately independent,

```
  Var(|A| − |B|)  =  2 · Var|A|  =  2 s_cross² (1 − 2/π)  ,
  z_gamma  =  γ  /  ( s_cross · sqrt( 2 (1 − 2/π) ) ) .                   # z_gamma() in the module
```

(The two direction nulls share the same paired-permutation draws, so a small positive
`Cov(|A|,|B|)` is neglected; validation median |Δz| = 0.02.)

---

## 7. Validation (6 scenarios, K_frozen excluded)

`no_regulation, A_to_B, multistate_A_to_B, multistate_A_B, kramp_A_to_B, kramp_A_B`,
t1 ∈ {1, 10}, t2 = 20; 360 pair-steps. Analytic z vs the stored 5000-shuffle z:

| statistic | corr | median \|Δz\| | max \|Δz\| | slope |
|---|---|---|---|---|
| step1 | 0.9998 | 0.062 | 0.78 | 1.000 |
| z_div | 0.9999 | 0.012 | 0.04 | 0.998 |
| z_het | 0.9999 | **0.004** | 0.62 | 1.001 |
| z_het_naive (single-draw centre) | 0.9907 | 0.607 | 2.57 | 1.020 |
| z_d_het | 0.9999 | 0.015 | 0.07 | 1.004 |
| z_d_div | 0.9994 | 0.018 | 0.16 | 0.983 |
| z_rho_cross | 1.0000 | 0.015 | 0.25 | 1.001 |
| z_gamma | 0.9999 | 0.018 | 0.19 | 1.002 |
| **overall (excl. naive)** | **0.9999** | **0.016** | — | — |

The permutation nulls are Gaussian in this regime (|skew| ≈ 0.04, empirical 99th percentile within
~1 % of 2.326 σ), so the Gaussian z → p mapping is valid. On real LARRY data the agreement is the
same (median |Δz| ≈ 0.02–0.09; `validate` numbers on the completed 5000-shuffle gene sets).

---

## 8. Where it breaks

- **Few exchangeable units** (small `m_eff` — few sibling clones at a timepoint, or one clone
  dominating the weight). The CLT margin shrinks and the null skews; use analytic 3rd/4th
  permutation moments + Cornish–Fisher, or a saddlepoint approximation of the permutation CDF.
  Not an issue for LARRY (1,076–1,918 sibling clones/timepoint) or the sims.
- **`z_reg_gated`** — the null reassigns Y *sibling blocks* (block structure), so its analytic SD
  is only approximate. `z_reg_gated()` uses `SD_HET_T1` as a proxy.
- **The heterogeneity centre must be averaged** (§5). A single re-pairing leaves ~0.6 z-units of
  error.

---

## 9. Files

| file | purpose |
|---|---|
| `analytic_zscores.py` | the module: `m_eff_*`, `z_signed`, `z_abs`, `z_gamma`, `z_reg_gated`, `analytic_twin_score_inputs`, `DEFAULT_SD` |
| `validate_analytic_zscores.py` | §7 table + scatter grid `validate_analytic_zscores.png` from the drift-sim rep JSONs |
