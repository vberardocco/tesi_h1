# R9 · Checks for intrinsic dimension (steps 2 and 3) — rules

Written on 5 Nov, **before** computing the curves on the real data. **Post hoc** analysis: it does not change R9
("Prediction P2, which we derived from Altan 2021, is not confirmed in any of the 15 cells (5 estimators × 3
sessions)") and must be stated in this way in every file and every slide. Prompt: `prompt_r9_controlli.md` of 5 Nov.
Step 1 (estimators verified, 21 rows out of 21): condition `9_verifica_stimatori_id`.

## Data

- **H1**: input (a) of R9 in S1, S2, S5 (pipeline of notebook 02: Gaussian 50 ms = 2.5 bins, z-score; bins of the
  movement phases in temporal order; `literature_cache/phase2/S*_a.npy`).
- **Twins**: the 5 random-phase twins of R9 per session (`S*_d0..d4.npy`).
- **Noise-free control**: expected rates of the generator, smoothed as (a) and with z-score, without Poisson.
- **Noisy control**: Poisson counts of the generator with preprocessing (a), at three shared fractions.
- Only input (a): (b) GPFA and (c) condition means cannot be reproduced in a simple way on the generator.

## Generator (literature2, new functions)

1. Latents: d + 1 Gaussian noise signals smoothed in time with σ_lat bins, normalized point by point
   (points on S^d). d = 6 (7 signals); d = 3 (4 signals) for T3.
2. Map: F = tanh(Z W + b), 64 units, W ~ N(0, 1) · 0.7 · √7, b ~ U(−0.5; 0.5); S = F A, A ~ N(0, 1) 64 × 160;
   per-channel z-score.
3. Rates: λ = (mean rate of the channel in S1) · exp(g · S − g²/2) · 0.02 s; Poisson counts. The 160 rates
   are those of the channels of S1 on the bins of input (a).
4. Preprocessing as (a): Gaussian σ = 2.5 bins, per-channel z-score (channels with zero variance removed).
5. 5507 bins in temporal order, 160 channels (as (a) of S1). σ_lat ∈ {5, 10} bins, both reported. 3 seeds.

**Gain g.** It is chosen (bisection, seed 0) so that the shared fraction per channel **measured as in slide 5**
(FA on the smoothed counts σ = 50 ms and with z-score, K by cross-validation with 5 contiguous blocks on {5, 10, 15, 20, 30},
median of the fraction per channel ||w_i||² / (||w_i||² + ψ_i)) is 0.167 (H1, S1), then 0.45 and 0.65. It is calibrated for
each σ_lat. Declared difference: for H1 the 16.7% is measured on the whole session, here on the 5507 generated bins.
For each level we also report the **true fraction**: median per channel of var(smoothed expected rate) / var(smoothed
counts).

## Tests (tests/test_literature2.py), to be passed before the curves

| Test | Data | Expected |
|---|---|---|
| T1 | generator, each level and σ_lat, seed 0 | measured FA fraction within ± 0.02 of the requested value |
| T2 | smoothed expected rates, without Poisson, d = 6, one bin out of 10 and out of 20 | TwoNN and MLE10 between 4.5 and 7.5 in all seeds |
| T3 | as T2 with d = 3 | TwoNN and MLE10 between 2.5 and 4.5; median below that of T2 |

In tests T2 and T3 the estimate per seed is the median over the initial offsets.

## Curves (estimators TwoNN and MLE10, those verified at step 1)

- **Thinning (main)**: one bin out of m, m ∈ {1, 2, 5, 10, 20}, all offsets 0..m − 1 (on the order of the
  rows). For each dataset: median and [min, max] over the offsets; for the controls also over the 3 seeds.
- **Random sampling**: n ∈ {500, 1000, 2000, 5000, all}, 10 draws (seeds 0-9), as R9.
- **Theiler**: all points, w ∈ {0, 1, 3, 5, 10, 20, 40} bins, with `literature2.stime_theiler` (group = trial for H1 and
  twins, a single group for the controls). It does not enter the rules.
- CorrInt only if the total time stays below 30 minutes; DANCo not.

## Readings (on the thinning curve)

For a dataset and an estimator, M10 and M20 are the medians at m = 10 and m = 20 (for the controls: medians over offsets and
seeds, separately for σ_lat and for level).

- **Stable**: max(M10, M20) / min(M10, M20) ≤ 1.25.
- **Around d**: M10 and M20 both in [d − 1.5; d + 1.5].
- **Sensitivity (condition for reading the rest)**: the noise-free control (d = 6) must be stable and around
  6 for TwoNN and for MLE10, for both σ_lat. If it is not: **stop and ask**.
- **Sentence A**, for an estimator: the control with the H1 fraction (0.167) is stable and around 6 for both σ_lat,
  and H1 is not (stable and around 6) in any of the three sessions.
- **Sentence B**, for an estimator: the control with the H1 fraction is not stable or not around 6 for at least one σ_lat.
- Other case (control fine and H1 stable and around 6 in at least one session): neither A nor B, it is reported as it is.
- Final sentence: A (or B) if it holds for both estimators; otherwise it is reported estimator by estimator.
- **H1 against twins**: for each point of the curves we report whether the median of H1 falls in [min, max] of the 5 twins
  of the same session (min and max of their medians).

Possible sentences:
- A. "At equal noise, an intrinsic dimension of 6 would have been seen; on H1 there is no clear intrinsic dimension
  of that order."
- B. "With the signal-to-noise ratio and the sampling of H1, TwoNN and MLE do not estimate the intrinsic dimension
  even when it exists by construction."

Outputs: `results/literature/r9_checks.csv` and `.md`, `results/literature/r9_curve.png`; rows in `h1_runs.csv` (keys as
in `literature_step9b.py`, condition `9c_*`). No push without asking; the deck is not touched without OK.

## Post hoc modification (5 Nov, before the curves on the real data; Valerio's choice, "option 1")

Reason: with the committed generator (3176d14) two conditions written above cannot be met.
- The FA fraction of slide 5 does not exceed ~0.59: it rises with g up to g ≈ 1.4 (0.587 with σ_lat 5, 0.574 with σ_lat 10) and then
  falls (0.44 and 0.20 at g = 2; literature_cache/r9c/scan_g.json). The true fraction instead reaches 0.65 already at g ≈ 1.05.
- T2 per seed fails: at level 0.45 MLE10 7.6-8.1; at level 0.167 with σ_lat 5 MLE10 7.51 in one seed (threshold 7.5).
  The noise-free control depends on g (curvature of exp(g S)); the Levina-Bickel MLE sits ~10% above MacKay-Ghahramani.

What changes:
1. **Levels**: 0.167 (H1) and 0.45 as before; the third level, 0.65, is replaced by the **maximum reachable**,
   called 0.58, with g = 1.4 fixed for both σ_lat (measured FA fraction 0.587 and 0.574, seed 0). T1 holds for the three.
2. **Only the H1 level (0.167) decides**: sentences A and B already used it. 0.45 and 0.58 are descriptive.
3. **T2 and T3** are checked only at the g of level 0.167, on the **medians over the 3 seeds** at m = 10 and m = 20 (as the
   sensitivity rule), not seed by seed. The value per seed of 7.51 (MLE10, σ_lat 5) is declared.
4. **Sensitivity**: the noise-free control at the g of level 0.167, stable and around 6 for TwoNN and MLE10, for
   both σ_lat. The noise-free control at the other g is reported, without effect on the readings.
