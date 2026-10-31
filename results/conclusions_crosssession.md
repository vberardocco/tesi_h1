# Cross-session few-shot adaptation on FALCON H1 — conclusions

Generated on 31 October 2026 from the run registry (`results/h1_runs.csv`, experiment `crosssession`) and the
analysis in `results/crosssession/` (`report.md`, `table_conditions.*`, `stats_*.csv`, `fig_A_*`, `fig_B_*`).
Protocol and statistical plan: `docs/crosssession_protocol.md`. All seven methods were completed: UMAP, CEBRA and
LFADS with three seeds each, the deterministic methods with one. A full re-run of CEBRA and LFADS reproduced every
logged R² value exactly (1,512 of 1,512 per-fold and concatenated values identical).

## Setup in brief

- **Data.** The 14 held-out calibration files (S6–S12, two files per day, 6–20 days after the last held-in session)
  each contain exactly 3 trials. Rotation is leave-one-trial-out: 2 trials adapt, 1 trial tests (3 folds per session).
- **Metric.** Variance-weighted R² (the `r2_score(multioutput="variance_weighted")` used by the falcon_challenge
  evaluator) on the bins with `eval_mask`. It is computed on the concatenated predictions of the three folds. The
  unit of analysis is the session (n = 14), after averaging over seeds.
- **Methods.** Each method and its linear readout were fit once on the 13 held-in sessions and then frozen. The
  hyperparameters and latent dimensionality are those chosen in the earlier notebooks: PCA d = 25, FA d = 15,
  GPFA d = 25, Isomap d = 3, UMAP d = 3, CEBRA-Time d = 3, LFADS 25 factors.
- **Readout.** Ridge regression on the standardized latents with 1,000 ms of causal history. Its α was chosen by
  leave-one-day-out cross-validation on the held-in sessions.
- **Conditions.**
  - zero_shot;
  - recenter (input recentred with the mean of the 2 adaptation trials);
  - align (recenter + moment matching of latent mean and covariance);
  - fewshot (align + ridge refit of the readout towards the held-in weights);
  - scratch (method and readout fit on the 2 trials only; not run for LFADS);
  - ridge_rates (a common reference: ridge on the smoothed rates of the 2 trials);
  - for FA only, fa_native (FA fit on the 2 trials, loadings rotated onto the held-in loadings with Procrustes).
- **Tuning.** λ and the scratch and ridge_rates α were chosen on a pseudo-few-shot task built from held-in data:
  a model trained on S0–S3 and 20 pseudo-sessions of 3 trials taken from S4–S5. The held-out sessions were never
  used for tuning.

## 1. Zero-shot performance and its change with days

- **Level.** Median zero-shot R² across the 14 held-out sessions:

  | Method | Median zero-shot R² |
  |---|---|
  | LFADS | 0.108 |
  | PCA | 0.079 |
  | FA | 0.034 |
  | GPFA | 0.030 |
  | Isomap, UMAP, CEBRA | −0.001 |

  For comparison, the same frozen readouts reach the following leave-one-day-out R² inside the held-in set:

  | Method | Held-in leave-one-day-out R² |
  |---|---|
  | GPFA | 0.205 |
  | PCA | 0.199 |
  | LFADS | 0.174–0.195 (by seed) |
  | FA | 0.071 |
  | 3-D maps | ≈ 0 |

- **Trend with days (descriptive; least-squares slope with 95% CI).**
  - **PCA:** zero-shot R² decreased by 0.0041 per day (95% CI −0.0080 to −0.0002; Spearman ρ = −0.53, p = 0.050).
    The mean of the two files fell from 0.125 at day 6 to 0.032 at day 20.
  - **LFADS:** the slope was the same, −0.0041 per day, but the CI includes zero (−0.0089 to +0.0007;
    ρ = −0.40, p = 0.16). The day means went from 0.144 at day 6 to 0.028 at day 20.
  - **FA and GPFA:** no trend was detectable (CIs include zero). GPFA zero-shot was erratic: −0.043 at day 6,
    0.096 at day 17.
- **The decline is not monotonic.** Day 14 (S10) scored higher than days 7–13 for PCA and LFADS in every condition,
  and for GPFA after recentering (Fig. A). FA and GPFA zero-shot did not show this. With only 7 days, these
  estimates are imprecise. The data are compatible with a moderate decline for PCA and LFADS, but they
  do not establish a time course.

## 2. Recovery by condition, within each method

Paired Wilcoxon tests between consecutive conditions, Holm-corrected within each method. Below, "x/14" is the
number of sessions in which the later condition was better, and Δ is the median paired difference.

- **recenter vs zero_shot.** Improved all four 25-D/15-D methods.

  | Method | Δ | Sessions better | p (Holm) | Median after recentering |
  |---|---|---|---|---|
  | GPFA | +0.099 | 14/14 | 0.0005 | 0.127 |
  | PCA | +0.023 | 13/14 | 0.002 | 0.118 |
  | FA | +0.009 | 14/14 | 0.0006 | 0.042 |
  | LFADS | +0.009 | 9/14 | 0.04 | 0.125 |

  GPFA, whose input is not z-scored, gains the most. After recentering it has the highest median of all
  method × condition cells, and recenter is GPFA's best condition in all 14 sessions.
- **align vs recenter (moment matching).** The result is mixed.

  | Method | Δ | Sessions better | p (Holm) | Effect |
  |---|---|---|---|---|
  | FA | +0.005 | 13/14 | 0.006 | small gain |
  | PCA | +0.0006 | 8/14 | 1.0 | no change |
  | LFADS | −0.013 | 5/14 | 0.04 | worse |
  | GPFA | −0.062 | 0/14 | 0.0005 | worse |

  On the held-in pseudo-sessions (1–5 days from the training days), align was equal to or better than recenter for
  every method (PCA 0.189 → 0.213, LFADS 0.173 → 0.207, GPFA 0.205 → 0.206). That benefit did not carry over to
  held-out sessions 6–20 days away.
- **fewshot vs align.** The pseudo-few-shot selected strong pulls towards the held-in weights: λ = 3.2·10⁶ (PCA),
  10⁷ (GPFA), and the edge of the grid (10⁹) for FA and LFADS. At that edge the validation curve is flat and equal
  to align. As a result, fewshot is practically identical to align: all median differences are ≤ 0.002 R². Some are
  statistically detectable (GPFA +0.0006, 11/14, p = 0.049; LFADS, Isomap), but none is large enough to matter.
  Refitting the readout on two trials did not add measurable decoding performance under this regularization.
- **scratch vs fewshot.** For every method with a scratch condition, scratch was worse than fewshot in 13–14 of
  14 sessions (Holm p ≤ 0.001). The median paired loss was 0.020 (CEBRA) to 0.14 (PCA). Median scratch R² was
  ≤ 0 for every method (−0.033 to −0.007), and the ridge_rates reference reached −0.016 [IQR −0.038, 0.015]. With
  this decoder, two trials are not enough to build a decoder, or a manifold plus decoder, from scratch. The
  held-in model is what makes positive R² possible.
- **fa_native vs align (FA).** fa_native was worse in all 14 sessions (Δ = −0.026, p = 0.0006). Its median (0.025)
  was below FA zero-shot (0.034).
- **The three 3-D maps.** For Isomap, UMAP and CEBRA, some within-method differences reach significance, but all
  are smaller than 0.002 R² around a level of ≈ 0. They carry no practical information.

## 3. Differences between methods

The tests are pre-specified at the align and fewshot conditions. Friedman tests, with sessions as blocks:

| Condition | χ²(6) | p | Kendall's W |
|---|---|---|---|
| align | 66.1 | 2.6·10⁻¹² | 0.79 |
| fewshot | 66.3 | 2.3·10⁻¹² | 0.79 |

Pairwise Wilcoxon tests, Holm-corrected within each condition, give the same picture at both conditions:

- **PCA and LFADS do not differ** (align: LFADS − PCA = −0.016, p = 1.0). They are the two best pipelines (median
  0.108 each at align).
- **Both are better than GPFA and FA** in 12–13 of 14 sessions (p ≤ 0.014). GPFA and FA do not differ from each
  other (p = 1.0).
- **All four 25-D/15-D pipelines are better than every 3-D map** (p ≤ 0.009). Differences among Isomap, UMAP and
  CEBRA are sometimes significant but are at most 0.003 R² around zero.

**This comparison is between frozen pipelines, not between methods at matched dimensionality.** The three nonlinear
maps were frozen at d = 3, the dimension chosen in notebooks 03/04 for phase recognition and visualization. An
exploratory check, not in the pre-specified plan (`diag_3d_within_session.csv`), tested decoding within a single
held-in session with the same readout. The 3-D latents explained little velocity even there, measured optimistically
with α chosen on the test trials:

| Latents | Within-session R² |
|---|---|
| UMAP (3-D) | 0.04–0.07 |
| Isomap (3-D) | 0.02–0.04 |
| First 3 principal components | 0.01–0.02 |
| PCA (25-D) | 0.22–0.26 |
| GPFA (25-D) | 0.22–0.30 |

So the near-zero results of Isomap, UMAP and CEBRA are tied to their dimensionality. The data do not say whether
these methods would transfer across sessions at a larger d.

## 4. Limitations

- **Few trials.** Each fold adapts on 2 trials: 1,442–1,933 bins, of which 1,390–1,876 are under eval_mask. That
  is set against 170 channels and up to 1,251 readout weights per degree of freedom (25 latents × 50 lags +
  intercept), or 8,501 for ridge_rates. Each session has only 3 folds; each test fold is a single trial; and there
  are 14 sessions on 7 days.
- **Local calibration split, not FALCON's hidden evaluation.** Performance is measured on the held-out calibration
  trials with leave-one-trial-out, and each trial is processed as a separate sequence. Decoder history restarts
  at every trial. Smoothing, GPFA, LFADS and CEBRA use bins after the decoded bin within a trial, so the pipelines
  are not causal. R² is computed per file rather than per day as in the FALCON evaluator. The numbers are therefore
  not comparable with the FALCON leaderboard.
- **Moment matching is blind to rotations.** On synthetic data, a pure rotation leaves the covariance matched
  (difference < 10⁻⁶) while the pointwise relative error stays at 1.5. A non-isotropic gain is not inverted exactly
  either (pointwise error 0.10). Whether this explains why align did not help on the held-out sessions was not
  tested.
- **LFADS has no scratch condition.** Two trials give about 80 segments of 600 ms, too few to train the network.
  The LFADS hyperparameters had been selected on a single session with 12 trials; here LFADS was trained on 13
  sessions for the same fixed 200 epochs. LFADS results depend on the seed: the median per-session SD across the
  three seeds is 0.021 R² at align. For UMAP and CEBRA the SD is ≈ 0, because their readouts predict close to the
  mean.
- **The tuning setting differs from the evaluation setting.** λ and α were tuned on pseudo-sessions 1–5 days from
  their training days, while held-out gaps are 6–20 days. For FA and LFADS the optimal λ lies at the edge of the
  grid, which makes fewshot equal to align.
- **Other fixed choices.**
  - The channel set (170 channels active on held-in data) is fixed; channels that changed on held-out days are not
    handled separately.
  - Isomap and UMAP were fit on at most 5,000 held-in points, because of memory.
  - Only one readout family (ridge with history) was tested.
