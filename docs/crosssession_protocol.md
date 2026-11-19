# Few-shot cross-session protocol on FALCON H1 — choices and analysis plan

Written on 31 Oct 2026 **before** looking at the held-out results (the only exception: a trial run of the code with
PCA in a temporary folder, whose numbers were not used for any choice). Code:
`src/thesis_h1/crosssession.py`, `scripts/xs_run.py`, `scripts/xs_analysis.py`; tests in `tests/test_alignment.py`.

## Data
- 13 held-in calib sessions (S0-S5) and 14 held-out calib sessions (S6-S12, 2 files per day), FALCON bins
  (`io.load_session(bins="falcon")` = `load_nwb`). Every held-out file has exactly 3 trials (checked).
- Channels: the 170 of 176 with mean rate ≥ 0.5 Hz on the held-in sessions (same criterion as 02_new), for all methods.
- **Every trial is a sequence of its own**: smoothing, GPFA and LFADS chunks, CEBRA windows and the decoder history
  stay inside the trial. The adaptation (`adapt`) receives only the 2 adaptation trials; the test trial enters only in
  `predict_trial`. The first second of each trial has an incomplete decoder history (zeros = mean), the same for all.

## Methods and hyperparameters frozen (from the notebooks)
| method | input | latent dim. | settings |
|---|---|---|---|
| PCA | smoothed rates σ 50 ms, z-score | 25 | fitted on the bins of the movement phases (03, 04 §1) |
| FA | z-scored counts | 15 | K* of 02_new §9, `svd_method="lapack"` |
| GPFA | square root of counts | 25 | 02_new §10: 1 s windows at the start of the movement phases, 500 iterations, τ₀ 100 ms; latents orthonormalized from 2 s chunks every 1 s (04 §7b) |
| Isomap | 25 PCs | 3 | k = 10, one movement bin out of 5 (03); on held-in at most 5000 random points (memory) |
| UMAP | 25 PCs | 3 | n_neighbors 10, min_dist 0.1, same points as Isomap |
| CEBRA | 25 PCs | 3 | CEBRA-Time of 04: offset10-model, batch 512, lr 3e-4, τ 1, Δ 10, cosine, 1000 steps, fixed seed |
| LFADS | counts | 25 | LF_HP of 04 §11b-§12: gen 64, dropout 0.05, KL × 0.3, CD 0, 200 epochs, 600 ms chunks every 400 ms |

The **readout** is the linear decoder of 05 (ridge with 1000 ms of causal history) applied to the latents standardized
with the held-in mean and standard deviation. Its α is chosen leave-one-day-out on the held-in sessions. Seeds: 3 for
UMAP, CEBRA, LFADS (0, 1, 2); 1 for the deterministic methods.

## Conditions (for each fold: 2 trials adapt, 1 tests)
1. `zero_shot`: held-in model and readout as they are.
2. `recenter`: method input minus the mean of the 2 trials plus the held-in mean.
3. `align`: recenter + moment matching on the latents, f(z) = (z − μ_a) C_a^(−1/2) C_s^(1/2) + μ_s (symmetric roots via
   eigh, eps = 1e-8 × mean eigenvalue on the diagonal); μ_a, C_a from the latents of the 2 trials, μ_s, C_s from the
   held-in latents. For FA also `fa_native`: FA (K = 15) on the 2 trials, loadings rotated onto the held-in ones with
   orthogonal Procrustes.
4. `fewshot`: align + readout re-estimated on the 2 trials with min ||Y − [Z 1]B||² + λ||B − B_src||² (intercept
   included: for λ → ∞ it returns to align).
5. `scratch`: method and readout estimated only on the 2 trials, same latent dimension (not for LFADS: 2 trials are
   ~80 chunks of 600 ms, too few to train the network; 04 §13b shows that LFADS already gets worse with 4 trials).
- Common reference `ridge_rates` (method "rates"): ridge with the same history on the smoothed rates (σ 50 ms) of the 2 trials.

## Choice of λ and α (without touching the held-out sessions)
Pseudo-few-shot on the held-in sessions: method and readout estimated on days S0-S3; pseudo-sessions = blocks of 3
consecutive trials from the 4 files of S4 and S5 (20 pseudo-sessions), same rotation and same conditions. λ (fewshot),
α (scratch) and α (ridge_rates) = the grid value (half-decade step) with the highest mean of the concatenated R². Chosen
with seed 0 and used for all seeds. Minival is not used.

## Metric and statistical unit
Variance-weighted R² (= `r2_score(multioutput="variance_weighted")` of falcon_challenge) on the bins with eval_mask:
per fold and on the concatenated predictions of the 3 folds (`fold = "concat"`), which is the main measure. For
methods with several seeds, seeds are averaged first. Statistical unit: the session (n = 14).

## Analysis (fixed before the results)
- **Table**: median and IQR of the concatenated R² per method × condition, with n.
- **Figure A**: one panel per method, R² as a function of days since the last held-in session, one curve per condition
  (mean of the two files of each day, points = sessions), ridge_rates as a common line.
- **Figure B**: fewshot condition, one curve per method as a function of days.
- **Tests within each method**: paired Wilcoxon (two-sided) between consecutive conditions: zero_shot→recenter,
  recenter→align, align→fewshot, fewshot→scratch (LFADS: the first three); for FA also align→fa_native. Holm
  correction within each method. We report W, raw and corrected p, effect size = paired rank-biserial correlation and
  median difference.
- **Tests between methods**, at the align and fewshot conditions: Friedman with sessions as blocks (χ², p, Kendall's W),
  then pairwise Wilcoxon with Holm within each condition (same measures). If LFADS is not complete, the Friedman test is
  run on the complete methods and LFADS is reported separately.
- **Descriptive** (not a hypothesis test): slope of the zero-shot R² against days (least squares, with 95% CI)
  and Spearman's ρ, per method.
- **Adaptation bins**: per session and fold, bins of the 2 trials (all and with eval_mask) relative to the 170 channels,
  the latent dimension and the number of readout weights (H × dim + 1).

## Deviations and additions (second round, 31 Oct 2026 afternoon, requested after the morning results, commit 48ca831)

Written before seeing the second-round results (code: commit 95122b6). Everything that follows is an addition
to the plan of the morning of 31 Oct; the new comparisons are marked `added_v2` in the tables.

1. **Choice of λ: it was already free of the pseudo-session.** Verified in the code (`tune`, `fit_bundle(..., "pseudo")`):
   for all 7 methods, CEBRA and LFADS included, the method (manifold) and the readout of the pseudo-few-shot are
   estimated only on days S0-S3; the pseudo-sessions come from S4 and S5, so every pseudo-session is excluded from
   both. This is stricter than leave-one-session-out (two whole days are excluded). No correction.
2. **λ grid extended** from 10^-1 - 10^9 to 10^-2 - 10^15 (half-decade step). The R²(λ) curve of every
   method is saved (`tune/*.lambda_curve.csv`, `lambda_curves.csv`, figure C) and compared with the asymptote λ → ∞,
   which is align (for ridge_channels: recenter). Rule: if the optimum stays at the upper end even so, the optimum is
   λ → ∞ and fewshot coincides with align; this is reported as a result, and the grid is not extended further. λ and α
   are chosen again for all methods (also for those whose optimum was already interior), with the new grid; the results
   of 31 Oct for PCA, FA, GPFA, LFADS are replaced in the registry (same keys: the last row counts).
3. **fewshot_gain** (new condition): on top of align, for each output DoF y ≈ g·ŷ + o with least squares on the bins
   of the 2 adaptation trials (2 parameters per DoF, no hyperparameter). Test added in every method:
   align→fewshot_gain (Holm within the method, together with the others).
4. **ridge_channels** (new reference): the causal ridge decoder of 05 (`models/ridge_h50.npz`, commit b797b3e:
   176 channels, per-channel z-score with held-in mean and standard deviation, 1000 ms of history, α 10^4.5, estimated on the whole held-in
   calib) with zero_shot, recenter (mean of the 2 trials in place of the held-in one), fewshot (recenter + ridge toward
   its weights, λ chosen with the pseudo-few-shot) and fewshot_gain (recenter + gain and offset). For the pseudo-few-shot
   the same decoder is re-estimated with the same recipe on days S0-S3. Difference from 05: here it is applied trial by
   trial like all the others (the history restarts at each trial). Tests: zero_shot→recenter, recenter→fewshot,
   recenter→fewshot_gain; it enters the between-method comparisons at fewshot and fewshot_gain (not at align, which it lacks).
5. **Maps of comparable dimension.** Isomap, UMAP and CEBRA were at d = 3 (choice of 03/04 to recognize the phases).
   The rule used for the other methods without a likelihood to validate (PCA in 03 §2, LFADS in 04) was to
   take the d of GPFA chosen with cross-validation on the held-in sessions (02_new §10): **d = 25**. FA has its own
   cross-validation (K* = 15). The same rule is applied to the three maps: d = 25, with all other hyperparameters
   unchanged. The d = 3 maps are redone with the new λ, grid and fewshot_gain and recorded with experiment
   `crosssession_d3`; their rows of 31 Oct (experiment `crosssession`, latent_dim 3) stay in the registry but
   the analysis excludes them by filtering on the dimension. Friedman and pairwise comparisons are redone with the
   d = 25 maps (main analysis) and, as a variant, with the d = 3 ones.
6. **Moment-matching diagnosis** (exploratory). In each of the 13 held-in sessions, blocks of 3 consecutive trials
   with the same rotation as the held-out sessions (2 trials adapt, 1 tests); source = the other trials of the same
   session: readout re-estimated on them (held-in α scaled by the number of bins), input moments and mean from their
   latents. The method (manifold) stays the held-in one. Conditions no_adapt, recenter, align; experiment
   `crosssession_mmdiag`, seed 0, main dimensions. Test: Wilcoxon align vs recenter (n = 13), Holm over the 7 methods.
   Rule fixed now: if align gets worse also within the session, where there is no drift between days, the drop on the
   held-out sessions is attributed to the behavioural content of the 2 trials and not to drift.
7. **Behaviour coverage** (descriptive). For each fold, in the 2 adaptation trials: movement phases (8),
   conditions phase × task version (16; the version is the parity of the trial, verified from the sign of g3 in Grasp on
   all held-in and held-out trials), movement directions DoF × sign (14; covered with at least 10 bins beyond the 75th
   percentile of |v| of the held-in pool), variance of the velocity relative to the pool. Held-out sessions,
   pseudo-sessions and blocks of the diagnosis are compared. In addition, exploratory: Δ align − recenter per fold, separating the folds with the
   2 trials of the same version.

### Notes after the run of the second round (31 Oct 2026 evening)

All runs of the second round finished without failures (v2 queues, 16:01-17:38; code 95122b6 for
ridge_channels, c6efc91 for the rest: they differ only in `xs_analysis.py` and in this file). The results of PCA, FA,
GPFA and LFADS at the morning conditions are bit-identical (same λ for PCA and GPFA), except fewshot of FA and LFADS,
where λ goes from 10^9 to 10^15 (both at the asymptote) and the R² changes by at most 4·10^-5.

8. **The date of the additions was wrong.** In the first versions of this file, of the code and of the tables the
   additions were dated "1/11" and marked `added_20261101`; they are from the afternoon of 31 Oct (queue logs). Corrected to
   `added_v2` and "second round"; no number changes.
9. **λ grid, outcome of the rule of point 2.** FA and LFADS have the optimum at the upper end even with the grid
   up to 10^15, with the curve rising monotonically toward the asymptote: by the rule, fewshot = align. For all the
   others the optimum is interior (PCA 10^6.5, GPFA 10^7, Isomap 25 10^4.5, UMAP 25 10^5.5, CEBRA 25 10^6, d = 3 maps
   10^5-10^5.5, ridge_channels 10^5).
10. **Two graphical corrections, no number changes.** Figure A: the lower limit of the y axis cut the zero-shot
    of ridge_channels on days 6 and 20; it now includes all points. Figure C: the marker of the chosen λ was missing for
    PCA, Isomap 25, UMAP 25 and CEBRA 3 (the half-decade λ values): the curve keys in the JSON are rounded to 4 digits
    ("3162") and the comparison with the chosen λ was exact; the relative tolerance is now 10^-3 (two neighbouring grid points
    differ by a factor of 3.16). In figure C the y axis is cut at −0.08,
    and this is written under the axis.
11. **Exploratory checks added AFTER seeing the results** (`scripts/xs_checks.py`, commit 81f5a1a; not in the
    plan, no hypothesis test). Two results needed an explanation before writing about them:
    a) fewshot_gain makes align worse in almost all sessions and in all methods. Decomposition on the held-out sessions (seed 0,
       main dimensions, same folds): gain only, offset only, and a gain estimated on the test trial itself
       (upper bound, not a condition); in addition the gain estimated on each trial alone.
    b) UMAP and CEBRA stay at R² ≈ 0 even at d = 25. In the 13 held-in sessions, with the held-in model: decoding
       within the session (readout on the trials except the last 3, α leave-one-trial-out, test on the last 3) and fraction
       of the latent variance explained by the session identity.
    Files: `check_gain_decomposition.csv`, `check_gain_per_trial.csv`, `check_within_session.csv`,
    `check_session_variance.csv`; final section of `report.md`.
12. **Limit of the moment-matching diagnosis for UMAP and CEBRA.** The readout of the diagnosis uses the held-in α scaled
    by the number of bins (point 6); for UMAP and CEBRA that α is at the upper end of the grid (the held-in readout predicts
    the mean), so within the session they start from R² ≈ 0.01-0.02 and the diagnosis says little about them.

## Stability of FA subspaces on the held-out sessions (addition of 2 Nov 2026)

Written on 2 Nov 2026 **before** computing any index or angle on the blocks of 3 trials (held-in or held-out).
Already known to the writer: the zero-shot R² per session (`sessions_r2.csv`) and the angles between S1, S2 and S5 on
whole sessions (replications, commit 1d032e9), where the Elsayed-style chance level for FA turned out to be almost uniform. Code:
`src/thesis_h1/stability.py`, `scripts/xs_stability.py`; tests in `tests/test_stability.py`. No number from this
analysis enters the choices of the cross-session protocol.

### Blocks, channels and normalization
- **Block = 3 consecutive trials.** Held-out: the 3 calib trials of each file (14 blocks). Held-in: non-overlapping
  blocks from the start of each file, the same as in the within-session diagnosis (`diag_within`, 54 blocks; the
  leftover trials stay out). In this way held-in and held-out are estimated with the same number of trials.
- **Channels: the same 170** active on the held-in sessions, in all blocks.
- **FA as in the decoder**: K = 15, `svd_method="lapack"`, `random_state=0`, raw counts.
- **Two normalizations**, because the span of the loadings changes if the channels are rescaled:
  a) *block against block*: z-score within the block (block mean and sd), as in the replications;
  b) *block against the frozen FA of the decoder*: the normalization that the decoder applies to the held-out sessions in
     zero-shot, i.e. mean and sd of the 13 held-in sessions of the model `FA_d15_heldin_s0` (the mean does not change the loadings,
     the sd matters). For the reference held-in blocks the FA estimated without their day is used (leave-one-day-out,
     same recipe), with its normalization.
- **Silent channels.** In a) the sd of a channel in the block does not fall below sd_min = 0.1, i.e. the Poisson sd of a
  channel at 0.5 Hz (the active-channel threshold) in 20 ms bins: √(0.5 · 0.02) = 0.1. In this way a channel with zero sd does not
  divide by zero and one with 1-2 spikes does not become a huge z value. In b) the sd is the held-in one and a channel silent
  in the block stays a constant column (FA gives it zero loadings; verified on synthetic data). We report
  how many (channel, block) pairs have sd = 0 and how many sd < sd_min.

### Measures
- L = loadings of the block (170 × 15), S = L Lᵀ its shared covariance.
- **Alignment index (primary)**, from Elsayed et al. 2016 applied to S:
  I(A→B) = tr(Q_Bᵀ S_A Q_B) / Σ_{i≤k} λ_i(S_A), with Q_B the orthonormal basis of the first k eigenvectors of S_B.
  Pair index = ½ [I(A→B) + I(B→A)]. It is 1 if each subspace captures all the shared variance
  of the other that k directions can capture. With k = 15, Q_B is a basis of span(L_B) and the denominator is tr(S_A).
- **Mean principal angle (secondary)**: mean of the k principal angles between span(Q_A) and span(Q_B), in degrees.
- **Sensitivity to k = 5**: the first 5 eigenvectors of L Lᵀ from the fit with K = 15, without redoing FA with K = 5. FA
  loadings have no order (they are defined up to a rotation); directions are ordered by shared
  variance.

### Elsayed-style chance level
- Random subspaces Q_r = orth(U Λ^½ r), r Gaussian 170 × k, from a covariance C = U Λ Uᵀ. For the pair (A, B):
  index chance level = ½ [I(A→Q_r1) + I(B→Q_r2)], with Q_r1 and Q_r2 independent; angle chance level = mean angle between
  Q_r1 and Q_r2. 500 draws per pair, fixed seed.
- Three C, as in the replications:
  - *all bins*: mean of the data covariances of the two members of the pair, in the normalization of the comparison;
  - *condition means*: covariance of the means of the 16 conditions (8 phases × version, 1 s windows
    from the start of the phase), computed in each held-in session and averaged over the 13; in a) with the z-score of the
    session, in b) with the normalization of the decoder. It is the same C for all pairs;
  - *uniform*: identity.
- For the index (higher = more aligned) the **95th percentile** counts; for the angle the 5th.

### Comparisons
1. **Within-session reference**: the pairs of blocks of the same held-in session (95 pairs). It is the maximum
   of alignment that can be measured with 3 trials.
2. **Same day, different file**: pairs of blocks from two files of the same day, held-in and held-out.
3. **Between days**: all the other pairs, as a function of |Δ days| (dates from `sessions.csv`, from 0 to 39).
4. **Held-out against the frozen FA of the decoder** (normalization b): it is the primary x axis of figure F, because it is
   the FA that produces the zero-shot. Reference: each held-in block against the leave-one-day-out FA of its day.
5. **Held-out against the 54 held-in blocks** (normalization a): median per held-out session, descriptive.

### Rule on estimability (fixed now)
The median of the index over the 95 pairs of the within-session reference, at k = 15, is compared with the median over the
same pairs of the 95th percentile of chance, for each of the three versions. If it does not exceed all three, the FA
subspace at K = 15 is not estimable from 3 trials: this is written as a result, and tests and figures are read on the
sensitivity at k = 5 (computed in any case), with the same rule. The same check, descriptive, for the angle (reference below
the 5th percentile of chance).

### Tests
- **Stability (figure E): Mantel test.** Statistic: Spearman's ρ between the index and |Δ days| over all pairs of
  blocks **except the 95 within the same session**, which are the reference band: if they stayed at distance 0 the
  test would measure "same session or not" instead of the decline with days. Pairs same day, different file stay
  at distance 0. p: permutation of the dates among the 13 days (the pairs of a same day are not independent),
  10 000 permutations, seed 0, two-sided. A single test, on all blocks; the three pair types (held-in–held-in,
  held-in–held-out, held-out–held-out) only descriptive.
- **Slope per day**: least squares of the index on |Δ days|, on the same pairs as the Mantel test. 95% CI with
  bootstrap over days (2 000 resamples, seed 0): the 13 days are resampled with replacement, the pairs between copies of different days
  and the pairs same day, different file inside each copy are kept.
- **Figure F: Spearman's ρ** between the index against the frozen FA (k = 15) and the zero-shot R² of FA (n = 14),
  p by permutation (10 000, seed 0, two-sided). Repeated on the means per day (n = 7) and as partial Spearman
  with respect to the days since the last held-in session (rank residuals after regression on the ranks of the days, p by
  permutation of the residuals). It is the only test on the relationship.
- **Descriptive**: the same index against the zero-shot of PCA, GPFA, LFADS and ridge_channels and against the gain
  align − recenter of FA; the median index against the held-in blocks (comparison 5) against the zero-shot of FA.
- **Power.** With n = 14 only |ρ| ≳ 0.54 is significant (Spearman, α = 0.05 two-sided); with the means per day
  (n = 7) only |ρ| ≳ 0.79. A null result reads "relationship not detectable with these data", not "no
  relationship". The script recomputes the exact thresholds by permutation and reports them.

### Figures and files
- **Figure E**: index (k = 15) against |Δ days|; points = median per pair of days, colour by pair type;
  band of the within-session reference (IQR of the 95 pairs) and lines of the 95th percentile of chance; panel with
  the mean angle. **Figure F**: index against the frozen FA on the abscissa, zero-shot R² of FA on the ordinate (14 points,
  colour = days since the last held-in session), with the band of the held-in reference (blocks against the leave-one-day-out FA);
  panel with align − recenter; descriptive panel with the other methods.
- **Registry** (`h1_runs.csv`): experiment `fa_stability`, method `FA`, latent_dim 15, decoder NA, split
  `calib3_block`, k_shot 3 (trials that estimate the subspace), H_ms NA, seed 0, fold NA; condition `vs_frozen_fa`
  (held-out), `vs_lodo_fa` (held-in blocks), `vs_heldin_blocks` (median per held-out); session = FALCON key
  of the file, fold = number of the block in the file (0 for the held-out; NA for `vs_heldin_blocks`). Corrected on 2 Nov before
  the run: the registry accepts only FALCON keys as session.
  Metrics: index and angle at k = 15 and 5, percentiles of chance. The pairs are in
  `results/crosssession/stability/pairs.csv`, tests and summaries in `results/crosssession/stability/`.
- **Limits to declare**: 14 sessions over 7 days; the index uses all 3 trials, including those on which the zero-shot is
  evaluated (it is a descriptive correlation, nothing is estimated on the index); the held-out blocks are the first 3
  trials of the file, the held-in ones come from the whole file.

### Notes after the run of the stability analysis (2 Nov 2026)
- Run with code cdf90b1 (10:47-11:13; clean registry, 2 844 `fa_stability` rows). Only change with respect to the
  plan, made before the run: the registry keys (session = file, fold = block), see above.
- Before the run, a full trial with 10 chance draws in a temporary folder (outside the registry) to
  check the code and the layout of the figures; the observed indices and angles do not depend on the draws.
- The frozen FA of the decoder is reproduced from the recipe (max |Δ loading| = 5.7·10⁻¹⁵); no channel with sd = 0 in the
  blocks, 543 (channel, block) pairs out of 11 560 with sd < sd_min.

## Table B: ridge on FA latents on minival (addition of 2 Nov 2026)

Written **before** the evaluation. Decision of the user (2 Nov): a single evaluation on minival, with the choices already
fixed, and no choice after seeing the result.
- **Model**: `FA_d15_heldin_s0` as it is. FA with K = 15 on the 170 active channels, z-scored with held-in mean and sd;
  latents standardized with the held-in moments; ridge readout with 1000 ms of history, α = 10⁶ chosen leave-one-day-out
  on the held-in sessions (R² 0.0709); estimated on all 13 held-in calib files. It is exported to `models/fa15_ridge_h50.npz` by
  `src/thesis_h1/fa_decoder.py`. The tests (`tests/test_fa_decoder.py`) verify that the latents coincide with
  `FactorAnalysis.transform` (10⁻¹⁴), that the predictions on a trial coincide with the frozen readout and that
  streaming coincides with the block computation. The script rechecks the npz against the frozen model before evaluating.
- **Evaluation**: `scripts/05_fa_minival.py`, like `scripts/05_ridge_minival_h50.py`. The reported number is
  "Held In R2 Mean" of the official FALCON evaluator in streaming with batch 4. Batch 1 and the block computation with
  our evaluator serve only to check that they coincide.
- **Difference declared beforehand**: the readout was trained with the history restarting at each trial. The FALCON
  evaluator for H1 is continual (it does not call `on_done`), so the history runs through the whole file, as for ridge_h50.
  Only the first 49 bins of each trial change (in the tests on held-in trials, predictions differ by at most 0.004). It is not
  corrected: it would be a different model from the one of the leave-one-day-out 0.071.
- **Channels**: 170 active for FA, 176 for ridge_h50.
- **Registry**: experiment `table_b`, decoder `ridge`, H_ms 1000, condition `calib_heldin`; split `minival` and
  `calib_lodo`. Method `FA` (latent_dim 15) and `counts`: the ridge on channels (ridge_h50) is imported from the log of 30 Oct
  (commit b797b3e), without a new evaluation.
- **Table B** (`results/table_B.{csv,md,tex}`): R² on minival (mean of the 6 days) and SD over days; leave-one-day-out R²
  on the held-in sessions (FA only: for ridge_h50 history and α were chosen on 80/20 of the calib); days and DoF.
