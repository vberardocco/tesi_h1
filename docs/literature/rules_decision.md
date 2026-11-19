# Decision rules (Step 0) - fixed on 3 Nov 2026, before the results of Steps 1-3

R1 - Gallego 2018: variance of the first 12 PCs on 66 channels (square root, 20 ms bins, 50 ms Gaussian, concatenated trials, PCA without z-score), 20 draws per session. "in line" if the median is >= 60%, otherwise "below". We also report whether it falls within 73.4 +/- 2 x 6.5% (60.4-86.4%).
R2 - Silvernagel 2026: components for 80% on 96 channels (30 ms Gaussian, without z-score). "in line" if the median falls in 31-38; "higher" if > 38; "lower" if < 31.
R3 - Oby 2019 and Degenhart 2020: FA dimensions for 95% of the shared variance (45 ms bins, raw counts, K by cross-validation). "in line" if the median falls in 5-11 (90 channels) and in 4-8 (75 channels).
R4 - Degenhart 2020: shared population fraction trace(WW^T)/trace(WW^T + Psi), reported next to the 32%, without a verdict.
R5 - Parallel Analysis with circular shift (minimum shift 2 s, excluding +/-2 s around multiples of the median trial duration): existence in the weak sense if k_PA >= 1 in S1, S2 and S5 with both pipelines. It is a minimum threshold: in the thesis and in the slides we report k_PA and k_PA/N.
R6 - Saturation (S1 only, N = 20-160): "saturates" if the model with a plateau has an AIC lower by at least 2 than linear and power; otherwise "does not saturate in the observed range" (expected outcome).
R7 - Clock + neurons: neurons add information beyond time if (clock + GPFA) - (clock only) > 0 in 3 sessions out of 3, with p < 0.05 (permutation of the latents across trials) in each.

# Phase 2 (5 Nov)

Rules written and committed on 5 Nov 2026, before any computation of phase 2. Text by Valerio, unchanged.

R8 - Sensitivity of Parallel Analysis to preprocessing.
- Grid, pipeline of notebook 02: sigma in {20, 50, 100} ms x z-score in {yes, no}.
- Grid, Gallego pipeline (square root): sigma in {20, 50, 100} ms.
- For each session: k_PA with circular shift and k_PA / N.
- Verdict "robust existence" if k_PA >= 1 in all combinations and in all 3 sessions.
- The number is reported as a min-max interval over the grid, per session. No "best" combination is chosen.
- Check: the combination sigma = 50 ms with z-score must reproduce 19 / 17 / 15, and Gallego with sigma = 50 ms must reproduce 11 / 9 / 11. If it does not reproduce them, stop and say so before going on.

R9 - Intrinsic dimension estimators (as in Altan et al. 2021).
- Estimators: TwoNN; Levina-Bickel MLE with k = 10 and k = 20; correlation dimension (Grassberger-Procaccia); DANCo, if available in scikit-dimension (skdim) and if it runs in reasonable time, otherwise we write that it was skipped and why.
- Inputs, all with FALCON bins: (a) smoothed single-trial data, active channels; (b) GPFA latents (25 dimensions); (c) condition means (16 conditions); (d) 5 linear Gaussian twins with the same covariance as (a); (e) TME surrogates (see R10), for (c).
- Sampling: n in {500, 1000, 2000, 5000, all points}, 10 draws per n, fixed seeds. If an input has fewer points, the possible values of n are used.
- "Stable estimate" if, from n = 1000 to all points, the median changes by less than 20% and the interquartile range across draws is below 20% of the median.
- "Distinguishable from the twin" if, with all points, the median of the data falls outside the interval [min, max] of the twins.
- Declared prediction (Altan): on (a) the estimates are unstable and not distinguishable from the twin, on (b) and (c) they are stable. The verdict is "replicates Altan" if it goes this way, "does not replicate" otherwise, with the detail of where it does not match.

R10 - TME null for non-linear structure.
- Reference: Elsayed & Cunningham 2017, Nature Neuroscience, "Structure in neural population recordings: an expected byproduct of simpler phenomena?".
- Implementation: first a reliable Python implementation; if there is none, the method ported from the official MATLAB code (github.com/gamaleldin/TME): maximum entropy with the three marginal covariances (time, neurons, conditions) of the T x N x C tensor of condition means.
- Validation before use: the surrogates must reproduce the three marginal covariances of the data with a relative error (Frobenius norm) below 5%. The error is reported. If validation fails, R10 is "not applicable": non-validated surrogates are not used.
- Note: by construction TME preserves the covariance between neurons, so it also preserves the PCA spectrum. For this reason it is NOT useful to test linear dimensionality, but only what goes beyond the marginal covariances. Here it is used for the non-linear structure of slide 8.
- Statistic: Isomap - PCA advantage in distance reconstruction at d = 3. It is the same metric as notebook 03, computed on the tensor of condition means with FALCON bins, both for the data and for 100 TME surrogates.
- p = (1 + number of surrogates with statistic >= data) / (100 + 1).
- Verdict: "non-linear structure beyond the marginals" if p < 0.05 in 3 sessions out of 3; "explained by the marginals" if p >= 0.05 in 3 out of 3; "mixed" otherwise.
- We also report the Isomap - PCA advantage on the single trial with FALCON bins, against the 5 Gaussian twins (as in notebook 03): it is the direct update of the numbers +3.6 and -9.6 of slide 8.

R7, check of p (no new rule): if the permutations are 99, rerun with 999 and the same seeds; if they are more, the number is written. The verdict of R7 does not change.

## Operational clarifications (5 Nov, written before the results)

Proposed by Claude Code in the time estimate of the afternoon of 5 Nov; Valerio gave the go-ahead with the recommended choices. They do not change the text of the rules: they say how the rules apply where the text is not enough.

P1 (R9, inputs with fewer than 1000 points). The condition means (c) and their TME surrogates (e) have 800 points (16 conditions x 50 bins): for them stability is evaluated from n = 500 to all points.
P2 (R9, how the criteria are computed). "The median changes by less than 20%": |median(all points) - median(n_min)| / median(n_min) < 0.20, with n_min = 1000 (500 for P1). The interquartile range must be below 20% of the median at every n between n_min and the last n with 10 draws (with all points there is a single estimate). "Distinguishable": the value of the data with all points outside [min, max] of the 5 twins with all points (for (c) the same comparison with the 5 TME surrogates, reported without entering the prediction). Altan's prediction is checked per estimator and session; "replicates Altan" only if it holds in all cells, otherwise "does not replicate" with the list of the cells that do not match. If DANCo is skipped, the verdict holds on the other estimators.
P3 (R10, validation). The 5% applies to the expected marginal covariances of the fitted maximum-entropy distribution (exact computation): it is the check that the implementation is correct. We also report, without effect on applicability, the error of the mean of the 100 surrogates and the median error of the single surrogate. Reason: in a 50 x 160 x 16 simulation sampling noise alone gives 22-71% on the single surrogate and 2-7% on the mean of 100.
P4 (R10, data and statistic). Tensor of condition means = the PSTH of Step 4b (8 movement phases x odd/even trials, 1 s windows from the start of the phase, 50 ms Gaussian, z-score), T = 50, C = 16. TME of type TNC with the tensor of means M.TNC, as 'surrogate-TNC' in demo.m; seeds 0-99. Statistic as in notebook 03: PCA with 25 dimensions of the points, Isomap with k = 10 and precomputed distances, residual variance of the distances, advantage = (PCA - Isomap) at d = 3, on all 800 points.
P5 (R10, single trial). Pipeline of notebook 03 with FALCON bins: 50 ms Gaussian, z-score, PCA with 25 dimensions estimated on the bins of the movement phases, one bin out of 5, 5 random-phase twins (seeds 0-4), S1, S2, S5. Check: with legacy bins in S1 it must reproduce +3.6 and -9.6 of notebook 03.
P6 (R9, inputs and estimators). (a) pipeline of notebook 02 (50 ms Gaussian, z-score), 160-166 channels, on the bins of the movement phases (as TwoNN in notebook 03); (b) F_GP12["tutti"] of the replication on the same bins; (d) random-phase twins of the whole session (notebook 03), seeds 0-4, same bins; (e) the first 5 TME surrogates of R10 (seeds 0-4). skdim 0.5.0: TwoNN with 10% discarded, MLE with fit(n_neighbors=k) and mean of the local estimates (Levina-Bickel), CorrInt with k1 = 10 and k2 = 20, DANCo with k = 10. Seeds of the draws 0-9.
P7 (R9, DANCo timing). DANCo runs in parallel processes. If after draws 0-4 the expected time of the R9 block exceeds 2 hours, DANCo stops at 5 draws per n and the reduction is noted.
P8 (R8). Bins as in Step 2: pipeline 02 on all bins of the session, Gallego pipeline on the bins of the trials; 200 surrogates, seed 0 for each combination; k_PA = k_seq (first consecutive eigenvalues above threshold).
