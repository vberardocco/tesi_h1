# Cross-session few-shot adaptation on FALCON H1 — conclusions (second round)

Generated on 31 October 2026 (evening) from the run registry (`results/h1_runs.csv`, experiments `crosssession`,
`crosssession_d3`, `crosssession_mmdiag`) and the analysis in `results/crosssession/` (`report.md`,
`table_conditions*.{csv,md,tex}`, `stats_*.csv`, `fig_A`–`fig_D`, `check_*.csv`). Protocol: `docs/crosssession_protocol.md`;
the additions of the second round are listed there under "Deviations and additions" (written before the second-round results)
and "Notes after the run of the second round" (written after them). This file replaces the first-round conclusions of commit 48ca831.

## What changed since the first round (48ca831)

1. **λ selection was already leakage-free** for all seven methods, CEBRA and LFADS included: the pseudo-few-shot fits
   both the method and the readout on days S0–S3 only, and the pseudo-sessions come from S4–S5. Each pseudo-session is
   therefore excluded from everything it is scored against, which is stricter than leave-one-session-out. No rerun was
   needed for this reason.
2. **The λ grid was widened** from 10⁻¹–10⁹ to 10⁻²–10¹⁵, and the validation curve R²(λ) is saved for every method
   (`lambda_curves.csv`, Fig. C).
3. **New condition fewshot_gain:** a per-DoF gain and offset on top of align, fit by least squares on the 2 adaptation
   trials (14 parameters in total).
4. **New reference ridge_channels:** the causal channel-level ridge decoder of notebook 05 (`models/ridge_h50.npz`),
   run through zero_shot, recenter, fewshot and fewshot_gain.
5. **Isomap, UMAP and CEBRA at d = 25** (the rule used for PCA and LFADS) are now the main analysis; the d = 3 runs are
   kept as experiment `crosssession_d3`.
6. **Two exploratory analyses:** a moment-matching diagnostic inside held-in sessions, and the behavioural coverage of
   the 2 adaptation trials.
7. **Reproducibility.** Everything that was rerun reproduced the first-round values bit for bit: PCA, FA, GPFA, LFADS
   and the three 3-D maps, all conditions. The only exception is fewshot for FA and LFADS, whose λ moved from 10⁹ to
   10¹⁵ (both at the asymptote); their R² changed by at most 4·10⁻⁵.

## Setup in brief

- **Data and folds.**
  - 14 held-out calibration files (S6–S12, two per day, 6–20 days after the last held-in session), 3 trials each.
  - Leave-one-trial-out: 2 trials adapt, 1 trial tests (3 folds per session).
  - Each fold adapts on 1,390–1,876 bins under eval_mask.
- **Metric.** Variance-weighted R² as in the FALCON evaluator, on eval_mask bins, on the concatenated predictions of the
  three folds. The unit of analysis is the session (n = 14), after averaging over seeds.
- **Methods.** Each method and its linear readout were fit once on the 13 held-in sessions and frozen.
  - Latent dimensions: PCA 25, FA 15, GPFA 25, Isomap 25, UMAP 25, CEBRA-Time 25, LFADS 25.
  - The readout is ridge on standardized latents with 1,000 ms of causal history; its α is chosen leave-one-day-out
    on the held-in sessions.
  - Seeds: 3 for UMAP, CEBRA and LFADS; 1 for the deterministic methods.
- **Conditions.**
  - zero_shot.
  - recenter: input recentred with the mean of the 2 trials.
  - align: recenter + moment matching of the latent mean and covariance.
  - fewshot: align + ridge refit of the readout towards the held-in weights, with weight λ.
  - fewshot_gain: align + per-DoF gain and offset.
  - scratch: method and readout fit on the 2 trials only; not run for LFADS.
  - For FA only, fa_native.
  - Two references: ridge_rates (ridge on the smoothed rates of the 2 trials) and ridge_channels.
- **Tuning.** λ and the α of scratch and ridge_rates were chosen on 20 held-in pseudo-sessions (model on S0–S3,
  pseudo-sessions from S4–S5). The held-out sessions were never used for any choice.

## 1. Zero-shot performance and its change with days

Unchanged from the first round for the four methods that decode at all. Median zero-shot R²:

| Method | Median zero-shot R² |
|---|---|
| LFADS | 0.108 |
| PCA | 0.079 |
| FA | 0.034 |
| GPFA | 0.030 |
| Isomap (d = 25) | 0.011 |
| UMAP, CEBRA (d = 25) | −0.001 |
| ridge_channels | −0.073 [IQR −0.162, 0.055] |

- **PCA.** R² fell by 0.0041 per day (95% CI −0.0080 to −0.0002; Spearman ρ = −0.53, p = 0.050).
- **LFADS.** Same slope, but the CI includes zero (−0.0089 to +0.0007).
- **No detectable trend** for FA, GPFA, the maps or ridge_channels (all CIs include zero).
- **The decline is not monotonic.** Day 14 is higher than days 7–13 for PCA and LFADS (Fig. A).
- **ridge_channels** has the worst zero-shot of all pipelines: its day means reach −0.21 and −0.22 on days 6 and 20.
  The channel-level decoder, used with the held-in channel means, does not transfer at all without recentering.

## 2. Recovery by condition, within each method

Paired Wilcoxon tests between consecutive conditions, Holm-corrected within each method. "x/14" is the number of
sessions in which the later condition is better; Δ is the median paired difference.

**Change in the Holm families.** The second round adds align→fewshot_gain to every method's family. For LFADS this
moves two results above 0.05: zero_shot→recenter (+0.009, 9/14) goes from p_Holm = 0.041 to 0.061, and recenter→align
(−0.013, 5/14) from 0.041 to 0.071. The raw p values are unchanged (0.020 and 0.035).

### recenter vs zero_shot

| Method | Δ | Sessions better | p (Holm) | Median after recentering |
|---|---|---|---|---|
| ridge_channels | +0.183 | 14/14 | 0.0004 | 0.146 |
| GPFA | +0.099 | 14/14 | 0.0006 | 0.127 |
| PCA | +0.023 | 13/14 | 0.002 | 0.118 |
| Isomap (d = 25) | +0.010 | 13/14 | 0.003 | 0.025 |
| FA | +0.009 | 14/14 | 0.0007 | 0.042 |
| LFADS | +0.009 | 9/14 | 0.061 | 0.125 |

Recentering remains the largest single gain, and it is what makes ridge_channels usable at all.

### align vs recenter (moment matching)

Same values as in the first round; only LFADS's Holm p changed (see above):

| Method | Δ | Sessions better | p (Holm) | Effect |
|---|---|---|---|---|
| FA | +0.005 | 13/14 | 0.006 | small gain |
| PCA | +0.0006 | 8/14 | 1.0 | no change |
| LFADS | −0.013 | 5/14 | 0.071 | worse, n.s. after Holm |
| GPFA | −0.062 | 0/14 | 0.0006 | worse |

Section 5 tests where the loss comes from.

### fewshot vs align, with the wider λ grid

The validation curves (Fig. C, `lambda_curves.csv`) settle the question raised in the first round.

- **FA and LFADS.** Even with the grid extended to 10¹⁵, the optimum stays at the upper edge, and the curve rises
  monotonically to the λ → ∞ asymptote, which is align. Following the rule fixed before the rerun, fewshot = align for
  these two methods. On the held-out sessions their fewshot and align values are identical.
- **PCA and GPFA.** The optimum is interior (10^6.5 and 10⁷), but its gain over the asymptote on the pseudo-sessions is
  2.5·10⁻⁵ and 4.7·10⁻⁵ R². On the held-out sessions:
  - PCA: fewshot − align = +0.0002, 8/14, p = 1.0;
  - GPFA: +0.0006, 11/14, p_Holm = 0.049.
- **Result.** For the four methods that decode, refitting the readout on 2 trials adds nothing measurable over align.
  Each fold has 1,390–1,876 bins under eval_mask. The readout has 1,251 weights per DoF (751 for FA), so the ratio of
  bins to weights is 1.1–1.9. With this much data, 2 trials are not enough to refit the readout: the
  pseudo-few-shot selects a pull towards the held-in weights so strong that the refit coincides, in practice, with
  align.
- **The maps at d = 25** have interior optima with larger pseudo-session gains: Isomap +0.048 (at λ = 10^4.5), UMAP
  and CEBRA +0.003, all at levels near zero. On the held-out sessions none of them differs from align (Isomap −0.0009,
  7/14; UMAP +0.001, 9/14; CEBRA +0.0004, 7/14; all p_Holm ≥ 0.4).
- **ridge_channels** has an interior optimum (λ = 10⁵, +0.008 over recenter on the pseudo-sessions). On the held-out
  sessions, fewshot − recenter = +0.007, 9/14, p_Holm = 0.58.

### fewshot_gain vs align: a two-parameter-per-DoF recalibration makes things worse

fewshot_gain was worse than align for every method:

| Method | Median fewshot_gain R² | Δ vs align | Sessions better | p (Holm) |
|---|---|---|---|---|
| PCA | 0.024 | −0.105 | 0/14 | 0.0006 |
| FA | −0.014 | −0.076 | 0/14 | 0.0007 |
| GPFA | −0.003 | −0.082 | 1/14 | 0.0007 |
| LFADS | 0.004 | −0.082 | 0/14 | 0.0005 |
| UMAP (d = 25) | −0.034 | −0.033 | 0/14 | 0.0006 |
| CEBRA (d = 25) | −0.053 | −0.051 | 0/14 | 0.0006 |
| Isomap (d = 25) | −0.033 | −0.045 | 4/14 | 0.089 |
| ridge_channels (vs recenter) | 0.088 | −0.050 | 1/14 | 0.002 |

The same happens on the held-in pseudo-sessions, where fewshot_gain also fell well below align (PCA 0.213 → 0.125,
GPFA 0.206 → 0.059, LFADS 0.207 → 0.068, ridge_channels 0.274 → 0.212 relative to recenter). So the loss is not
specific to the held-out days.

**Where the loss comes from** (exploratory, added after seeing the results; `check_gain_decomposition.csv`,
`check_gain_per_trial.csv`; seed 0, same folds). Median R² over the 14 held-out sessions:

| Method | align (ridge_channels: recenter) | gain + offset (= fewshot_gain) | gain only | offset only | gain + offset fit on the test trial (upper bound) |
|---|---|---|---|---|---|
| PCA | 0.108 | 0.025 | 0.045 | 0.102 | 0.189 |
| GPFA | 0.071 | −0.003 | 0.010 | 0.057 | 0.194 |
| LFADS (seed 0) | 0.107 | 0.011 | 0.032 | 0.101 | 0.198 |
| FA | 0.050 | −0.014 | −0.001 | 0.034 | 0.111 |
| ridge_channels | 0.146 | 0.088 | 0.100 | 0.125 | 0.263 |

- **Most of the loss comes from the gain.** In median R², the offset alone costs 0.006–0.021; the gain alone costs 0.046–0.075
  for these five pipelines.
- **The gain fit on the test trial itself would help a lot.** It reaches 0.19–0.20 for PCA, GPFA and LFADS. So the
  predictions are under- or mis-scaled, but the right scaling for one trial is not the right scaling for another.
- **Single-trial gains vary across the trials of a session by an amount comparable to the gain itself.** A per-DoF gain fit on
  one trial at a time has a median of 1.19 for PCA, with a median SD of 0.90 across the 3 trials of the same session
  (GPFA 1.02 ± 0.72, LFADS 1.16 ± 0.79, ridge_channels 0.76 ± 0.52).
- **Gains fit on the 2 adaptation trials** spread from 0.20 to 2.31 for PCA (5th–95th percentile over folds and
  DoF).

The obstacle is therefore not only the ratio of bins to weights: even 2 parameters per DoF, fit on 1,390–1,876 bins,
do not transfer from 2 trials to the third. This check cannot say whether the trial-to-trial spread of the gain is a
real change in the neural-to-kinematic relation or estimation noise. The velocity is strongly autocorrelated, so the
effective sample of a single trial is much smaller than its bin count.

### scratch, fa_native and ridge_rates

- **scratch** was worse than fewshot for PCA, FA, GPFA, Isomap and UMAP:
  - 0–2 of 14 sessions better;
  - median loss 0.021 (UMAP) to 0.14 (PCA);
  - p_Holm ≤ 0.016.
- **CEBRA (d = 25) is the exception:** scratch − fewshot = −0.013, 3/14 better, p_Holm = 0.15 (n.s.), around a
  level of ≈ 0.
- **Median scratch R²** was ≤ 0 for every method (−0.028 to −0.007).
- **ridge_rates** stayed at −0.016 [−0.038, 0.015].
- **fa_native** was worse than align in all 14 sessions (Δ = −0.026, p_Holm = 0.0007).

## 3. Differences between methods

### Friedman tests (sessions as blocks)

Main analysis, maps at d = 25:

| Condition | Methods | χ² | p | Kendall's W |
|---|---|---|---|---|
| align | 7 | 68.8 (df 6) | 7.2·10⁻¹³ | 0.82 |
| fewshot | 7 + ridge_channels | 68.0 (df 7) | 3.7·10⁻¹² | 0.69 |
| fewshot_gain | 7 + ridge_channels | 42.5 (df 7) | 4.2·10⁻⁷ | 0.43 |

Variant with the maps at d = 3:

| Condition | χ² | p | W |
|---|---|---|---|
| align | 66.1 | 2.6·10⁻¹² | 0.79 |
| fewshot | 72.7 | 4.3·10⁻¹³ | 0.74 |
| fewshot_gain | 42.8 | 3.6·10⁻⁷ | 0.44 |

### Pairwise Wilcoxon tests (Holm within condition), align

- **PCA and LFADS do not differ** (LFADS − PCA = −0.016, 5/14, p_Holm = 0.93). Both have a median of 0.108.
- **Both beat FA and GPFA** in 12–13 of 14 sessions (p_Holm ≤ 0.009). FA and GPFA do not differ (p_Holm = 0.93).
- **All four beat Isomap at d = 25.** GPFA vs Isomap is the closest pair: Isomap − GPFA = −0.043, 3/14,
  p_Holm = 0.020.
- **Isomap at d = 25 beats UMAP and CEBRA at d = 25** in 14/14 sessions (Δ ≈ +0.024, p_Holm = 0.003).
- **UMAP and CEBRA** differ by 0.0002 R² around zero.
- **With the maps at d = 3,** the ordering of the non-map methods is the same, and all three maps sit at ≈ 0.

### fewshot, with ridge_channels added

- ridge_channels has the highest median: 0.150 [0.075, 0.233].
- Its advantage over the best latent pipelines does not survive Holm correction:

  | Comparison | Δ | Sessions better | p (Holm) |
  |---|---|---|---|
  | ridge_channels − PCA | +0.057 | 11/14 | 0.29 |
  | ridge_channels − LFADS | +0.046 | 11/14 | 0.54 |
  | ridge_channels − FA | +0.109 | 12/14 | 0.058 |
  | ridge_channels − GPFA | +0.094 | 12/14 | 0.12 |

- It beats the three maps (13/14, p_Holm = 0.011).
- In the d = 3 variant, whose Holm family is different, ridge_channels − FA reaches p_Holm = 0.042; the other three
  comparisons in the table stay non-significant.
- At fewshot_gain, ridge_channels beats every latent pipeline (13–14/14, p_Holm ≤ 0.008), because it loses less from
  the gain refit.
- **Day profile.** ridge_channels has the largest day-to-day swings of any pipeline (Fig. B): from 0.26 on day 6 to
  −0.003 on day 7.

## 4. The nonlinear maps at comparable dimensionality

- **Isomap** improved with d: from ≈ 0 at d = 3 to a median of 0.023 at align with d = 25 (0.025 after recentering).
  That is still about a fifth of PCA's 0.108 and below FA's 0.050.
- **UMAP and CEBRA-Time** stayed at R² ≈ 0 at d = 25 in zero_shot, recenter, align and fewshot, and below zero in
  fewshot_gain and scratch.
- **The failure is already present inside the held-in set.** For UMAP and CEBRA at d = 25 (all three seeds), the
  leave-one-day-out choice of the held-in readout's α lands at the top of its grid (10⁸). There the readout predicts
  essentially the mean: leave-one-day-out R² ≈ 0.000. Isomap at d = 25 reaches only 0.005 leave-one-day-out
  (α = 10⁷). For comparison, PCA reaches 0.199, GPFA 0.205 and LFADS 0.174–0.195.
- **So the near-zero held-out results of UMAP and CEBRA are not caused by the held-out days.** Their latents, as fit
  here, do not support a readout that generalizes across held-in days.

**Within a session the map latents do carry velocity** (exploratory, added after seeing the results;
`check_within_session.csv`, `check_session_variance.csv`; seed 0). Setup: in each of the 13 held-in sessions, a readout
trained on all trials but the last 3 (α leave-one-trial-out) and tested on the last 3.

| Latents (d) | Within-session R², median [min, max] | Leave-one-day-out R², held-in readout | Share of latent variance explained by session identity |
|---|---|---|---|
| PCA (25) | 0.245 [0.134, 0.316] | 0.199 | 0.20 |
| GPFA (25) | 0.249 [0.164, 0.294] | 0.205 | 0.23 |
| LFADS (25) | 0.217 [0.143, 0.277] | 0.192 | 0.27 |
| FA (15) | 0.120 [0.043, 0.259] | 0.071 | 0.20 |
| Isomap (25) | 0.204 [0.055, 0.325] | 0.005 | 0.19 |
| UMAP (25) | 0.134 [−0.222, 0.234] | −0.000 | 0.80 |
| CEBRA-Time (25) | 0.132 [0.084, 0.216] | −0.000 | 0.998 |

The share of variance is the between-session η² of the standardized latents, averaged over dimensions; for the
velocity itself it is 0.0001.

- **Not a bug.** The near-zero cross-session results are not caused by broken latents. Inside a session, all three
  maps at d = 25 decode velocity: Isomap at 0.204 (PCA 0.245), UMAP and CEBRA at about 0.13.
- **UMAP and CEBRA-Time.** Their latents are dominated by session identity: 80% and 99.8% of their variance is
  between-session differences, against 19–27% for the other methods. A readout trained across days then has
  nothing that is common to all days, and the held-in α goes to the top of the grid.
- **Mechanism for CEBRA-Time.** This fits its objective: with time-contrastive training on pooled sessions, separating
  sessions is a cheap way to tell distant time points apart. This is an interpretation, not tested here.
- **Isomap.** Its session share (0.19) is like PCA's, yet its leave-one-day-out R² is only 0.005. Whatever makes its
  latents day-specific is not a shift of the session means, and this check does not identify it.

The first-round statement "the data do not say whether these methods would transfer at a larger d" is now answered
for d = 25:
- Isomap transfers weakly.
- UMAP and CEBRA-Time, with the hyperparameters frozen from notebooks 03/04, do not transfer.

Other hyperparameters (UMAP neighbours, CEBRA's conditional distribution and training length) were not explored.

## 5. Why does moment matching not help on the held-out sessions?

### Within-session diagnostic (exploratory)

The diagnostic was set up before the second-round results (`crosssession_mmdiag`, Fig. D):
- In each of the 13 held-in sessions, blocks of 3 consecutive trials are rotated exactly like the held-out sessions.
- The "source" is the rest of the same session, so there is no day-to-day drift to correct.

| Method | No adaptation | recenter | align | Δ align − recenter | Sessions better | p_Holm (n = 13) |
|---|---|---|---|---|---|---|
| GPFA | 0.236 | 0.234 | 0.164 | −0.065 | 0/13 | 0.002 |
| LFADS | 0.226 | 0.229 | 0.194 | −0.037 | 1/13 | 0.002 |
| PCA | 0.229 | 0.232 | 0.218 | −0.011 | 0/13 | 0.002 |
| FA | 0.094 | 0.098 | 0.100 | +0.0006 | 10/13 | 0.080 |
| Isomap (d = 25) | 0.092 | 0.091 | 0.095 | +0.004 | 11/13 | 0.065 |

UMAP and CEBRA start from 0.01–0.02 even here, because the diagnostic reuses their edge-of-grid held-in α. They carry
little information (protocol note 12).

### Interpretation (rule fixed before the results)

The rule fixed before the results was: if align is worse than recenter even inside a session, where there is no
day-to-day drift, then the loss on the held-out sessions comes from the behavioural content of the 2 adaptation trials
and not from drift. The condition is met for GPFA, LFADS and PCA (and for UMAP: −0.005, 2/13 better, p_Holm = 0.03,
at a level of 0.02).

- **GPFA.** The within-session loss (−0.065) matches the held-out loss (−0.062).
- **LFADS.** The within-session loss (−0.037) is larger than the held-out loss (−0.013).
- **PCA.** It loses 0.011 inside a session, and the net effect on the held-out sessions is about zero (+0.0006).
- **Conclusion.** For these three methods, the loss on the held-out sessions comes from matching the latent moments
  to those of 2 trials, not from day-to-day drift.
- **FA and Isomap** do not lose inside a session, and FA is also the one method with a consistent align gain on the
  held-out sessions.
- **Consistency with the pseudo-sessions.** There, 1–5 days from the training days, align beat recenter for PCA
  (0.189 → 0.213) and LFADS (0.173 → 0.207). This fits a balance between two effects: a correction that pays off when
  there is drift, and a cost of estimating the moments from 2 trials that is present with or without drift. It
  remains an interpretation, not a test.

### Behavioural coverage (descriptive)

The coverage counts do not identify what in the trial content matters:

| Set | Folds | Median conditions (of 16) | Min conditions | Directions (of 14), median / min | Median velocity-variance ratio |
|---|---|---|---|---|---|
| held-out | 42 | 16 | 8 | 14 / 13 | 0.89 |
| pseudo-sessions | 60 | 16 | 8 | 14 / 13 | 0.90 |
| within-session blocks | 162 | 16 | 8 | 14 / 13 | 0.91 |

- **Conditions** are phase × task version; the version is set by trial parity.
- **One third of the folds** (14/42 held-out) have both adaptation trials of the same task version, and so 8 of 16
  conditions. These are the folds that test on the middle trial.
- **Velocity variance.** The 2 trials carry about 0.9 of the pool's velocity variance (median over DoF).
- **Same coverage everywhere.** Coverage is the same in the held-out, pseudo-session and within-session folds, yet
  for PCA and LFADS align helps in the pseudo-sessions and hurts inside sessions.
- **Splitting folds by number of task versions** gives no consistent pattern (Δ align − recenter, one version vs
  two):
  - inside held-in sessions, GPFA and LFADS lose more when both trials share a version (−0.063 vs −0.034;
    −0.035 vs −0.018);
  - on the held-out sessions, only GPFA does (−0.066 vs −0.049), while PCA and LFADS do better with one version
    (+0.024 vs −0.005; +0.003 vs −0.022).

So the "content of the 2 trials" that matters is not captured by which conditions or directions they contain.

## 6. Limitations

- **Few trials.**
  - 2 adaptation trials per fold (1,390–1,876 bins under eval_mask);
  - 3 folds per session, each test fold a single trial;
  - 14 sessions on 7 days.
- **Not FALCON's hidden evaluation.**
  - Trials are processed as separate sequences, so decoder history restarts at every trial.
  - Smoothing, GPFA, LFADS and CEBRA see bins after the decoded bin within a trial (non-causal).
  - R² is per file, not per day.
  - The numbers are not comparable with the FALCON leaderboard.
- **ridge_channels.** It is applied trial by trial here, unlike in notebook 05, so its numbers differ from the 05
  evaluation. It uses all 176 channels; the latent methods use the 170 active ones.
- **Tuning on a different gap.** λ and α were tuned on pseudo-sessions 1–5 days from their training days; held-out
  gaps are 6–20 days.
- **Exploratory analyses after the results.** The fewshot_gain decomposition, the within-session decoding of the maps
  and the session-variance measure were added after seeing the second-round results (protocol note 11). They explain
  results; they are not tests.
- **Moment matching is blind to rotations.** On synthetic data, a pure rotation leaves the covariance matched while
  the pointwise error stays large.
- **LFADS.**
  - No scratch condition.
  - Hyperparameters selected on one session, trained here for a fixed 200 epochs on 13 sessions.
  - Seed-dependent: the median per-session SD across seeds is 0.021 R² at align (first round).
- **Other fixed choices.**
  - The fixed 170-channel set.
  - Isomap and UMAP fit on at most 5,000 held-in points.
  - A single readout family (ridge with history).
  - Map hyperparameters frozen from the earlier notebooks.

## 7. Stability of FA subspaces and zero-shot R² (added 2 November 2026)

Plan written before any index was computed (`docs/crosssession_protocol.md`, section of 2/11; commit 26be985), run with
code cdf90b1 (registry experiment `fa_stability`). Full numbers in `results/crosssession/stability/report.md`; figures
`fig_E_stability_by_days_k15.png` (k = 5: `_k5`) and `fig_F_alignment_vs_zero_shot.png`.

- **Setup.**
  - FA with K = 15, as in the decoder, on blocks of 3 consecutive trials: 54 held-in blocks and the 14 held-out files.
    Held-in and held-out subspaces are therefore estimated from the same amount of data.
  - Primary measure: alignment index (Elsayed et al. 2016) on the shared covariance L Lᵀ, mean of both directions;
    1 = same subspace. Secondary: mean principal angle.
  - Chance: random subspaces drawn from a covariance (all bins, condition means, uniform), 500 draws per pair.
- **A 3-trial FA subspace is estimable.** Two blocks of the same held-in session have a median index of 0.50
  [IQR 0.48, 0.53]. That is far above the 95th percentile of all three chance versions (0.10–0.18). The angle is 54°,
  against a chance 5th percentile of 73–74°. So the estimability rule passes and the tests use k = 15 (k = 5: 0.71
  against ≤ 0.15).
- **Same day ≈ same session.** Blocks from the two files of one day reach 0.49 (held-in) and 0.52 (held-out), the
  same level as the same-session reference.
- **The subspaces drift with the days between blocks.** Same-session pairs are excluded; 2,183 pairs remain.
  - Mantel test: ρ = −0.50, p = 0.0008 (permutation of the 13 days).
  - Slope: −0.0030 index per day (95% CI −0.0051 to −0.0017, bootstrap on days).
  - The angle (ρ = +0.60) and k = 5 (ρ = −0.55) agree.
  - Descriptively, most of the drop comes in the first ~10 days. The median index is 0.49 at 0 days, 0.46 at 1–5,
    0.41 at 6–10, then 0.40–0.42 up to 20 days and 0.37 at 31–39.
  - The index stays far above chance at every distance: the lowest median for a pair of days is 0.32. The shared
    subspace changes in part, not completely.
- **Held-out sessions against the decoder's frozen FA.** The comparison uses the decoder's normalization.
  - Median index 0.295 (range 0.233–0.432), median angle 59.7°.
  - This is below the held-in reference: held-in blocks against the FA fit without their day give a median of 0.347
    (IQR 0.312–0.381) and 54.0°.
  - 12 of 14 sessions exceed all three chance thresholds. The two S8 files do not exceed the all-bins one, which is
    high on that day (0.33 and 0.30).
- **No detectable link with zero-shot R² (the test).**
  - Index against FA zero-shot R²: ρ = +0.19, p = 0.51 (n = 14).
  - Day means: ρ = +0.39, p = 0.40 (n = 7).
  - Partial on days since the held-in sessions: ρ = +0.18, p = 0.54.
  - With n = 14 only |ρ| ≥ 0.54 is detectable (n = 7: ≥ 0.79). The result reads "not detectable with these data",
    not "no relation".
  - k = 5 (ρ = +0.20) and the angle (ρ = +0.02) agree.
- **Descriptive.**
  - The same index does not track the zero-shot of PCA, GPFA or LFADS (ρ −0.13 to −0.02) or of ridge_channels
    (−0.37).
  - It does not track the align − recenter gain of FA either (−0.24).
  - The median index against the 54 held-in blocks agrees with the frozen-FA index (ρ = +0.80), and it is also
    unrelated to FA zero-shot (+0.01).
- **Reading.** The shared FA subspace of 3 trials is reproducible within a day and drifts gradually over days. How
  far a held-out session has drifted from the decoder's FA does not predict how well the frozen decoder works on it,
  within what 14 sessions can detect. This fits section 1, where FA zero-shot has no trend with days. One possible
  reason, not tested: FA zero-shot is close to its floor (median 0.034, range 0.012–0.064), so there is little
  between-session variance left to explain.
- **Limitations.**
  - 14 sessions on 7 days.
  - The index uses all 3 trials, including the ones on which zero-shot is evaluated. It is a descriptive
    correlation: nothing is fit on the index.
  - The held-out blocks are the first 3 trials of each file; the held-in blocks come from the whole file.
  - Block-versus-block pairs use each block's own z-score (correlation structure). The comparison with the frozen FA
    uses the decoder's normalization.
