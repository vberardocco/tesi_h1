# Phase 2 · R10 · TME null for non-linear structure

Commit 880a3fc. Rules: docs/literature/rules_decision.md (commit 1aa7c5d), clarifications P3-P5. TME: src/thesis_h1/tme.py, port of the official MATLAB code (github.com/gamaleldin/TME), type TNC.

**Note.** By construction the TME preserves the covariance between neurons, so it also preserves the PCA spectrum. For this reason it is not useful to test linear dimensionality, but only what goes beyond the marginal covariances. Here it is used for the non-linear structure of slide 8.

## Validation of the surrogates (relative Frobenius error on the marginal covariances)

| Session | T × N × C | expected: time / neurons / conditions | mean of 100 surrogates | single surrogate (median) | valid |
| --- | --- | --- | --- | --- | --- |
| S1 | 50 × 160 × 16 | 0.000% / 0.000% / 0.000% | 4.5% / 10.0% / 5.6% | 23.1% / 78.7% / 43.9% | yes |
| S2 | 50 × 166 × 16 | 0.000% / 0.000% / 0.000% | 4.7% / 9.6% / 6.1% | 26.1% / 76.6% / 46.7% | yes |
| S5 | 50 × 165 × 16 | 0.000% / 0.000% / 0.000% | 4.4% / 9.3% / 5.3% | 22.9% / 75.2% / 42.4% | yes |

The 5% criterion (P3) applies to the expected covariances of the fitted distribution. The mean of the surrogates and the single surrogate deviate through sampling noise alone (the surrogates are centred on the tensor of the data means, M.TNC).

## Statistic: Isomap − PCA advantage at d = 3 on the condition means

| Session | data | TME null: median [95th percentile; max] | p |
| --- | --- | --- | --- |
| S1 | +8.1 | +8.0 [+18.6; +24.0] | 0.495 |
| S2 | +4.5 | +6.6 [+21.6; +32.6] | 0.614 |
| S5 | +0.1 | +12.5 [+27.5; +45.0] | 0.950 |

**R10**: explained by the marginals.

## Single trial with FALCON bins (updates +3.6 and −9.6 of slide 8)

| Session | points | data, d = 3 | twins, mean [min, max] | twins below the data |
| --- | --- | --- | --- | --- |
| S1 | 1099 | +5.3 | −9.2 [−13.5, −5.3] | 5 of 5 |
| S2 | 1037 | +6.2 | −9.7 [−10.6, −8.3] | 5 of 5 |
| S5 | 1137 | +0.9 | −9.0 [−10.7, −5.8] | 5 of 5 |

Check with legacy bins in S1 (1106 points) against notebook 03: data +3.58 (expected +3.6), twins mean −9.55 (expected −9.6), best twin −5.35 (expected −5.3): reproduced.
