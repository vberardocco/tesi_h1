# Step 4 · Robustness checks

Commit fe12d79.

## 4a · Activity threshold at 1 Hz instead of 0.5 Hz (S1)

Cunningham & Yu 2014 (p. 1506): neurons below about 1 spike/s "should typically be excluded".

| Threshold | channels | PR (σ 50 ms, z-score) | components for 80% | K* (FA, CV) | FA dimensions for 95% |
| --- | --- | --- | --- | --- | --- |
| 0.5 Hz | 160 | 106.1 | 100 | 15 | 14 |
| 1 Hz | 148 | 96.7 | 92 | 15 | 14 |

Differences (1 Hz − 0.5 Hz): channels -12, PR -9.4, components for 80% -8, K* +0, dimensions for 95% +0. Channels lost (original indices): [1, 2, 9, 10, 11, 12, 22, 26, 27, 30, 73, 80].

Cross-validation log-likelihood per K (mean over the 5 blocks):

| Threshold | K = 2 | K = 5 | K = 10 | K = 15 | K = 20 | K = 30 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 Hz | -226.6431 | -226.4798 | -226.3340 | -226.3154 | -226.3309 | -226.3919 |
| 1 Hz | -209.5671 | -209.4026 | -209.2504 | -209.2285 | -209.2388 | -209.2921 |

## 4b · Condition means (16 conditions × 50 bins, z-score, σ 50 ms)

| Session | channels | repetitions per condition | first 2 PCs | first 3 PCs | first 6 PCs | first 10 PCs | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 160 | 7-8 | 25.7% | 31.3% | 43.0% | 51.8% | 19.7 |
| S2 | 166 | 7-8 | 30.1% | 34.5% | 45.2% | 54.0% | 16.1 |
| S5 | 165 | 7-8 | 24.5% | 29.1% | 39.5% | 49.3% | 21.6 |

Comparisons: Churchland 2012 uses 6 PCs (k = 6) and 2-3 jPCA planes capture 50-70% of the total variance; Pandarinath 2015: the first jPCA plane captures 61% (T6) and 27% (T7), 28% on average in the macaque. The variance of the first 2 PCs is the maximum over any plane, so it is a ceiling for the first jPCA plane. Conditions, windows and preprocessing are different (16 conditions, 1 s from the start of the phase, z-score): the comparison is only indicative.
