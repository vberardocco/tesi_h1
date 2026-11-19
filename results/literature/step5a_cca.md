# Step 5a · CCA as in Gallego 2020 (optional)

Commit 4a66c54. Square root, 30 ms bins, Gaussian σ 50 ms, PCA with 10 dimensions per session; windows of 0.99 s from the start of the movement phases, 16 conditions, 3 repetitions per condition and per half; 20 draws (median [min-max]). Normalized = between sessions / mean of the two values within the same day.

| Comparison | days | aligned (CCA) | unaligned | aligned, normalized | unaligned, normalized | swapped conditions (control) |
| --- | --- | --- | --- | --- | --- | --- |
| S1, same day | 0 | 0.32 [0.30-0.34] | 0.21 [0.17-0.26] | — | — | — |
| S2, same day | 0 | 0.31 [0.29-0.32] | 0.22 [0.18-0.25] | — | — | — |
| S5, same day | 0 | 0.28 [0.24-0.31] | 0.17 [0.14-0.21] | — | — | — |
| S1-S2 | 5 | 0.31 [0.28-0.34] | 0.16 [0.12-0.20] | 0.99 [0.90-1.09] | 0.73 [0.57-0.94] | 0.23 [0.21-0.25] |
| S1-S5 | 12 | 0.28 [0.26-0.31] | 0.10 [0.07-0.13] | 0.95 [0.88-1.02] | 0.51 [0.36-0.72] | 0.22 [0.20-0.25] |
| S2-S5 | 7 | 0.28 [0.27-0.30] | 0.09 [0.07-0.12] | 0.96 [0.85-1.13] | 0.49 [0.32-0.62] | 0.22 [0.20-0.25] |

Post hoc variant: the same CCAs on the means of the 3 repetitions of each condition (16 mean windows per half, 528 rows).

| Comparison | days | aligned (CCA) | unaligned | aligned, normalized | unaligned, normalized | swapped conditions (control) |
| --- | --- | --- | --- | --- | --- | --- |
| S1, same day | 0 | 0.50 [0.46-0.54] | 0.43 [0.36-0.48] | — | — | — |
| S2, same day | 0 | 0.49 [0.45-0.53] | 0.39 [0.35-0.47] | — | — | — |
| S5, same day | 0 | 0.46 [0.43-0.50] | 0.35 [0.29-0.39] | — | — | — |
| S1-S2 | 5 | 0.49 [0.45-0.54] | 0.28 [0.22-0.32] | 0.99 [0.93-1.06] | 0.68 [0.51-0.79] | 0.38 [0.32-0.43] |
| S1-S5 | 12 | 0.45 [0.42-0.49] | 0.17 [0.13-0.21] | 0.93 [0.85-1.10] | 0.43 [0.32-0.57] | 0.36 [0.33-0.42] |
| S2-S5 | 7 | 0.44 [0.41-0.48] | 0.16 [0.10-0.25] | 0.94 [0.86-1.04] | 0.42 [0.30-0.68] | 0.37 [0.34-0.41] |

Canonical correlations per component, single trials (median over draws):

| Comparison | CC1 | CC2 | CC3 | CC4 | CC5 | CC6 | CC7 | CC8 | CC9 | CC10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 0.72 | 0.58 | 0.50 | 0.41 | 0.33 | 0.25 | 0.17 | 0.12 | 0.06 | 0.02 |
| S2 | 0.77 | 0.67 | 0.46 | 0.35 | 0.26 | 0.20 | 0.15 | 0.11 | 0.07 | 0.02 |
| S5 | 0.74 | 0.49 | 0.39 | 0.32 | 0.26 | 0.21 | 0.17 | 0.13 | 0.07 | 0.02 |
| S1-S2 | 0.76 | 0.59 | 0.46 | 0.38 | 0.30 | 0.22 | 0.17 | 0.11 | 0.06 | 0.02 |
| S1-S5 | 0.72 | 0.55 | 0.40 | 0.33 | 0.27 | 0.21 | 0.15 | 0.12 | 0.05 | 0.02 |
| S2-S5 | 0.74 | 0.53 | 0.39 | 0.33 | 0.26 | 0.21 | 0.15 | 0.10 | 0.06 | 0.03 |

Gallego 2020 (monkey CL, up to ~750 days, M1 at 10 dimensions): "the normalized similarity of the latent dynamics after alignment with CCA was 0.93 ± 0.03", against 0.38 ± 0.14 without alignment. The comparison with H1 is qualitative: different data, conditions (targets versus phases) and windows.
Variance explained by the 10 PCs on the windows: S1 32.4%, S2 33.2%, S5 35.5%.
