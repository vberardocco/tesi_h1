# Step 5d · GPFA on 61 channels (optional, S1 only)

Commit 0d4b8bf. Procedure of notebook 02 (sec. 10.2) on 61 channels drawn at random among the 160 active ones, 10 draws; 119 windows of 1 s, square root of counts, 500 iterations, 5 contiguous blocks of windows, score = log-likelihood per bin on the held-out windows.

| | d chosen (notebook rule) | d with the highest mean score |
| --- | --- | --- |
| H1, S1, 61 channels (median [min-max] over 10 draws) | 22.5 [20-25] | 25.0 [25-25] |
| H1, S1, 160 channels (replication) | 25 | — |
| Yu 2009, 61 units (18 single units + 43 multiunits), leave-neuron-out | p* = 10 (8-12 per target) | — |

Mean score on the held-out windows, per d (median over draws, difference from d = 2):

| d = 2 | d = 5 | d = 8 | d = 10 | d = 12 | d = 15 | d = 20 | d = 25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0.0000 | +0.1305 | +0.1982 | +0.2301 | +0.2527 | +0.2710 | +0.2962 | +0.3026 |

Caveats: Yu 2009 chooses p* with the leave-neuron-out error, here with the log-likelihood on the held-out windows; Yu estimates one target at a time, here all phases together. In the tests (tests/test_literature.py) the same procedure on Poisson counts with an exponential of 3 latents chooses 8 factors: with non-linear data the d chosen by cross-validation is a ceiling for the latent dimension, not an estimate.

**At the edge of the grid**: the highest mean score is at d = 25 in 10 draws out of 10, and the curve still rises between d = 20 and 25. The d of H1 with 61 channels is therefore at least the one reported: a lower bound, not a closed estimate. Higher values are needed to find the maximum (notebook 02 went up to 50).
