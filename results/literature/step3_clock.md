# Step 3 · Clock + neurons (R7)

Commit 5ee65a3. Same points, blocks and classifier as the replication (8 movement phases, one point every 100 ms, 5 blocks of 3 trials, 10 neighbors, balanced accuracy, chance 12.5%). (c): clock weight chosen by cross-validation on the training trials only. Null: latents permuted across whole test trials (at the same time from the trial start), clock held fixed, 200 times.

| Session | points | (a) clock | (b) GPFA 25 | (c) clock + GPFA | (c) − (a) | null of (c): median [95th] | p | clock weight per block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 1099 | 50.1 | 49.9 | 71.3 | +21.2 | 51.1 [59.9] | 0.010 | 0.5, 4, 2, 0.5, 1 |
| S2 | 1037 | 47.2 | 54.4 | 73.5 | +26.3 | 47.0 [57.1] | 0.010 | 2, 2, 0.5, 1, 1 |
| S5 | 1137 | 63.8 | 42.6 | 75.4 | +11.6 | 61.1 [68.0] | 0.010 | 4, 2, 4, 4, 4 |

**R7**: yes ((c) − (a) > 0 with p < 0.05 in 3 sessions out of 3).

Variants of (c) with fixed weight (no verdict):

| Session | (c), w = 1 | − (a) | p | (c), z-score of all columns (w = 0.2) | − (a) | p |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 74.1 | +23.9 | 0.010 | 64.4 | +14.3 | 0.010 |
| S2 | 76.5 | +29.3 | 0.010 | 62.6 | +15.4 | 0.010 |
| S5 | 70.9 | +7.2 | 0.010 | 58.3 | -5.5 | 0.010 |

Optional: balanced accuracy within the deciles of time from the trial start ((a) / (b) / (c)):

| Decile | S1 | S2 | S5 |
| --- | --- | --- | --- |
| 1 | 79 / 41 / 56 | 67 / 25 / 48 | 100 / 22 / 100 |
| 2 | 39 / 11 / 29 | 47 / 28 / 52 | 33 / 12 / 50 |
| 3 | 37 / 36 / 51 | 46 / 37 / 64 | 63 / 36 / 82 |
| 4 | 25 / 44 / 58 | 34 / 48 / 64 | 30 / 34 / 48 |
| 5 | 35 / 41 / 77 | 32 / 78 / 81 | 39 / 38 / 53 |
| 6 | 26 / 54 / 49 | 25 / 94 / 95 | 52 / 58 / 63 |
| 7 | 33 / 45 / 61 | 24 / 61 / 67 | 32 / 51 / 64 |
| 8 | 23 / 59 / 85 | 38 / 65 / 63 | 39 / 38 / 71 |
| 9 | 47 / 60 / 86 | 27 / 70 / 68 | 48 / 43 / 63 |
| 10 | 83 / 56 / 85 | 54 / 66 / 86 | 73 / 49 / 88 |
