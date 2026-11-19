# Step 1g · Cheater and Frankenstein (exploratory, post hoc)

Commit f46b6c5. Added on 3 Nov after seeing the results: it does not change any R1-R7 verdict and is not in the committed rules. Pipeline 1a; circular-shift null, 20 surrogates per draw; values in % of the variance (R1 = first 12 PCs).

## Cheater (channels chosen by variance alone; these are not H1 values)

| Session | Group | share of the 12 most variable (%) | max/min variance | R1 data | R1 null | data − null |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | A | 79.0 | 131.3 | 79.5 | 79.3 | +0.2 |
| S1 | B | 24.9 | 2.8 | 39.0 | 28.6 | +10.3 |
| S2 | A | 82.4 | 105.0 | 83.0 | 82.7 | +0.3 |
| S2 | B | 25.0 | 2.7 | 38.6 | 28.4 | +10.2 |
| S5 | A | 81.5 | 95.5 | 82.1 | 81.7 | +0.4 |
| S5 | B | 25.3 | 2.5 | 45.1 | 29.5 | +15.6 |

Random draws of 66 channels (200): r² between the share of the 12 most variable and R1 on the data: S1 0.84 (medians: share 43.7%, R1 51.6%); S2 0.84 (medians: share 44.6%, R1 52.3%); S5 0.66 (medians: share 43.5%, R1 54.6%)

## Frankenstein (66 channels, 20 draws; median [min-max])

### R1 without z-score

| Session | data | phases from different trials | null | data − null | shuffled − null | share |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 51.4 [47.9-55.2] | 46.8 [42.2-50.6] | 45.3 [40.8-49.0] | +6.2 [+4.6-+7.3] | +1.4 [+0.8-+2.4] | 24% [14%-33%] |
| S2 | 51.7 [49.1-55.4] | 48.4 [44.2-51.8] | 46.0 [42.2-50.2] | +6.1 [+4.2-+7.2] | +2.2 [+1.0-+3.0] | 35% [16%-54%] |
| S5 | 55.0 [50.7-57.2] | 48.6 [43.7-51.2] | 46.6 [41.4-49.3] | +8.6 [+6.5-+11.3] | +2.0 [+1.0-+2.9] | 23% [13%-29%] |

Frankenstein (22 channels from each session, S1 structure): 46.9 [40.8-50.2]; its null 45.1 [39.1-49.0]; difference +1.7 [+1.1-+3.0].

### R1 with z-score

| Session | data | phases from different trials | null | data − null | shuffled − null | share |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 31.2 [29.6-33.5] | 27.1 [26.1-28.2] | 24.4 [24.2-24.8] | +6.7 [+5.2-+8.7] | +2.7 [+1.8-+3.5] | 40% [29%-49%] |
| S2 | 31.4 [30.5-33.2] | 28.0 [27.3-29.2] | 24.5 [24.3-24.7] | +7.0 [+6.1-+8.5] | +3.5 [+2.8-+4.6] | 49% [41%-61%] |
| S5 | 32.9 [30.7-35.0] | 27.3 [26.4-28.7] | 24.8 [24.5-25.3] | +8.0 [+6.1-+10.1] | +2.5 [+1.5-+3.6] | 30% [20%-43%] |

Frankenstein (22 channels from each session, S1 structure): 27.2 [26.5-28.0]; its null 24.4 [24.2-24.5]; difference +2.8 [+2.2-+3.6].

Cases (channel × segment, sum over the 20 draws). Phases from different trials: S1 same key 283800, same phase 1320, same trial 0; S2 same key 277200, same phase 1320, same trial 0; S5 same key 278520, same phase 1320, same trial 0. Frankenstein: same key 285120, same phase 0, channel mean 0.

The share is nonlinear: in testing, a scenario halfway between task and internal structure gives about 30%. The dimensions above the null are not counted in the shuffled versions (channels that draw the same trial retain a small correlation). Shuffling the trials loses the part that requires simultaneous recording, but also the signal specific to each condition; slow common drift and noise common across channels remain possible.

## Check: cov(even mean, odd mean) / single-trial variance (median over channels)

| Session | phase by phase | from the trial start only |
| --- | --- | --- |
| S1 | 2.5% | 1.3% |
| S2 | 3.9% | 1.3% |
| S5 | 2.5% | 1.1% |

Alignment from the trial start only is not used for the Frankenstein.
