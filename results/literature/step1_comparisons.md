# Step 1 · Dimensionality comparisons on equal terms

Commit a277e83. FALCON bins (replication rep/19250108T110520 and sessions S2, S5), active channels ≥ 0.5 Hz, trial bins without Intertrial and without the bin with an empty label. Values: median [min-max] over 20 draws of channels (seeds 0-19); with 'all channels' a single value. Null: independent circular shift per channel (criteria of Step 2), 20 surrogates per draw; we report the median over draws of the median of the surrogates and the difference data − null computed draw by draw. The null is an extra column: it is not in the committed rules (7cde125) and does not change the verdicts.

## Rules R1-R4

| Rule | Source | Criterion | Source value | S1 | S2 | S5 | null S1 / S2 / S5 | data − null S1 / S2 / S5 | Verdict S1 / S2 / S5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Gallego 2018 | variance of the first 12 PCs, 66 channels, square root, σ 50 ms, no z-score (%) | ≥ 60 (mean 73.4 ± 6.5) | 51.4 [47.8-55.2] | 52.0 [48.8-55.8] | 54.4 [51.2-59.6] | 45.2 / 46.0 / 45.2 | +6.2 [+4.6-+7.3] / +6.2 [+4.8-+7.5] / +9.0 [+6.2-+10.4] | below / below / below |
| R2 | Silvernagel 2026 | components for 80%, 96 channels, σ 30 ms, no z-score | 31-38 | 35.0 [33.0-39.0] | 35.0 [32.0-37.0] | 33.5 [32.0-37.0] | 37.0 / 36.5 / 35.5 | -2.0 [-2.0--1.0] / -2.0 [-2.0--1.0] / -2.0 [-3.0--1.0] | in line / in line / in line |
| R3 | Oby 2019 | FA dimensions for 95% shared, 90 channels, 45 ms, K* (K* = 12, 8, 10) | 5-11 (mean 8.4) | 11.0 [11.0-11.0] | 8.0 [8.0-8.0] | 9.0 [9.0-9.0] | — | — | in line / in line / in line |
| R3 | Degenhart 2020 | FA dimensions for 95% shared, 75 channels, 45 ms, K* (K* = 10, 8, 10) | 4-8 | 9.0 [9.0-10.0] | 8.0 [8.0-8.0] | 9.0 [9.0-9.0] | — | — | higher / in line / higher |
| R4 | Degenhart 2020 | population shared fraction, 75 channels, 45 ms, K*, no z-score (K* = 10, 8, 10) | 32% | 14.3% [10.6%-17.6%] | 13.8% [11.6%-20.3%] | 15.9% [13.8%-18.2%] | — | — | no verdict (R4) |

R3 requires both conditions (Oby 5-11 at 90 channels and Degenhart 4-8 at 75 channels): S1 not in line, S2 in line, S5 not in line. K* is the maximum of the cross-validation curve averaged over 5 draws (seeds 0-4); the K* of the single draws are in step1c_cv.csv.

R1, band 73.4 ± 2 × 6.5% (60.4-86.4%): S1 outside, S2 outside, S5 outside.

## Variants (no verdict)

| Source | Criterion | S1 | S2 | S5 | null S1 / S2 / S5 | data − null S1 / S2 / S5 |
| --- | --- | --- | --- | --- | --- | --- |
| variant 1a | as R1 with z-score | 31.2 [29.6-33.4] | 31.9 [30.4-34.1] | 33.4 [31.3-35.1] | 24.5 / 24.5 / 24.9 | +6.7 / +7.5 / +8.4 |
| variant 1a | as R1 with all active channels | 35.9 [35.9-35.9] | 36.6 [36.6-36.6] | 39.9 [39.9-39.9] | 25.0 / 25.3 / 24.0 | +10.9 / +11.3 / +15.8 |
| variant 1a | as R1 with all active channels e z-score | 21.5 [21.5-21.5] | 21.9 [21.9-21.9] | 23.5 [23.5-23.5] | 12.8 / 12.6 / 13.1 | +8.6 / +9.3 / +10.4 |
| variant 1b | as R2 with z-score | 67.0 [66.0-68.0] | 67.0 [66.0-67.0] | 66.0 [64.0-66.0] | 71.0 / 71.0 / 71.0 | -4.0 / -4.0 / -5.0 |
| variant 1b | as R2 with all active channels | 58.0 [58.0-58.0] | 57.0 [57.0-57.0] | 56.0 [56.0-56.0] | 61.0 / 61.0 / 60.0 | -3.0 / -4.0 / -4.0 |
| variant 1b | as R2 with all active channels e z-score | 107.0 [107.0-107.0] | 110.0 [110.0-110.0] | 108.0 [108.0-108.0] | 115.0 / 118.0 / 117.0 | -8.0 / -8.0 / -9.0 |
| variant 1c | Oby with K = 10 | 9.0 [9.0-10.0] | 9.0 [9.0-10.0] | 9.0 [9.0-9.0] | — | — |
| variant 1c | Oby with K = 20 | 18.0 [18.0-18.0] | 18.0 [18.0-19.0] | 18.0 [18.0-18.0] | — | — |
| variant 1c | Oby with K*, no z-score | 10.5 [10.0-11.0] | 7.0 [7.0-8.0] | 9.0 [8.0-9.0] | — | — |
| variant 1c | Degenhart with K = 10 | 9.0 [9.0-10.0] | 9.0 [9.0-10.0] | 9.0 [9.0-9.0] | — | — |
| variant 1c | Degenhart with K = 20 | 18.0 [18.0-18.0] | 18.0 [18.0-18.0] | 18.0 [18.0-18.0] | — | — |
| variant 1c | Degenhart with K*, no z-score | 9.0 [8.0-9.0] | 7.0 [7.0-7.0] | 9.0 [9.0-9.0] | — | — |
| variant 1c | population shared fraction, 90 channels | 15.4% [12.8%-17.5%] | 13.5% [10.2%-16.7%] | 16.3% [13.8%-18.0%] | — | — |
| variant 1c | shared fraction per channel, median, 75 channels | 4.5% [3.1%-5.0%] | 3.7% [2.1%-4.5%] | 4.5% [3.2%-6.5%] | — | — |

## 1e · PCs for 60% (post hoc analysis, does not replace R1)

Gallego 2018 chooses m = 12 because it guarantees "at least 60%" in all datasets, with a mean of 73.4%: many of their datasets reach 60% with fewer than 12 PCs. Silvernagel 2026 (p. 7): 16 and 13 for monkeys O and U with the threshold at 60%.

| Criterion | Source value | S1 | S2 | S5 | null S1 / S2 / S5 [min-max] | data − null S1 / S2 / S5 |
| --- | --- | --- | --- | --- | --- | --- |
| PCs for 60%, pipeline 1a, 66 channels | m = 12 (Gallego); 16 and 13 (Silvernagel) | 16.0 [15.0-18.0] | 16.0 [14.0-18.0] | 15.0 [13.0-17.0] | 19.0 [17.0-20.5] / 18.0 [16.0-21.0] / 18.0 [15.0-19.0] | -2.0 [-3.0--1.0] / -2.0 [-3.0--2.0] / -3.0 [-4.0--2.0] |
| PCs for 60%, pipeline 1b, 96 channels | 16 and 13 | 19.5 [18.0-23.0] | 19.0 [17.0-20.0] | 18.0 [17.0-20.0] | 22.0 [21.0-25.0] / 21.0 [19.0-23.0] / 21.5 [20.0-23.0] | -3.0 [-3.0--2.0] / -2.0 [-3.0--2.0] / -3.0 [-4.0--2.0] |
| PCs for 60%, pipeline 1a, all active channels | — | 33.0 [33.0-33.0] | 33.0 [33.0-33.0] | 31.0 [31.0-31.0] | 41.0 [41.0-42.0] / 42.0 [42.0-42.0] / 41.0 [41.0-42.0] | -8.0 [-8.0--8.0] / -9.0 [-9.0--9.0] / -10.0 [-10.0--10.0] |
| PCs for 60%, pipeline 1b, all active channels | — | 31.0 [31.0-31.0] | 30.0 [30.0-30.0] | 28.0 [28.0-28.0] | 36.0 [36.0-36.0] / 35.0 [35.0-35.0] / 36.0 [35.0-36.0] | -5.0 [-5.0--5.0] / -5.0 [-5.0--5.0] / -8.0 [-8.0--8.0] |

With all channels the null is over 20 surrogates: median [min-max] of the surrogates.

## R5 · Parallel Analysis (Step 2)

| Session | Pipeline | N | k_PA (circular shift) | k_PA/N | k_PA (Altan shuffle) |
| --- | --- | --- | --- | --- | --- |
| S1 | 02 | 160 | 19 | 11.9% | 50 |
| S1 | gallego | 160 | 11 | 6.9% | 18 |
| S2 | 02 | 166 | 17 | 10.2% | 51 |
| S2 | gallego | 166 | 9 | 5.4% | 11 |
| S5 | 02 | 165 | 15 | 9.1% | 47 |
| S5 | gallego | 165 | 11 | 6.7% | 12 |

R5: yes (minimum threshold k_PA ≥ 1 in all sessions and pipelines). Details in step2_pa.md.

## R6 · Saturation (1d, S1 only, N = 20-160)

| Measure | AIC linear | AIC power | AIC plateau | Verdict |
| --- | --- | --- | --- | --- |
| dim95 | 4.79 | 0.94 | -0.90 | does not saturate in the observed range |
| n80 | -6.84 | -7.28 | -5.33 | does not saturate in the observed range |
| var12 | 36.48 | -0.46 | 8.23 | does not saturate in the observed range |

Figure: step1_saturation.png.
