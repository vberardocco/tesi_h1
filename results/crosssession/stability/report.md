# Stability of FA subspaces on the held-out sessions

Plan: `docs/crosssession_protocol.md`, section of 2/11/2026 (written before any index was computed). Code: `src/tesi_h1/stability.py`, `scripts/xs_stability.py`, tests in `tests/test_stability.py`.

## Data

- 68 blocks of 3 consecutive trials (54 held-in, 14 held-out), 170 channels, FA K = 15; 2278 pairs of blocks.
- The recipe of the decoder's frozen FA was reproduced exactly (max |Δ loading| = 5.7e-15).
- Silent channels: (channel, block) cases with sd = 0 in the block z-score: 0; with sd below sd_min = 0.1: 543 (of 11560). With the decoder normalization, silent channels in the blocks vs the model: 0.
- Condition means: conditions per held-in file [16].

## Estimability rule (same-session reference vs chance)

| k | same-session index, median [IQR] (95 pairs) | chance 95th perc. (median over pairs) | passes | same-session angle, median | chance 5th perc. | passes |
|---|---|---|---|---|---|---|
| 15 | 0.503 [0.476, 0.532] | allbins 0.164, cond 0.183, unif 0.099 | yes | 54.3° | allbins 73.4°, cond 72.6°, unif 74.0° | yes |
| 5 | 0.708 [0.660, 0.733] | allbins 0.122, cond 0.145, unif 0.067 | yes | 42.6° | allbins 79.2°, cond 78.3°, unif 79.7° | yes |

Primary k for the tests: **15**.

## Stability across days (Mantel test, same-session pairs excluded)

2183 pairs (134 same day, other file, at distance 0).

| measure | Spearman ρ with days apart | p (permutation of days) | slope per day [95% CI, bootstrap on days] |
|---|---|---|---|
| index_k15 (primary test) | -0.497 | 0.0008 | -0.00302 [-0.00505, -0.00166] |
| angle_k15 | +0.601 | 0.0002 | +0.17899 [+0.11171, +0.28458] |
| index_k5 | -0.552 | 0.0004 | -0.00433 [-0.00748, -0.00257] |
| angle_k5 | +0.517 | 0.0001 | +0.26728 [+0.15686, +0.46930] |

By pair type (k = 15, descriptive):

| pair type | pairs | days apart | median index | median angle | Spearman with days |
|---|---|---|---|---|---|
| held-in–held-in | 1336 | 0–19 | 0.437 | 58.4° | -0.46 |
| held-in–held-out | 756 | 6–39 | 0.388 | 61.4° | -0.13 |
| held-out–held-out | 91 | 0–14 | 0.402 | 61.6° | -0.44 |

Same day, other file: held-in–held-in: 127 pairs, median index 0.492, angle 55.0°; held-out–held-out: 7 pairs, median index 0.520, angle 56.6°.

## Alignment with the decoder's frozen FA and zero-shot R² (Fig. F)

- Held-out blocks vs frozen FA (k = 15): median index 0.295 [0.233, 0.432], median angle 59.7°.
- Reference, held-in blocks vs the FA fit without their day: median index 0.347 [IQR 0.312, 0.381], median angle 54.0°.
- Chance 95th percentile (median over the 14): allbins 0.230, cond 0.157, unif 0.102; held-out sessions above all three: 12/14.

| test | ρ | p | significant only if abs(ρ) ≥ |
|---|---|---|---|
| index vs FA zero-shot R², sessions (n = 14) — **the test** | +0.191 | 0.5129 | 0.538 |
| same, day means (n = 7) | +0.393 | 0.3956 | 0.786 |
| same, partial on days since held-in | +0.180 | 0.5394 | — |
| sensitivity: index_k5 vs FA zero-shot | +0.200 | 0.4940 | — |
| secondary: angle_k15 vs FA zero-shot | +0.020 | 0.9475 | — |

Descriptive Spearman ρ (no test): PCA_zero_shot -0.12; GPFA_zero_shot -0.02; LFADS_zero_shot -0.13; ridge_channels_zero_shot -0.37; FA_align_minus_recenter -0.24; median_vs_heldin_blocks__FA_zero_shot +0.01; median_vs_heldin_blocks__vs_frozen +0.80.

A null result reads as "not detectable with these data" (n = 14 detects only |ρ| ≳ 0.54, n = 7 only |ρ| ≳ 0.79), not as "no relation".

Files: `blocks.csv`, `pairs.csv`, `model_pairs.csv`, `heldout_sessions.csv`, `tests.json`, `fig_E_stability_by_days_k15.png`, `fig_E_stability_by_days_k5.png`, `fig_F_alignment_vs_zero_shot.png`; registry rows with esperimento `fa_stability`.
