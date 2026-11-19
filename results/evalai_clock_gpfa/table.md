# Clock + online GPFA as a velocity decoder on FALCON H1 (minival, local test)

R² as in the official FALCON evaluator (`Held In R2 Mean`: variance-weighted R² on the eval_mask bins, per day, mean over the 6 held-in days), falcon_challenge 1.0.2 evaluator in streaming, batch 1. Nothing was submitted to EvalAI.

**API**: for H1 the evaluator is *continual*: it never calls `on_done` or `observe`. The decoder only receives `reset(dataset_tags)` at the start of each group of files and `predict(counts)` at every 20 ms bin, inter-trial bins included. No trial boundaries, so c) and d) cannot be evaluated.

**GPFA**: GPFA_d25_heldin_s0 from the cross-session protocol, reused without refit (d = 25, √counts of the 170 active channels, 13 held-in calib files, fitted on 1 s windows at the start of the movement phases). Online latents with `gpfa_online` and `default_window`: W = 100 bins = 2000 ms (default_window ceiling; 3·τ_max would be 474 bins). τ from 72 to 3157 ms (median 235 ms; 6 latents above 1 s). Orthonormalized latents, whole file = one segment.

**Ridge on the latents**: lags 0..49 (1000 ms), z-score per latent, fit on the eval_mask bins; α chosen as for ridge_h50 (80/20 per file, grid 10^0..10^6).

| Variant | minival R² | SD over days | S0 | S1 | S2 | S3 | S4 | S5 | calib 80/20 validation R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a) ridge on channels | **0.4483** | 0.0427 | 0.521 | 0.439 | 0.461 | 0.445 | 0.449 | 0.375 | 0.3922 |
| b) GPFA online | **0.2614** | 0.0300 | 0.289 | 0.274 | 0.291 | 0.262 | 0.202 | 0.250 | 0.2787 |
| c) clock + online GPFA | not evaluable | |  |  |  |  |  |  | |
| d) clock alone | not evaluable | |  |  |  |  |  |  | |

R² per degree of freedom (mean over days): a) ridge on channels: tx 0.299, ty 0.136, tz 0.317, rx 0.433, g1 0.547, g2 0.424, g3 0.367; b) GPFA online: tx 0.130, ty -0.031, tz 0.142, rx 0.316, g1 0.296, g2 0.222, g3 0.227.

**Checks**
- a) official evaluator 0.448332 (expected 0.448332); streaming − block max |diff| 5.6e-10.
- b) batch 1 0.261442, batch 4 0.261442; wrapper in streaming − block max |diff| 2.6e-17 (batch 1), 2.6e-17 (batch 4).
- Point 4: `StreamingClockGPFA` bin by bin against the batch version (`gpfa_online` + `design_matrix` + ridge) on S1_set_1 (19250108T110520), 1691 bins: max |diff| 2.0e-17 (< 1e-9, ok).
- Causal filters recomputed at wrapper start-up in 36 s.
