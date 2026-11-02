# Table B — causal decoders, calib held-in → minival (FALCON evaluator, streaming)

| decoder | input | minival R² (mean of 6 days) | SD over days | held-in calib, leave-one-day-out R² |
|---|---|---|---|---|
| Ridge on channels (ridge_h50) | 176 channels, 1000 ms history | 0.448 | 0.043 | — |
| Ridge on FA latents | FA, K = 15, 170 active channels, 1000 ms history | 0.100 | 0.012 | 0.071 |

Per day and per DoF (minival):

| decoder | S0 | S1 | S2 | S3 | S4 | S5 | tx | ty | tz | rx | g1 | g2 | g3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ridge on channels (ridge_h50) | 0.521 | 0.439 | 0.461 | 0.445 | 0.449 | 0.375 | 0.298 | 0.136 | 0.317 | 0.433 | 0.547 | 0.424 | 0.367 |
| Ridge on FA latents | 0.115 | 0.101 | 0.084 | 0.107 | 0.084 | 0.110 | 0.020 | 0.006 | 0.041 | 0.166 | 0.113 | 0.065 | 0.100 |

- Both models are fit once on all 13 held-in calib files and evaluated once on minival (ridge_h50: 30/10/2026, commit b797b3e; FA: run of this script).
- Ridge on channels: history and α chosen on an 80/20 split of the calib (0.392 on the last 20%), no leave-one-day-out value.
- FA latents: the frozen FA model of the cross-session protocol (FA_d15_heldin_s0); α = 10⁶ chosen leave-one-day-out on the held-in days. Its readout was trained with the history restarting at each trial; the H1 evaluator is continual, so at evaluation the history runs through the file (as for ridge_h50).
