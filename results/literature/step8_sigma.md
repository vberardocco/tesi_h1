# Fase 2 · R8 · Parallel Analysis al variare del preprocessing

Commit 880a3fc. Rules: docs/literature/rules_decision.md (commit 1aa7c5d). Circular shift, 200 surrogates, seed 0 for each combination; k_PA = first consecutive eigenvalues above the 95th percentile.

Check: 02 with σ = 50 ms and z-score = 19 / 17 / 15, Gallego with σ = 50 ms = 11 / 9 / 11: reproduced.

| Session | 02 σ 20 ms z-score | 02 σ 20 ms no z | 02 σ 50 ms z-score | 02 σ 50 ms no z | 02 σ 100 ms z-score | 02 σ 100 ms no z | Gallego σ 20 ms | Gallego σ 50 ms | Gallego σ 100 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 (N = 160) | 21 (13.1%) | 9 (5.6%) | 19 (11.9%) | 10 (6.2%) | 17 (10.6%) | 9 (5.6%) | 11 (6.9%) | 11 (6.9%) | 10 (6.2%) |
| S2 (N = 166) | 20 (12.0%) | 5 (3.0%) | 17 (10.2%) | 4 (2.4%) | 15 (9.0%) | 3 (1.8%) | 8 (4.8%) | 9 (5.4%) | 7 (4.2%) |
| S5 (N = 165) | 16 (9.7%) | 9 (5.5%) | 15 (9.1%) | 9 (5.5%) | 14 (8.5%) | 7 (4.2%) | 11 (6.7%) | 11 (6.7%) | 10 (6.1%) |

Min-max interval over the grid, per session (k_PA; k_PA/N):

- S1: 9-21; 5.6-13.1%
- S2: 3-20; 1.8-12.0%
- S5: 7-16; 4.2-9.7%

**R8**: robust existence (k_PA ≥ 1 in 27 combinations out of 27).

## Shared fraction per channel (FA, K by cross-validation), for slide 5

As in sec. 9.5 of notebook 02. No verdict.

| Session | bins | σ (ms) | K* | share per channel, median | population share | dimensions for 95% |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | falcon | counts | 15 | 3.9% | 6.2% | 14 |
| S1 | falcon | 20 | 20 | 9.5% | 13.1% | 18 |
| S1 | falcon | 50 | 20 | 16.7% | 20.6% | 18 |
| S1 | falcon | 100 | 15 | 18.9% | 23.5% | 14 |
| S2 | falcon | counts | 10 | 2.8% | 4.7% | 9 |
| S2 | falcon | 20 | 20 | 9.2% | 13.2% | 18 |
| S2 | falcon | 50 | 20 | 18.2% | 21.1% | 18 |
| S2 | falcon | 100 | 15 | 20.8% | 24.4% | 14 |
| S5 | falcon | counts | 15 | 3.2% | 6.2% | 14 |
| S5 | falcon | 20 | 20 | 9.0% | 15.0% | 18 |
| S5 | falcon | 50 | 15 | 15.4% | 20.6% | 13 |
| S5 | falcon | 100 | 15 | 22.0% | 26.5% | 13 |
| S1 | legacy | counts | 15 | 3.8% | 6.2% | 14 |
| S1 | legacy | 50 | 20 | 16.7% | 20.6% | 18 |

Check with legacy bins (S1) against notebook 02 (3.8% and 16.7%): σ 0: 3.8% (reproduced), σ 50: 16.7% (reproduced).
