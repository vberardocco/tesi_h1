# Phase 2 · R9 · Intrinsic dimension estimators (as in Altan 2021)

Commit 880a3fc. Rules: docs/literature/rules_decision.md (commit 1aa7c5d), clarifications P1, P2, P6, P7. skdim 0.5.0. Duration of the block: 104 min; DANCo with 5 draws per n (reduction P7).

Values with all points; in brackets the change of the median from n_min to all points and the maximum relative IQR (criterion: both below 20%).

| Session | Estimator | (a) data | twins [min, max] | (a) stable | (a) distinguishable | (b) GPFA | (b) stable | (c) means | (c) stable | TME [min, max] | Altan prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | TwoNN | 20.1 (1062%, 6%) | [28.4, 29.5] | no | yes | 5.9 (274%, 5%) | no | 28.0 (1294%, 3%) | no | [24.2, 27.6] | does not hold |
| S1 | MLE10 | 1.7 (87%, 6%) | [1.7, 1.7] | no | no | 1.6 (71%, 3%) | no | 1.6 (28%, 1%) | no | [1.6, 1.6] | does not hold |
| S1 | MLE20 | 2.8 (82%, 3%) | [2.7, 2.7] | no | yes | 2.1 (66%, 2%) | no | 2.4 (34%, 2%) | no | [2.5, 2.7] | does not hold |
| S1 | CorrInt | 16.9 (23%, 3%) | [7.9, 8.5] | no | yes | 3.7 (57%, 1%) | no | 6.3 (49%, 8%) | no | [7.6, 9.0] | does not hold |
| S1 | DANCo | 1.9 (70%, 70%) | [1.9, 1.9] | no | yes | 1.9 (51%, 1%) | no | 1.8 (11%, 0%) | yes | [1.8, 1.8] | does not hold |
| S2 | TwoNN | 20.5 (1104%, 6%) | [29.5, 30.1] | no | yes | 8.0 (439%, 3%) | no | 28.6 (1323%, 3%) | no | [23.6, 28.5] | does not hold |
| S2 | MLE10 | 1.7 (87%, 4%) | [1.6, 1.6] | no | yes | 1.5 (70%, 3%) | no | 1.7 (28%, 1%) | no | [1.7, 1.7] | does not hold |
| S2 | MLE20 | 2.8 (82%, 2%) | [2.6, 2.7] | no | yes | 2.0 (67%, 1%) | no | 2.5 (33%, 2%) | no | [2.6, 2.7] | does not hold |
| S2 | CorrInt | 15.3 (31%, 4%) | [7.7, 8.2] | no | yes | 3.2 (63%, 2%) | no | 6.1 (54%, 3%) | no | [7.4, 9.7] | does not hold |
| S2 | DANCo | 1.8 (70%, 2%) | [1.8, 1.8] | no | yes | 1.9 (51%, 1%) | no | 1.8 (11%, 0%) | yes | [1.8, 1.8] | does not hold |
| S5 | TwoNN | 20.8 (1063%, 4%) | [28.7, 29.2] | no | yes | 7.7 (379%, 3%) | no | 26.8 (1233%, 3%) | no | [24.1, 26.4] | does not hold |
| S5 | MLE10 | 1.7 (88%, 8%) | [1.6, 1.7] | no | yes | 1.6 (72%, 6%) | no | 1.7 (26%, 1%) | no | [1.7, 1.7] | does not hold |
| S5 | MLE20 | 2.8 (83%, 3%) | [2.6, 2.7] | no | yes | 2.2 (66%, 2%) | no | 2.4 (33%, 1%) | no | [2.6, 2.8] | does not hold |
| S5 | CorrInt | 14.9 (34%, 3%) | [7.7, 8.5] | no | yes | 4.1 (50%, 3%) | no | 5.3 (53%, 4%) | no | [6.5, 7.8] | does not hold |
| S5 | DANCo | 1.8 (71%, 41%) | [1.8, 1.8] | no | yes | 1.9 (53%, 1%) | no | 1.8 (11%, 0%) | yes | [1.8, 1.8] | does not hold |

**R9**: prediction P2 not confirmed. Prediction P2, which we derived from Altan 2021, is not confirmed in any of the 15 cells (5 estimators × 3 sessions). [Text corrected on 5 Nov: Altan has no table with these cells, the grid is ours.]

Where it does not hold:

- S1, TwoNN: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S1, MLE10: (b) unstable, (c) unstable
- S1, MLE20: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S1, CorrInt: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S1, DANCo: (a) distinguishable from the twins, (b) unstable
- S2, TwoNN: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S2, MLE10: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S2, MLE20: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S2, CorrInt: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S2, DANCo: (a) distinguishable from the twins, (b) unstable
- S5, TwoNN: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S5, MLE10: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S5, MLE20: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S5, CorrInt: (a) distinguishable from the twins, (b) unstable, (c) unstable
- S5, DANCo: (a) distinguishable from the twins, (b) unstable
