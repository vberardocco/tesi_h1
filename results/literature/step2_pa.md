# Step 2 · Parallel Analysis

Commit c8291b8. 200 surrogates per null; threshold = 95th percentile of the same rank.
k_seq = first consecutive eigenvalues above threshold (main definition); k_all = all ranks above threshold.

| Session | Pipeline | Null | N | bins | k_seq | k_seq/N | k_all | k_all/N | trial duration (s) | shifts excluded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 02 | shift | 160 | 12233 | 19 | 11.9% | 19 | 11.9% | 15.64 | 25.1% |
| S1 | 02 | shuffle | 160 | 12233 | 50 | 31.2% | 50 | 31.2% | 15.64 | — |
| S1 | gallego | shift | 160 | 12179 | 11 | 6.9% | 11 | 6.9% | 15.64 | 25.1% |
| S1 | gallego | shuffle | 160 | 12179 | 18 | 11.2% | 18 | 11.2% | 15.64 | — |
| S2 | 02 | shift | 166 | 11646 | 17 | 10.2% | 17 | 10.2% | 15.02 | 26.3% |
| S2 | 02 | shuffle | 166 | 11646 | 51 | 30.7% | 51 | 30.7% | 15.02 | — |
| S2 | gallego | shift | 166 | 11553 | 9 | 5.4% | 9 | 5.4% | 15.02 | 26.3% |
| S2 | gallego | shuffle | 166 | 11553 | 11 | 6.6% | 12 | 7.2% | 15.02 | — |
| S5 | 02 | shift | 165 | 12078 | 15 | 9.1% | 15 | 9.1% | 15.66 | 25.4% |
| S5 | 02 | shuffle | 165 | 12078 | 47 | 28.5% | 47 | 28.5% | 15.66 | — |
| S5 | gallego | shift | 165 | 12032 | 11 | 6.7% | 11 | 6.7% | 15.66 | 25.4% |
| S5 | gallego | shuffle | 165 | 12032 | 12 | 7.3% | 12 | 7.3% | 15.66 | — |

Difference shuffle − circular shift (k_seq): S1 02: +31; S1 gallego: +7; S2 02: +34; S2 gallego: +2; S5 02: +32; S5 gallego: +1

**R5**: yes (k_seq ≥ 1 with the circular shift in all three sessions and in both pipelines; minimum 9). Minimum threshold: k_PA and k_PA/N are reported.
