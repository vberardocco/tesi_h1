# Protocollo cross-sessione: risultati (generato da scripts/xs_analysis.py)

Registro: /Users/valerio/tesi_h1_work/results/h1_runs.csv. R² concatenato sui 3 fold, media sui semi; unità = sessione. Dimensioni principali: PCA 25, FA 15, GPFA 25, Isomap 25, UMAP 25, CEBRA 25, LFADS 25.

## Mediana [IQR] (n) per metodo × condizione

| method | zero_shot | recenter | align | fa_native | fewshot | fewshot_gain | scratch | ridge_rates |
|---|---|---|---|---|---|---|---|---|
| PCA (d = 25) | 0.079 [0.055, 0.098] (14) | 0.118 [0.083, 0.142] (14) | 0.108 [0.065, 0.158] (14) | -- | 0.108 [0.067, 0.158] (14) | 0.024 [-0.040, 0.059] (14) | -0.028 [-0.050, 0.006] (14) | -- |
| FA (d = 15) | 0.034 [0.023, 0.047] (14) | 0.042 [0.038, 0.053] (14) | 0.050 [0.043, 0.061] (14) | 0.025 [0.020, 0.032] (14) | 0.050 [0.043, 0.061] (14) | -0.014 [-0.052, -0.004] (14) | -0.018 [-0.020, -0.005] (14) | -- |
| GPFA (d = 25) | 0.030 [0.007, 0.046] (14) | 0.127 [0.110, 0.131] (14) | 0.071 [0.032, 0.097] (14) | -- | 0.072 [0.033, 0.097] (14) | -0.003 [-0.067, 0.031] (14) | -0.007 [-0.020, 0.000] (14) | -- |
| Isomap (d = 25) | 0.011 [0.004, 0.022] (14) | 0.025 [0.018, 0.029] (14) | 0.023 [0.015, 0.029] (14) | -- | 0.024 [-0.007, 0.049] (14) | -0.033 [-0.067, 0.033] (14) | -0.024 [-0.038, -0.012] (14) | -- |
| UMAP (d = 25) | -0.001 [-0.001, -0.001] (14) | -0.001 [-0.001, -0.001] (14) | -0.001 [-0.001, -0.001] (14) | -- | -0.000 [-0.002, 0.004] (14) | -0.034 [-0.042, -0.027] (14) | -0.025 [-0.037, -0.005] (14) | -- |
| CEBRA (d = 25) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -- | -0.001 [-0.003, 0.002] (14) | -0.053 [-0.056, -0.037] (14) | -0.012 [-0.023, -0.006] (14) | -- |
| LFADS (d = 25) | 0.108 [0.075, 0.134] (14) | 0.125 [0.102, 0.143] (14) | 0.108 [0.082, 0.135] (14) | -- | 0.108 [0.082, 0.135] (14) | 0.004 [-0.028, 0.049] (14) | -- | -- |
| ridge_channels | -0.073 [-0.162, 0.055] (14) | 0.146 [0.096, 0.196] (14) | -- | -- | 0.150 [0.075, 0.233] (14) | 0.088 [0.068, 0.137] (14) | -- | -- |
| rates | -- | -- | -- | -- | -- | -- | -- | -0.016 [-0.038, 0.015] (14) |

## Mappe a d = 3 (esperimento crosssession_d3)

| method | zero_shot | recenter | align | fewshot | fewshot_gain | scratch |
|---|---|---|---|---|---|---|
| Isomap (d = 3) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.001, -0.001] (14) | 0.000 [-0.001, 0.004] (14) | -0.025 [-0.041, -0.023] (14) | -0.033 [-0.046, -0.018] (14) |
| UMAP (d = 3) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.003, 0.001] (14) | -0.037 [-0.044, -0.033] (14) | -0.023 [-0.043, -0.020] (14) |
| CEBRA (d = 3) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.002 [-0.003, -0.001] (14) | -0.040 [-0.052, -0.029] (14) | -0.021 [-0.028, -0.010] (14) |

## Wilcoxon tra condizioni consecutive (Holm dentro ogni metodo)

Differenze = to − from; r_rb > 0 e n_better = sessioni in cui `to` è migliore. added_v2 = confronto aggiunto nella seconda tornata (non nel piano della mattina del 31/10).

| method | from | to | n | W | p | p_holm | r_rb | median_diff | n_better | added_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| PCA | zero_shot | recenter | 14 | 3 | 0.00061 | 0.00183 | 0.943 | 0.0228 | 13 | False |
| PCA | recenter | align | 14 | 47 | 0.761 | 1 | -0.105 | 0.000623 | 8 | False |
| PCA | align | fewshot | 14 | 48 | 0.808 | 1 | 0.0857 | 0.00018 | 8 | False |
| PCA | fewshot | scratch | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.14 | 0 | False |
| PCA | align | fewshot_gain | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.105 | 0 | True |
| FA | zero_shot | recenter | 14 | 0 | 0.000122 | 0.000732 | 1 | 0.00858 | 14 | False |
| FA | recenter | align | 14 | 8 | 0.00305 | 0.0061 | 0.848 | 0.00529 | 13 | False |
| FA | align | fewshot | 14 | 0 | nan | nan | 0 | 0 | 0 | False |
| FA | fewshot | scratch | 14 | 0 | 0.000122 | 0.000732 | -1 | -0.0631 | 0 | False |
| FA | align | fa_native | 14 | 0 | 0.000122 | 0.000732 | -1 | -0.0264 | 0 | False |
| FA | align | fewshot_gain | 14 | 0 | 0.000122 | 0.000732 | -1 | -0.0755 | 0 | True |
| GPFA | zero_shot | recenter | 14 | 0 | 0.000122 | 0.00061 | 1 | 0.0988 | 14 | False |
| GPFA | recenter | align | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0615 | 0 | False |
| GPFA | align | fewshot | 14 | 21 | 0.0494 | 0.0494 | 0.6 | 0.000565 | 11 | False |
| GPFA | fewshot | scratch | 14 | 1 | 0.000244 | 0.000732 | -0.981 | -0.0828 | 1 | False |
| GPFA | align | fewshot_gain | 14 | 2 | 0.000366 | 0.000732 | -0.962 | -0.0821 | 1 | True |
| Isomap | zero_shot | recenter | 14 | 3 | 0.00061 | 0.00305 | 0.943 | 0.00951 | 13 | False |
| Isomap | recenter | align | 14 | 37 | 0.358 | 0.715 | -0.295 | -0.00184 | 4 | False |
| Isomap | align | fewshot | 14 | 47 | 0.761 | 0.761 | -0.105 | -0.000914 | 7 | False |
| Isomap | fewshot | scratch | 14 | 9 | 0.00403 | 0.0161 | -0.829 | -0.0493 | 2 | False |
| Isomap | align | fewshot_gain | 14 | 18 | 0.0295 | 0.0886 | -0.657 | -0.0448 | 4 | True |
| UMAP | zero_shot | recenter | 14 | 44 | 0.626 | 1 | 0.162 | 6.69e-05 | 9 | False |
| UMAP | recenter | align | 14 | 42 | 0.542 | 1 | 0.2 | 7.79e-06 | 9 | False |
| UMAP | align | fewshot | 14 | 28 | 0.135 | 0.406 | 0.467 | 0.000971 | 9 | False |
| UMAP | fewshot | scratch | 14 | 2 | 0.000366 | 0.00146 | -0.962 | -0.0211 | 1 | False |
| UMAP | align | fewshot_gain | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0334 | 0 | True |
| CEBRA | zero_shot | recenter | 14 | 23 | 0.0676 | 0.148 | 0.562 | 2.32e-05 | 10 | False |
| CEBRA | recenter | align | 14 | 0 | 0.000122 | 0.00061 | -1 | -7.33e-05 | 0 | False |
| CEBRA | align | fewshot | 14 | 50 | 0.903 | 0.903 | 0.0476 | 0.000368 | 7 | False |
| CEBRA | fewshot | scratch | 14 | 21 | 0.0494 | 0.148 | -0.6 | -0.0127 | 3 | False |
| CEBRA | align | fewshot_gain | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0514 | 0 | True |
| LFADS | zero_shot | recenter | 14 | 16 | 0.0203 | 0.0608 | 0.695 | 0.00855 | 9 | False |
| LFADS | recenter | align | 14 | 19 | 0.0353 | 0.0706 | -0.638 | -0.0128 | 5 | False |
| LFADS | align | fewshot | 14 | 0 | nan | nan | 0 | 0 | 0 | False |
| LFADS | align | fewshot_gain | 14 | 0 | 0.000122 | 0.000488 | -1 | -0.0821 | 0 | True |
| ridge_channels | zero_shot | recenter | 14 | 0 | 0.000122 | 0.000366 | 1 | 0.183 | 14 | True |
| ridge_channels | recenter | fewshot | 14 | 43 | 0.583 | 0.583 | 0.181 | 0.00671 | 9 | True |
| ridge_channels | recenter | fewshot_gain | 14 | 5 | 0.00122 | 0.00244 | -0.905 | -0.0504 | 1 | True |

Mappe a d = 3:

| method | from | to | n | W | p | p_holm | r_rb | median_diff | n_better |
|---|---|---|---|---|---|---|---|---|---|
| Isomap | zero_shot | recenter | 14 | 20 | 0.0419 | 0.0419 | 0.619 | 2.39e-05 | 10 |
| Isomap | recenter | align | 14 | 7 | 0.00232 | 0.00696 | 0.867 | 1.2e-05 | 11 |
| Isomap | align | fewshot | 14 | 11 | 0.00671 | 0.0134 | 0.79 | 0.00197 | 11 |
| Isomap | fewshot | scratch | 14 | 1 | 0.000244 | 0.00122 | -0.981 | -0.0348 | 1 |
| Isomap | align | fewshot_gain | 14 | 1 | 0.000244 | 0.00122 | -0.981 | -0.0243 | 1 |
| UMAP | zero_shot | recenter | 14 | 37 | 0.358 | 0.715 | 0.295 | -1.21e-06 | 6 |
| UMAP | recenter | align | 14 | 4 | 0.000854 | 0.00256 | -0.924 | -3.62e-06 | 2 |
| UMAP | align | fewshot | 14 | 50 | 0.903 | 0.903 | 0.0476 | 7.29e-05 | 7 |
| UMAP | fewshot | scratch | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0222 | 0 |
| UMAP | align | fewshot_gain | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0358 | 0 |
| CEBRA | zero_shot | recenter | 14 | 52 | 1 | 1 | 0.00952 | -8.15e-07 | 6 |
| CEBRA | recenter | align | 14 | 21.5 | 0.0516 | 0.155 | -0.59 | -8.94e-07 | 4 |
| CEBRA | align | fewshot | 14 | 35 | 0.296 | 0.592 | -0.333 | -0.00103 | 5 |
| CEBRA | fewshot | scratch | 14 | 1 | 0.000244 | 0.000977 | -0.981 | -0.0197 | 1 |
| CEBRA | align | fewshot_gain | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0387 | 0 |

## Friedman tra metodi

| condition | methods | n | k | chi2 | p | kendall_W |
|---|---|---|---|---|---|---|
| align | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS | 14 | 7 | 68.8 | 7.15e-13 | 0.819 |
| fewshot | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS,ridge_channels | 14 | 8 | 68 | 3.66e-12 | 0.694 |
| fewshot_gain | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS,ridge_channels | 14 | 8 | 42.5 | 4.21e-07 | 0.433 |

Con le mappe a d = 3:

| condition | methods | n | k | chi2 | p | kendall_W |
|---|---|---|---|---|---|---|
| align | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS | 14 | 7 | 66.1 | 2.58e-12 | 0.787 |
| fewshot | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS,ridge_channels | 14 | 8 | 72.7 | 4.26e-13 | 0.741 |
| fewshot_gain | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS,ridge_channels | 14 | 8 | 42.8 | 3.63e-07 | 0.437 |

## Wilcoxon tra metodi a coppie (Holm dentro ogni condizione)

Differenze = method_b − method_a; r_rb > 0 e n_better = sessioni in cui method_b è migliore.

| condition | method_a | method_b | n | W | p | p_holm | r_rb | median_diff | n_better |
|---|---|---|---|---|---|---|---|---|---|
| align | PCA | FA | 14 | 3 | 0.00061 | 0.00488 | -0.943 | -0.0635 | 2 |
| align | PCA | GPFA | 14 | 6 | 0.00171 | 0.00854 | -0.886 | -0.0439 | 2 |
| align | PCA | Isomap | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0844 | 0 |
| align | PCA | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.109 | 0 |
| align | PCA | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.109 | 0 |
| align | PCA | LFADS | 14 | 49 | 0.855 | 0.926 | 0.0667 | -0.0156 | 5 |
| align | FA | GPFA | 14 | 40 | 0.463 | 0.926 | 0.238 | 0.00715 | 8 |
| align | FA | Isomap | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0275 | 0 |
| align | FA | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0509 | 0 |
| align | FA | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0512 | 0 |
| align | FA | LFADS | 14 | 1 | 0.000244 | 0.00256 | 0.981 | 0.0582 | 13 |
| align | GPFA | Isomap | 14 | 11 | 0.00671 | 0.0201 | -0.79 | -0.0427 | 3 |
| align | GPFA | UMAP | 14 | 3 | 0.00061 | 0.00488 | -0.943 | -0.0726 | 2 |
| align | GPFA | CEBRA | 14 | 3 | 0.00061 | 0.00488 | -0.943 | -0.0729 | 2 |
| align | GPFA | LFADS | 14 | 1 | 0.000244 | 0.00256 | 0.981 | 0.0449 | 13 |
| align | Isomap | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0235 | 0 |
| align | Isomap | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0238 | 0 |
| align | Isomap | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.092 | 14 |
| align | UMAP | CEBRA | 14 | 6 | 0.00171 | 0.00854 | -0.886 | -0.000218 | 2 |
| align | UMAP | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.11 | 14 |
| align | CEBRA | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.11 | 14 |
| fewshot | PCA | FA | 14 | 3 | 0.00061 | 0.011 | -0.943 | -0.0637 | 2 |
| fewshot | PCA | GPFA | 14 | 7 | 0.00232 | 0.0278 | -0.867 | -0.0444 | 2 |
| fewshot | PCA | Isomap | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.0878 | 0 |
| fewshot | PCA | UMAP | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.109 | 0 |
| fewshot | PCA | CEBRA | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.109 | 0 |
| fewshot | PCA | LFADS | 14 | 48 | 0.808 | 0.853 | 0.0857 | -0.0161 | 5 |
| fewshot | PCA | ridge_channels | 14 | 20 | 0.0419 | 0.293 | 0.619 | 0.0571 | 11 |
| fewshot | FA | GPFA | 14 | 39 | 0.426 | 0.853 | 0.257 | 0.00749 | 8 |
| fewshot | FA | Isomap | 14 | 12 | 0.00854 | 0.0854 | -0.771 | -0.0312 | 4 |
| fewshot | FA | UMAP | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.0509 | 0 |
| fewshot | FA | CEBRA | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.0538 | 0 |
| fewshot | FA | LFADS | 14 | 1 | 0.000244 | 0.00488 | 0.981 | 0.0582 | 13 |
| fewshot | FA | ridge_channels | 14 | 10 | 0.00525 | 0.0577 | 0.81 | 0.109 | 12 |
| fewshot | GPFA | Isomap | 14 | 14 | 0.0134 | 0.121 | -0.733 | -0.0512 | 4 |
| fewshot | GPFA | UMAP | 14 | 3 | 0.00061 | 0.011 | -0.943 | -0.073 | 2 |
| fewshot | GPFA | CEBRA | 14 | 3 | 0.00061 | 0.011 | -0.943 | -0.0733 | 2 |
| fewshot | GPFA | LFADS | 14 | 1 | 0.000244 | 0.00488 | 0.981 | 0.0446 | 13 |
| fewshot | GPFA | ridge_channels | 14 | 14 | 0.0134 | 0.121 | 0.733 | 0.0937 | 12 |
| fewshot | Isomap | UMAP | 14 | 27 | 0.119 | 0.543 | -0.486 | -0.0224 | 5 |
| fewshot | Isomap | CEBRA | 14 | 26 | 0.104 | 0.543 | -0.505 | -0.0268 | 4 |
| fewshot | Isomap | LFADS | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.0976 | 14 |
| fewshot | Isomap | ridge_channels | 14 | 3 | 0.00061 | 0.011 | 0.943 | 0.135 | 13 |
| fewshot | UMAP | CEBRA | 14 | 31 | 0.194 | 0.581 | -0.41 | -0.000934 | 4 |
| fewshot | UMAP | LFADS | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.109 | 14 |
| fewshot | UMAP | ridge_channels | 14 | 3 | 0.00061 | 0.011 | 0.943 | 0.147 | 13 |
| fewshot | CEBRA | LFADS | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.106 | 14 |
| fewshot | CEBRA | ridge_channels | 14 | 3 | 0.00061 | 0.011 | 0.943 | 0.148 | 13 |
| fewshot | LFADS | ridge_channels | 14 | 25 | 0.0906 | 0.543 | 0.524 | 0.0458 | 11 |
| fewshot_gain | PCA | FA | 14 | 33 | 0.241 | 1 | -0.371 | -0.0588 | 4 |
| fewshot_gain | PCA | GPFA | 14 | 30 | 0.173 | 1 | -0.429 | -0.0115 | 4 |
| fewshot_gain | PCA | Isomap | 14 | 30 | 0.173 | 1 | -0.429 | -0.0227 | 4 |
| fewshot_gain | PCA | UMAP | 14 | 26 | 0.104 | 1 | -0.505 | -0.0647 | 4 |
| fewshot_gain | PCA | CEBRA | 14 | 20 | 0.0419 | 0.712 | -0.619 | -0.0873 | 3 |
| fewshot_gain | PCA | LFADS | 14 | 49 | 0.855 | 1 | -0.0667 | 0.00737 | 8 |
| fewshot_gain | PCA | ridge_channels | 14 | 1 | 0.000244 | 0.00586 | 0.981 | 0.0847 | 13 |
| fewshot_gain | FA | GPFA | 14 | 44 | 0.626 | 1 | 0.162 | 0.0187 | 8 |
| fewshot_gain | FA | Isomap | 14 | 47 | 0.761 | 1 | 0.105 | 0.012 | 8 |
| fewshot_gain | FA | UMAP | 14 | 42 | 0.542 | 1 | -0.2 | -0.0112 | 5 |
| fewshot_gain | FA | CEBRA | 14 | 15 | 0.0166 | 0.315 | -0.714 | -0.0335 | 3 |
| fewshot_gain | FA | LFADS | 14 | 21 | 0.0494 | 0.791 | 0.6 | 0.0356 | 11 |
| fewshot_gain | FA | ridge_channels | 14 | 1 | 0.000244 | 0.00586 | 0.981 | 0.131 | 13 |
| fewshot_gain | GPFA | Isomap | 14 | 48 | 0.808 | 1 | -0.0857 | -0.00189 | 6 |
| fewshot_gain | GPFA | UMAP | 14 | 39 | 0.426 | 1 | -0.257 | -0.0343 | 5 |
| fewshot_gain | GPFA | CEBRA | 14 | 22 | 0.058 | 0.87 | -0.581 | -0.0606 | 4 |
| fewshot_gain | GPFA | LFADS | 14 | 29 | 0.153 | 1 | 0.448 | 0.014 | 10 |
| fewshot_gain | GPFA | ridge_channels | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.0972 | 14 |
| fewshot_gain | Isomap | UMAP | 14 | 45 | 0.67 | 1 | -0.143 | -0.00349 | 6 |
| fewshot_gain | Isomap | CEBRA | 14 | 37 | 0.358 | 1 | -0.295 | -0.022 | 6 |
| fewshot_gain | Isomap | LFADS | 14 | 26 | 0.104 | 1 | 0.505 | 0.0302 | 9 |
| fewshot_gain | Isomap | ridge_channels | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.126 | 14 |
| fewshot_gain | UMAP | CEBRA | 14 | 10 | 0.00525 | 0.105 | -0.81 | -0.0168 | 4 |
| fewshot_gain | UMAP | LFADS | 14 | 19 | 0.0353 | 0.635 | 0.638 | 0.0276 | 11 |
| fewshot_gain | UMAP | ridge_channels | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.126 | 14 |
| fewshot_gain | CEBRA | LFADS | 14 | 8 | 0.00305 | 0.0641 | 0.848 | 0.0601 | 12 |
| fewshot_gain | CEBRA | ridge_channels | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.157 | 14 |
| fewshot_gain | LFADS | ridge_channels | 14 | 2 | 0.000366 | 0.00806 | 0.962 | 0.098 | 13 |

Con le mappe a d = 3:

| condition | method_a | method_b | n | W | p | p_holm | r_rb | median_diff | n_better |
|---|---|---|---|---|---|---|---|---|---|
| align | PCA | FA | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0635 | 2 |
| align | PCA | GPFA | 14 | 6 | 0.00171 | 0.00854 | -0.886 | -0.0439 | 2 |
| align | PCA | Isomap | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.109 | 0 |
| align | PCA | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.109 | 0 |
| align | PCA | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.109 | 0 |
| align | PCA | LFADS | 14 | 49 | 0.855 | 1 | 0.0667 | -0.0156 | 5 |
| align | FA | GPFA | 14 | 40 | 0.463 | 1 | 0.238 | 0.00715 | 8 |
| align | FA | Isomap | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0512 | 0 |
| align | FA | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0512 | 0 |
| align | FA | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0512 | 0 |
| align | FA | LFADS | 14 | 1 | 0.000244 | 0.00293 | 0.981 | 0.0582 | 13 |
| align | GPFA | Isomap | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0728 | 2 |
| align | GPFA | UMAP | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0728 | 2 |
| align | GPFA | CEBRA | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0728 | 2 |
| align | GPFA | LFADS | 14 | 1 | 0.000244 | 0.00293 | 0.981 | 0.0449 | 13 |
| align | Isomap | UMAP | 14 | 5 | 0.00122 | 0.00732 | -0.905 | -6.62e-05 | 2 |
| align | Isomap | CEBRA | 14 | 6 | 0.00171 | 0.00854 | -0.886 | -6.97e-05 | 3 |
| align | Isomap | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.11 | 14 |
| align | UMAP | CEBRA | 14 | 37 | 0.358 | 1 | 0.295 | 4.99e-06 | 9 |
| align | UMAP | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.11 | 14 |
| align | CEBRA | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.11 | 14 |
| fewshot | PCA | FA | 14 | 3 | 0.00061 | 0.0104 | -0.943 | -0.0637 | 2 |
| fewshot | PCA | GPFA | 14 | 7 | 0.00232 | 0.0232 | -0.867 | -0.0444 | 2 |
| fewshot | PCA | Isomap | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.111 | 0 |
| fewshot | PCA | UMAP | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.111 | 0 |
| fewshot | PCA | CEBRA | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.109 | 0 |
| fewshot | PCA | LFADS | 14 | 48 | 0.808 | 1 | 0.0857 | -0.0161 | 5 |
| fewshot | PCA | ridge_channels | 14 | 20 | 0.0419 | 0.209 | 0.619 | 0.0571 | 11 |
| fewshot | FA | GPFA | 14 | 39 | 0.426 | 1 | 0.257 | 0.00749 | 8 |
| fewshot | FA | Isomap | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.0498 | 0 |
| fewshot | FA | UMAP | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.0522 | 0 |
| fewshot | FA | CEBRA | 14 | 0 | 0.000122 | 0.00342 | -1 | -0.0519 | 0 |
| fewshot | FA | LFADS | 14 | 1 | 0.000244 | 0.00464 | 0.981 | 0.0582 | 13 |
| fewshot | FA | ridge_channels | 14 | 10 | 0.00525 | 0.042 | 0.81 | 0.109 | 12 |
| fewshot | GPFA | Isomap | 14 | 3 | 0.00061 | 0.0104 | -0.943 | -0.0717 | 2 |
| fewshot | GPFA | UMAP | 14 | 3 | 0.00061 | 0.0104 | -0.943 | -0.0748 | 2 |
| fewshot | GPFA | CEBRA | 14 | 3 | 0.00061 | 0.0104 | -0.943 | -0.0747 | 2 |
| fewshot | GPFA | LFADS | 14 | 1 | 0.000244 | 0.00464 | 0.981 | 0.0446 | 13 |
| fewshot | GPFA | ridge_channels | 14 | 14 | 0.0134 | 0.0806 | 0.733 | 0.0937 | 12 |
| fewshot | Isomap | UMAP | 14 | 12 | 0.00854 | 0.0598 | -0.771 | -0.00312 | 2 |
| fewshot | Isomap | CEBRA | 14 | 9 | 0.00403 | 0.0363 | -0.829 | -0.0028 | 2 |
| fewshot | Isomap | LFADS | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.107 | 14 |
| fewshot | Isomap | ridge_channels | 14 | 4 | 0.000854 | 0.0104 | 0.924 | 0.147 | 13 |
| fewshot | UMAP | CEBRA | 14 | 48 | 0.808 | 1 | -0.0857 | -0.000155 | 7 |
| fewshot | UMAP | LFADS | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.11 | 14 |
| fewshot | UMAP | ridge_channels | 14 | 3 | 0.00061 | 0.0104 | 0.943 | 0.148 | 13 |
| fewshot | CEBRA | LFADS | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.109 | 14 |
| fewshot | CEBRA | ridge_channels | 14 | 3 | 0.00061 | 0.0104 | 0.943 | 0.15 | 13 |
| fewshot | LFADS | ridge_channels | 14 | 25 | 0.0906 | 0.362 | 0.524 | 0.0458 | 11 |
| fewshot_gain | PCA | FA | 14 | 33 | 0.241 | 1 | -0.371 | -0.0588 | 4 |
| fewshot_gain | PCA | GPFA | 14 | 30 | 0.173 | 1 | -0.429 | -0.0115 | 4 |
| fewshot_gain | PCA | Isomap | 14 | 31 | 0.194 | 1 | -0.41 | -0.0669 | 4 |
| fewshot_gain | PCA | UMAP | 14 | 24 | 0.0785 | 1 | -0.543 | -0.0769 | 4 |
| fewshot_gain | PCA | CEBRA | 14 | 23 | 0.0676 | 1 | -0.562 | -0.0734 | 3 |
| fewshot_gain | PCA | LFADS | 14 | 49 | 0.855 | 1 | -0.0667 | 0.00737 | 8 |
| fewshot_gain | PCA | ridge_channels | 14 | 1 | 0.000244 | 0.00635 | 0.981 | 0.0847 | 13 |
| fewshot_gain | FA | GPFA | 14 | 44 | 0.626 | 1 | 0.162 | 0.0187 | 8 |
| fewshot_gain | FA | Isomap | 14 | 51 | 0.952 | 1 | -0.0286 | -0.0113 | 6 |
| fewshot_gain | FA | UMAP | 14 | 27 | 0.119 | 1 | -0.486 | -0.0155 | 4 |
| fewshot_gain | FA | CEBRA | 14 | 22 | 0.058 | 0.928 | -0.581 | -0.0243 | 3 |
| fewshot_gain | FA | LFADS | 14 | 21 | 0.0494 | 0.89 | 0.6 | 0.0356 | 11 |
| fewshot_gain | FA | ridge_channels | 14 | 1 | 0.000244 | 0.00635 | 0.981 | 0.131 | 13 |
| fewshot_gain | GPFA | Isomap | 14 | 38 | 0.391 | 1 | -0.276 | -0.0121 | 5 |
| fewshot_gain | GPFA | UMAP | 14 | 32 | 0.217 | 1 | -0.39 | -0.0343 | 4 |
| fewshot_gain | GPFA | CEBRA | 14 | 27 | 0.119 | 1 | -0.486 | -0.0454 | 4 |
| fewshot_gain | GPFA | LFADS | 14 | 29 | 0.153 | 1 | 0.448 | 0.014 | 10 |
| fewshot_gain | GPFA | ridge_channels | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.0972 | 14 |
| fewshot_gain | Isomap | UMAP | 14 | 26 | 0.104 | 1 | -0.505 | -0.0112 | 5 |
| fewshot_gain | Isomap | CEBRA | 14 | 21 | 0.0494 | 0.89 | -0.6 | -0.0149 | 3 |
| fewshot_gain | Isomap | LFADS | 14 | 19 | 0.0353 | 0.67 | 0.638 | 0.0416 | 11 |
| fewshot_gain | Isomap | ridge_channels | 14 | 1 | 0.000244 | 0.00635 | 0.981 | 0.116 | 13 |
| fewshot_gain | UMAP | CEBRA | 14 | 37 | 0.358 | 1 | -0.295 | -0.003 | 5 |
| fewshot_gain | UMAP | LFADS | 14 | 12 | 0.00854 | 0.171 | 0.771 | 0.0341 | 11 |
| fewshot_gain | UMAP | ridge_channels | 14 | 1 | 0.000244 | 0.00635 | 0.981 | 0.138 | 13 |
| fewshot_gain | CEBRA | LFADS | 14 | 9 | 0.00403 | 0.0846 | 0.829 | 0.0449 | 12 |
| fewshot_gain | CEBRA | ridge_channels | 14 | 0 | 0.000122 | 0.00342 | 1 | 0.14 | 14 |
| fewshot_gain | LFADS | ridge_channels | 14 | 2 | 0.000366 | 0.00806 | 0.962 | 0.098 | 13 |

## Zero-shot contro i giorni (descrittivo)

| method | condition | n | slope_per_day | ci_low | ci_high | spearman_rho | spearman_p | r2_first_day | r2_last_day |
|---|---|---|---|---|---|---|---|---|---|
| PCA | zero_shot | 14 | -0.0041 | -0.00801 | -0.000201 | -0.532 | 0.0504 | 0.125 | 0.0321 |
| FA | zero_shot | 14 | -4.29e-05 | -0.002 | 0.00191 | -0.0709 | 0.81 | 0.0438 | 0.0437 |
| GPFA | zero_shot | 14 | 0.00309 | -0.00349 | 0.00968 | 0.292 | 0.31 | -0.0427 | 0.0263 |
| Isomap | zero_shot | 14 | 0.000527 | -0.00126 | 0.00231 | 0.0886 | 0.763 | 0.0154 | 0.00563 |
| UMAP | zero_shot | 14 | 9.65e-06 | -3.95e-05 | 5.88e-05 | 0.0975 | 0.74 | -0.000895 | -0.00095 |
| CEBRA | zero_shot | 14 | 1.64e-05 | -2.67e-05 | 5.94e-05 | 0.23 | 0.428 | -0.00121 | -0.00101 |
| LFADS | zero_shot | 14 | -0.0041 | -0.00887 | 0.000677 | -0.399 | 0.158 | 0.144 | 0.0278 |
| ridge_channels | zero_shot | 14 | 0.00315 | -0.0138 | 0.0202 | 0.115 | 0.695 | -0.209 | -0.222 |
| rates | ridge_rates | 14 | 9.02e-05 | -0.00615 | 0.00633 | 0.0177 | 0.952 | 0.0468 | 0.00604 |

## Pseudo-few-shot sulle held-in (scelte congelate) e curve R²(λ)

Ogni pseudo-sessione (S4, S5) è esclusa dalla stima di metodo e readout, che usano solo S0-S3. lam_interior = ottimo dentro la griglia (10^-2 - 10^15); fewshot_minus_asymptote = R²(λ scelto) − R²(λ -> ∞).

| method | dim | lodo_r2 | pseudo_zero_shot | pseudo_recenter | pseudo_align | pseudo_fewshot_gain | pseudo_fewshot | lam | lam_interior | fewshot_minus_asymptote | pseudo_scratch | pseudo_ridge_rates | alpha_scratch | alpha_at_edge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | 25 | 0.1936 | 0.09944 | 0.1886 | 0.2129 | 0.1251 | 0.2129 | 3.162e+06 | True | 2.506e-05 | 0.02248 | nan | 3.162e+04 | False |
| FA | 15 | 0.0746 | 0.07596 | 0.09468 | 0.1001 | 0.053 | 0.1001 | 1e+15 | False | -1.741e-13 | 0.00913 | nan | 3.162e+04 | False |
| GPFA | 25 | 0.189 | 0.1076 | 0.2052 | 0.2063 | 0.05901 | 0.2064 | 1e+07 | True | 4.67e-05 | 0.01135 | nan | 1e+05 | False |
| Isomap | 25 | -0.0005563 | 0.004688 | 0.005645 | 0.003122 | 0.02992 | 0.05115 | 3.162e+04 | True | 0.04803 | 0.0249 | nan | 3.162e+04 | False |
| UMAP | 25 | -0.0004053 | -0.001364 | -0.001219 | -0.0009585 | -0.01718 | 0.001779 | 3.162e+05 | True | 0.002737 | 0.004359 | nan | 1e+05 | False |
| CEBRA | 25 | 0.0001037 | -0.0009004 | 2.515e-05 | -0.0007062 | -0.04174 | 0.001791 | 1e+06 | True | 0.002498 | -0.00145 | nan | 3.162e+05 | False |
| LFADS | 25 | 0.1736 | 0.1673 | 0.1727 | 0.2068 | 0.06788 | 0.2068 | 1e+15 | False | -8.234e-13 | nan | nan | nan | None |
| Isomap | 3 | -8.172e-05 | -0.001879 | -0.00188 | -0.001879 | -0.03666 | -0.0001717 | 1e+05 | True | 0.001707 | -0.007393 | nan | 3.162e+04 | False |
| UMAP | 3 | -5.275e-05 | -0.001824 | -0.001829 | -0.001824 | -0.03524 | 6.748e-05 | 1e+05 | True | 0.001891 | -0.007162 | nan | 3.162e+04 | False |
| CEBRA | 3 | -6.848e-05 | -0.001913 | -0.001907 | -0.001909 | -0.05303 | -0.001173 | 3.162e+05 | True | 0.000736 | -0.0171 | nan | 1e+05 | False |
| ridge_channels | nan | nan | 0.02845 | 0.2743 | nan | 0.2118 | 0.2827 | 1e+05 | True | 0.008345 | nan | nan | nan | None |
| rates | nan | nan | nan | nan | nan | nan | nan | nan | None | nan | nan | 0.03832 | 1e+05 | False |

## Bin dei 2 trial di adattamento (canali attivi: 170)

Tutti i bin: min 1442, mediana 1647, max 1933; con eval_mask: min 1390, mediana 1611, max 1876 (42 fold).

| method | latent_dim | readout_weights_per_dof | bins_mask_min | ratio_min |
|---|---|---|---|---|
| PCA | 25 | 1251 | 1390 | 1.11 |
| FA | 15 | 751 | 1390 | 1.85 |
| GPFA | 25 | 1251 | 1390 | 1.11 |
| Isomap | 25 | 1251 | 1390 | 1.11 |
| UMAP | 25 | 1251 | 1390 | 1.11 |
| CEBRA | 25 | 1251 | 1390 | 1.11 |
| LFADS | 25 | 1251 | 1390 | 1.11 |
| Isomap/UMAP/CEBRA (d = 3) | 3 | 151 | 1390 | 9.21 |
| rates | 170 | 8501 | 1390 | 0.164 |
| ridge_channels | 176 | 8801 | 1390 | 0.158 |

## Copertura del comportamento nei 2 trial di adattamento

Condizioni = fase di movimento (8) × versione del task (dispari/pari: g3 in Grasp di segno opposto); direzioni = DoF × verso (14) con almeno 10 bin oltre il 75° percentile di |v| del pool held-in; var_ratio = varianza della velocità nei 2 trial / varianza nel pool (mediana sui DoF). versions_1 = fold con i 2 trial della stessa versione.

| set | folds | versions_1 | conditions_median | conditions_min | directions_median | directions_min | var_ratio_median |
|---|---|---|---|---|---|---|---|
| pool held-in | 1 | 0 | 16 | 16 | 14 | 14 | 1 |
| held-out | 42 | 14 | 16 | 8 | 14 | 13 | 0.886 |
| pseudo (S4, S5) | 60 | 20 | 16 | 8 | 14 | 13 | 0.904 |
| held-in, diagnosi | 162 | 54 | 16 | 8 | 14 | 13 | 0.905 |

Δ = R²(align) − R²(recenter) sul trial di test, per numero di versioni nei 2 trial (esplorativo):

| method | where | n_versions | folds | delta_median | delta_mean | n_better |
|---|---|---|---|---|---|---|
| PCA | held-out | 1 | 14 | 0.024 | 0.0143 | 10 |
| PCA | held-out | 2 | 28 | -0.00502 | -0.025 | 13 |
| FA | held-out | 1 | 14 | 0.00794 | 0.0123 | 13 |
| FA | held-out | 2 | 28 | 0.00348 | 0.00312 | 18 |
| GPFA | held-out | 1 | 14 | -0.0661 | -0.0563 | 3 |
| GPFA | held-out | 2 | 28 | -0.0492 | -0.075 | 6 |
| Isomap | held-out | 1 | 14 | -0.00602 | -0.00327 | 5 |
| Isomap | held-out | 2 | 28 | -0.000617 | 0.000784 | 13 |
| UMAP | held-out | 1 | 14 | 9.18e-05 | 0.000146 | 10 |
| UMAP | held-out | 2 | 28 | -1.67e-05 | -4.95e-05 | 11 |
| CEBRA | held-out | 1 | 14 | -4.08e-05 | -5.75e-05 | 1 |
| CEBRA | held-out | 2 | 28 | -8.7e-05 | -6.92e-05 | 5 |
| LFADS | held-out | 1 | 14 | 0.00308 | 0.00264 | 7 |
| LFADS | held-out | 2 | 28 | -0.0224 | -0.0339 | 9 |
| PCA | held-in, diagnosi | 1 | 54 | -0.00258 | -0.0147 | 24 |
| PCA | held-in, diagnosi | 2 | 108 | -0.0127 | -0.0161 | 34 |
| FA | held-in, diagnosi | 1 | 54 | 0.00219 | 0.00223 | 36 |
| FA | held-in, diagnosi | 2 | 108 | 0.00024 | 0.000303 | 57 |
| GPFA | held-in, diagnosi | 1 | 54 | -0.0628 | -0.0823 | 11 |
| GPFA | held-in, diagnosi | 2 | 108 | -0.0342 | -0.0567 | 17 |
| Isomap | held-in, diagnosi | 1 | 54 | 0.00706 | 0.0121 | 47 |
| Isomap | held-in, diagnosi | 2 | 108 | 0.00129 | -0.00474 | 58 |
| UMAP | held-in, diagnosi | 1 | 54 | 0.00176 | -0.00609 | 35 |
| UMAP | held-in, diagnosi | 2 | 108 | 6.25e-05 | -0.0546 | 56 |
| CEBRA | held-in, diagnosi | 1 | 54 | 0.00176 | 0.00259 | 49 |
| CEBRA | held-in, diagnosi | 2 | 108 | 0.000878 | 0.00104 | 81 |
| LFADS | held-in, diagnosi | 1 | 54 | -0.0346 | -0.0446 | 16 |
| LFADS | held-in, diagnosi | 2 | 108 | -0.0182 | -0.0281 | 31 |

## Diagnosi del moment matching (esplorativa, aggiunta nella seconda tornata)

Dentro ogni sessione held-in: sorgente = gli altri trial della stessa sessione, blocchi di 3 trial con la stessa rotazione delle held-out (nessuna deriva tra giorni). Wilcoxon align vs recenter; Holm sui 7 metodi (solo dentro la sessione).

| method | where | n | no_adapt | recenter | align | delta_median | n_better | W | p | p_holm | r_rb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | held-in, within session | 13 | 0.229 | 0.232 | 0.218 | -0.0112 | 0 | 0 | 0.000244 | 0.00171 | -1 |
| PCA | held-out (6-20 days) | 14 | nan | 0.118 | 0.108 | 0.000623 | 8 | 47 | 0.761 | nan | -0.105 |
| FA | held-in, within session | 13 | 0.0944 | 0.0978 | 0.0997 | 0.000642 | 10 | 20 | 0.0803 | 0.0803 | 0.56 |
| FA | held-out (6-20 days) | 14 | nan | 0.0421 | 0.0495 | 0.00529 | 13 | 8 | 0.00305 | nan | 0.848 |
| GPFA | held-in, within session | 13 | 0.236 | 0.234 | 0.164 | -0.0649 | 0 | 0 | 0.000244 | 0.00171 | -1 |
| GPFA | held-out (6-20 days) | 14 | nan | 0.127 | 0.0714 | -0.0615 | 0 | 0 | 0.000122 | nan | -1 |
| Isomap | held-in, within session | 13 | 0.0918 | 0.0912 | 0.0947 | 0.00387 | 11 | 15 | 0.0327 | 0.0654 | 0.67 |
| Isomap | held-out (6-20 days) | 14 | nan | 0.0248 | 0.0228 | -0.00184 | 4 | 37 | 0.358 | nan | -0.295 |
| UMAP | held-in, within session | 13 | 0.0184 | 0.0185 | 0.00782 | -0.00512 | 2 | 10 | 0.0105 | 0.0315 | -0.78 |
| UMAP | held-out (6-20 days) | 14 | nan | -0.00106 | -0.000942 | 7.79e-06 | 9 | 42 | 0.542 | nan | 0.2 |
| CEBRA | held-in, within session | 13 | 0.0111 | 0.0112 | 0.0123 | 0.00156 | 13 | 0 | 0.000244 | 0.00171 | 1 |
| CEBRA | held-out (6-20 days) | 14 | nan | -0.00115 | -0.0012 | -7.33e-05 | 0 | 0 | 0.000122 | nan | -1 |
| LFADS | held-in, within session | 13 | 0.226 | 0.229 | 0.194 | -0.0371 | 1 | 1 | 0.000488 | 0.00195 | -0.978 |
| LFADS | held-out (6-20 days) | 14 | nan | 0.125 | 0.108 | -0.0128 | 5 | 19 | 0.0353 | nan | -0.638 |

## Verifica sui latenti a 3 dimensioni dentro una sessione (31/10, esplorativa)

Diagnostic (exploratory, not pre-registered), 2026-10-31 14:20: is the near-zero held-in readout of the 3-D maps a bug?
Within one held-in session: readout (H = 50 bins, ridge) trained on trials 1-12, tested on trials 13-15 (eval_mask),
latents from the held-in models (Isomap refit with the same settings); alpha = best on the test trials (optimistic upper bound).
session,latents,dim,r2_within_session,alpha
S5_set_1,PCA,25,0.262,1e4
S5_set_1,GPFA,25,0.302,1e4
S5_set_1,UMAP,3,0.036,1e4
S5_set_1,Isomap,3,0.023,1e4
S5_set_1,PCA first 3 PCs,3,0.010,1e5
S1_set_1,PCA,25,0.219,1e4
S1_set_1,GPFA,25,0.221,1e5
S1_set_1,UMAP,3,0.071,1e-1
S1_set_1,Isomap,3,0.044,1e3
S1_set_1,PCA first 3 PCs,3,0.019,1e5

## Controlli esplorativi dopo la seconda tornata (scripts/xs_checks.py; seme 0, dimensioni principali)

**Da dove viene il calo di fewshot_gain.** Mediana sulle 14 held-out dell'R² concatenato. base = align (ridge_channels: recenter); gain_offset = fewshot_gain; gain_only = solo guadagno per DoF (offset = intercetta held-in); offset_only = solo offset; oracle_test = guadagno e offset stimati sul trial di test stesso (limite superiore, non una condizione). gain_offset_worse = sessioni in cui fewshot_gain è sotto base.

| method | n | base | gain_offset | gain_only | offset_only | oracle_test | gain_offset_worse |
|---|---|---|---|---|---|---|---|
| PCA | 14 | 0.108 | 0.0245 | 0.0453 | 0.102 | 0.189 | 14 |
| FA | 14 | 0.0495 | -0.0143 | -0.00146 | 0.034 | 0.111 | 14 |
| GPFA | 14 | 0.0714 | -0.00297 | 0.0103 | 0.0572 | 0.194 | 13 |
| Isomap | 14 | 0.0228 | -0.0329 | -0.0147 | 0.0018 | 0.109 | 10 |
| UMAP | 14 | -0.00095 | -0.0348 | -0.0192 | -0.0219 | 0.0626 | 12 |
| CEBRA | 14 | -0.00125 | -0.0471 | -0.0245 | -0.022 | 0.059 | 14 |
| LFADS | 14 | 0.107 | 0.0112 | 0.032 | 0.101 | 0.198 | 14 |
| ridge_channels | 14 | 0.146 | 0.0879 | 0.1 | 0.125 | 0.263 | 13 |

Guadagno per DoF stimato su un trial alla volta: media e deviazione standard tra i 3 trial della stessa sessione (mediane su sessioni e DoF); g2 = guadagno stimato sui 2 trial di adattamento (percentili su fold e DoF).

| method | g_mean_median | g_sd_between_trials_median | g2_p5 | g2_median | g2_p95 |
|---|---|---|---|---|---|
| PCA | 1.19 | 0.895 | 0.197 | 1.16 | 2.31 |
| FA | 1.58 | 1.86 | -0.313 | 1.86 | 4.81 |
| GPFA | 1.02 | 0.719 | 0.207 | 1.04 | 2.14 |
| Isomap | 4.4 | 6.03 | -2.47 | 4.81 | 14.4 |
| UMAP | 55.2 | 136 | -114 | 58.6 | 415 |
| CEBRA | 18.5 | 197 | -326 | 6.4 | 384 |
| LFADS | 1.16 | 0.787 | 0.142 | 1.16 | 2.23 |
| ridge_channels | 0.757 | 0.522 | 0.167 | 0.72 | 1.35 |

**Decodifica dentro la sessione e varianza tra sessioni dei latenti.** In ogni sessione held-in: readout stimato sui trial tranne gli ultimi 3 (α leave-one-trial-out, griglia del readout held-in), R² sugli ultimi 3. eta2_session = quota della varianza dei latenti standardizzati spiegata dalla sessione (media sulle dimensioni); lodo_r2_heldin = readout held-in, leave-one-day-out. alpha_at_edge = sessioni con α all'estremo della griglia.

| method | dim | n | r2_within_median | r2_within_min | r2_within_max | alpha_at_edge | eta2_session | lodo_r2_heldin |
|---|---|---|---|---|---|---|---|---|
| PCA | 25 | 13 | 0.245 | 0.134 | 0.316 | 0 | 0.197 | 0.199 |
| FA | 15 | 13 | 0.12 | 0.0428 | 0.259 | 0 | 0.195 | 0.0709 |
| GPFA | 25 | 13 | 0.249 | 0.164 | 0.294 | 0 | 0.228 | 0.205 |
| Isomap | 25 | 13 | 0.204 | 0.0549 | 0.325 | 0 | 0.19 | 0.00534 |
| UMAP | 25 | 13 | 0.134 | -0.222 | 0.234 | 0 | 0.798 | -0.000228 |
| CEBRA | 25 | 13 | 0.132 | 0.0835 | 0.216 | 0 | 0.998 | -0.000213 |
| LFADS | 25 | 13 | 0.217 | 0.143 | 0.277 | 0 | 0.269 | 0.192 |

Riferimento: per la velocità la stessa quota è 0.000.
