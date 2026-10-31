# Protocollo cross-sessione: risultati (generato da scripts/xs_analysis.py)
Registro: /Users/valerio/tesi_h1_work/results/h1_runs.csv. R² concatenato sui 3 fold, media sui semi; unità = sessione.
## Mediana [IQR] (n) per metodo × condizione
| method | zero_shot | recenter | align | fa_native | fewshot | scratch | ridge_rates |
|---|---|---|---|---|---|---|---|
| PCA | 0.079 [0.055, 0.098] (14) | 0.118 [0.083, 0.142] (14) | 0.108 [0.065, 0.158] (14) | -- | 0.108 [0.067, 0.158] (14) | -0.028 [-0.050, 0.006] (14) | -- |
| FA | 0.034 [0.023, 0.047] (14) | 0.042 [0.038, 0.053] (14) | 0.050 [0.043, 0.061] (14) | 0.025 [0.020, 0.032] (14) | 0.050 [0.043, 0.061] (14) | -0.018 [-0.020, -0.005] (14) | -- |
| GPFA | 0.030 [0.007, 0.046] (14) | 0.127 [0.110, 0.131] (14) | 0.071 [0.032, 0.097] (14) | -- | 0.072 [0.033, 0.097] (14) | -0.007 [-0.020, 0.000] (14) | -- |
| Isomap | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.001, -0.001] (14) | -- | 0.000 [-0.001, 0.004] (14) | -0.033 [-0.046, -0.018] (14) | -- |
| UMAP | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -- | -0.001 [-0.003, 0.001] (14) | -0.023 [-0.043, -0.020] (14) | -- |
| CEBRA | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -0.001 [-0.002, -0.001] (14) | -- | -0.002 [-0.003, -0.001] (14) | -0.021 [-0.028, -0.010] (14) | -- |
| LFADS | 0.108 [0.075, 0.134] (14) | 0.125 [0.102, 0.143] (14) | 0.108 [0.082, 0.135] (14) | -- | 0.108 [0.082, 0.135] (14) | -- | -- |
| rates | -- | -- | -- | -- | -- | -- | -0.016 [-0.038, 0.015] (14) |
## Wilcoxon tra condizioni consecutive (Holm dentro ogni metodo)
Differenze = to − from; r_rb > 0 e n_better = sessioni in cui `to` è migliore.

| method | from | to | n | W | p | p_holm | r_rb | median_diff | n_better |
|---|---|---|---|---|---|---|---|---|---|
| PCA | zero_shot | recenter | 14 | 3 | 0.00061 | 0.00183 | 0.943 | 0.0228 | 13 |
| PCA | recenter | align | 14 | 47 | 0.761 | 1 | -0.105 | 0.000623 | 8 |
| PCA | align | fewshot | 14 | 48 | 0.808 | 1 | 0.0857 | 0.00018 | 8 |
| PCA | fewshot | scratch | 14 | 0 | 0.000122 | 0.000488 | -1 | -0.14 | 0 |
| FA | zero_shot | recenter | 14 | 0 | 0.000122 | 0.00061 | 1 | 0.00858 | 14 |
| FA | recenter | align | 14 | 8 | 0.00305 | 0.0061 | 0.848 | 0.00529 | 13 |
| FA | align | fewshot | 14 | 42 | 0.807 | 0.807 | 0.0769 | 1.19e-07 | 7 |
| FA | fewshot | scratch | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0631 | 0 |
| FA | align | fa_native | 14 | 0 | 0.000122 | 0.00061 | -1 | -0.0264 | 0 |
| GPFA | zero_shot | recenter | 14 | 0 | 0.000122 | 0.000488 | 1 | 0.0988 | 14 |
| GPFA | recenter | align | 14 | 0 | 0.000122 | 0.000488 | -1 | -0.0615 | 0 |
| GPFA | align | fewshot | 14 | 21 | 0.0494 | 0.0494 | 0.6 | 0.000565 | 11 |
| GPFA | fewshot | scratch | 14 | 1 | 0.000244 | 0.000488 | -0.981 | -0.0828 | 1 |
| Isomap | zero_shot | recenter | 14 | 20 | 0.0419 | 0.0419 | 0.619 | 2.39e-05 | 10 |
| Isomap | recenter | align | 14 | 7 | 0.00232 | 0.00696 | 0.867 | 1.2e-05 | 11 |
| Isomap | align | fewshot | 14 | 11 | 0.00671 | 0.0134 | 0.79 | 0.00197 | 11 |
| Isomap | fewshot | scratch | 14 | 1 | 0.000244 | 0.000977 | -0.981 | -0.0348 | 1 |
| UMAP | zero_shot | recenter | 14 | 37 | 0.358 | 0.715 | 0.295 | -1.21e-06 | 6 |
| UMAP | recenter | align | 14 | 4 | 0.000854 | 0.00256 | -0.924 | -3.62e-06 | 2 |
| UMAP | align | fewshot | 14 | 50 | 0.903 | 0.903 | 0.0476 | 7.29e-05 | 7 |
| UMAP | fewshot | scratch | 14 | 0 | 0.000122 | 0.000488 | -1 | -0.0222 | 0 |
| CEBRA | zero_shot | recenter | 14 | 52 | 1 | 1 | 0.00952 | -8.15e-07 | 6 |
| CEBRA | recenter | align | 14 | 21.5 | 0.0516 | 0.155 | -0.59 | -8.94e-07 | 4 |
| CEBRA | align | fewshot | 14 | 35 | 0.296 | 0.592 | -0.333 | -0.00103 | 5 |
| CEBRA | fewshot | scratch | 14 | 1 | 0.000244 | 0.000977 | -0.981 | -0.0197 | 1 |
| LFADS | zero_shot | recenter | 14 | 16 | 0.0203 | 0.0405 | 0.695 | 0.00855 | 9 |
| LFADS | recenter | align | 14 | 19 | 0.0353 | 0.0405 | -0.638 | -0.0128 | 5 |
| LFADS | align | fewshot | 14 | 14 | 0.0134 | 0.0403 | 0.733 | 2.59e-06 | 11 |

## Friedman tra metodi
| condition | methods | n | k | chi2 | p | kendall_W |
|---|---|---|---|---|---|---|
| align | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS | 14 | 7 | 66.1 | 2.58e-12 | 0.787 |
| fewshot | PCA,FA,GPFA,Isomap,UMAP,CEBRA,LFADS | 14 | 7 | 66.3 | 2.3e-12 | 0.79 |

## Wilcoxon tra metodi a coppie (Holm dentro ogni condizione)
Differenze = method_b − method_a; r_rb > 0 e n_better = sessioni in cui method_b è migliore.

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
| fewshot | PCA | FA | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0637 | 2 |
| fewshot | PCA | GPFA | 14 | 7 | 0.00232 | 0.0139 | -0.867 | -0.0444 | 2 |
| fewshot | PCA | Isomap | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.111 | 0 |
| fewshot | PCA | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.111 | 0 |
| fewshot | PCA | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.109 | 0 |
| fewshot | PCA | LFADS | 14 | 48 | 0.808 | 1 | 0.0857 | -0.0161 | 5 |
| fewshot | FA | GPFA | 14 | 39 | 0.426 | 1 | 0.257 | 0.00749 | 8 |
| fewshot | FA | Isomap | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0498 | 0 |
| fewshot | FA | UMAP | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0522 | 0 |
| fewshot | FA | CEBRA | 14 | 0 | 0.000122 | 0.00256 | -1 | -0.0519 | 0 |
| fewshot | FA | LFADS | 14 | 1 | 0.000244 | 0.00293 | 0.981 | 0.0582 | 13 |
| fewshot | GPFA | Isomap | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0717 | 2 |
| fewshot | GPFA | UMAP | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0748 | 2 |
| fewshot | GPFA | CEBRA | 14 | 3 | 0.00061 | 0.0061 | -0.943 | -0.0747 | 2 |
| fewshot | GPFA | LFADS | 14 | 1 | 0.000244 | 0.00293 | 0.981 | 0.0446 | 13 |
| fewshot | Isomap | UMAP | 14 | 12 | 0.00854 | 0.0342 | -0.771 | -0.00312 | 2 |
| fewshot | Isomap | CEBRA | 14 | 9 | 0.00403 | 0.0201 | -0.829 | -0.0028 | 2 |
| fewshot | Isomap | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.107 | 14 |
| fewshot | UMAP | CEBRA | 14 | 48 | 0.808 | 1 | -0.0857 | -0.000155 | 7 |
| fewshot | UMAP | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.11 | 14 |
| fewshot | CEBRA | LFADS | 14 | 0 | 0.000122 | 0.00256 | 1 | 0.109 | 14 |

## Zero-shot contro i giorni (descrittivo)
| method | condition | n | slope_per_day | ci_low | ci_high | spearman_rho | spearman_p | r2_first_day | r2_last_day |
|---|---|---|---|---|---|---|---|---|---|
| PCA | zero_shot | 14 | -0.0041 | -0.00801 | -0.000201 | -0.532 | 0.0504 | 0.125 | 0.0321 |
| FA | zero_shot | 14 | -4.29e-05 | -0.002 | 0.00191 | -0.0709 | 0.81 | 0.0438 | 0.0437 |
| GPFA | zero_shot | 14 | 0.00309 | -0.00349 | 0.00968 | 0.292 | 0.31 | -0.0427 | 0.0263 |
| Isomap | zero_shot | 14 | 1.24e-05 | -2.96e-05 | 5.44e-05 | 0.0886 | 0.763 | -0.00127 | -0.00111 |
| UMAP | zero_shot | 14 | 1.15e-05 | -3.05e-05 | 5.35e-05 | 0.0797 | 0.786 | -0.00129 | -0.00112 |
| CEBRA | zero_shot | 14 | 1.25e-05 | -2.89e-05 | 5.39e-05 | 0.0797 | 0.786 | -0.00129 | -0.00111 |
| LFADS | zero_shot | 14 | -0.0041 | -0.00887 | 0.000677 | -0.399 | 0.158 | 0.144 | 0.0278 |
| rates | ridge_rates | 14 | 9.02e-05 | -0.00615 | 0.00633 | 0.0177 | 0.952 | 0.0468 | 0.00604 |

## Bin dei 2 trial di adattamento (canali attivi: 170)
Tutti i bin: min 1442, mediana 1647, max 1933; con eval_mask: min 1390, mediana 1611, max 1876 (42 fold).

| method | latent_dim | readout_weights_per_dof | bins_mask_min | ratio_min |
|---|---|---|---|---|
| PCA | 25 | 1251 | 1390 | 1.11 |
| FA | 15 | 751 | 1390 | 1.85 |
| GPFA | 25 | 1251 | 1390 | 1.11 |
| Isomap | 3 | 151 | 1390 | 9.21 |
| UMAP | 3 | 151 | 1390 | 9.21 |
| CEBRA | 3 | 151 | 1390 | 9.21 |
| LFADS | 25 | 1251 | 1390 | 1.11 |
| rates | 170 | 8501 | 1390 | 0.164 |

## Verifica sui latenti a 3 dimensioni (esplorativa, non nel piano)
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

## Pseudo-few-shot sulle held-in (scelte congelate)
| method | heldin_alpha_src | heldin_lodo_r2 | pseudo_zero_shot | pseudo_recenter | pseudo_align | pseudo_fa_native | pseudo_fewshot | lam | lam_at_edge | pseudo_scratch | pseudo_ridge_rates | alpha_scratch | alpha_at_edge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | 1e+06 | 0.199 | 0.0994 | 0.189 | 0.213 | nan | 0.213 | 3.16e+06 | False | 0.0225 | nan | 3.16e+04 | False |
| FA | 1e+06 | 0.0709 | 0.076 | 0.0947 | 0.1 | 0.0711 | 0.1 | 1e+09 | True | 0.00913 | nan | 3.16e+04 | False |
| GPFA | 1e+06 | 0.205 | 0.108 | 0.205 | 0.206 | nan | 0.206 | 1e+07 | False | 0.0113 | nan | 1e+05 | False |
| Isomap | 1e+08 | -7.02e-05 | -0.00188 | -0.00188 | -0.00188 | nan | -0.000172 | 1e+05 | False | -0.00739 | nan | 3.16e+04 | False |
| UMAP | 1e+08 | -6.83e-05 | -0.00182 | -0.00183 | -0.00182 | nan | 6.75e-05 | 1e+05 | False | -0.00716 | nan | 3.16e+04 | False |
| CEBRA | 1e+08 | -6.07e-05 | -0.00191 | -0.00191 | -0.00191 | nan | -0.00117 | 3.16e+05 | False | -0.0171 | nan | 1e+05 | False |
| LFADS | 1e+06 | 0.192 | 0.167 | 0.173 | 0.207 | nan | 0.207 | 1e+09 | True | nan | nan | nan | None |
| rates | nan | nan | nan | nan | nan | nan | nan | nan | None | nan | 0.0383 | 1e+05 | False |
