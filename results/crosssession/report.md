# Protocollo cross-sessione: risultati (generato da scripts/xs_analysis.py)
Registro: /Users/valerio/tesi_h1_work/results/h1_runs.csv. R² concatenato sui 3 fold, media sui semi; unità = sessione.
## Mediana [IQR] (n) per metodo × condizione
| method | zero_shot | recenter | align | fewshot | scratch | ridge_rates |
|---|---|---|---|---|---|---|
| PCA | 0.079 [0.055, 0.098] (14) | 0.118 [0.083, 0.142] (14) | 0.108 [0.065, 0.158] (14) | 0.108 [0.067, 0.158] (14) | -0.028 [-0.050, 0.006] (14) | -- |
| rates | -- | -- | -- | -- | -- | -0.016 [-0.038, 0.015] (14) |
## Wilcoxon tra condizioni consecutive (Holm dentro ogni metodo)
| method | from | to | n | W | p | p_holm | r_rb | median_diff | n_better |
|---|---|---|---|---|---|---|---|---|---|
| PCA | zero_shot | recenter | 14 | 3 | 0.00061 | 0.00183 | 0.943 | 0.0228 | 13 |
| PCA | recenter | align | 14 | 47 | 0.761 | 1 | -0.105 | 0.000623 | 8 |
| PCA | align | fewshot | 14 | 48 | 0.808 | 1 | 0.0857 | 0.00018 | 8 |
| PCA | fewshot | scratch | 14 | 0 | 0.000122 | 0.000488 | -1 | -0.14 | 0 |

## Friedman tra metodi
(vuoto)

## Wilcoxon tra metodi a coppie (Holm dentro ogni condizione)
(vuoto)

## Zero-shot contro i giorni (descrittivo)
| method | condition | n | slope_per_day | ci_low | ci_high | spearman_rho | spearman_p | r2_first_day | r2_last_day |
|---|---|---|---|---|---|---|---|---|---|
| PCA | zero_shot | 14 | -0.0041 | -0.00801 | -0.000201 | -0.532 | 0.0504 | 0.125 | 0.0321 |
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

## Pseudo-few-shot sulle held-in (scelte congelate)
| method | heldin_alpha_src | heldin_lodo_r2 | pseudo_zero_shot | pseudo_recenter | pseudo_align | pseudo_fewshot | lam | lam_at_edge | pseudo_scratch | pseudo_ridge_rates | alpha_scratch | alpha_at_edge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | 1e+06 | 0.199 | 0.0994 | 0.189 | 0.213 | 0.213 | 3.16e+06 | False | 0.0225 | nan | 3.16e+04 | False |
| rates | nan | nan | nan | nan | nan | nan | nan | None | nan | 0.0383 | 1e+05 | False |
