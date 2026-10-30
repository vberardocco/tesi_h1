# Repliche su 19250113T120811 (S2) e 19250120T115044 (S5): confronto

Generato da `rep/tabelle.py` dalle uscite dei notebook eseguiti, da `decoder.json` e da `angles.pkl`.

**Modifiche del 30/10.**
- §4: aggiunta la diagnosi del caso "dalla covarianza" (§4b). La covarianza usata dal caso è la stessa dei dati su cui
  sono stimati i sottospazi, con lo stesso preprocessing (differenza massima 0), e il campionamento la riproduce. Lo
  spettro però è piatto (PR tra 87 e 145 su 155 canali; i primi 10 autovalori tengono l'11-23% della varianza): per
  questo il caso è quasi uniforme. Non c'è un errore da correggere e il caso non è stato rifatto: i numeri del §4 e
  `angoli.csv` sono gli stessi di prima.
- §7: aggiunto il controllo sulla cinematica (variazione della posizione contro velocità) dentro i trial e in
  eval_mask, con la cross-correlazione. In S5 la correlazione resta bassa (tx 0.59, tz 0.46) senza ritardi; torna a
  0.98-0.99 con una media mobile di 300 ms: lo scarto è rumore veloce nella posizione, non nella velocità.

Note per leggere le tabelle:
- **originale (io.py)**: i notebook 02_new e 04_deep sulla sessione di riferimento, con i bin di io.py (una riga di
  sfasamento rispetto a FALCON). **rif. S1, S2, S5**: le repliche con i bin di FALCON, stessi iperparametri, senza sweep.
- Nel ricalcolo della sessione di riferimento cambiano anche i blocchi della cross-validation del 04: `GroupKFold`
  divide i trial in base al numero di punti di valutazione, che con la riga spostata passa da 1106 a 1099. La
  differenza tra "originale" e "rif. S1" quindi non è solo l'effetto dei bin.
- Il §6 (CEBRA-Behavior, PLS, kin_r2) è stato ricalcolato solo sulla sessione di riferimento: "—" nelle altre.
- Intervalli tra parentesi: minimo e massimo sui semi. Isomap e UMAP restano fuori (esplorativi).

## 1. Decoder (ridge causale, 1000 ms di storia, R² pesato per varianza sul 20% finale, eval_mask)

Formato di minival: R² medio e per grado di libertà. Zero-shot: allenata sulle held-in degli altri giorni, α = 10^4,5. Held-in 80%: primo 80% di tutti i file held-in, giorno incluso, α = 10^4,5. Solo la sessione: primo 80% della sola sessione, α scelto con cross-validation interna (5 blocchi di trial).

| sessione | condizione | R² medio | tx | ty | tz | rx | g1 | g2 | g3 | α |
|---|---|---|---|---|---|---|---|---|---|---|
| rif. S1 (S1_set_1) | zero-shot (senza il suo giorno) | 0.174 | 0.114 | -0.427 | -0.213 | 0.068 | 0.149 | 0.196 | 0.264 | 10^4.5 |
| rif. S1 (S1_set_1) | held-in 80% (giorno incluso) | 0.305 | 0.210 | -0.512 | 0.299 | 0.331 | 0.307 | 0.311 | 0.293 | 10^4.5 |
| rif. S1 (S1_set_1) | solo la sessione, 80% | 0.291 | -0.111 | -0.572 | 0.392 | 0.240 | 0.378 | 0.351 | 0.170 | 10^4.0 |
| S2 (S2_set_1) | zero-shot (senza il suo giorno) | 0.097 | 0.209 | -2.820 | -0.430 | 0.162 | -0.071 | 0.059 | 0.336 | 10^4.5 |
| S2 (S2_set_1) | held-in 80% (giorno incluso) | 0.401 | 0.283 | -0.622 | -0.233 | 0.359 | 0.415 | 0.431 | 0.399 | 10^4.5 |
| S2 (S2_set_1) | solo la sessione, 80% | 0.344 | 0.247 | -0.368 | -0.292 | 0.269 | 0.446 | 0.366 | 0.270 | 10^4.5 |
| S5 (S5_set_1) | zero-shot (senza il suo giorno) | 0.192 | 0.237 | -1.059 | -0.309 | 0.462 | 0.329 | 0.344 | -0.281 | 10^4.5 |
| S5 (S5_set_1) | held-in 80% (giorno incluso) | 0.452 | 0.392 | 0.331 | 0.349 | 0.543 | 0.423 | 0.445 | 0.441 | 10^4.5 |
| S5 (S5_set_1) | solo la sessione, 80% | 0.338 | 0.374 | 0.288 | 0.186 | 0.409 | 0.376 | 0.292 | 0.319 | 10^4.5 |

## 2. Dimensionalità e sottospazi (02)

**Dati**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| canali attivi (≥ 0,5 Hz) | 160 | 160 | 166 | 165 |
| frequenza media (Hz) | 14.2 | 14.2 | 14.7 | 16.8 |

**PCA (σ = 50 ms, tutti i bin)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| participation ratio | 106.1 | 106.1 | 104.0 | 94.4 |
| varianza della PC1 (%) | 3.9 | 3.9 | 4.1 | 4.8 |
| varianza delle PC1-3 (%) | 8.5 | 8.6 | 9.5 | 11.0 |
| componenti per l'80% / 90% / 95% | 100 / 125 / 140 | 100 / 125 / 140 | 103 / 129 / 145 | 100 / 126 / 143 |

**PCA sulle medie per condizione**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| participation ratio | 19.7 | 19.7 | 16.1 | 21.6 |
| componenti per il 95% | 78 | 78 | 78 | 79 |
| PR con 1 / 7 ripetizioni mediate (finestre allineate) | 50.1 / 20.6 | 50.1 / 20.6 | 47.6 / 16.9 | 48.5 / 22.6 |

**FA (conteggi, K* = 15)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| PR della parte condivisa | 9.5 | 9.5 | 9.4 | 9.1 |
| dimensioni condivise per il 95% | 14 | 14 | 14 | 14 |
| prime 3 direzioni (% della varianza condivisa) | 46 | 46 | 47 | 48 |
| quota condivisa per canale, mediana | 0.038 | 0.039 | 0.034 | 0.032 |

**FA nelle finestre (K* = 10)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| PR della parte condivisa | 7.0 | 7.0 | 6.6 | 6.9 |

**GPFA (d = 25)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| PR della parte condivisa | 12.2 | 12.2 | 10.5 | 11.6 |
| dimensioni condivise per il 95% | 21 | 21 | 21 | 21 |
| quota condivisa per canale, mediana (%) | 3.2 | 3.2 | 3.8 | 3.9 |
| scala temporale più breve / più lunga (ms) | 48.0 / 754.0 | 48.0 / 754.0 | 61.0 / 1125.0 | 51.0 / 1268.0 |

**Traiettorie nelle finestre**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| consistenza: PCA smussata | 0.1167 | 0.1167 | 0.1422 | 0.1533 |
| consistenza: FA (K = 25) | 0.0387 | 0.0387 | 0.0448 | 0.0577 |
| consistenza: GPFA (d = 25) | 0.1369 | 0.1369 | 0.1542 | 0.1549 |
| seghettatura: GPFA (d = 25) | 0.0108 | 0.0108 | 0.0044 | 0.0049 |

## 3. CEBRA, LFADS, GPFA (04)

**Dati**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| punti di valutazione | 1106 | 1099 | 1037 | 1137 |

**Fase riconosciuta (%), caso 12,5**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| PCA, d = 3 | 32.4 | 30.9 | 37.8 | 28.3 |
| CEBRA-Time, 3 dimensioni | 37.7 (35.3-40.6) | 35.5 (33.0-37.3) | 40.5 (37.7-42.4) | 31.2 (29.6-33.0) |
| CEBRA-Time, 4 dimensioni | 40.4 (38.5-42.7) | 40.3 (38.7-41.6) | 44.0 (41.0-46.9) | 34.9 (33.3-37.0) |
| tetto: 25 PC smussati | 42.2 | 41.4 | 46.7 | 36.9 |
| tetto alla pari: media sulla finestra | 44.9 | 43.6 | 48.0 | 39.7 |
| orologio (solo il tempo dall'inizio del trial) | 48.5 | 50.1 | 47.2 | 63.8 |
| CEBRA-Time 3D sui gemelli | 12.5 (10.7-13.9) | 13.1 (12.3-13.9) | 12.8 (11.6-14.1) | 14.1 (12.7-16.3) |

**CEBRA-Time**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| consistenza tra allenamenti, 3D (R², %) | 63.4 (46.9-83.1) | 66.0 (51.3-83.0) | 65.3 (53.7-84.9) | 64.5 (51.3-81.5) |
| consistenza tra allenamenti, 4D (R², %) | 75.1 (66.8-83.9) | 75.9 (67.1-84.1) | 68.0 (58.4-79.3) | 75.2 (64.6-85.2) |
| vicini nella catena del tempo, Δ = 10 (%) | 7.9 | 8.2 | 8.1 | 8.4 |
| vicini della stessa fase in altri trial, Δ = 10 (%) | 33.1 | 32.9 | 35.7 | 25.1 |
| seghettatura, 4D | 0.074 (0.066-0.080) | 0.074 (0.065-0.082) | 0.072 (0.059-0.077) | 0.077 (0.072-0.084) |
| consistenza delle traiettorie, 4D | 0.106 (0.086-0.127) | 0.106 (0.088-0.129) | 0.113 (0.090-0.128) | 0.091 (0.086-0.097) |

**§6, solo rif.: fase (%)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| CEBRA-Behavior, 3 dimensioni | 25.6 (22.4-27.9) | 23.8 (21.5-25.8) | — | — |
| CEBRA-Behavior, 4 dimensioni | 28.7 (27.3-30.1) | 30.3 (28.3-32.4) | — | — |
| PLS, 2 componenti | 27.3 | 26.8 | — | — |
| PLS, 3 componenti | 28.3 | 29.5 | — | — |
| distacco CEBRA-Time − CEBRA-Behavior, 3D / 4D (punti) | 12.1 / 11.7 | 11.7 / 10.0 | — | — |
| CEBRA-Behavior 3D, etichette mescolate nel tempo | 15.1 (12.9-17.3) | 15.1 (13.4-16.3) | — | — |
| CEBRA-Behavior 3D, etichette scambiate tra trial | 29.8 (28.1-31.3) | 29.2 (27.7-30.7) | — | — |

**§6, solo rif.: kin_r2, velocità (R², %)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| CEBRA-Time, 4 dimensioni | 1.5 (-1.1-3.8) | 5.1 (3.6-6.3) | — | — |
| CEBRA-Behavior, 4 dimensioni | -1.1 (-5.4-1.2) | -0.6 (-3.0-1.4) | — | — |
| PLS, 3 componenti | 4.6 | 7.3 | — | — |
| tetto, 25 dimensioni | 19.4 | 20.5 | — | — |

**§6, solo rif.: kin_r2, posizione (R², %)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| CEBRA-Time, 4 dimensioni | -27.8 (-29.3--26.4) | -33.0 (-34.0--31.8) | — | — |
| CEBRA-Behavior, 4 dimensioni | -23.1 (-24.8--21.9) | -26.9 (-28.3--25.7) | — | — |
| PLS, 3 componenti | -19.3 | -23.2 | — | — |
| tetto, 25 dimensioni | -16.1 | -23.7 | — | — |

**§12: fase, tetto con tutti i 25 fattori (%)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| FA | 25.8 | 21.2 | 28.0 | 22.4 |
| GPFA come nel §7b | 50.7 | 50.2 | 53.2 | 42.6 |
| GPFA alla pari | 51.4 | 49.9 | 54.4 | 42.6 |
| LFADS (5 semi) | 46.5 (44.7-47.7) | 45.7 (44.9-46.4) | 52.3 (50.5-53.0) | 42.5 (39.7-45.2) |
| 25 PC smussati (parte A) | 42.2 | 41.4 | 46.7 | 36.9 |

**§12: fase, primi 3 fattori (%)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| FA | 17.3 | 19.7 | 22.0 | 19.8 |
| GPFA alla pari | 30.9 | 31.7 | 37.9 | 26.2 |
| LFADS (5 semi) | 20.9 (19.1-22.4) | 21.1 (17.6-27.6) | 25.1 (21.8-28.3) | 19.6 (15.8-23.9) |
| 25 PC smussati (parte A) | 32.4 | 30.9 | 37.8 | 28.3 |

**§12: fase senza SnapTo, tetto (%), caso 14,3**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| GPFA alla pari | 58.5 | 56.9 | 62.3 | 49.0 |
| LFADS (5 semi) | 53.1 (50.6-54.1) | 52.2 (50.6-53.1) | 59.7 (58.2-60.6) | 47.5 (45.3-50.8) |

**§12: bit/spike sui 40 canali nascosti, trial nuovi**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| smoothing + GLM (riferimento) | 0.0416 | 0.0417 | 0.0247 | 0.0563 |
| FA | 0.0358 | 0.0357 | 0.0227 | 0.0368 |
| GPFA alla pari | 0.0422 | 0.0426 | 0.0294 | 0.0602 |
| LFADS (5 semi) | 0.0388 [0.0383, 0.0393] | 0.0389 [0.0386, 0.0395] | 0.0292 [0.0280, 0.0305] | 0.0575 [0.0572, 0.0579] |
| GPFA + regressione di Poisson | 0.0474 | 0.0479 | 0.0339 | 0.0642 |
| LFADS sopra lo smoothing + GLM (blocco × seme) | 4 su 25 | 5 su 25 | 25 su 25 | 20 su 25 |

**§13: bit/spike, trial nuovi / trial visti**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| smoothing + GLM | 0.4430 | 0.4425 | 0.3051 | 0.5301 |
| GPFA alla pari | 0.6638 | 0.6673 | 0.5831 | 0.7424 |
| LFADS (5 semi) | 0.4021 [0.3929, 0.4136] | 0.3995 [0.3921, 0.4079] | 0.3557 [0.3290, 0.3707] | 0.5214 [0.5167, 0.5260] |

**§13: bit/spike con canali nascosti spostati nel tempo (trial nuovi)**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| smoothing + GLM | -0.0234 | -0.0236 | -0.0230 | -0.0229 |
| GPFA alla pari | -0.0122 | -0.0135 | -0.0145 | -0.0199 |
| LFADS (3 semi) | -0.0179 [-0.0190, -0.0162] | -0.0177 [-0.0183, -0.0173] | -0.0138 [-0.0148, -0.0132] | -0.0169 [-0.0171, -0.0166] |

**§13: gemelli**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| LFADS sul gemello, fase con tutti i fattori (%) | 11.0 (7.8-13.9) | 11.9 (11.2-12.8) | 12.9 (11.1-14.2) | 13.3 (10.9-15.8) |

**§12: stabilità**

| misura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| LFADS tra semi, 25 fattori (R², %) | 88.7 (86.7-89.5) | 88.4 (86.1-90.0) | 89.7 (88.2-91.1) | 90.3 (89.1-91.5) |
| consistenza delle traiettorie: GPFA alla pari | 0.118 | 0.114 | 0.157 | 0.153 |
| consistenza delle traiettorie: LFADS | 0.109 (0.090-0.135) | 0.130 (0.095-0.184) | 0.086 (0.051-0.122) | 0.091 (0.056-0.132) |

## 4. Angoli principali tra sessioni (angolo medio, gradi; 155 canali attivi in tutte e tre)

Metà – metà: le due metà della stessa sessione (trial 1-2, 5-6, … contro 3-4, 7-8, …), il pavimento del rumore. Caso dalla covarianza: sottospazi presi dalla covarianza dei dati (Elsayed et al. 2016), 500 estrazioni. Uniforme: sottospazi casuali uniformi negli stessi canali, per confronto.

| metodo | rif. S1 – S2 | rif. S1 – S5 | S2 – S5 | metà – metà (media) | metà – metà rif. S1 | metà – metà S2 | metà – metà S5 | caso dalla covarianza (mediana) | uniforme (mediana) |
|---|---|---|---|---|---|---|---|---|---|
| PCA, K = 10 | 46.5 | 50.7 | 46.9 | 40.7 | 43.7 | 41.0 | 37.3 | 75.8 | 77.5 |
| PCA, K = 25 | 51.9 | 55.5 | 53.2 | 50.0 | 50.8 | 49.9 | 49.4 | 67.2 | 69.3 |
| FA, K = 10 | 36.4 | 38.9 | 40.0 | 34.7 | 32.8 | 34.4 | 36.8 | 77.3 | 77.5 |
| FA, K = 15 | 42.8 | 42.8 | 45.3 | 40.3 | 40.4 | 41.6 | 39.0 | 74.1 | 74.4 |
| FA, K = 25 | 46.8 | 47.8 | 47.2 | 46.9 | 46.3 | 48.1 | 46.3 | 69.0 | 69.3 |
| GPFA, K = 10 | 49.8 | 49.8 | 50.0 | 42.5 | 42.9 | 42.7 | 41.8 | 75.0 | 77.5 |
| GPFA, K = 25 | 48.4 | 49.7 | 49.8 | 46.3 | 45.1 | 47.6 | 46.2 | 65.4 | 69.3 |

I primi tre angoli principali (gradi):

| metodo | rif. S1 – S2 | rif. S1 – S5 | S2 – S5 |
|---|---|---|---|
| PCA, K = 10 | 25 / 29 / 32 | 23 / 30 / 34 | 22 / 28 / 34 |
| PCA, K = 25 | 20 / 24 / 25 | 19 / 26 / 28 | 18 / 22 / 25 |
| FA, K = 10 | 13 / 18 / 25 | 17 / 17 / 30 | 14 / 18 / 26 |
| FA, K = 15 | 13 / 18 / 22 | 15 / 16 / 27 | 13 / 17 / 23 |
| FA, K = 25 | 12 / 16 / 18 | 14 / 15 / 22 | 12 / 16 / 20 |
| GPFA, K = 10 | 22 / 29 / 34 | 21 / 32 / 36 | 20 / 31 / 32 |
| GPFA, K = 25 | 16 / 22 / 23 | 14 / 23 / 24 | 14 / 22 / 24 |

**Attenzione:** il caso dalla covarianza è quasi uguale a quello uniforme, soprattutto per FA: la covarianza dei conteggi z-scorati a 20 ms è quasi piatta (domina il rumore di Poisson). Il confronto che conta qui è quello con le due metà della stessa sessione.

### 4b. Diagnosi del caso dalla covarianza (`rep/angles_diag.py`)

Per ogni sessione, la covarianza da cui il caso campiona (nel caso si usa la media delle due sessioni confrontate). Preprocessing, uguale per il caso e per i sottospazi: PCA, bin FALCON da 20 ms, smoothing σ = 50 ms, z-score, tutti i bin; FA, conteggi grezzi z-scorati, tutti i bin; GPFA, radice dei conteggi nelle 119 finestre da 1 s, senza z-score. Campionamento (`angles.py`, righe 122-127): autovalori e autovettori della covarianza media, M = U S^1/2, K vettori M r con r gaussiano, ortonormalizzati con QR, 500 coppie. L'errore relativo dei 20000 vettori è quello atteso dal campionamento finito.

| metodo | sessione | canali | PR | primi 10 (%) | primi 15 (%) | primi 25 (%) | differenza massima con la covarianza dei sottospazi | errore relativo, covarianza di 20000 vettori campionati |
|---|---|---|---|---|---|---|---|---|
| PCA | rif. S1 | 155 | 102.1 | 20.0 |  | 34.9 | 0 | 0.071 |
| PCA | S2 | 155 | 96.2 | 21.1 |  | 35.9 | 0 | 0.069 |
| PCA | S5 | 155 | 87.3 | 23.1 |  | 37.4 | 0 | 0.066 |
| FA | rif. S1 | 155 | 144.9 | 11.0 | 15.0 | 22.4 | 0 | 0.085 |
| FA | S2 | 155 | 145.5 | 10.9 | 14.8 | 22.3 | 0 | 0.085 |
| FA | S5 | 155 | 143.9 | 11.2 | 15.2 | 22.6 | 0 | 0.085 |
| GPFA | rif. S1 | 155 | 97.4 | 18.0 |  | 35.9 | 0 | 0.070 |
| GPFA | S2 | 155 | 97.4 | 18.0 |  | 35.8 | 0 | 0.070 |
| GPFA | S5 | 155 | 98.3 | 17.7 |  | 35.6 | 0 | 0.070 |

## 5. Controllo sul riconoscimento delle fasi (04, sessione di riferimento, bin di io.py)

Mappe rilette dalla cache del 04, nessun allenamento (`rep/fasi`).

| metodo | semi | accuratezza bilanciata (%) | accuratezza semplice (%) | errori su fasi adiacenti (%) | se sparsi a caso (%) |
|---|---|---|---|---|---|
| PCA, d = 3 | 1 | 32.4 | 33.9 | 25 | 24 |
| tetto: 25 PC smussati, 1 bin | 1 | 42.2 | 41.9 | 21 | 24 |
| CEBRA-Time, 3 dimensioni | 5 | 37.7 (35.3-40.6) | 38.0 (35.3-40.3) | 18 | 25 |
| CEBRA-Time, 4 dimensioni | 5 | 40.4 (38.5-42.7) | 40.5 (38.8-41.8) | 16 | 25 |
| CEBRA-Behavior, 3 dimensioni | 5 | 25.6 (22.4-27.9) | 27.3 (23.6-29.8) | 24 | 24 |
| CEBRA-Behavior, 4 dimensioni | 5 | 28.7 (27.3-30.1) | 29.6 (28.2-30.6) | 22 | 24 |
| PLS, 2 componenti | 1 | 27.3 | 30.4 | 21 | 25 |
| PLS, 3 componenti | 1 | 28.3 | 30.2 | 24 | 25 |
| orologio (solo il tempo dall'inizio del trial) | 1 | 48.5 | 57.4 | 84 | 26 |
| contesto: 25 PC dei conteggi, 1 bin (causale) | 1 | 22.1 | 24.6 | 29 | 25 |
| contesto: 25 PC dei conteggi, media causale 1000 ms | 1 | 56.5 | 58.1 | 25 | 25 |
| contesto: 25 PC dei conteggi, 10 × 100 ms causali (250 numeri) | 1 | 52.3 | 53.7 | 19 | 24 |
| contesto: 25 PC smussati, 10 × 100 ms causali (250 numeri) | 1 | 52.2 | 52.7 | 24 | 24 |
| caso: 1/K |  | 12.5 |  |  |  |
| caso: classe più frequente |  | 12.5 | 17.9 (Carry) |  |  |

## 6. Figure

Originale, poi le tre repliche.

| figura | originale (io.py) | rif. S1 | S2 | S5 |
|---|---|---|---|---|
| `02new_active_channels.png` | <img src="../../figures/02new_active_channels.png" width="260"> | <img src="../19250108T110520/figures/02new_active_channels.png" width="260"> | <img src="../19250113T120811/figures/02new_active_channels.png" width="260"> | <img src="../19250120T115044/figures/02new_active_channels.png" width="260"> |
| `02new_fa_shared.png` | <img src="../../figures/02new_fa_shared.png" width="260"> | <img src="../19250108T110520/figures/02new_fa_shared.png" width="260"> | <img src="../19250113T120811/figures/02new_fa_shared.png" width="260"> | <img src="../19250120T115044/figures/02new_fa_shared.png" width="260"> |
| `02new_gpfa_plane.png` | <img src="../../figures/02new_gpfa_plane.png" width="260"> | <img src="../19250108T110520/figures/02new_gpfa_plane.png" width="260"> | <img src="../19250113T120811/figures/02new_gpfa_plane.png" width="260"> | <img src="../19250120T115044/figures/02new_gpfa_plane.png" width="260"> |
| `02new_gpfa_repetitions.png` | <img src="../../figures/02new_gpfa_repetitions.png" width="260"> | <img src="../19250108T110520/figures/02new_gpfa_repetitions.png" width="260"> | <img src="../19250113T120811/figures/02new_gpfa_repetitions.png" width="260"> | <img src="../19250120T115044/figures/02new_gpfa_repetitions.png" width="260"> |
| `02new_gpfa_timescales.png` | <img src="../../figures/02new_gpfa_timescales.png" width="260"> | <img src="../19250108T110520/figures/02new_gpfa_timescales.png" width="260"> | <img src="../19250113T120811/figures/02new_gpfa_timescales.png" width="260"> | <img src="../19250120T115044/figures/02new_gpfa_timescales.png" width="260"> |
| `02new_normalization.png` | <img src="../../figures/02new_normalization.png" width="260"> | <img src="../19250108T110520/figures/02new_normalization.png" width="260"> | <img src="../19250113T120811/figures/02new_normalization.png" width="260"> | <img src="../19250120T115044/figures/02new_normalization.png" width="260"> |
| `02new_pca_phase_averages.png` | <img src="../../figures/02new_pca_phase_averages.png" width="260"> | <img src="../19250108T110520/figures/02new_pca_phase_averages.png" width="260"> | <img src="../19250113T120811/figures/02new_pca_phase_averages.png" width="260"> | <img src="../19250120T115044/figures/02new_pca_phase_averages.png" width="260"> |
| `02new_pca_scree.png` | <img src="../../figures/02new_pca_scree.png" width="260"> | <img src="../19250108T110520/figures/02new_pca_scree.png" width="260"> | <img src="../19250113T120811/figures/02new_pca_scree.png" width="260"> | <img src="../19250120T115044/figures/02new_pca_scree.png" width="260"> |
| `02new_pca_timecourse.png` | <img src="../../figures/02new_pca_timecourse.png" width="260"> | <img src="../19250108T110520/figures/02new_pca_timecourse.png" width="260"> | <img src="../19250113T120811/figures/02new_pca_timecourse.png" width="260"> | <img src="../19250120T115044/figures/02new_pca_timecourse.png" width="260"> |
| `02new_smoothing.png` | <img src="../../figures/02new_smoothing.png" width="260"> | <img src="../19250108T110520/figures/02new_smoothing.png" width="260"> | <img src="../19250113T120811/figures/02new_smoothing.png" width="260"> | <img src="../19250120T115044/figures/02new_smoothing.png" width="260"> |
| `04_behavior_maps.png` | <img src="../../figures/04_behavior_maps.png" width="260"> | <img src="../19250108T110520/figures/04_behavior_maps.png" width="260"> | — | — |
| `04_folds.png` | <img src="../../figures/04_folds.png" width="260"> | <img src="../19250108T110520/figures/04_folds.png" width="260"> | <img src="../19250113T120811/figures/04_folds.png" width="260"> | <img src="../19250120T115044/figures/04_folds.png" width="260"> |
| `04_lfads_cobps.png` | <img src="../../figures/04_lfads_cobps.png" width="260"> | <img src="../19250108T110520/figures/04_lfads_cobps.png" width="260"> | <img src="../19250113T120811/figures/04_lfads_cobps.png" width="260"> | <img src="../19250120T115044/figures/04_lfads_cobps.png" width="260"> |
| `04_lfads_data.png` | <img src="../../figures/04_lfads_data.png" width="260"> | <img src="../19250108T110520/figures/04_lfads_data.png" width="260"> | <img src="../19250113T120811/figures/04_lfads_data.png" width="260"> | <img src="../19250120T115044/figures/04_lfads_data.png" width="260"> |
| `04_lfads_hidden_example.png` | <img src="../../figures/04_lfads_hidden_example.png" width="260"> | <img src="../19250108T110520/figures/04_lfads_hidden_example.png" width="260"> | <img src="../19250113T120811/figures/04_lfads_hidden_example.png" width="260"> | <img src="../19250120T115044/figures/04_lfads_hidden_example.png" width="260"> |
| `04_lfads_overfit.png` | <img src="../../figures/04_lfads_overfit.png" width="260"> | <img src="../19250108T110520/figures/04_lfads_overfit.png" width="260"> | <img src="../19250113T120811/figures/04_lfads_overfit.png" width="260"> | <img src="../19250120T115044/figures/04_lfads_overfit.png" width="260"> |
| `04_lfads_phase.png` | <img src="../../figures/04_lfads_phase.png" width="260"> | <img src="../19250108T110520/figures/04_lfads_phase.png" width="260"> | <img src="../19250113T120811/figures/04_lfads_phase.png" width="260"> | <img src="../19250120T115044/figures/04_lfads_phase.png" width="260"> |
| `04_lfads_trajectories.png` | <img src="../../figures/04_lfads_trajectories.png" width="260"> | <img src="../19250108T110520/figures/04_lfads_trajectories.png" width="260"> | <img src="../19250113T120811/figures/04_lfads_trajectories.png" width="260"> | <img src="../19250120T115044/figures/04_lfads_trajectories.png" width="260"> |
| `04_phase_decoding.png` | <img src="../../figures/04_phase_decoding.png" width="260"> | <img src="../19250108T110520/figures/04_phase_decoding.png" width="260"> | <img src="../19250113T120811/figures/04_phase_decoding.png" width="260"> | <img src="../19250120T115044/figures/04_phase_decoding.png" width="260"> |
| `04_time_chain.png` | <img src="../../figures/04_time_chain.png" width="260"> | <img src="../19250108T110520/figures/04_time_chain.png" width="260"> | <img src="../19250113T120811/figures/04_time_chain.png" width="260"> | <img src="../19250120T115044/figures/04_time_chain.png" width="260"> |
| `04_trajectories.png` | <img src="../../figures/04_trajectories.png" width="260"> | <img src="../19250108T110520/figures/04_trajectories.png" width="260"> | <img src="../19250113T120811/figures/04_trajectories.png" width="260"> | <img src="../19250120T115044/figures/04_trajectories.png" width="260"> |
| `04_twin_maps.png` | <img src="../../figures/04_twin_maps.png" width="260"> | <img src="../19250108T110520/figures/04_twin_maps.png" width="260"> | <img src="../19250113T120811/figures/04_twin_maps.png" width="260"> | <img src="../19250120T115044/figures/04_twin_maps.png" width="260"> |

## 7. Cinematica: variazione della posizione contro velocità (`rep/kin_check.py`)

dkin[k] = (posizione[k+1] - posizione[k]) / 20 ms contro la velocità del file (target di FALCON). Tutti i campioni: come nel §1 del 04. Dentro i trial e in eval_mask: solo coppie di campioni consecutivi dello stesso trial, entrambi in eval_mask. Ritardo: quello che massimizza la correlazione tra -10 e +10 campioni (negativo: la velocità anticipa). Velocità ed eval_mask uguali a quelle di `load_nwb`; la posizione viene dallo stesso NWB.

| sessione | grado | r, tutti i campioni (come il 04) | r, dentro i trial e eval_mask (L = 0) | ritardo migliore (campioni da 20 ms) | r al ritardo migliore | r, media mobile 100 ms | r, media mobile 300 ms | pendenza dkin ~ vel | sd(dkin) / sd(vel) |
|---|---|---|---|---|---|---|---|---|---|
| rif. S1 | tx | 0.905 | 0.905 | -1 | 0.907 | 0.984 | 0.995 | 51.862 | 57.279 |
| rif. S1 | ty | 0.881 | 0.881 | -1 | 0.883 | 0.98 | 0.994 | 52.35 | 59.419 |
| rif. S1 | tz | 0.88 | 0.88 | -1 | 0.882 | 0.978 | 0.994 | 52.019 | 59.09 |
| rif. S1 | rx | 0.911 | 0.911 | -1 | 0.912 | 0.987 | 0.996 | 51.526 | 56.565 |
| rif. S1 | g1 | 0.729 | 0.736 | -2 | 0.739 | 0.945 | 0.988 | 53.348 | 72.464 |
| rif. S1 | g2 | 0.812 | 0.813 | -1 | 0.815 | 0.943 | 0.984 | 54.006 | 66.46 |
| rif. S1 | g3 | 0.875 | 0.875 | -1 | 0.878 | 0.972 | 0.988 | 55.37 | 63.27 |
| S2 | tx | 0.905 | 0.906 | -1 | 0.907 | 0.982 | 0.995 | 51.969 | 57.376 |
| S2 | ty | 0.911 | 0.911 | -1 | 0.912 | 0.983 | 0.996 | 51.736 | 56.781 |
| S2 | tz | 0.92 | 0.92 | -1 | 0.921 | 0.98 | 0.994 | 51.961 | 56.508 |
| S2 | rx | 0.922 | 0.921 | -1 | 0.922 | 0.987 | 0.996 | 51.451 | 55.847 |
| S2 | g1 | 0.843 | 0.846 | -2 | 0.849 | 0.964 | 0.99 | 53.202 | 62.894 |
| S2 | g2 | 0.807 | 0.808 | -1 | 0.81 | 0.944 | 0.984 | 54.347 | 67.238 |
| S2 | g3 | 0.908 | 0.908 | -1 | 0.911 | 0.976 | 0.988 | 55.453 | 61.089 |
| S5 | tx | 0.59 | 0.59 | -3 | 0.591 | 0.932 | 0.989 | 51.848 | 87.822 |
| S5 | ty | 0.764 | 0.764 | -1 | 0.765 | 0.972 | 0.994 | 51.745 | 67.728 |
| S5 | tz | 0.46 | 0.46 | -4 | 0.463 | 0.882 | 0.981 | 51.896 | 112.757 |
| S5 | rx | 0.763 | 0.764 | -2 | 0.766 | 0.974 | 0.995 | 51.529 | 67.487 |
| S5 | g1 | 0.696 | 0.696 | -1 | 0.698 | 0.944 | 0.987 | 53.3 | 76.551 |
| S5 | g2 | 0.678 | 0.679 | -1 | 0.681 | 0.923 | 0.982 | 54.111 | 79.699 |
| S5 | g3 | 0.863 | 0.863 | -1 | 0.866 | 0.973 | 0.987 | 55.479 | 64.282 |

<img src="../kin/kin_lag.png" width="820">

Angoli principali:

<img src="../angles/figures/angles_FA_10.png" width="420"> <img src="../angles/figures/angles_FA_15.png" width="420"> <img src="../angles/figures/angles_FA_25.png" width="420"> <img src="../angles/figures/angles_GPFA_10.png" width="420"> <img src="../angles/figures/angles_GPFA_25.png" width="420"> <img src="../angles/figures/angles_PCA_10.png" width="420"> <img src="../angles/figures/angles_PCA_25.png" width="420"> 
