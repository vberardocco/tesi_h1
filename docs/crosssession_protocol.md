# Protocollo cross-sessione few-shot su FALCON H1 — scelte e piano dell'analisi

Scritto il 31/10/2026 **prima** di guardare i risultati sulle held-out (unica eccezione: una prova del codice con la
PCA in una cartella temporanea, i cui numeri non sono stati usati per nessuna scelta). Codice:
`src/tesi_h1/crosssession.py`, `scripts/xs_run.py`, `scripts/xs_analysis.py`; prove in `tests/test_alignment.py`.

## Dati
- 13 sessioni held-in calib (S0-S5) e 14 held-out calib (S6-S12, 2 file per giorno), bin di FALCON
  (`io.load_session(bins="falcon")` = `load_nwb`). Ogni held-out ha esattamente 3 trial (controllato).
- Canali: i 170 su 176 con frequenza media ≥ 0.5 Hz sulle held-in (stesso criterio del 02_new), per tutti i metodi.
- **Ogni trial è una sequenza a sé**: smoothing, pezzi di GPFA e LFADS, finestre di CEBRA e storia del decoder stanno
  dentro il trial. L'adattamento (`adapt`) riceve solo i 2 trial di adattamento; il trial di test entra solo in
  `predict_trial`. Il primo secondo di ogni trial ha una storia del decoder incompleta (zeri = media), uguale per tutti.

## Metodi e iperparametri congelati (dai notebook)
| metodo | ingresso | dim. latente | impostazioni |
|---|---|---|---|
| PCA | rate smussati σ 50 ms, z-score | 25 | stimata sui bin delle fasi di movimento (03, 04 §1) |
| FA | conteggi z-scorati | 15 | K* del 02_new §9, `svd_method="lapack"` |
| GPFA | radice dei conteggi | 25 | 02_new §10: finestre da 1 s all'inizio delle fasi di movimento, 500 iterazioni, τ₀ 100 ms; latenti ortonormalizzati da pezzi di 2 s ogni 1 s (04 §7b) |
| Isomap | 25 PC | 3 | k = 10, un bin di movimento ogni 5 (03); sulle held-in al più 5000 punti a caso (memoria) |
| UMAP | 25 PC | 3 | n_neighbors 10, min_dist 0.1, stessi punti di Isomap |
| CEBRA | 25 PC | 3 | CEBRA-Time del 04: offset10-model, batch 512, lr 3e-4, τ 1, Δ 10, coseno, 1000 passi, seme corretto |
| LFADS | conteggi | 25 | LF_HP del 04 §11b-§12: gen 64, dropout 0.05, KL × 0.3, CD 0, 200 epoche, pezzi 600 ms ogni 400 ms |

Il **readout** è il decoder lineare del 05 (ridge con 1000 ms di storia causale) applicato ai latenti standardizzati
con media e deviazione held-in. Il suo α si sceglie leave-one-day-out sulle held-in. Semi: 3 per UMAP, CEBRA, LFADS
(0, 1, 2); 1 per i metodi deterministici.

## Condizioni (per ogni fold: 2 trial adattano, 1 testa)
1. `zero_shot`: modello e readout held-in così come sono.
2. `recenter`: ingresso del metodo meno la media dei 2 trial più la media held-in.
3. `align`: recenter + moment matching sui latenti, f(z) = (z − μ_a) C_a^(−1/2) C_s^(1/2) + μ_s (radici simmetriche via
   eigh, eps = 1e-8 × autovalore medio sulla diagonale); μ_a, C_a dai latenti dei 2 trial, μ_s, C_s dai latenti held-in.
   Per FA anche `fa_native`: FA (K = 15) sui 2 trial, loadings ruotate su quelle held-in con Procrustes ortogonale.
4. `fewshot`: align + readout ristimato sui 2 trial con min ||Y − [Z 1]B||² + λ||B − B_src||² (intercetta compresa:
   per λ → ∞ si torna ad align).
5. `scratch`: metodo e readout stimati solo sui 2 trial, stessa dimensione latente (non per LFADS: 2 trial sono ~80
   pezzi da 600 ms, troppo pochi per allenare la rete; il 04 §13b mostra che LFADS peggiora già con 4 trial).
- Riferimento comune `ridge_rates` (method "rates"): ridge con la stessa storia sui rate smussati (σ 50 ms) dei 2 trial.

## Scelta di λ e α (senza toccare le held-out)
Pseudo-few-shot sulle held-in: metodo e readout stimati sui giorni S0-S3; pseudo-sessioni = blocchi di 3 trial
consecutivi dei 4 file di S4 e S5 (20 pseudo-sessioni), stessa rotazione e stesse condizioni. λ (fewshot), α (scratch)
e α (ridge_rates) = valore della griglia (passo mezzo decennio) con la media più alta dell'R² concatenato. Scelti con
il seme 0 e usati per tutti i semi. Minival non si usa.

## Metrica e unità statistica
R² pesato per varianza (= `r2_score(multioutput="variance_weighted")` di falcon_challenge) sui bin con eval_mask:
per fold e sulle predizioni concatenate dei 3 fold (`fold = "concat"`), che è la misura principale. Per i metodi con
più semi si fa prima la media sui semi. Unità statistica: la sessione (n = 14).

## Analisi (fissata prima dei risultati)
- **Tabella**: mediana e IQR dell'R² concatenato per metodo × condizione, con n.
- **Figura A**: un pannello per metodo, R² in funzione dei giorni dall'ultima held-in, una curva per condizione
  (media dei due file di ogni giorno, punti = sessioni), ridge_rates come linea comune.
- **Figura B**: condizione fewshot, una curva per metodo in funzione dei giorni.
- **Test dentro ogni metodo**: Wilcoxon appaiato (a due code) tra condizioni consecutive: zero_shot→recenter,
  recenter→align, align→fewshot, fewshot→scratch (LFADS: le prime tre); per FA anche align→fa_native. Correzione di
  Holm dentro ogni metodo. Si riportano W, p grezzo e corretto, effect size = correlazione rank-biserial appaiata e
  differenza mediana.
- **Test tra metodi**, alle condizioni align e fewshot: Friedman con le sessioni come blocchi (χ², p, W di Kendall),
  poi Wilcoxon a coppie con Holm dentro ogni condizione (stesse misure). Se LFADS non è completo, il Friedman si fa
  sui metodi completi e LFADS si riporta a parte.
- **Descrittivo** (non è un test d'ipotesi): pendenza dell'R² zero-shot contro i giorni (minimi quadrati, con IC 95%)
  e ρ di Spearman, per metodo.
- **Bin di adattamento**: per sessione e fold, bin dei 2 trial (tutti e con eval_mask) rispetto ai 170 canali, alla
  dimensione latente e al numero di pesi del readout (H × dim + 1).
