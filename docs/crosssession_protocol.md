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

## Deviazioni e aggiunte (seconda tornata, 31/10/2026 pomeriggio, richieste dopo i risultati della mattina, commit 48ca831)

Scritte prima di vedere i risultati della seconda tornata (codice: commit 95122b6). Tutto ciò che segue è un'aggiunta
al piano della mattina del 31/10; i confronti nuovi sono marcati `added_v2` nelle tabelle.

1. **Scelta di λ: era già senza la pseudo-sessione.** Verificato nel codice (`tune`, `fit_bundle(..., "pseudo")`):
   per tutti e 7 i metodi, CEBRA e LFADS compresi, il metodo (manifold) e il readout dello pseudo-few-shot sono
   stimati solo sui giorni S0-S3; le pseudo-sessioni vengono da S4 e S5, quindi ogni pseudo-sessione è esclusa da
   entrambi. È più severo del leave-one-session-out (sono esclusi i due giorni interi). Nessuna correzione.
2. **Griglia di λ allargata** da 10^-1 - 10^9 a 10^-2 - 10^15 (passo mezzo decennio). Si salva la curva R²(λ) di ogni
   metodo (`tune/*.lambda_curve.csv`, `lambda_curves.csv`, figura C) e la si confronta con l'asintoto λ → ∞, che è
   align (per ridge_channels: recenter). Regola: se l'ottimo resta all'estremo alto anche così, l'ottimo è λ → ∞ e
   fewshot coincide con align; si riporta come risultato, non si allarga oltre. λ e α si scelgono di nuovo per tutti i
   metodi (anche per quelli con l'ottimo già interno), con la griglia nuova; i risultati del 31/10 per PCA, FA, GPFA,
   LFADS sono sostituiti nel registro (stesse chiavi: vale l'ultima riga).
3. **fewshot_gain** (nuova condizione): sopra align, per ogni DoF dell'uscita y ≈ g·ŷ + o con i minimi quadrati sui bin
   dei 2 trial di adattamento (2 parametri per DoF, nessun iperparametro). Test aggiunto in ogni metodo:
   align→fewshot_gain (Holm dentro il metodo, insieme agli altri).
4. **ridge_channels** (nuovo riferimento): il decoder ridge causale del 05 (`models/ridge_h50.npz`, commit b797b3e:
   176 canali, z-score per canale con media e deviazione held-in, 1000 ms di storia, α 10^4.5, stimato su tutto il calib
   held-in) con zero_shot, recenter (media dei 2 trial al posto di quella held-in), fewshot (recenter + ridge verso i
   suoi pesi, λ scelto con lo pseudo-few-shot) e fewshot_gain (recenter + guadagno e offset). Per lo pseudo-few-shot
   lo stesso decoder è ristimato con la stessa ricetta sui giorni S0-S3. Differenza dal 05: qui si applica trial per
   trial come tutti gli altri (la storia riparte a ogni trial). Test: zero_shot→recenter, recenter→fewshot,
   recenter→fewshot_gain; entra nei confronti tra metodi a fewshot e fewshot_gain (non ad align, che non ha).
5. **Mappe a dimensione confrontabile.** Isomap, UMAP e CEBRA erano a d = 3 (scelta del 03/04 per riconoscere le fasi).
   La regola usata per gli altri metodi senza una verosimiglianza da validare (PCA nel 03 §2, LFADS nel 04) è stata
   prendere il d della GPFA scelto con la cross-validation sulle held-in (02_new §10): **d = 25**. FA ha la sua
   cross-validation (K* = 15). Si applica la stessa regola alle tre mappe: d = 25, con tutti gli altri iperparametri
   invariati. Le mappe a d = 3 si rifanno con λ, griglia e fewshot_gain nuovi e si registrano con esperimento
   `crosssession_d3`; le loro righe del 31/10 (esperimento `crosssession`, latent_dim 3) restano nel registro ma
   l'analisi le esclude filtrando la dimensione. Friedman e confronti a coppie si rifanno con le mappe a d = 25
   (analisi principale) e, come variante, con quelle a d = 3.
6. **Diagnosi del moment matching** (esplorativa). In ognuna delle 13 sessioni held-in, blocchi di 3 trial consecutivi
   con la stessa rotazione delle held-out (2 trial adattano, 1 testa); sorgente = gli altri trial della stessa
   sessione: readout ristimato su di loro (α held-in scalato per il numero di bin), momenti e media d'ingresso dai loro
   latenti. Il metodo (manifold) resta quello held-in. Condizioni no_adapt, recenter, align; esperimento
   `crosssession_mmdiag`, seme 0, dimensioni principali. Test: Wilcoxon align vs recenter (n = 13), Holm sui 7 metodi.
   Regola fissata ora: se align peggiora anche dentro la sessione, dove non c'è deriva tra giorni, il calo sulle
   held-out si attribuisce al contenuto comportamentale dei 2 trial e non alla deriva.
7. **Copertura del comportamento** (descrittiva). Per ogni fold, nei 2 trial di adattamento: fasi di movimento (8),
   condizioni fase × versione del task (16; la versione è la parità del trial, verificata dal segno di g3 in Grasp su
   tutti i trial held-in e held-out), direzioni di movimento DoF × verso (14; coperta con almeno 10 bin oltre il 75°
   percentile di |v| del pool held-in), varianza della velocità rispetto al pool. Si confrontano held-out,
   pseudo-sessioni e blocchi della diagnosi. In più, esplorativo: Δ align − recenter per fold, separando i fold con i
   2 trial della stessa versione.

### Note dopo l'esecuzione della seconda tornata (31/10/2026 sera)

Tutti i run della seconda tornata sono finiti senza fallimenti (code v2, 16:01-17:38; codice 95122b6 per
ridge_channels, c6efc91 per il resto: differiscono solo in `xs_analysis.py` e in questo file). I risultati di PCA, FA,
GPFA e LFADS alle condizioni della mattina sono identici al bit (stesso λ per PCA e GPFA), tranne fewshot di FA e LFADS,
dove λ passa da 10^9 a 10^15 (entrambi all'asintoto) e l'R² cambia al più di 4·10^-5.

8. **La data delle aggiunte era sbagliata.** Nelle prime versioni di questo file, del codice e delle tabelle le
   aggiunte erano datate "1/11" e marcate `added_20261101`; sono del 31/10 pomeriggio (log delle code). Corretto in
   `added_v2` e "seconda tornata"; nessun numero cambia.
9. **Griglia di λ, esito della regola del punto 2.** FA e LFADS hanno l'ottimo all'estremo alto anche con la griglia
   fino a 10^15, con la curva che sale in modo monotono verso l'asintoto: per la regola, fewshot = align. Per tutti gli
   altri l'ottimo è interno (PCA 10^6.5, GPFA 10^7, Isomap 25 10^4.5, UMAP 25 10^5.5, CEBRA 25 10^6, mappe a d = 3
   10^5-10^5.5, ridge_channels 10^5).
10. **Due correzioni grafiche, nessun numero cambia.** Figura A: il limite inferiore dell'asse y tagliava lo zero-shot
    di ridge_channels nei giorni 6 e 20; ora comprende tutti i punti. Figura C: il marcatore del λ scelto mancava per
    PCA, Isomap 25, UMAP 25 e CEBRA 3 (i λ a mezzo decennio): le chiavi della curva nel JSON sono arrotondate a 4 cifre
    ("3162") e il confronto con il λ scelto era esatto; ora la tolleranza relativa è 10^-3 (due punti vicini della
    griglia differiscono di un fattore 3.16). Nella figura C l'asse y è tagliato a −0.08,
    ed è scritto sotto l'asse.
11. **Controlli esplorativi aggiunti DOPO aver visto i risultati** (`scripts/xs_checks.py`, commit 81f5a1a; non nel
    piano, nessun test d'ipotesi). Due risultati andavano spiegati prima di scriverne:
    a) fewshot_gain peggiora align in quasi tutte le sessioni e in tutti i metodi. Scomposizione sulle held-out (seme 0,
       dimensioni principali, stessi fold): solo guadagno, solo offset, e un guadagno stimato sul trial di test stesso
       (limite superiore, non una condizione); in più il guadagno stimato su ogni trial da solo.
    b) UMAP e CEBRA restano a R² ≈ 0 anche a d = 25. Nelle 13 sessioni held-in, con il modello held-in: decodifica
       dentro la sessione (readout sui trial tranne gli ultimi 3, α leave-one-trial-out, test sugli ultimi 3) e quota
       della varianza dei latenti spiegata dall'identità della sessione.
    File: `check_gain_decomposition.csv`, `check_gain_per_trial.csv`, `check_within_session.csv`,
    `check_session_variance.csv`; sezione finale di `report.md`.
12. **Limite della diagnosi del moment matching per UMAP e CEBRA.** Il readout della diagnosi usa l'α held-in scalato
    per il numero di bin (punto 6); per UMAP e CEBRA quell'α è all'estremo alto della griglia (il readout held-in predice
    la media), quindi dentro la sessione partono da R² ≈ 0.01-0.02 e la diagnosi dice poco su di loro.
