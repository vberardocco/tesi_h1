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

## Stabilità dei sottospazi FA sulle held-out (aggiunta del 2/11/2026)

Scritta il 2/11/2026 **prima** di calcolare qualsiasi indice o angolo sui blocchi da 3 trial (held-in o held-out).
Già noti a chi scrive: l'R² zero-shot per sessione (`sessions_r2.csv`) e gli angoli tra S1, S2 e S5 sulle sessioni
intere (repliche, commit 1d032e9), dove il caso alla Elsayed per la FA è risultato quasi uniforme. Codice:
`src/tesi_h1/stability.py`, `scripts/xs_stability.py`; prove in `tests/test_stability.py`. Nessun numero di questa
analisi entra nelle scelte del protocollo cross-sessione.

### Blocchi, canali e normalizzazione
- **Blocco = 3 trial consecutivi.** Held-out: i 3 trial di calib di ogni file (14 blocchi). Held-in: blocchi non
  sovrapposti dall'inizio di ogni file, gli stessi della diagnosi dentro la sessione (`diag_within`, 54 blocchi; i
  trial che avanzano restano fuori). Così held-in e held-out sono stimati con lo stesso numero di trial.
- **Canali: gli stessi 170** attivi sulle held-in, in tutti i blocchi.
- **FA come nel decoder**: K = 15, `svd_method="lapack"`, `random_state=0`, conteggi grezzi.
- **Due normalizzazioni**, perché lo span delle loading cambia se si riscalano i canali:
  a) *blocco contro blocco*: z-score dentro il blocco (media e sd del blocco), come nelle repliche;
  b) *blocco contro la FA congelata del decoder*: la normalizzazione che il decoder applica alle held-out in
     zero-shot, cioè media e sd delle 13 held-in del modello `FA_d15_heldin_s0` (la media non cambia le loading,
     conta la sd). Per i blocchi held-in di riferimento si usa la FA stimata senza il loro giorno (leave-one-day-out,
     stessa ricetta), con la sua normalizzazione.
- **Canali silenti.** In a) la sd di un canale nel blocco non scende sotto sd_min = 0.1, cioè la sd di Poisson di un
  canale a 0.5 Hz (la soglia dei canali attivi) in bin da 20 ms: √(0.5 · 0.02) = 0.1. Così un canale con sd nulla non
  divide per zero e uno con 1-2 spike non diventa un valore z enorme. In b) la sd è quella held-in e un canale silente
  nel blocco resta una colonna costante (la FA gli dà loading nulle; verificato su dati sintetici). Si riportano
  quanti (canale, blocco) hanno sd = 0 e quanti sd < sd_min.

### Misure
- L = loading del blocco (170 × 15), S = L Lᵀ la sua covarianza condivisa.
- **Indice di allineamento (primario)**, da Elsayed et al. 2016 applicato a S:
  I(A→B) = tr(Q_Bᵀ S_A Q_B) / Σ_{i≤k} λ_i(S_A), con Q_B la base ortonormale dei primi k autovettori di S_B.
  Indice della coppia = ½ [I(A→B) + I(B→A)]. Vale 1 se ciascun sottospazio cattura tutta la varianza condivisa
  dell'altro che k direzioni possono catturare. Con k = 15, Q_B è una base di span(L_B) e il denominatore è tr(S_A).
- **Angolo principale medio (secondario)**: media dei k angoli principali tra span(Q_A) e span(Q_B), in gradi.
- **Sensibilità a k = 5**: i primi 5 autovettori di L Lᵀ dal fit con K = 15, senza rifare la FA con K = 5. Le
  loading FA non hanno un ordine (sono definite a meno di una rotazione); le direzioni si ordinano per varianza
  condivisa.

### Caso alla Elsayed
- Sottospazi casuali Q_r = orth(U Λ^½ r), r gaussiano 170 × k, da una covarianza C = U Λ Uᵀ. Per la coppia (A, B):
  caso dell'indice = ½ [I(A→Q_r1) + I(B→Q_r2)], con Q_r1 e Q_r2 indipendenti; caso dell'angolo = angolo medio tra
  Q_r1 e Q_r2. 500 estrazioni per coppia, seme fisso.
- Tre C, come nelle repliche:
  - *tutti i bin*: media delle covarianze dei dati dei due elementi della coppia, nella normalizzazione del confronto;
  - *medie per condizione*: covarianza delle medie delle 16 condizioni (8 fasi × versione, finestre da 1 s
    dall'inizio della fase), calcolata in ogni sessione held-in e mediata sulle 13; in a) con lo z-score della
    sessione, in b) con la normalizzazione del decoder. È la stessa C per tutte le coppie;
  - *uniforme*: identità.
- Per l'indice (più alto = più allineato) conta il **95° percentile**; per l'angolo il 5°.

### Confronti
1. **Riferimento dentro la sessione**: le coppie di blocchi della stessa sessione held-in (95 coppie). È il massimo
   di allineamento che si può misurare con 3 trial.
2. **Stesso giorno, file diverso**: coppie di blocchi di due file dello stesso giorno, held-in e held-out.
3. **Tra giorni**: tutte le altre coppie, in funzione di |Δ giorni| (date di `sessions.csv`, da 0 a 39).
4. **Held-out contro la FA congelata del decoder** (normalizzazione b): è l'asse x primario della figura F, perché è
   la FA che produce lo zero-shot. Riferimento: ogni blocco held-in contro la FA leave-one-day-out del suo giorno.
5. **Held-out contro i 54 blocchi held-in** (normalizzazione a): mediana per held-out, descrittiva.

### Regola sulla stimabilità (fissata ora)
Si confronta la mediana dell'indice sulle 95 coppie del riferimento dentro la sessione, a k = 15, con la mediana sulle
stesse coppie del 95° percentile del caso, per ognuna delle tre versioni. Se non le supera tutte e tre, il sottospazio
FA a K = 15 non è stimabile da 3 trial: lo si scrive come risultato, e test e figure si leggono sulla sensibilità a
k = 5 (calcolata in ogni caso), con la stessa regola. Lo stesso controllo, descrittivo, per l'angolo (riferimento sotto
il 5° percentile del caso).

### Test
- **Stabilità (figura E): test di Mantel.** Statistica: ρ di Spearman tra l'indice e |Δ giorni| su tutte le coppie di
  blocchi **tranne le 95 dentro la stessa sessione**, che sono la banda di riferimento: se restassero a distanza 0 il
  test misurerebbe "stessa sessione o no" invece del calo con i giorni. Le coppie stesso giorno, file diverso restano
  a distanza 0. p: permutazione delle date tra i 13 giorni (le coppie di uno stesso giorno non sono indipendenti),
  10 000 permutazioni, seme 0, a due code. Un solo test, su tutti i blocchi; i tre tipi di coppia (held-in–held-in,
  held-in–held-out, held-out–held-out) solo descrittivi.
- **Pendenza per giorno**: minimi quadrati dell'indice su |Δ giorni|, sulle stesse coppie del Mantel. IC 95% con
  bootstrap sui giorni (2 000 ricampionamenti, seme 0): si ricampionano i 13 giorni con ripetizione, si tengono le
  coppie tra copie di giorni diversi e le coppie stesso giorno, file diverso dentro ogni copia.
- **Figura F: ρ di Spearman** tra l'indice contro la FA congelata (k = 15) e l'R² zero-shot della FA (n = 14),
  p per permutazione (10 000, seme 0, a due code). Ripetuto sulle medie per giorno (n = 7) e come Spearman parziale
  rispetto ai giorni dall'ultima held-in (residui dei ranghi dopo la regressione sui ranghi dei giorni, p per
  permutazione dei residui). È l'unico test sulla relazione.
- **Descrittivi**: lo stesso indice contro lo zero-shot di PCA, GPFA, LFADS e ridge_channels e contro il guadagno
  align − recenter della FA; l'indice mediano contro i blocchi held-in (confronto 5) contro lo zero-shot della FA.
- **Potenza.** Con n = 14 è significativo (Spearman, α = 0.05 a due code) solo |ρ| ≳ 0.54; con le medie per giorno
  (n = 7) solo |ρ| ≳ 0.79. Un risultato nullo si legge "relazione non rilevabile con questi dati", non "nessuna
  relazione". Lo script ricalcola le soglie esatte per permutazione e le riporta.

### Figure e file
- **Figura E**: indice (k = 15) contro |Δ giorni|; punti = mediana per coppia di giorni, colore per tipo di coppia;
  banda del riferimento dentro la sessione (IQR delle 95 coppie) e linee del 95° percentile del caso; pannello con
  l'angolo medio. **Figura F**: indice contro la FA congelata in ascissa, R² zero-shot della FA in ordinata (14 punti,
  colore = giorni dall'ultima held-in), con la banda del riferimento held-in (blocchi contro la FA leave-one-day-out);
  pannello con align − recenter; pannello descrittivo con gli altri metodi.
- **Registro** (`h1_runs.csv`): esperimento `fa_stability`, method `FA`, latent_dim 15, decoder NA, split
  `calib3_block`, k_shot 3 (trial che stimano il sottospazio), H_ms NA, seme 0, fold NA; condition `vs_frozen_fa`
  (held-out), `vs_lodo_fa` (blocchi held-in, sessione `S0_set_1#b0` ecc.), `vs_heldin_blocks` (mediana per held-out).
  Metriche: indice e angolo a k = 15 e 5, percentili del caso. Le coppie in
  `results/crosssession/stability/pairs.csv`, test e riassunti in `results/crosssession/stability/`.
- **Limiti da dichiarare**: 14 sessioni su 7 giorni; l'indice usa tutti e 3 i trial, compresi quelli su cui si valuta
  lo zero-shot (è una correlazione descrittiva, sull'indice non si stima niente); i blocchi held-out sono i primi 3
  trial del file, quelli held-in vengono da tutto il file.
