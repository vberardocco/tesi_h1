# Tesi — FALCON H1 · manifold neurali e decoding motorio

Pipeline di analisi su **FALCON Benchmark H1** (DANDI:000954) — *Human 7DoF Reach and Grasp
Motor BCI*, Ye, Collinger & Gaunt (2024).

Registrazioni intracorticali (2 array da 96 canali, area della mano della corteccia motoria)
da un partecipante umano con interfaccia cervello-computer, durante sessioni di calibrazione
open-loop di movimenti di *reach-and-grasp* a 7 gradi di libertà.

Il progetto riprende la struttura dell'analisi svolta su FALCON B1 (canto del fringuello,
DANDI:001046), riadattandola al nuovo dataset.

## Contenuto

| Notebook | Contenuto |
|---|---|
| [`01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb) | Struttura dei file NWB, inventario dei 3 split, anatomia di un trial e delle sue fasi, raster + firing rate, cinematica 7 DoF, `eval_mask`, verifica della modulazione neurale legata al task |
| [`02_new.ipynb`](notebooks/02_new.ipynb) | Metodi lineari, passo per passo: dagli `spike_times` a binning, selezione dei canali, smoothing, z-score; **PCA**, PCA sulle medie per condizione, **Factor Analysis**, **GPFA**, con un'analisi di sensibilità per ogni scelta |
| [`03_geometric.ipynb`](notebooks/03_geometric.ipynb) | Metodi geometrici non lineari: **Isomap** (parte A) e **UMAP** (parte B), con gemelli senza curvatura come termine di confronto e riconoscimento delle fasi su trial mai visti |
| [`04_deep.ipynb`](notebooks/04_deep.ipynb) | Metodi "deep": **CEBRA** (parte A) e **LFADS** (parte B), confrontati con la GPFA alla pari; tabella finale di tutti i metodi |

## Corrispondenza con la review (`manifolds-3.pdf`)

| Sezione della review | Dove viene verificata |
|---|---|
| §2.4 — dimensionalità *ambient / embedding / intrinsic* | 02_new §6, §8, §9.3, §9.6; 03 §5 |
| §3.1 — PCA e i suoi limiti | 02_new §6-§8 |
| §3.2 — Factor Analysis, struttura condivisa vs privata | 02_new §9 |
| §3.3 — GPFA e il prior temporale | 02_new §10 |
| §3.5 — i metodi lineari sovrastimano la dimensionalità intrinseca | 02_new §9.6; 03 §5 |
| Sezione 4 — metodi non lineari (Isomap, UMAP, CEBRA, LFADS) | 03, 04 |
| §3.4 e §5.1 — jPCA, rotazioni (Churchland 2012), *tangling* (Russo 2018) | non presenti nei notebook attuali |
| §5.5 / §6.2 — stabilità del manifold tra sessioni e drift | non presenti nei notebook attuali |

## Struttura

```
tesi_h1/
├── data/raw -> ~/000954         # symlink al dandiset scaricato (40 file NWB)
├── src/tesi_h1/io.py            # caricamento NWB, binning, segmentazione in trial e fasi
├── notebooks/                   # notebook eseguiti, con output e figure incorporate
├── figures/                     # figure esportate in PNG
└── environment.yml
```

## Risultati principali

**Dataset** (notebook 01)

- 40 sessioni, 176 unità multi-unit **identiche in tutte le sessioni**, bin nativi da 20 ms,
  ~64 minuti totali. Split held-in (training/validazione) e held-out (few-shot recalibration
  su giorni successivi).
- Ogni trial (~16 s) è una sequenza stereotipata di fasi etichettate (Reach, Orient, SnapTo,
  Shape, Grasp, Carry, Orient2, Release) — ~15 ripetizioni per fase per sessione. Sono le
  condizioni discrete che su B1 mancavano.
- `eval_mask` esclude quasi solo la fase `SnapTo`, in cui il braccio virtuale si riposiziona
  automaticamente: non è un movimento tentato dal partecipante.
- Il segnale c'è ma è **distribuito**: modulazione di popolazione fase-specifica, ma η² mediano
  0.015 per singola unità. È la premessa dell'analisi a livello di popolazione.

**Metodi lineari** (notebook 02_new, sessione `held-in-calib` 19250108T110520)

- **Il task alterna due versioni** tra trial dispari e pari (scoperto nel 03, §9.2): le ripetizioni
  di una fase non sono repliche. Medie e consistenza usano quindi **16 condizioni**, fase × versione,
  con 7-8 ripetizioni ciascuna.
- **Dimensionalità.** La PCA sui dati single-trial (σ = 50 ms) ha un participation ratio ≈ 106, ma
  sono soprattutto rumore: a parità di punti, mediando le ripetizioni allineate di ogni condizione
  il PR scende da ~50 a ~21 (con finestre casuali non scende). Con più ripetizioni scenderebbe
  ancora: ~20 è un tetto.
- **Factor Analysis** (conteggi grezzi a 20 ms): in un canale tipico solo il ~4% della varianza è
  condiviso con la popolazione; K\* = 15 fattori per cross-validation. La parte condivisa ha ~9.5
  dimensioni efficaci (~7 sugli stessi istanti delle medie), poche e stabili rispetto allo
  smoothing. Ma le traiettorie single-trial della FA non sono migliori di quelle della PCA:
  modellare il rumore non basta, serve modellare il tempo.
- **GPFA**: 25 fattori per cross-validation, scale di tempo da ~50 ms a centinaia di ms; ~12
  dimensioni efficaci. È il metodo lineare con le traiettorie single-trial migliori: seghettatura
  0.011, consistenza 0.137 contro 0.117 della PCA smussata e ~0.04 di PCA e FA sui conteggi grezzi.

**Metodi geometrici** (notebook 03)

- Il grafo dei vicini sui dati smussati è una catena del tempo: serve sottocampionare (un bin ogni
  100 ms) prima di Isomap e UMAP.
- **C'è struttura non lineare, ma modesta**: Isomap riproduce le distanze un po' meglio della PCA
  (+3.6 punti con 3 dimensioni), mentre su un gemello lineare e gaussiano perderebbe di ~10. Nessuna
  dimensione intrinseca piccola e netta nei dati single-trial.
- **UMAP** conserva in 2-3 coordinate l'informazione sulle fasi (42.6% contro 32.4% della PCA, a
  caso 12.5%), al livello del classificatore sulle 25 dimensioni, ma non aggiunge informazione: si
  usa per mostrare, non per misurare.

**Metodi deep** (notebook 04)

- **CEBRA-Time** non batte UMAP sul metro comune (37.7% contro 41.2% con 2 coordinate libere);
  CEBRA-Behavior con posizione e velocità fa peggio. Il tempo dall'inizio del trial, da solo,
  riconosce la fase al 48.5%.
- **LFADS**, con 12 trial di training, resta sotto la GPFA alla pari sia sulla fase (46.5% contro
  51.4%) sia sui canali nascosti (0.0388 contro 0.0422 bit per spike). Con il controller supera tutti
  sui canali nascosti (0.0631), perché segue un ritmo a ~18 Hz comune alla popolazione; con 28 trial
  raggiunge la GPFA sulla fase.
- **Quello che conta in questi dati è modellare il tempo, non la non linearità**: dalla FA alla
  GPFA si guadagnano ~25 punti sulla fase, la rete non lineare non aggiunge niente.

## Ambiente

```bash
conda env create -f environment.yml
conda activate tesi-manifolds
jupyter lab
```

I notebook sono già eseguiti: si aprono con tutti gli output e le figure visibili, senza bisogno
di rilanciarli. Per rieseguirli serve il dandiset in `data/raw` (symlink incluso). Tempi: 02_new
~20 minuti (cross-validation di FA e GPFA), 03 ~7 minuti, 04 ~15 minuti se i modelli allenati sono
già salvati su disco (senza, molte ore).

## Dati

```bash
dandi download DANDI:000954
```

L'uso dei dati è soggetto al Data Use Agreement di FALCON H1 (dati de-identificati, nessuna
redistribuzione).
