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
| [`02_baseline_manifolds.ipynb`](notebooks/02_baseline_manifolds.ipynb) | Manifold con i metodi lineari: dimensionalità *ambient/embedding/intrinsic*, **PCA**, **Factor Analysis**, **GPFA**, dinamica rotazionale (**jPCA**) e *tangling*, stabilità del manifold tra sessioni |

## Corrispondenza con la review (`manifolds-3.pdf`)

Il notebook 02 verifica sui dati i punti sviluppati nella parte teorica:

| Sezione della review | Dove viene verificata |
|---|---|
| §2.4 — dimensionalità *ambient / embedding / intrinsic* | 02 §1 (stimatore two-NN, validato su dati sintetici) |
| §3.1 — PCA e i suoi limiti | 02 §1, §3, §4 |
| §3.2 — Factor Analysis, struttura condivisa vs privata | 02 §2 |
| §3.3 — GPFA e il prior temporale | 02 §7, §8, §9 |
| §3.4 — metodi mirati: jPCA | 02 §6 |
| §3.5 — i metodi lineari sovrastimano la dimensionalità intrinseca | 02 §1 |
| §5.1 — corteccia motoria: rotazioni (Churchland 2012), *tangling* (Russo 2018) | 02 §6 |
| §5.5 / §6.2 — stabilità del manifold e drift temporale | 02 §10 |

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

**Manifold** (notebook 02, sessione `held-in-calib` 19250108T110520)

- **Dimensionalità, due contrasti.** Single-trial vs media per condizione: participation ratio
  ≈ 106 contro ≈ 11 — il segnale è a bassa dimensione, il rumore no. Embedding vs intrinsic:
  la PCA richiede ~106 dimensioni effettive dove il two-NN ne stima ~22 → **il manifold è curvo**.
- **Factor Analysis**: solo il ~5% della varianza di ciascuna unità è condivisa con la popolazione
  (conteggi a 20 ms); K\* ≈ 30 fattori condivisi. Ma le traiettorie single-trial di FA non sono
  migliori di quelle della PCA: modellare il rumore non basta, serve modellare il tempo.
- **GPFA**: `d*` ≈ 20 per cross-validation della log-likelihood held-out (concorde con FA);
  traiettorie molto più lisce e consistenti a parità di ingresso.
- **jPCA / tangling**: l'86% della dinamica lineare è rotazionale (~0.5 Hz); il *tangling* è
  ~100× inferiore a un controllo con il tempo rimescolato.
- **Stabilità tra sessioni**: alignment index 0.63 (stesso giorno) → 0.49 (settimane), contro
  0.26 atteso per caso. Substrato stabile + drift sistematico = la motivazione della
  ricalibrazione few-shot del benchmark.

## Ambiente

```bash
conda env create -f environment.yml
conda activate tesi-manifolds
jupyter lab
```

I notebook sono già eseguiti: si aprono con tutti gli output e le figure visibili, senza bisogno
di rilanciarli. Per rieseguirli serve il dandiset in `data/raw` (symlink incluso); il notebook 02
richiede ~6 minuti, quasi tutti nella cross-validation di GPFA.

## Dati

```bash
dandi download DANDI:000954
```

L'uso dei dati è soggetto al Data Use Agreement di FALCON H1 (dati de-identificati, nessuna
redistribuzione).
