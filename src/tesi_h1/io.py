"""
Caricamento e preparazione dei dati FALCON H1 (DANDI:000954).

FALCON H1 = "Human 7DoF Reach and Grasp Motor BCI" (Ye, Collinger, Gaunt, 2024).
Attivita' neurale registrata da 2 array intracorticali da 96 canali (area "hand knob",
corteccia motoria) in un partecipante umano con iBCI, durante sessioni di calibrazione
*open loop*: un cue audiovisivo indica un movimento di reach-and-grasp che il partecipante
tenta di eseguire, mentre un braccio virtuale mostra la cinematica di riferimento.

Struttura dei file NWB (verificata sui 40 file del dandiset):
  - units (176 unita')                : multi-unit spiking activity, solo `spike_times` (s).
  - acquisition['OpenLoopKinematics'] : (T, 7) posizione della cinematica di riferimento.
                                        Colonne: tx, ty, tz (traslazione 3D dell'end-effector),
                                        rx (orientamento/roll), g1, g2, g3 (3 gradi di presa).
  - acquisition['OpenLoopKinematicsVelocity'] : (T, 7) velocita' = **target ufficiale FALCON**.
  - acquisition['TrialNum']           : (T,) indice del trial a cui appartiene ogni bin.
  - acquisition['eval_mask']          : (T,) bool, True = timestep da usare in valutazione.
  - intervals['epochs']               : fasi del task (Presentation, Reach, Orient, SnapTo,
                                        Shape, Grasp, Carry, Release, Intertrial) con
                                        start_time / stop_time / tags.

ATTENZIONE al campo `rate`: NWB lo dichiara come 0.02, ma è il **passo temporale in secondi**
(bin da 20 ms → 50 Hz), non una frequenza. Il loader ufficiale di FALCON fa la stessa
assunzione (`timestamps = offset + arange(T) * rate`), quindi la manteniamo per coerenza.

Convenzioni di questo modulo:
  - Le matrici neurali sono sempre (n_time_bins, n_units).
  - Il binning e' a 20 ms, identico a quello del benchmark: nessun ricampionamento nascosto.

Binning: quale finestra di 20 ms finisce nella riga k.
  Sia t_k = offset + k * 20 ms l'istante del campione k di cinematica (offset = 0 in tutti i
  file di H1). La riga k dei conteggi si abbina sempre al campione k (kin, vel, trial_num,
  eval_mask hanno lo stesso indice); cambia solo quale finestra di tempo conta:

  - bins="falcon" (predefinita dal 30/10/2026): il bin k contiene gli spike in
    [t_k - 20 ms, t_k). Sono gli stessi conteggi di `falcon_challenge.dataloaders.load_nwb`
    (verificato su tutti i 40 file). Il bin k finisce all'istante del campione k, quindi un
    decoder che usa i bin fino al k-esimo vede solo spike precedenti al campione: e' causale.
  - bins="legacy": il bin k contiene gli spike in [t_k, t_k + 20 ms), cioe' i 20 ms DOPO il
    campione. E' la convenzione di questo modulo fino al 30/10/2026 (il commento nel codice
    diceva il contrario). Riga k legacy = riga k+1 falcon. Serve solo a rileggere le cache del
    notebook 04, le cui chiavi dipendono dai conteggi.

  Le funzioni che passano da un istante a un bin (`phase_windows`, `windows_spiketrains`)
  leggono la convenzione dalla `Session` (`session.bins`), tramite `time_to_bin` e `bin_start`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# Split del dataset FALCON H1 (nomi delle cartelle "sub-HumanPitt-<split>")
SPLITS = ("held-in-calib", "held-in-minival", "held-out-calib")

# Passo temporale nativo del benchmark: 20 ms
BIN_S = 0.02

# Convenzioni di binning (vedi la docstring del modulo); la prima e' quella predefinita
BINS = ("falcon", "legacy")

# Nomi dei 7 gradi di liberta' (dal campo `description` dei TimeSeries)
KIN_LABELS = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]
KIN_LONG = [
    "traslazione x", "traslazione y", "traslazione z", "orientamento (roll)",
    "presa g1 (pollice)", "presa g2 (indice)", "presa g3 (anulare/mignolo)",
]

# Fasi del task che comportano movimento effettivo (escludiamo Presentation* e Intertrial,
# in cui il partecipante osserva il cue e la cinematica e' ferma).
MOVEMENT_PHASES = ("Reach", "Orient", "SnapTo", "Shape", "Grasp", "Carry", "Orient2", "Release")

# Colori stabili per fase, usati in tutte le figure
PHASE_COLORS = {
    "Reach": "#1f77b4", "Orient": "#ff7f0e", "SnapTo": "#2ca02c", "Shape": "#d62728",
    "Grasp": "#9467bd", "Carry": "#8c564b", "Orient2": "#e377c2", "Release": "#7f7f7f",
    "Presentation": "#c7c7c7", "Intertrial": "#dddddd",
}


@dataclass
class Session:
    """Una sessione FALCON H1 caricata in memoria (i file sono <7 MB: si carica tutto)."""

    path: Path
    split: str
    session_id: str                 # es. "19250108T110520" (data anonimizzata dal DANDI)
    spikes: np.ndarray              # (T, n_units) conteggi di spike per bin da 20 ms
    times: np.ndarray               # (T,) tempo del *centro* di ogni bin, in secondi
    kin: np.ndarray                 # (T, 7) posizione
    vel: np.ndarray                 # (T, 7) velocita' (target FALCON)
    trial_num: np.ndarray           # (T,) indice di trial
    eval_mask: np.ndarray           # (T,) bool
    epochs: pd.DataFrame            # start_time, stop_time, tag, dur
    spike_times: list = field(repr=False, default_factory=list)  # per unita', tempi grezzi (s)
    bins: str = "falcon"            # convenzione di binning usata per `spikes` e `times` (BINS)
    offset: float = 0.0             # istante (s) del campione 0 di cinematica: t_k = offset + k * BIN_S

    @property
    def n_units(self) -> int:
        return self.spikes.shape[1]

    @property
    def n_bins(self) -> int:
        return self.spikes.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_bins * BIN_S

    @property
    def trial_ids(self) -> np.ndarray:
        return np.unique(self.trial_num)

    def __repr__(self) -> str:
        return (
            f"Session({self.split}/{self.session_id}, {self.n_units} unita', "
            f"{self.duration_s:.1f}s, {len(self.trial_ids)} trials)"
        )


# --------------------------------------------------------------------------------------
# Scoperta dei file e metadati leggeri
# --------------------------------------------------------------------------------------

def find_sessions(data_dir: str | Path) -> "list[Path]":
    """Tutti i file .nwb sotto data_dir, ordinati."""
    return sorted(Path(data_dir).rglob("*.nwb"))


def _infer_split_and_id(path: Path) -> "tuple[str, str]":
    sid = path.stem.split("ses-")[-1] if "ses-" in path.stem else path.stem
    split = next((s for s in SPLITS if s in str(path)), "unknown")
    return split, sid


def quick_meta(path: str | Path) -> dict:
    """Metadati di una sessione senza binnare gli spike (lettura lazy via h5py)."""
    from pynwb import NWBHDF5IO

    path = Path(path)
    split, sid = _infer_split_and_id(path)
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        n_bins = int(nwb.acquisition["OpenLoopKinematics"].data.shape[0])
        n_units = int(len(nwb.units.id))
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:])
        eval_mask = np.asarray(nwb.acquisition["eval_mask"].data[:]).astype(bool)
        n_epochs = int(len(nwb.intervals["epochs"].id))
        n_spikes = int(sum(len(nwb.units["spike_times"][i]) for i in range(n_units)))
    dur = n_bins * BIN_S
    return dict(
        split=split,
        session=sid,
        n_units=n_units,
        n_bins=n_bins,
        duration_s=round(dur, 1),
        n_trials=int(len(np.unique(trial_num))),
        n_epochs=n_epochs,
        eval_frac=round(float(eval_mask.mean()), 3),
        rate_hz=round(n_spikes / n_units / dur, 1),   # firing rate medio per unita'
        file_mb=round(path.stat().st_size / 1e6, 1),
    )


# --------------------------------------------------------------------------------------
# Caricamento completo
# --------------------------------------------------------------------------------------

def bin_units(spike_times: "list[np.ndarray]", bin_edges: np.ndarray) -> np.ndarray:
    """Conteggi di spike per bin, per unita' -> (n_bins, n_units). Stessa logica del
    loader ufficiale `falcon_challenge.dataloaders.bin_units`."""
    out = np.zeros((len(bin_edges) - 1, len(spike_times)), dtype=np.int16)
    for u, st in enumerate(spike_times):
        out[:, u], _ = np.histogram(st, bins=bin_edges)
    return out


def bin_edges(n_bins: int, offset: float = 0.0, bins: str = "falcon") -> np.ndarray:
    """I n_bins + 1 bordi dei bin da 20 ms, con t_k = offset + k * BIN_S (campione k di cinematica).

    falcon: bordi t_k - 20 ms ... t_k, cioe' il bin k copre [t_k - 20 ms, t_k)
            (come `falcon_challenge.dataloaders.bin_units` con i timestamp come fine del bin);
    legacy: bordi t_k ... t_k + 20 ms, cioe' il bin k copre [t_k, t_k + 20 ms).
    """
    k = np.arange(n_bins + 1)
    if bins == "falcon":
        return offset + (k - 1) * BIN_S
    if bins == "legacy":
        return offset + k * BIN_S
    raise ValueError(f"bins deve essere uno di {BINS}, non {bins!r}")


def load_session(path: str | Path, bins: str = "falcon") -> Session:
    """Carica un NWB FALCON H1 in una `Session`, binnando gli spike a 20 ms.

    bins="falcon" (predefinita): bin k = [t_k - 20 ms, t_k), conteggi identici a `load_nwb`.
    bins="legacy": bin k = [t_k, t_k + 20 ms), la convenzione usata fino al 30/10/2026 (cache del 04).
    """
    from pynwb import NWBHDF5IO

    if bins not in BINS:
        raise ValueError(f"bins deve essere uno di {BINS}, non {bins!r}")
    path = Path(path)
    split, sid = _infer_split_and_id(path)

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()

        kin_ts = nwb.acquisition["OpenLoopKinematics"]
        kin = np.asarray(kin_ts.data[:], dtype=np.float64)
        vel = np.asarray(nwb.acquisition["OpenLoopKinematicsVelocity"].data[:], dtype=np.float64)
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:]).astype(int)
        eval_mask = np.asarray(nwb.acquisition["eval_mask"].data[:]).astype(bool)

        n_bins = kin.shape[0]
        # `rate` vale 0.02 = passo in secondi (vedi docstring del modulo)
        offset = float(getattr(kin_ts, "offset", 0.0) or 0.0)
        # t_k = offset + k * 20 ms; il bin k e' [t_k - 20 ms, t_k) (falcon) o [t_k, t_k + 20 ms) (legacy)
        edges = bin_edges(n_bins, offset, bins)

        n_units = len(nwb.units.id)
        spike_times = [np.asarray(nwb.units["spike_times"][u]) for u in range(n_units)]
        spikes = bin_units(spike_times, edges)

        ep = nwb.intervals["epochs"].to_dataframe()
        epochs = pd.DataFrame({
            "start_time": ep["start_time"].to_numpy(dtype=float),
            "stop_time": ep["stop_time"].to_numpy(dtype=float),
            "tag": [t[0] for t in ep["tags"]],
        })
        epochs["dur"] = epochs.stop_time - epochs.start_time

    return Session(
        path=path,
        split=split,
        session_id=sid,
        spikes=spikes,
        times=edges[1:] - BIN_S / 2,         # centro del bin (falcon: t_k - 10 ms; legacy: t_k + 10 ms)
        kin=kin,
        vel=vel,
        trial_num=trial_num,
        eval_mask=eval_mask,
        epochs=epochs,
        spike_times=spike_times,
        bins=bins,
        offset=offset,
    )


def time_to_bin(session: Session, t: float) -> int:
    """Indice del bin che comincia all'istante t (arrotondato alla griglia dei 20 ms).

    falcon: il bin k comincia a t_k - 20 ms, quindi k = (t - offset) / 20 ms + 1;
    legacy: il bin k comincia a t_k,         quindi k = (t - offset) / 20 ms.
    """
    shift = 1 if session.bins == "falcon" else 0
    return int(round((t - session.offset) / BIN_S)) + shift


def bin_start(session: Session, k: int) -> float:
    """Istante (s) in cui comincia il bin k: l'inverso di `time_to_bin`."""
    shift = 1 if session.bins == "falcon" else 0
    return session.offset + (k - shift) * BIN_S


# --------------------------------------------------------------------------------------
# Feature neurali
# --------------------------------------------------------------------------------------

def smooth_rates(counts: np.ndarray, sigma_ms: float = 50.0, bin_s: float = BIN_S) -> np.ndarray:
    """Firing rate in Hz: smoothing gaussiano lungo il tempo + conversione conteggi -> spikes/s."""
    from scipy.ndimage import gaussian_filter1d

    x = counts.astype(np.float32)
    if sigma_ms and sigma_ms > 0:
        x = gaussian_filter1d(x, sigma=(sigma_ms / 1000.0) / bin_s, axis=0, mode="nearest")
    return x / bin_s


def firing_rates(session: Session, sigma_ms: float = 50.0) -> np.ndarray:
    """Scorciatoia: conteggi 20 ms -> smoothing -> firing rate (n_bins, n_units) in Hz."""
    return smooth_rates(session.spikes, sigma_ms=sigma_ms)


def active_units(session: Session, min_hz: float = 0.5) -> np.ndarray:
    """Indici delle unita' con firing rate medio >= min_hz.

    Le unita' quasi silenti non portano informazione e destabilizzano GPFA
    (fattori degeneri), quindi vengono escluse dalle analisi di manifold."""
    hz = session.spikes.sum(0) / session.duration_s
    return np.where(hz >= min_hz)[0]


# --------------------------------------------------------------------------------------
# Segmentazione in trial e in fasi del task
# --------------------------------------------------------------------------------------

def trial_slices(session: Session) -> "list[tuple[int, int]]":
    """Per ogni trial (da TrialNum), gli indici di bin (start, stop) — stop escluso."""
    tn = session.trial_num
    bounds = np.flatnonzero(np.diff(tn) != 0) + 1
    edges = np.concatenate([[0], bounds, [len(tn)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def phase_per_bin(session: Session) -> np.ndarray:
    """Etichetta di fase (stringa) per ogni bin; "" dove nessuna epoch copre il bin."""
    lab = np.full(session.n_bins, "", dtype=object)
    idx = np.clip(np.round(session.times / BIN_S - 0.5).astype(int), 0, session.n_bins - 1)
    t = session.times
    for _, e in session.epochs.iterrows():
        m = (t >= e.start_time) & (t < e.stop_time)
        lab[m] = e.tag
    _ = idx
    return lab


def phase_windows(
    session: Session,
    tags: "tuple[str, ...]" = MOVEMENT_PHASES,
    win_s: float = 1.0,
    align: str = "start",
) -> "list[dict]":
    """Finestre di lunghezza fissa allineate all'inizio (o alla fine) di ogni fase.

    Serve come "trial" corto e ripetuto per GPFA: le fasi sono stereotipate e si
    ripetono a ogni trial, quindi giocano lo stesso ruolo del motivo di canto in B1.
    Vengono scartate le finestre che escono dai bordi della registrazione.

    Ritorna una lista di dict con: tag, trial, start (bin), stop (bin).
    La finestra copre sempre gli stessi istanti; il suo indice di bin dipende da `session.bins`
    (con "falcon" e' un bin piu' avanti che con "legacy": il bin k+1 falcon = il bin k legacy).
    """
    n_win = int(round(win_s / BIN_S))
    out = []
    for _, e in session.epochs.iterrows():
        if e.tag not in tags:
            continue
        if align == "start":
            a = time_to_bin(session, e.start_time)            # il bin che comincia all'inizio della fase
        else:
            a = time_to_bin(session, e.stop_time) - n_win     # l'ultimo bin finisce alla fine della fase
        b = a + n_win
        if a < 0 or b > session.n_bins:
            continue
        out.append(dict(tag=e.tag, trial=int(session.trial_num[a]), start=a, stop=b))
    return out


def windows_spiketrains(session: Session, windows: "list[dict]", units=None) -> list:
    """Spike train per GPFA (elephant): lista di finestre, ognuna lista di `neo.SpikeTrain`
    (una per unita'), con tempi relativi all'inizio della finestra."""
    import neo
    import quantities as pq

    if units is None:
        units = np.arange(session.n_units)
    out = []
    for w in windows:
        t0 = bin_start(session, w["start"])                   # inizio vero della finestra (dipende da session.bins)
        t1 = bin_start(session, w["stop"])
        dur = (t1 - t0) * pq.s
        trains = []
        for u in units:
            st = session.spike_times[u]
            sel = st[(st >= t0) & (st < t1)] - t0
            trains.append(neo.SpikeTrain(sel * pq.s, t_start=0 * pq.s, t_stop=dur))
        out.append(trains)
    return out


def speed(session: Session, dims=(0, 1, 2)) -> np.ndarray:
    """Modulo della velocita' sulle dimensioni indicate (default: traslazione 3D)."""
    return np.linalg.norm(session.vel[:, list(dims)], axis=1)
