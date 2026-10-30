"""SUPERATO dal 30/10/2026: non usarlo piu'. io.py ora fa da solo i bin di FALCON (default bins="falcon",
controllo su tutti i 40 file in scripts/io_bins_check.py) e phase_windows / windows_spiketrains seguono la
convenzione della sessione. Con io.py nuovo questo file sposterebbe i bin una seconda volta e il controllo in
fondo fallirebbe. Resta qui come traccia di come sono state eseguite le repliche del 29-30/10 (build_nb.py,
angles.py lo caricano con exec): per rifarle basta io.load_session con il default, senza questo file.
"""

"""Bin come FALCON dentro i notebook di replica (si esegue con exec() nella prima cella).

FALCON (load_nwb): il bin k copre [t_k - 20 ms, t_k) e si abbina al campione di cinematica k, t_k = offset + k·20 ms.
io.py (fino al 29/10): il bin k copre [t_k, t_k + 20 ms), cioè 20 ms DOPO il campione. Qui si sostituiscono, solo nel
kernel della replica, le tre funzioni di io che dipendono dalla convenzione:

- load_session: conteggi e centri dei bin con i bordi di FALCON (controllo: identici a load_nwb);
- phase_windows: la finestra che parte all'inizio di una fase comincia un bin più avanti (il bin k+1 copre [t_k, t_k+1));
- windows_spiketrains: tempi relativi all'inizio vero della finestra.

Il file io.py non viene toccato (la sua correzione è un passo a parte, entro il 2/11).
"""
import dataclasses as _dc
import numpy as _np
from tesi_h1 import io as _io

if not hasattr(_io, "_orig_load_session"):
    _io._orig_load_session = _io.load_session
    _io._orig_phase_windows = _io.phase_windows
    _io._orig_windows_spiketrains = _io.windows_spiketrains


def _load_session_falcon(path):
    s_ = _io._orig_load_session(path)
    off_ = float(s_.times[0] - _io.BIN_S / 2)                    # offset del file (0 in tutti i file di H1)
    edges_ = off_ + (_np.arange(s_.n_bins + 1) - 1) * _io.BIN_S  # bordi: t_k - 20 ms ... t_k
    return _dc.replace(s_, spikes=_io.bin_units(s_.spike_times, edges_), times=edges_[1:] - _io.BIN_S / 2)


def _phase_windows_falcon(session, *a, **k):
    out_ = []
    for w_ in _io._orig_phase_windows(session, *a, **k):
        st_, sp_ = w_["start"] + 1, w_["stop"] + 1
        if sp_ <= session.n_bins:
            out_.append(dict(w_, start=st_, stop=sp_, trial=int(session.trial_num[st_])))
    return out_


def _windows_spiketrains_falcon(session, windows, units=None):
    return _io._orig_windows_spiketrains(session, [dict(w_, start=w_["start"] - 1, stop=w_["stop"] - 1) for w_ in windows], units)


_io.load_session = _load_session_falcon
_io.phase_windows = _phase_windows_falcon
_io.windows_spiketrains = _windows_spiketrains_falcon

# controllo: i conteggi coincidono con quelli del caricatore ufficiale di FALCON
from falcon_challenge.dataloaders import load_nwb as _load_nwb
from falcon_challenge.config import FalconTask as _FT
_chk_counts = _load_nwb(SESS, dataset=_FT.h1)[0]
_chk_s = _io.load_session(SESS)
assert _np.array_equal(_chk_s.spikes, _chk_counts.astype(_chk_s.spikes.dtype)), "bin diversi da FALCON"
print(f"REPLICA {SESS.name}: bin di FALCON (controllo con load_nwb superato), figure in {FIGS}")
del _chk_counts, _chk_s
