"""Controllo del binning di io.py (30/10/2026): su tutti i file NWB di H1.

1. io.load_session (default bins="falcon") da' conteggi identici a falcon_challenge.dataloaders.load_nwb.
2. bins="legacy" e' la stessa matrice spostata di una riga: riga k legacy = riga k+1 falcon.
3. Le finestre di fase coprono gli stessi istanti con le due convenzioni (indice falcon = legacy + 1)
   e windows_spiketrains da' gli stessi spike.

Uso (dalla radice del progetto, ambiente tesi-manifolds):
    python scripts/io_bins_check.py [cartella dei dati, default data/raw]
Esce con codice 1 se un controllo fallisce.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesi_h1 import io                                       # noqa: E402
from falcon_challenge.config import FalconTask               # noqa: E402
from falcon_challenge.dataloaders import load_nwb            # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw"
files = io.find_sessions(DATA)
print(f"{len(files)} file NWB in {DATA}")
bad = []
for p in files:
    s = io.load_session(p)                                   # default: bins="falcon"
    leg = io.load_session(p, bins="legacy")
    ref = load_nwb(p, dataset=FalconTask.h1)[0]              # conteggi del caricatore ufficiale (uint8)
    ok_falcon = s.bins == "falcon" and s.spikes.shape == ref.shape and np.array_equal(s.spikes.astype(int), ref.astype(int))
    ok_legacy = leg.bins == "legacy" and np.array_equal(leg.spikes[:-1], s.spikes[1:])
    ok_times = np.allclose(leg.times - s.times, io.BIN_S)    # i centri dei bin legacy sono 20 ms dopo
    wf = io.phase_windows(s, win_s=1.0, align="start")
    wl = io.phase_windows(leg, win_s=1.0, align="start")
    wl_in = [w for w in wl if w["stop"] + 1 <= s.n_bins]     # legacy: finestre che con falcon non escono dalla sessione
    ok_win = (len(wf) == len(wl_in)
              and all(a["start"] == b["start"] + 1 and a["stop"] == b["stop"] + 1 and a["tag"] == b["tag"]
                      for a, b in zip(wf, wl_in))
              and all(io.bin_start(s, a["start"]) == io.bin_start(leg, b["start"]) for a, b in zip(wf, wl_in)))
    units = np.arange(20)                                    # spike train: bastano 20 canali per vedere i tempi
    stf = io.windows_spiketrains(s, wf, units=units)
    stl = io.windows_spiketrains(leg, wl_in, units=units)
    ok_st = all(np.array_equal(np.asarray(x.magnitude), np.asarray(y.magnitude)) and x.t_stop == y.t_stop
                for a, b in zip(stf, stl) for x, y in zip(a, b))
    ok = ok_falcon and ok_legacy and ok_times and ok_win and ok_st
    if not ok:
        bad.append(p.name)
    print(f"{'ok ' if ok else 'NO '} {p.parent.name[-14:]:>14} {s.session_id}: {s.n_bins:5d} bin, offset {s.offset:g}, "
          f"= load_nwb {ok_falcon}, legacy spostato di 1 {ok_legacy}, centri {ok_times}, "
          f"finestre {len(wf)} {ok_win}, spike train {ok_st}")
print(f"\nFile controllati: {len(files)}, falliti: {len(bad)} {bad if bad else ''}")
sys.exit(1 if bad or not files else 0)
