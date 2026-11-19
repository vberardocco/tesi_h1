"""Check of the binning in io.py (30 Oct 2026): on all H1 NWB files.

1. io.load_session (default bins="falcon") gives counts identical to falcon_challenge.dataloaders.load_nwb.
2. bins="legacy" is the same matrix shifted by one row: legacy row k = falcon row k+1.
3. The phase windows cover the same instants under the two conventions (falcon index = legacy + 1)
   and windows_spiketrains gives the same spikes.

Usage (from the project root, tesi-manifolds environment):
    python scripts/io_bins_check.py [data folder, default data/raw]
Exits with code 1 if a check fails.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thesis_h1 import io                                       # noqa: E402
from falcon_challenge.config import FalconTask               # noqa: E402
from falcon_challenge.dataloaders import load_nwb            # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw"
files = io.find_sessions(DATA)
print(f"{len(files)} NWB files in {DATA}")
bad = []
for p in files:
    s = io.load_session(p)                                   # default: bins="falcon"
    leg = io.load_session(p, bins="legacy")
    ref = load_nwb(p, dataset=FalconTask.h1)[0]              # counts from the official loader (uint8)
    ok_falcon = s.bins == "falcon" and s.spikes.shape == ref.shape and np.array_equal(s.spikes.astype(int), ref.astype(int))
    ok_legacy = leg.bins == "legacy" and np.array_equal(leg.spikes[:-1], s.spikes[1:])
    ok_times = np.allclose(leg.times - s.times, io.BIN_S)    # legacy bin centres are 20 ms later
    wf = io.phase_windows(s, win_s=1.0, align="start")
    wl = io.phase_windows(leg, win_s=1.0, align="start")
    wl_in = [w for w in wl if w["stop"] + 1 <= s.n_bins]     # legacy: windows that do not fall outside the session under falcon
    ok_win = (len(wf) == len(wl_in)
              and all(a["start"] == b["start"] + 1 and a["stop"] == b["stop"] + 1 and a["tag"] == b["tag"]
                      for a, b in zip(wf, wl_in))
              and all(io.bin_start(s, a["start"]) == io.bin_start(leg, b["start"]) for a, b in zip(wf, wl_in)))
    units = np.arange(20)                                    # spike trains: 20 channels are enough to check the timing
    stf = io.windows_spiketrains(s, wf, units=units)
    stl = io.windows_spiketrains(leg, wl_in, units=units)
    ok_st = all(np.array_equal(np.asarray(x.magnitude), np.asarray(y.magnitude)) and x.t_stop == y.t_stop
                for a, b in zip(stf, stl) for x, y in zip(a, b))
    ok = ok_falcon and ok_legacy and ok_times and ok_win and ok_st
    if not ok:
        bad.append(p.name)
    print(f"{'ok ' if ok else 'NO '} {p.parent.name[-14:]:>14} {s.session_id}: {s.n_bins:5d} bins, offset {s.offset:g}, "
          f"= load_nwb {ok_falcon}, legacy shifted by 1 {ok_legacy}, centres {ok_times}, "
          f"windows {len(wf)} {ok_win}, spike trains {ok_st}")
print(f"\nFiles checked: {len(files)}, failed: {len(bad)} {bad if bad else ''}")
sys.exit(1 if bad or not files else 0)
