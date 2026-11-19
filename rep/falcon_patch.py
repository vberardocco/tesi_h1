"""SUPERSEDED as of 30 Oct 2026: do not use it any more. io.py now produces the FALCON bins by itself (default bins="falcon",
checked on all 40 files in scripts/io_bins_check.py) and phase_windows / windows_spiketrains follow the
convention of the session. With the new io.py this file would shift the bins a second time and the check at the
end would fail. It is kept here as a record of how the replications of 29-30 Oct were run (build_nb.py,
angles.py load it with exec): to redo them, io.load_session with the default is enough, without this file.
"""

"""FALCON bins inside the replication notebooks (run with exec() in the first cell).

FALCON (load_nwb): bin k covers [t_k - 20 ms, t_k) and is paired with kinematics sample k, t_k = offset + k·20 ms.
io.py (until 29 Oct): bin k covers [t_k, t_k + 20 ms), i.e. 20 ms AFTER the sample. Here, only in the
replication kernel, the three io functions that depend on the convention are replaced:

- load_session: counts and bin centers with the FALCON edges (check: identical to load_nwb);
- phase_windows: the window starting at the onset of a phase begins one bin later (bin k+1 covers [t_k, t_k+1));
- windows_spiketrains: times relative to the true start of the window.

The file io.py is not touched (its correction is a separate step, by 2 Nov).
"""
import dataclasses as _dc
import numpy as _np
from thesis_h1 import io as _io

if not hasattr(_io, "_orig_load_session"):
    _io._orig_load_session = _io.load_session
    _io._orig_phase_windows = _io.phase_windows
    _io._orig_windows_spiketrains = _io.windows_spiketrains


def _load_session_falcon(path):
    s_ = _io._orig_load_session(path)
    off_ = float(s_.times[0] - _io.BIN_S / 2)                    # file offset (0 in all H1 files)
    edges_ = off_ + (_np.arange(s_.n_bins + 1) - 1) * _io.BIN_S  # edges: t_k - 20 ms ... t_k
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

# check: the counts match those of the official FALCON loader
from falcon_challenge.dataloaders import load_nwb as _load_nwb
from falcon_challenge.config import FalconTask as _FT
_chk_counts = _load_nwb(SESS, dataset=_FT.h1)[0]
_chk_s = _io.load_session(SESS)
assert _np.array_equal(_chk_s.spikes, _chk_counts.astype(_chk_s.spikes.dtype)), "bins differ from FALCON"
print(f"REPLICATION {SESS.name}: FALCON bins (check against load_nwb passed), figures in {FIGS}")
del _chk_counts, _chk_s
