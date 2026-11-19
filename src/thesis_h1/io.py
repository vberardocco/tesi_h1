"""
Loading and preparation of the FALCON H1 data (DANDI:000954).

FALCON H1 = "Human 7DoF Reach and Grasp Motor BCI" (Ye, Collinger, Gaunt, 2024).
Neural activity recorded from 2 intracortical arrays of 96 channels (hand knob area,
motor cortex) in a human participant with an iBCI, during *open loop* calibration sessions:
an audiovisual cue indicates a reach-and-grasp movement that the participant
attempts to perform, while a virtual arm shows the reference kinematics.

Structure of the NWB files (verified on the 40 files of the dandiset):
  - units (176 units)                 : multi-unit spiking activity, only `spike_times` (s).
  - acquisition['OpenLoopKinematics'] : (T, 7) position of the reference kinematics.
                                        Columns: tx, ty, tz (3D translation of the end-effector),
                                        rx (orientation/roll), g1, g2, g3 (3 grip degrees of freedom).
  - acquisition['OpenLoopKinematicsVelocity'] : (T, 7) velocity = **official FALCON target**.
  - acquisition['TrialNum']           : (T,) index of the trial each bin belongs to.
  - acquisition['eval_mask']          : (T,) bool, True = timestep to be used in evaluation.
  - intervals['epochs']               : task phases (Presentation, Reach, Orient, SnapTo,
                                        Shape, Grasp, Carry, Release, Intertrial) with
                                        start_time / stop_time / tags.

WARNING about the `rate` field: NWB declares it as 0.02, but it is the **time step in seconds**
(20 ms bins → 50 Hz), not a frequency. The official FALCON loader makes the same
assumption (`timestamps = offset + arange(T) * rate`), so we keep it for consistency.

Conventions of this module:
  - Neural matrices are always (n_time_bins, n_units).
  - Binning is at 20 ms, identical to that of the benchmark: no hidden resampling.

Binning: which 20 ms window ends in row k.
  Let t_k = offset + k * 20 ms be the time of kinematics sample k (offset = 0 in all
  H1 files). Row k of the counts is always paired with sample k (kin, vel, trial_num,
  eval_mask share the same index); only the time window that is counted changes:

  - bins="falcon" (default since 30/10/2026): bin k contains the spikes in
    [t_k - 20 ms, t_k). These are the same counts as `falcon_challenge.dataloaders.load_nwb`
    (verified on all 40 files). Bin k ends at the time of sample k, so a
    decoder that uses the bins up to the k-th sees only spikes preceding the sample: it is causal.
  - bins="legacy": bin k contains the spikes in [t_k, t_k + 20 ms), i.e. the 20 ms AFTER the
    sample. This is the convention of this module until 30/10/2026 (the comment in the code
    said the opposite). Legacy row k = falcon row k+1. It is only needed to re-read the caches of
    notebook 04, whose keys depend on the counts.

  The functions that go from a time to a bin (`phase_windows`, `windows_spiketrains`)
  read the convention from the `Session` (`session.bins`), through `time_to_bin` and `bin_start`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# Splits of the FALCON H1 dataset (folder names "sub-HumanPitt-<split>")
SPLITS = ("held-in-calib", "held-in-minival", "held-out-calib")

# Native time step of the benchmark: 20 ms
BIN_S = 0.02

# Binning conventions (see the module docstring); the first is the default
BINS = ("falcon", "legacy")

# Names of the 7 degrees of freedom (from the `description` field of the TimeSeries)
KIN_LABELS = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]
KIN_LONG = [
    "translation x", "translation y", "translation z", "orientation (roll)",
    "grip g1 (thumb)", "grip g2 (index)", "grip g3 (ring/little)",
]

# Task phases that involve actual movement (we exclude Presentation* and Intertrial,
# in which the participant observes the cue and the kinematics are still).
MOVEMENT_PHASES = ("Reach", "Orient", "SnapTo", "Shape", "Grasp", "Carry", "Orient2", "Release")

# Stable colors per phase, used in all figures
PHASE_COLORS = {
    "Reach": "#1f77b4", "Orient": "#ff7f0e", "SnapTo": "#2ca02c", "Shape": "#d62728",
    "Grasp": "#9467bd", "Carry": "#8c564b", "Orient2": "#e377c2", "Release": "#7f7f7f",
    "Presentation": "#c7c7c7", "Intertrial": "#dddddd",
}


@dataclass
class Session:
    """A FALCON H1 session loaded in memory (files are <7 MB: everything is loaded)."""

    path: Path
    split: str
    session_id: str                 # e.g. "19250108T110520" (date anonymized by DANDI)
    spikes: np.ndarray              # (T, n_units) spike counts per 20 ms bin
    times: np.ndarray               # (T,) time of the *center* of each bin, in seconds
    kin: np.ndarray                 # (T, 7) position
    vel: np.ndarray                 # (T, 7) velocity (FALCON target)
    trial_num: np.ndarray           # (T,) trial index
    eval_mask: np.ndarray           # (T,) bool
    epochs: pd.DataFrame            # start_time, stop_time, tag, dur
    spike_times: list = field(repr=False, default_factory=list)  # per unit, raw times (s)
    bins: str = "falcon"            # binning convention used for `spikes` and `times` (BINS)
    offset: float = 0.0             # time (s) of kinematics sample 0: t_k = offset + k * BIN_S

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
            f"Session({self.split}/{self.session_id}, {self.n_units} units, "
            f"{self.duration_s:.1f}s, {len(self.trial_ids)} trials)"
        )


# --------------------------------------------------------------------------------------
# File discovery and lightweight metadata
# --------------------------------------------------------------------------------------

def find_sessions(data_dir: str | Path) -> "list[Path]":
    """All .nwb files under data_dir, sorted."""
    return sorted(Path(data_dir).rglob("*.nwb"))


def _infer_split_and_id(path: Path) -> "tuple[str, str]":
    sid = path.stem.split("ses-")[-1] if "ses-" in path.stem else path.stem
    split = next((s for s in SPLITS if s in str(path)), "unknown")
    return split, sid


def quick_meta(path: str | Path) -> dict:
    """Metadata of a session without binning the spikes (lazy read via h5py)."""
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
        rate_hz=round(n_spikes / n_units / dur, 1),   # mean firing rate per unit
        file_mb=round(path.stat().st_size / 1e6, 1),
    )


# --------------------------------------------------------------------------------------
# Full loading
# --------------------------------------------------------------------------------------

def bin_units(spike_times: "list[np.ndarray]", bin_edges: np.ndarray) -> np.ndarray:
    """Spike counts per bin, per unit -> (n_bins, n_units). Same logic as the
    official loader `falcon_challenge.dataloaders.bin_units`."""
    out = np.zeros((len(bin_edges) - 1, len(spike_times)), dtype=np.int16)
    for u, st in enumerate(spike_times):
        out[:, u], _ = np.histogram(st, bins=bin_edges)
    return out


def bin_edges(n_bins: int, offset: float = 0.0, bins: str = "falcon") -> np.ndarray:
    """The n_bins + 1 edges of the 20 ms bins, with t_k = offset + k * BIN_S (kinematics sample k).

    falcon: edges t_k - 20 ms ... t_k, i.e. bin k covers [t_k - 20 ms, t_k)
            (like `falcon_challenge.dataloaders.bin_units` with the timestamps as the end of the bin);
    legacy: edges t_k ... t_k + 20 ms, i.e. bin k covers [t_k, t_k + 20 ms).
    """
    k = np.arange(n_bins + 1)
    if bins == "falcon":
        return offset + (k - 1) * BIN_S
    if bins == "legacy":
        return offset + k * BIN_S
    raise ValueError(f"bins must be one of {BINS}, not {bins!r}")


def load_session(path: str | Path, bins: str = "falcon") -> Session:
    """Load a FALCON H1 NWB into a `Session`, binning the spikes at 20 ms.

    bins="falcon" (default): bin k = [t_k - 20 ms, t_k), counts identical to `load_nwb`.
    bins="legacy": bin k = [t_k, t_k + 20 ms), the convention used until 30/10/2026 (caches of 04).
    """
    from pynwb import NWBHDF5IO

    if bins not in BINS:
        raise ValueError(f"bins must be one of {BINS}, not {bins!r}")
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
        # `rate` is 0.02 = step in seconds (see the module docstring)
        offset = float(getattr(kin_ts, "offset", 0.0) or 0.0)
        # t_k = offset + k * 20 ms; bin k is [t_k - 20 ms, t_k) (falcon) or [t_k, t_k + 20 ms) (legacy)
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
        times=edges[1:] - BIN_S / 2,         # bin center (falcon: t_k - 10 ms; legacy: t_k + 10 ms)
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
    """Index of the bin that starts at time t (rounded to the 20 ms grid).

    falcon: bin k starts at t_k - 20 ms, so k = (t - offset) / 20 ms + 1;
    legacy: bin k starts at t_k,         so k = (t - offset) / 20 ms.
    """
    shift = 1 if session.bins == "falcon" else 0
    return int(round((t - session.offset) / BIN_S)) + shift


def bin_start(session: Session, k: int) -> float:
    """Time (s) at which bin k starts: the inverse of `time_to_bin`."""
    shift = 1 if session.bins == "falcon" else 0
    return session.offset + (k - shift) * BIN_S


# --------------------------------------------------------------------------------------
# Neural features
# --------------------------------------------------------------------------------------

def smooth_rates(counts: np.ndarray, sigma_ms: float = 50.0, bin_s: float = BIN_S) -> np.ndarray:
    """Firing rate in Hz: Gaussian smoothing along time + conversion counts -> spikes/s."""
    from scipy.ndimage import gaussian_filter1d

    x = counts.astype(np.float32)
    if sigma_ms and sigma_ms > 0:
        x = gaussian_filter1d(x, sigma=(sigma_ms / 1000.0) / bin_s, axis=0, mode="nearest")
    return x / bin_s


def firing_rates(session: Session, sigma_ms: float = 50.0) -> np.ndarray:
    """Shortcut: 20 ms counts -> smoothing -> firing rate (n_bins, n_units) in Hz."""
    return smooth_rates(session.spikes, sigma_ms=sigma_ms)


def active_units(session: Session, min_hz: float = 0.5) -> np.ndarray:
    """Indices of the units with mean firing rate >= min_hz.

    Nearly silent units carry no information and destabilize GPFA
    (degenerate factors), so they are excluded from the manifold analyses."""
    hz = session.spikes.sum(0) / session.duration_s
    return np.where(hz >= min_hz)[0]


# --------------------------------------------------------------------------------------
# Segmentation into trials and task phases
# --------------------------------------------------------------------------------------

def trial_slices(session: Session) -> "list[tuple[int, int]]":
    """For each trial (from TrialNum), the bin indices (start, stop), stop excluded."""
    tn = session.trial_num
    bounds = np.flatnonzero(np.diff(tn) != 0) + 1
    edges = np.concatenate([[0], bounds, [len(tn)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def phase_per_bin(session: Session) -> np.ndarray:
    """Phase label (string) for each bin; "" where no epoch covers the bin."""
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
    """Fixed-length windows aligned to the start (or end) of each phase.

    They serve as short, repeated "trials" for GPFA: the phases are stereotyped and
    repeat at every trial, so they play the same role as the song motif in B1.
    Windows that fall outside the edges of the recording are discarded.

    Returns a list of dicts with: tag, trial, start (bin), stop (bin).
    The window always covers the same instants; its bin index depends on `session.bins`
    (with "falcon" it is one bin later than with "legacy": falcon bin k+1 = legacy bin k).
    """
    n_win = int(round(win_s / BIN_S))
    out = []
    for _, e in session.epochs.iterrows():
        if e.tag not in tags:
            continue
        if align == "start":
            a = time_to_bin(session, e.start_time)            # the bin that starts at the beginning of the phase
        else:
            a = time_to_bin(session, e.stop_time) - n_win     # the last bin ends at the end of the phase
        b = a + n_win
        if a < 0 or b > session.n_bins:
            continue
        out.append(dict(tag=e.tag, trial=int(session.trial_num[a]), start=a, stop=b))
    return out


def windows_spiketrains(session: Session, windows: "list[dict]", units=None) -> list:
    """Spike trains for GPFA (elephant): list of windows, each a list of `neo.SpikeTrain`
    (one per unit), with times relative to the start of the window."""
    import neo
    import quantities as pq

    if units is None:
        units = np.arange(session.n_units)
    out = []
    for w in windows:
        t0 = bin_start(session, w["start"])                   # true start of the window (depends on session.bins)
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
    """Magnitude of the velocity over the given dimensions (default: 3D translation)."""

    return np.linalg.norm(session.vel[:, list(dims)], axis=1)
