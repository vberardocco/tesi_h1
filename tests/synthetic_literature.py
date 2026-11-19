"""Synthetic data for tests/test_literature.py (sessions with trials and phases as in H1)."""
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter1d

from thesis_h1 import literature

PHASES_A = ["Presentation", "Reach", "Presentation", "Orient", "SnapTo", "Presentation", "Shape", "Presentation",
          "Grasp", "Presentation", "Carry", "Presentation", "Orient2", "Presentation", "Release"]
PHASES_B = ["Presentation", "Reach", "Orient", "SnapTo", "Presentation", "Shape", "Presentation", "Grasp",
          "Presentation", "Carry", "Orient2", "Presentation", "Release"]
DURATION = {"Presentation": 70, "Reach": 70, "Orient": 60, "SnapTo": 25, "Shape": 30, "Grasp": 40, "Carry": 70,
          "Orient2": 70, "Release": 40}


def _shape(name, j, u):
    """Latent j during the segment `name`, as a function of the relative time u in [0, 1]: fixed for (name, j)."""
    r = np.random.default_rng(zlib.crc32(f"{name}|{j}".encode()))
    a, ph = r.normal(size=3), r.uniform(0, 2 * np.pi, 3)
    return r.normal() + sum(a[m] * np.sin(np.pi * (m + 1) * u + ph[m]) for m in range(3))


def session(seed, task=True, N=100, n_trial=15, depth=0.15, seed_weights=123, cut_last=True):
    """A synthetic session: Poisson counts at 20 ms with 4 latents.

    task=True: the latents depend only on the phase (with the next phase for Presentation) and on the relative
    time within the segment; task=False: 4 smooth latents independent of the task (session seed).
    Channel weights and baseline rates depend only on seed_weights; durations (±45%), sequence variants and
    noise on seed. Every trial after the first starts with 4 Release bins; Intertrial at the start and at the end,
    plus a first bin without a label, as in H1. With cut_last the last trial ends without Release, as
    in S1 (the key "Presentation>end" appears in a single trial). Returns (counts, phase, trial)."""
    rp = np.random.default_rng(seed_weights)
    W = rp.normal(size=(N, 4)) * depth
    base = rp.uniform(2, 25, N) * literature.BIN_S
    rng = np.random.default_rng(seed)
    phase, trial, lat = [""], [1], [np.zeros(4)]
    phase += ["Intertrial"] * 10
    trial += [1] * 10
    lat += [np.zeros(4)] * 10
    for t in range(1, n_trial + 1):
        seq = PHASES_A if rng.random() < 0.5 else PHASES_B
        if cut_last and t == n_trial:
            seq = seq[:-1]
        if t > 1:
            phase += ["Release"] * 4
            trial += [t] * 4
            lat += [np.zeros(4)] * 4
        for i, f in enumerate(seq):
            L = max(2, int(round(DURATION[f] * (1 + rng.uniform(-0.45, 0.45)))))
            name = f
            if f == "Presentation":
                name = "Presentation>" + next((g for g in seq[i + 1:] if g != "Presentation"), "end")
            u = (np.arange(L) + 0.5) / L
            lat += list(np.stack([_shape(name, j, u) for j in range(4)], 1))
            phase += [f] * L
            trial += [t] * L
    phase += ["Intertrial"] * 20
    trial += [n_trial] * 20
    lat += [np.zeros(4)] * 20
    lat = np.array(lat)
    if task:
        lat = (lat - lat.mean(0)) / lat.std(0)
    else:
        lat = gaussian_filter1d(rng.normal(size=lat.shape), 15, axis=0)
        lat = (lat - lat.mean(0)) / lat.std(0)
    rate = base[None, :] * np.exp(lat @ W.T - 0.5 * (W ** 2).sum(1)[None, :])
    return rng.poisson(rate), np.array(phase, dtype=object), np.array(trial)


def sessions_1g(task=True, semi=(1, 2, 3), **kw):
    """Three synthetic sessions as literature.Data1g (pipeline 1a), all channels 'active'."""
    out = {}
    for name, s in zip(("S1", "S2", "S5"), semi):
        counts, phase, trial = session(s, task=task, **kw)
        out[name] = literature.Data1g(Y=literature.pipeline_gallego(counts), phase=phase, trial=trial,
                                  active=np.arange(counts.shape[1]))
    return out


SEQ_CLOCK = ["Presentation", "Reach", "Presentation", "Orient", "SnapTo", "Presentation", "Shape", "Presentation",
                "Grasp", "Presentation", "Carry", "Presentation", "Orient2", "Presentation", "Release"]


def session_clock(seed, informative, n_trial=15, d=25, variation=0.4):
    """Step 3: 15 trials with the H1 phase sequence; returns (clock, latents T × d, phase, trial).

    informative=False (T7): equal phase durations in all trials, so the phase is a function of the time since
    the start of the trial alone; the latents are smooth functions of the clock (equal in every trial) plus noise.
    informative=True (T8): durations varying by ±variation, so the clock is not enough; each phase has its own
    mean vector in the latents (plus smooth noise), so the latents carry the phase."""
    rng = np.random.default_rng(seed)
    rf = np.random.default_rng(1000)                           # fixed structure: shapes and means per phase
    freq, phase0 = rf.uniform(0.05, 0.6, d), rf.uniform(0, 2 * np.pi, d)
    means = {f: rf.normal(size=d) for f in sorted(set(SEQ_CLOCK))}   # sorted: the order of a set of strings changes between processes

    clock, phase, trial = [], [], []
    for t in range(1, n_trial + 1):
        n = 0
        for f in SEQ_CLOCK:
            L = DURATION[f] if not informative else max(3, int(round(DURATION[f] * (1 + rng.uniform(-variation, variation)))))
            phase += [f] * L
            trial += [t] * L
            clock += list((n + np.arange(L)) * literature.BIN_S)
            n += L
    clock, phase, trial = np.array(clock), np.array(phase, dtype=object), np.array(trial)
    noise = gaussian_filter1d(rng.normal(size=(len(clock), d)), 5, axis=0)
    noise /= noise.std(0)
    if informative:
        lat = np.stack([means[f] for f in phase]) + 1.5 * noise
    else:
        lat = np.sin(2 * np.pi * freq[None, :] * clock[:, None] + phase0[None, :]) + 1.0 * noise
    return clock, lat, phase, trial
