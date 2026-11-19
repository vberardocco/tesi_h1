"""Dimensionality comparisons with the literature thresholds (docs/literature/prompt.md, Steps 1 and 2).

Small functions, each with a test on synthetic data in tests/test_literature.py. Conventions:

- time × channel matrices; counts at 20 ms with the FALCON bins (io.load_session, bins="falcon");
- "trial bins": bins with a phase label different from Intertrial and from the empty label;
- PCA = eigenvalues of the covariance matrix of the chosen bins (with z-score: of the correlation matrix);
- null = independent circular shift per channel over the whole session: uniform shift between 2 s and
  T − 2 s, excluding ±2 s around every multiple of the median trial duration (Step 2, rule R5);
- FA = sklearn FactorAnalysis with svd_method="lapack" and random_state=0, as in notebook 02.
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

BIN_S = 0.02
ESCLUSE = ("Intertrial", "")                      # labels outside the trials
SESSIONS = {"S1": "19250108T110520", "S2": "19250113T120811", "S5": "19250120T115044"}
KEYS = {"S1": "S1_set_1", "S2": "S2_set_1", "S5": "S5_set_1"}   # FALCON keys for the registry
DATA = Path.home() / "000954"
REP = Path.home() / "thesis_h1_work/rep"


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def name_base(tag: str) -> str:
    """'Presentation3' -> 'Presentation' (as base_tag in notebook 04): one label for all presentations."""
    return "Presentation" if tag.startswith("Presentation") else tag


@dataclass
class Session:
    name: str                        # "S1", "S2", "S5"
    sid: str                         # "19250108T110520"
    key: str                      # FALCON key, for the registry
    counts: np.ndarray               # T × N counts at 20 ms (FALCON bins), active channels only
    active: np.ndarray               # original indices (0-175) of the active channels
    phase: np.ndarray                 # T phase labels (single Presentation, "" if none)
    trial: np.ndarray                # T trial numbers (from TrialNum)
    spike_times: list = field(repr=False, default_factory=list)   # spike times of the active channels (s)
    t0: float = 0.0                  # the session covers [t0, t1)
    t1: float = 0.0

    @property
    def mask(self) -> np.ndarray:
        """Trial bins: without Intertrial and without the empty label."""
        return ~np.isin(self.phase, ESCLUSE)

    @property
    def duration_trial(self) -> int:
        """Median trial duration, in bins (from TrialNum, over the whole session)."""
        return duration_median(self.trial)

    @property
    def rate_hz(self) -> np.ndarray:
        return self.counts.sum(0) / (len(self.counts) * BIN_S)


def duration_median(trial) -> int:
    _, n = np.unique(np.asarray(trial), return_counts=True)
    return int(round(np.median(n)))


def load(name: str, min_hz: float = 0.5, check_rep: bool = True) -> Session:
    """A session from the NWB file with the FALCON bins; the counts are compared with X_counts of the replication."""
    from . import io
    sid = SESSIONS[name]
    s = io.load_session(DATA / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_ses-{sid}.nwb",
                        bins="falcon")
    act = io.active_units(s, min_hz=min_hz)
    counts = s.spikes[:, act]
    if check_rep:
        rep = pickle.load(open(REP / sid / "02_rep.pkl", "rb"))["arrays"]
        if not (np.array_equal(rep["ACTIVE"], act) and np.array_equal(rep["X_counts"], counts)):
            raise ValueError(f"{sid}: counts differ from X_counts of the replication")
    phase = np.array([name_base(p) for p in io.phase_per_bin(s)], dtype=object)
    return Session(name=name, sid=sid, key=KEYS[name], counts=counts, active=act, phase=phase,
                    trial=s.trial_num.copy(), spike_times=[s.spike_times[u] for u in act],
                    t0=float(s.times[0] - BIN_S / 2), t1=float(s.times[-1] + BIN_S / 2))


# --------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------
def smooth(X, sigma_ms: float, bin_s: float = BIN_S) -> np.ndarray:
    """Gaussian smoothing along time, channel by channel (mode="nearest" as in notebook 02)."""
    return gaussian_filter1d(np.asarray(X, float), (sigma_ms / 1000) / bin_s, axis=0, mode="nearest")


def pipeline_gallego(counts) -> np.ndarray:
    """1a: square root of the counts, then Gaussian with σ = 50 ms (Gallego 2018)."""
    return smooth(np.sqrt(np.asarray(counts, float)), 50)


def pipeline_silvernagel(counts) -> np.ndarray:
    """1b: counts, Gaussian with σ = 30 ms (Silvernagel 2026)."""
    return smooth(counts, 30)


def pipeline_02(counts) -> np.ndarray:
    """Notebook 02: counts, Gaussian with σ = 50 ms, in Hz (the z-score is applied afterwards)."""
    return smooth(counts, 50) / BIN_S


# --------------------------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------------------------
def eigenvalues(X, zscore: bool = False) -> np.ndarray:
    """Eigenvalues of the column covariance (of the correlation with zscore), in decreasing order."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    if zscore:
        X = X / X.std(0, ddof=1)
    C = X.T @ X / (len(X) - 1)
    return np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)


def share_first(ev, k: int) -> float:
    """Fraction of the variance in the first k eigenvalues."""
    ev = np.asarray(ev, float)
    return float(ev[:k].sum() / ev.sum())


def n_per(ev, q: float) -> int:
    """Minimum number of components with cumulative variance ≥ q."""
    ev = np.asarray(ev, float)
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), q) + 1)


def participation_ratio(ev) -> float:
    ev = np.asarray(ev, float)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


# --------------------------------------------------------------------------------------------
# Nulls
# --------------------------------------------------------------------------------------------
def shifts_allowed(T: int, duration_trial: int, minimum: int = 100, window: int = 100):
    """Allowed shifts (in bins) for the circular shift and excluded fraction.

    Uniform between `minimum` and T − `minimum` (2 s = 100 bins), excluding those within ±`window` bins of a
    multiple m · duration_trial, m ≥ 1."""
    s = np.arange(minimum, T - minimum + 1)
    multiple = np.arange(1, T // duration_trial + 2) * duration_trial
    forbidden = (np.abs(s[:, None] - multiple[None, :]) <= window).any(1)
    return s[~forbidden], float(forbidden.mean())


def circular_shift(X, rng, allowed) -> np.ndarray:
    """Each column circularly shifted by a value drawn from `allowed`, independently per column."""
    X = np.asarray(X)
    T, N = X.shape
    sh = rng.choice(allowed, size=N)
    idx = (np.arange(T)[:, None] - sh[None, :]) % T
    return X[idx, np.arange(N)[None, :]]


def shuffle_channels(X, rng) -> np.ndarray:
    """Altan 2021 shuffle: the bins of each channel permuted independently."""
    return rng.permuted(np.asarray(X), axis=0)


def spectra(Y, mask, zscore: bool, rng, allowed, n_surr: int, null: str = "shift"):
    """Spectrum of the data (bins in `mask`) and of n_surr surrogates.

    Y is the whole preprocessed session (T × N): the null shifts (or shuffles) the whole session,
    then takes the same bins as the data."""
    ev = eigenvalues(Y[mask], zscore)
    if null == "shift":
        sur = [eigenvalues(circular_shift(Y, rng, allowed)[mask], zscore) for _ in range(n_surr)]
    elif null == "shuffle":
        sur = [eigenvalues(shuffle_channels(Y, rng)[mask], zscore) for _ in range(n_surr)]
    else:
        raise ValueError(f"unknown null: {null!r}")
    return ev, np.array(sur)


def parallel_analysis(ev, ev_null, q: float = 95):
    """k_PA: eigenvalues of the data above the q-th percentile of the eigenvalue of the same rank in the surrogates.

    Returns (k_seq, k_all, thresholds): k_seq counts the first consecutive eigenvalues above the threshold
    (it stops at the first one that does not exceed it); k_all counts all ranks above the threshold."""
    ev = np.asarray(ev, float)
    thresholds = np.percentile(np.asarray(ev_null, float), q, axis=0)
    above = ev > thresholds
    k_seq = len(ev) if above.all() else int(np.argmin(above))
    return k_seq, int(above.sum()), thresholds


# --------------------------------------------------------------------------------------------
# Channel draws and summaries
# --------------------------------------------------------------------------------------------
def extract(n_tot: int, n: int, rng) -> np.ndarray:
    """n distinct columns out of n_tot, in increasing order."""
    return np.sort(rng.choice(n_tot, size=n, replace=False))


def summary(v) -> tuple[float, float, float]:
    """(median, minimum, maximum)."""
    v = np.asarray(v, float)
    return float(np.median(v)), float(v.min()), float(v.max())


# --------------------------------------------------------------------------------------------
# Rebinning and Factor Analysis
# --------------------------------------------------------------------------------------------
def ribinna(spike_times, t0: float, t1: float, bin_s: float):
    """Counts in bins of bin_s seconds starting from t0, from the spike times in [t0, t1).

    Returns (counts n_bin × N, edges, complete): the last bin may exceed t1 (complete = False);
    all spikes in [t0, t1) are counted."""
    n = int(np.ceil((t1 - t0) / bin_s - 1e-9))
    edges = t0 + np.arange(n + 1) * bin_s
    out = np.zeros((n, len(spike_times)), dtype=np.int64)
    for c, st in enumerate(spike_times):
        st = np.asarray(st)
        out[:, c] = np.histogram(st[(st >= t0) & (st < t1)], bins=edges)[0]
    complete = edges[1:] <= t1 + 1e-9
    return out, edges, complete


def mask_ribinnata(mask, t0: float, edges, bin_s: float = BIN_S) -> np.ndarray:
    """For new bins with `edges`: true if the bin center falls in a bin of bin_s seconds (from t0) with mask true."""
    centers = (np.asarray(edges)[:-1] + np.asarray(edges)[1:]) / 2
    k = np.floor((centers - t0) / bin_s + 1e-9).astype(int)
    ok = (k >= 0) & (k < len(mask))
    out = np.zeros(len(centers), bool)
    out[ok] = np.asarray(mask, bool)[k[ok]]
    return out


def counts_45ms(s: "Session"):
    """1c: counts at 45 ms rebinned from the spike times; only whole bins with the center in the trial bins."""
    c, edges, complete = ribinna(s.spike_times, s.t0, s.t1, 0.045)
    m = complete & mask_ribinnata(s.mask, s.t0, edges)
    return c[m].astype(float), m


def fa_fit(X, K: int):
    from sklearn.decomposition import FactorAnalysis
    return FactorAnalysis(n_components=K, svd_method="lapack", random_state=0).fit(X)


def ledermann(N: int, K: int) -> bool:
    """Ledermann bound: with K factors FA is identifiable only if (N − K)² ≥ N + K."""
    return (N - K) ** 2 >= N + K


def _cv_one(X, K, tr, te):
    return fa_fit(X[tr], K).score(X[te])


def fa_cv(X, grid, n_blocks: int = 5, n_jobs: int = 1) -> np.ndarray:
    """Mean log-likelihood over the held-out blocks, for each K of the grid (contiguous blocks)."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    folds = list(KFold(n_splits=n_blocks, shuffle=False).split(X))
    ll = Parallel(n_jobs=n_jobs)(delayed(_cv_one)(X, K, tr, te) for K in grid for tr, te in folds)
    return np.array(ll).reshape(len(grid), n_blocks).mean(1)


def eigenvalues_shared(f) -> np.ndarray:
    """Eigenvalues of W Wᵀ (the same, nonzero, as those of Wᵀ W), in decreasing order."""
    W = f.components_.T
    return np.clip(np.linalg.eigvalsh(W.T @ W)[::-1], 0, None)


def dim_shared(f, q: float = 0.95) -> int:
    """Dimensions for the fraction q of the shared variance (Williamson 2016, Oby 2019)."""
    return n_per(eigenvalues_shared(f), q)


def share_population(f) -> float:
    """trace(W Wᵀ) / trace(W Wᵀ + Ψ) (Degenhart 2020)."""
    tr = float((f.components_ ** 2).sum())
    return tr / (tr + float(f.noise_variance_.sum()))


def share_channels(f) -> np.ndarray:
    """Shared fraction of each channel: ||w_i||² / (||w_i||² + ψ_i)."""
    sh = (f.components_ ** 2).sum(0)
    return sh / (sh + f.noise_variance_)


# --------------------------------------------------------------------------------------------
# Saturation (R6)
# --------------------------------------------------------------------------------------------
def _aic(rss, n, k):
    return float(n * np.log(max(rss, 1e-300) / n) + 2 * k)


def models_saturation(N, y) -> dict:
    """Least squares of three models and AIC = n ln(RSS/n) + 2k.

    linear y = a + b N (k = 2); power y = a N^b (k = 2); plateau y = A + B exp(−N/τ) (k = 3).
    For the plateau a grid of initial τ is tried and the fit with minimum RSS is kept."""
    from scipy.optimize import curve_fit
    N = np.asarray(N, float)
    y = np.asarray(y, float)
    n = len(y)
    out = {}
    b, a = np.polyfit(N, y, 1)
    rss = float(((y - (a + b * N)) ** 2).sum())
    out["linear"] = dict(par=(a, b), rss=rss, aic=_aic(rss, n, 2))

    def pot(x, a, b):
        return a * x ** b
    p0 = np.polyfit(np.log(N), np.log(np.clip(y, 1e-9, None)), 1)[::-1]
    p0 = (np.exp(p0[0]), p0[1])
    try:
        par, _ = curve_fit(pot, N, y, p0=p0, maxfev=20000)
    except RuntimeError:
        par = p0
    rss = float(((y - pot(N, *par)) ** 2).sum())
    out["power"] = dict(par=tuple(par), rss=rss, aic=_aic(rss, n, 2))

    def plat(x, A, B, tau):
        return A + B * np.exp(-x / tau)
    best = None
    for tau0 in (5, 10, 20, 40, 80, 160, 320, 640, 1280):
        e = np.exp(-N / tau0)
        B0, A0 = np.polyfit(e, y, 1)
        try:
            par, _ = curve_fit(plat, N, y, p0=(A0, B0, tau0), maxfev=20000)
        except RuntimeError:
            continue
        rss = float(((y - plat(N, *par)) ** 2).sum())
        if np.isfinite(rss) and (best is None or rss < best[1]):
            best = (tuple(par), rss)
    out["plateau"] = dict(par=best[0], rss=best[1], aic=_aic(best[1], n, 3))
    aic_p = out["plateau"]["aic"]
    out["saturates"] = bool(aic_p <= out["linear"]["aic"] - 2 and aic_p <= out["power"]["aic"] - 2)
    return out


# --------------------------------------------------------------------------------------------
# 1g: phase segments, shuffled trials, Frankenstein, even/odd check, Cheater
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    trial: int
    start: int                      # index of the first bin in the session
    end: int                        # index after the last bin
    phase: str                        # phase label
    name: str                        # phase, or "Presentation>" + next phase name / "Presentation>end"
    key: tuple                    # (name, occurrence of the name in the trial, from 1)

    @property
    def n(self) -> int:
        return self.end - self.start


def segments(phase, trial, mask) -> list[Segment]:
    """Contiguous segments of the same phase within each trial (bins in `mask`), in time order.

    The first segment of a trial is discarded if it is Release (it is the tail of the previous trial)."""
    phase = np.asarray(phase).astype(str)
    trial = np.asarray(trial)
    mask = np.asarray(mask, bool)
    out = []
    for t in np.unique(trial[mask]):
        idx = np.flatnonzero((trial == t) & mask)
        cuts = np.flatnonzero((np.diff(idx) != 1) | (phase[idx[1:]] != phase[idx[:-1]])) + 1
        runs = np.split(idx, cuts)
        if phase[runs[0][0]] == "Release":
            runs = runs[1:]
        phases = [phase[c[0]] for c in runs]
        occ = Counter()
        for i, c in enumerate(runs):
            f = phases[i]
            if f == "Presentation":
                after = next((g for g in phases[i + 1:] if g != "Presentation"), "end")
                name = f"Presentation>{after}"
            else:
                name = f
            occ[name] += 1
            out.append(Segment(int(t), int(c[0]), int(c[-1]) + 1, f, name, (name, occ[name])))
    return out


def bin_of_segments(segs) -> np.ndarray:
    """Session indices of the bins covered by the segments, in the order of the segments."""
    return np.concatenate([np.arange(g.start, g.end) for g in segs])


def matrix_ricampionamento(L_da: int, L_a: int) -> np.ndarray:
    """M (L_a × L_da): linear interpolation on the normalized time (bin centers), constant ends."""
    u_da = (np.arange(L_da) + 0.5) / L_da
    u_a = (np.arange(L_a) + 0.5) / L_a
    M = np.zeros((L_a, L_da))
    if L_da == 1:
        M[:, 0] = 1
        return M
    pos = np.clip(u_a * L_da - 0.5, 0, L_da - 1)
    i0 = np.minimum(np.floor(pos).astype(int), L_da - 2)
    w = pos - i0
    M[np.arange(L_a), i0] = 1 - w
    M[np.arange(L_a), i0 + 1] += w
    return M


def _indices(segs):
    per_key, per_phase = defaultdict(list), defaultdict(list)
    for g in segs:
        per_key[g.key].append(g)
        per_phase[g.phase].append(g)
    return per_key, per_phase


def shuffle_trial(Y, segs, rng):
    """'Phases from different trials': for each channel and segment, the segment with the same key from another trial
    drawn at random, resampled to the same duration; if the key is missing in the other trials, a segment of the
    same phase from another trial; if that is missing too, the segment of the same trial.

    Y: whole preprocessed session (T × N). Returns the matrix over the bins of the segments (in order) and the
    number of (channel, segment) per case: "key", "phase", "same"."""
    Y = np.asarray(Y, float)
    N = Y.shape[1]
    per_key, per_phase = _indices(segs)
    out = np.empty((sum(g.n for g in segs), N))
    cases = Counter({"key": 0, "phase": 0, "same": 0})
    pos = 0
    for g in segs:
        cand, kind = [h for h in per_key[g.key] if h.trial != g.trial], "key"
        if not cand:
            cand, kind = [h for h in per_phase[g.phase] if h.trial != g.trial], "phase"
        if not cand:
            cand, kind = [g], "same"
        choice = rng.integers(len(cand), size=N)
        for j in np.unique(choice):
            h = cand[j]
            col = np.flatnonzero(choice == j)
            out[pos:pos + g.n, col] = matrix_ricampionamento(h.n, g.n) @ Y[h.start:h.end][:, col]
        cases[kind] += N
        pos += g.n
    return out, dict(cases)


def frankenstein(Ys: dict, segs: dict, channels: dict, structure: str, rng):
    """Frankenstein: channels from different sessions put back onto the trial structure of `structure`.

    Ys[s]: whole preprocessed session s (T_s × N_s); segs[s]: its segments; channels[s]: columns of Ys[s]
    assigned to s. For each channel and each segment of `structure`, from a random trial of the assigned
    session the segment with the same key is taken, resampled to the duration of the segment of
    `structure`; if the key is missing, a segment with the same phase; if that is missing too, the channel
    mean in that session. At the end the mean of each channel is removed.
    Output columns: in the order of the sessions in `channels`, then of the columns. Returns (matrix, cases)."""
    S = segs[structure]
    order = [(s, c) for s in channels for c in channels[s]]
    out = np.empty((sum(g.n for g in S), len(order)))
    cases = Counter({"key": 0, "phase": 0, "mean": 0})
    idx = {s: _indices(segs[s]) for s in channels}
    means = {s: np.asarray(Ys[s], float).mean(0) for s in channels}
    pos = 0
    for g in S:
        j0 = 0
        for s, cols in channels.items():
            cols = np.asarray(cols)
            dest = np.arange(j0, j0 + len(cols))
            j0 += len(cols)
            per_key, per_phase = idx[s]
            cand, kind = per_key.get(g.key, []), "key"
            if not cand:
                cand, kind = per_phase.get(g.phase, []), "phase"
            cases[kind if cand else "mean"] += len(cols)
            if not cand:
                out[pos:pos + g.n, dest] = means[s][cols][None, :]
                continue
            choice = rng.integers(len(cand), size=len(cols))
            for j in np.unique(choice):
                h = cand[j]
                k = np.flatnonzero(choice == j)
                out[pos:pos + g.n, dest[k]] = matrix_ricampionamento(h.n, g.n) @ Ys[s][h.start:h.end][:, cols[k]]
        pos += g.n
    return out - out.mean(0), dict(cases)


def check_even_odd(Y, segs, mode: str = "phase") -> np.ndarray:
    """For each channel: cov(mean of the even trials, mean of the odd trials) / variance of the single trial.

    mode="phase": each segment resampled to the median duration of its key, keys aligned across trials
    (a key is included if it is present in at least one even and one odd trial); mode="start": the bins of the segments of
    each trial from the start, cut to the length of the shortest trial."""
    Y = np.asarray(Y, float)
    trials = sorted({g.trial for g in segs})
    if mode == "phase":
        per_key = defaultdict(list)
        for g in segs:
            per_key[g.key].append(g)
        lung = {k: max(1, int(round(np.median([g.n for g in v])))) for k, v in per_key.items()}
        keys = [k for k, v in per_key.items()
                  if any(g.trial % 2 == 0 for g in v) and any(g.trial % 2 == 1 for g in v)]
        even, odd, single = [], [], defaultdict(list)
        for k in keys:
            ric = {g.trial: matrix_ricampionamento(g.n, lung[k]) @ Y[g.start:g.end] for g in per_key[k]}
            even.append(np.mean([v for t, v in ric.items() if t % 2 == 0], 0))
            odd.append(np.mean([v for t, v in ric.items() if t % 2 == 1], 0))
            for t, v in ric.items():
                single[t].append(v)
        A, B = np.concatenate(even), np.concatenate(odd)
        var1 = np.mean([np.concatenate(v).var(0, ddof=1) for v in single.values()], 0)
    elif mode == "start":
        per_trial = {t: Y[bin_of_segments([g for g in segs if g.trial == t])] for t in trials}
        L = min(len(v) for v in per_trial.values())
        A = np.mean([v[:L] for t, v in per_trial.items() if t % 2 == 0], 0)
        B = np.mean([v[:L] for t, v in per_trial.items() if t % 2 == 1], 0)
        var1 = np.mean([v[:L].var(0, ddof=1) for v in per_trial.values()], 0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    cov = ((A - A.mean(0)) * (B - B.mean(0))).sum(0) / (len(A) - 1)
    return cov / var1


def group_A(var, n_high: int = 12, n_low: int = 54) -> np.ndarray:
    """Cheater, group A: the most variable plus the least variable channels (indices, from the variance only)."""
    o = np.argsort(var)
    return np.sort(np.concatenate([o[len(o) - n_high:], o[:n_low]]))


def group_B(var, n: int = 66) -> np.ndarray:
    """Cheater, group B: n contiguous channels in variance order with the smallest max/min ratio."""
    o = np.argsort(var)
    v = np.asarray(var, float)[o]
    i = int(np.argmin(v[n - 1:] / v[:len(v) - n + 1]))
    return np.sort(o[i:i + n])


def share_more_variables(var, k: int = 12) -> float:
    """Share of the total variance in the k most variable channels."""
    v = np.sort(np.asarray(var, float))[::-1]
    return float(v[:k].sum() / v.sum())


# --------------------------------------------------------------------------------------------
# Complete procedures (used by the scripts and the tests)
# --------------------------------------------------------------------------------------------
MEASURES = {
    "var12": lambda ev: 100 * share_first(ev, 12),     # % of variance in the first 12 PCs (R1)
    "n60": lambda ev: n_per(ev, 0.60),                  # components for 60% (1e)
    "n80": lambda ev: n_per(ev, 0.80),                  # components for 80% (R2)
}


def measures_con_null(ev, sur, measures=MEASURES) -> dict:
    """For each measure: value on the data, median (and min, max) over the surrogates, difference data − median."""
    out = {}
    for name, f in measures.items():
        d = float(f(ev))
        v = [f(e) for e in sur]
        nn = float(np.median(v))
        out.update({name: d, f"{name}_null": nn, f"{name}_diff": d - nn,
                    f"{name}_null_min": float(np.min(v)), f"{name}_null_max": float(np.max(v))})
    return out


def log(recs, metrics, **keys):
    """Registry: one row per draw (seed = seed) and one summary row (seed = None) with median, min, max.

    keys: all keys of results.log_run except seed."""
    from . import results
    results.check_keys(seed=None, **keys)
    for r in recs:
        results.log_run({m: r[m] for m in metrics if m in r}, seed=r["seed"], **keys)
    summ = {}
    for m in metrics:
        if m in recs[0]:
            med, lo, hi = summary([r[m] for r in recs])
            summ.update({f"{m}_median": med, f"{m}_min": lo, f"{m}_max": hi})
    summ["n_draws"] = len(recs)
    results.log_run(summ, seed=None, **keys)
    return summ


def draws_con_null(Y, mask, n_channels, semi, duration_trial, zscore=False, n_surr=20,
                         measures=MEASURES) -> list[dict]:
    """For each seed: n_channels drawn at random (same seed -> same channels and same shifts), measures
    on the data and median over n_surr circular-shift surrogates of the same channels.

    Y: whole preprocessed session (T × N); n_channels=None = all channels, without drawing."""
    allowed, _ = shifts_allowed(len(Y), duration_trial)
    out = []
    for seed in semi:
        rng = np.random.default_rng(seed)
        col = np.arange(Y.shape[1]) if n_channels is None else extract(Y.shape[1], n_channels, rng)
        ev, sur = spectra(Y[:, col], mask, zscore, rng, allowed, n_surr)
        out.append({"seed": int(seed), "n_channels": len(col), **measures_con_null(ev, sur, measures)})
    return out


def r1(Z, zscore=False) -> float:
    return 100 * share_first(eigenvalues(Z, zscore), 12)


def r1_null(Y, rows, zscore, rng, allowed, n_surr) -> float:
    """Median of R1 over n_surr surrogates: circular shift of Y (T × N), then the chosen rows."""
    return float(np.median([r1(circular_shift(Y, rng, allowed)[rows], zscore) for _ in range(n_surr)]))


@dataclass
class Data1g:
    """A session for 1g: Y = whole session preprocessed with pipeline 1a (T × N active channels)."""
    Y: np.ndarray
    phase: np.ndarray
    trial: np.ndarray
    active: np.ndarray               # original channel indices (for the Frankenstein)
    mask: np.ndarray = None

    def __post_init__(self):
        if self.mask is None:
            self.mask = ~np.isin(np.asarray(self.phase).astype(str), ESCLUSE)
        self.segs = segments(self.phase, self.trial, self.mask)
        self.rows = bin_of_segments(self.segs)
        self.duration = duration_median(self.trial)


def frankenstein_1g(sess: dict, seed: int, n_channels: int = 66, n_surr: int = 20, structure: str = None) -> dict:
    """1g, one draw: real data, phases from different trials and null for each session, plus the Frankenstein.

    sess: {name: Data1g}. The Frankenstein uses n_channels distinct indices among the channels active in all
    sessions, n_channels / len(sess) per session, on the trial structure of `structure` (the first)."""
    names = list(sess)
    structure = structure or names[0]
    rec = {"seed": int(seed)}
    for i, s in enumerate(names):
        d = sess[s]
        rng = np.random.default_rng([seed, i])
        col = extract(d.Y.shape[1], n_channels, rng)
        Yc = d.Y[:, col]
        mix, cases = shuffle_trial(Yc, d.segs, rng)
        allowed, _ = shifts_allowed(len(Yc), d.duration)
        for z, suff in ((False, ""), (True, "_z")):
            data, mx = r1(Yc[d.rows], z), r1(mix, z)
            null = r1_null(Yc, d.rows, z, rng, allowed, n_surr)
            rec.update({f"{s}_data{suff}": data, f"{s}_shuffled{suff}": mx, f"{s}_null{suff}": null,
                        f"{s}_share{suff}": (mx - null) / (data - null)})
        rec.update({f"{s}_cases_{k}": v for k, v in cases.items()})
    rng = np.random.default_rng([seed, 99])
    common = sess[names[0]].active
    for s in names[1:]:
        common = np.intersect1d(common, sess[s].active)
    chosen = rng.choice(common, size=n_channels, replace=False)
    per = n_channels // len(names)
    channels = {s: np.searchsorted(sess[s].active, np.sort(chosen[i * per:(i + 1) * per])) for i, s in enumerate(names)}
    for s in names:
        assert np.all(np.isin(sess[s].active[channels[s]], chosen))
    F, cases = frankenstein({s: sess[s].Y for s in names}, {s: sess[s].segs for s in names}, channels, structure, rng)
    allowed, _ = shifts_allowed(len(F), sess[structure].duration)
    for z, suff in ((False, ""), (True, "_z")):
        rec[f"frank{suff}"] = r1(F, z)
        rec[f"frank_null{suff}"] = r1_null(F, np.arange(len(F)), z, rng, allowed, n_surr)
    rec.update({f"frank_cases_{k}": v for k, v in cases.items()})
    return rec


def cheater(Y, mask, duration_trial, n_surr: int = 20, n_random: int = 200, n: int = 66) -> dict:
    """1g, Cheater: groups chosen only from the channel variance (pipeline 1a, bins in mask).

    For groups A and B: share of the variance in their 12 most variable channels, R1 on the data and on the null
    (median of n_surr surrogates, seed 0). On n_random random draws (seeds 0, 1, ...): r² between that
    share and R1 on the data."""
    Z = Y[mask]
    var = Z.var(0, ddof=1)
    allowed, _ = shifts_allowed(len(Y), duration_trial)
    out = {}
    for name, g in (("A", group_A(var, 12, n - 12)), ("B", group_B(var, n))):
        rng = np.random.default_rng(0)
        out[f"{name}_share12"] = 100 * share_more_variables(var[g], 12)
        out[f"{name}_data"] = r1(Z[:, g])
        out[f"{name}_null"] = r1_null(Y[:, g], mask, False, rng, allowed, n_surr)
        out[f"{name}_ratio_var"] = float(var[g].max() / var[g].min())
    q, r = [], []
    for seed in range(n_random):
        col = extract(Y.shape[1], n, np.random.default_rng(seed))
        q.append(100 * share_more_variables(var[col], 12))
        r.append(r1(Z[:, col]))
    out["random_share12_median"], out["random_data_median"] = float(np.median(q)), float(np.median(r))
    out["r2"] = float(np.corrcoef(q, r)[0, 1] ** 2)
    return out


# --------------------------------------------------------------------------------------------
# Step 3: clock + neurons (rule R7)
# --------------------------------------------------------------------------------------------
PHASES_MOVEMENT = ("Reach", "Orient", "SnapTo", "Shape", "Grasp", "Carry", "Orient2", "Release")   # io.MOVEMENT_PHASES
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)   # weight of the clock relative to the latent block in (c)


def points_phase(phase, step: int = 5) -> np.ndarray:
    """Evaluation points of the replication (notebook 04, par. 1.2): bins of the 8 movement phases with index a multiple of 5."""
    b = np.flatnonzero(np.isin(np.asarray(phase).astype(str), PHASES_MOVEMENT))
    return b[b % step == 0]


def blocks_trial(b, trial, n: int = 5) -> list:
    """The 5 blocks of the replication: GroupKFold on the trials of the points (3 test trials per block)."""
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n).split(b, groups=np.asarray(trial)[b]))


def _knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=10)


def _bacc(y, p) -> float:
    import warnings
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(y, p) * 100)


def combine(clock, lat, ref, w: float) -> np.ndarray:
    """Input (c): standardized clock × w, standardized latents / √d; means and deviations from the bins `ref`.

    w = inf: clock only; w = 0: latents only (standardized)."""
    clock = np.asarray(clock, float).reshape(-1, 1)
    lat = np.asarray(lat, float)
    c = (clock - clock[ref].mean()) / clock[ref].std()
    L = (lat - lat[ref].mean(0)) / lat[ref].std(0) / np.sqrt(lat.shape[1])
    if np.isinf(w):
        return c
    if w == 0:
        return L
    return np.hstack([w * c, L])


def choose_weight(clock, lat, phase, trial, b_tr, weights=WEIGHTS, n_inner: int = 4) -> float:
    """Clock weight by inner cross-validation on the training points only (blocks of trials).
    At equal accuracy the highest weight wins (closest to the clock alone)."""
    from sklearn.model_selection import GroupKFold
    phase = np.asarray(phase)
    folds = list(GroupKFold(n_splits=n_inner).split(b_tr, groups=np.asarray(trial)[b_tr]))
    acc = {}
    for w in weights:
        X = combine(clock, lat, b_tr, w)
        pred = np.empty(len(b_tr), dtype=object)
        for trn, tst in folds:
            pred[tst] = _knn().fit(X[b_tr[trn]], phase[b_tr[trn]]).predict(X[b_tr[tst]])
        acc[w] = _bacc(phase[b_tr], pred)
    best = max(acc.values())
    return max(w for w, a in acc.items() if a >= best - 1e-9)


def latents_permutati(lat, trial, clock, perm: dict, bin_s: float = BIN_S) -> np.ndarray:
    """Latents with swapped trials: each bin of trial i takes the latent of trial perm[i] at the same time
    from the start of the trial (the clock stays); if that trial is shorter, its last bin."""
    lat = np.asarray(lat)
    trial = np.asarray(trial)
    out = lat.copy()
    start = {t: int(np.flatnonzero(trial == t)[0]) for t in np.unique(trial)}
    lung = {t: int((trial == t).sum()) for t in np.unique(trial)}
    for i, j in perm.items():
        idx = np.flatnonzero(trial == i)
        u = np.rint(np.asarray(clock)[idx] / bin_s).astype(int)
        out[idx] = lat[start[j] + np.minimum(u, lung[j] - 1)]
    return out


def clock_neurons(clock, lat_block: dict, phase, trial, n_perm: int = 200, seed: int = 0,
                     weights_fixed=(1.0, 0.2)) -> dict:
    """Step 3: balanced accuracy (%) with (a) the clock only, (b) the latents as in the replication, (c) clock +
    latents with the weight chosen on the training trials; null of (c) − (a): latents permuted between whole test
    trials (within each block), clock fixed, n_perm times; p = (1 + #{null ≥ observed}) / (1 + n_perm).

    lat_block[k]: latents of the whole session from the model of block k (T × d). weights_fixed: variants of (c) with
    fixed weight (1 = clock and latent block with the same variance; 0.2 = z-score of all columns)."""
    phase = np.asarray(phase).astype(str)
    trial = np.asarray(trial)
    b = points_phase(phase)
    folds = blocks_trial(b, trial)
    y = phase[b]
    pred = {k: np.empty(len(b), dtype=object) for k in ("a", "b", "c", *[f"c_w{w:g}" for w in weights_fixed])}
    models, chosen = [], []
    for k, (trn, tst) in enumerate(folds):
        lat = lat_block[k]
        btr, bte = b[trn], b[tst]
        Xa = np.asarray(clock, float)[:, None]
        pred["a"][tst] = _knn().fit(Xa[btr], phase[btr]).predict(Xa[bte])
        pred["b"][tst] = _knn().fit(lat[btr], phase[btr]).predict(lat[bte])
        w = choose_weight(clock, lat, phase, trial, btr)
        chosen.append(w)
        fit = {}
        for name, ww in (("c", w), *[(f"c_w{x:g}", x) for x in weights_fixed]):
            X = combine(clock, lat, btr, ww)
            fit[name] = (ww, _knn().fit(X[btr], phase[btr]))
            pred[name][tst] = fit[name][1].predict(X[bte])
        models.append((btr, bte, fit))
    acc = {k: _bacc(y, p) for k, p in pred.items()}
    rng = np.random.default_rng(seed)
    nulls = {k: [] for k in acc if k.startswith("c")}
    test_trials = [np.unique(trial[b[tst]]) for _, tst in folds]
    for _ in range(n_perm):
        pp = {k: np.empty(len(b), dtype=object) for k in nulls}
        for k, ((trn, tst), (btr, bte, fit)) in enumerate(zip(folds, models)):
            tt = test_trials[k]
            latp = latents_permutati(lat_block[k], trial, clock, dict(zip(tt, rng.permutation(tt))))
            for name, (ww, clf) in fit.items():
                X = combine(clock, latp, btr, ww)            # latp changes only the bins of the test trials
                pp[name][tst] = clf.predict(X[bte])
        for name in nulls:
            nulls[name].append(_bacc(y, pp[name]))
    out = {"acc_a": acc["a"], "acc_b": acc["b"], "n_points": len(b), "n_classes": len(np.unique(y)),
           "chance": 100 / len(np.unique(y)), "n_perm": n_perm}
    for name, v in nulls.items():
        v = np.array(v)
        suff = "" if name == "c" else name[1:]
        out.update({f"acc_c{suff}": acc[name], f"diff_c_a{suff}": acc[name] - acc["a"],
                    f"null_c_median{suff}": float(np.median(v)), f"null_c_p95{suff}": float(np.percentile(v, 95)),
                    f"null_c_max{suff}": float(v.max()),
                    f"p{suff}": float((1 + (v >= acc[name] - 1e-9).sum()) / (1 + n_perm))})
    out.update({f"weight_block{k}": float(w) for k, w in enumerate(chosen)})
    # optional: accuracy within the deciles of the clock
    dec = np.quantile(np.asarray(clock)[b], np.linspace(0, 1, 11))
    q = np.clip(np.searchsorted(dec, np.asarray(clock)[b], side="right") - 1, 0, 9)
    for i in range(10):
        m = q == i
        out.update({f"decile{i + 1}_{k}": _bacc(y[m], pred[k][m]) for k in ("a", "b", "c")})
    return out


# --------------------------------------------------------------------------------------------
# Step 5a: CCA as in Gallego 2020
# --------------------------------------------------------------------------------------------
def correlations_canonical(A, B) -> np.ndarray:
    """Canonical correlations between the columns of A and B (same rows), in decreasing order."""
    A = np.asarray(A, float) - np.mean(A, 0)
    B = np.asarray(B, float) - np.mean(B, 0)
    qa, qb = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)


def correlations_non_aligned(A, B) -> np.ndarray:
    """|correlation| between column i of A and column i of B (the sign of the PCs is arbitrary)."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return np.array([abs(np.corrcoef(A[:, i], B[:, i])[0, 1]) for i in range(A.shape[1])])


def windows_phase(epochs, trial_di, t0: float, bin_s: float, n_bin: int, n_tot: int, phases=PHASES_MOVEMENT) -> list:
    """Windows of n_bin bins (of bin_s seconds, starting from t0) from the start of each movement phase.

    epochs: DataFrame with start_time and tag; trial_di(t): trial at time t. Condition = (phase, odd/even)
    as in par. 8 of notebook 02. Windows that fall outside the session are discarded."""
    out = []
    for _, e in epochs.iterrows():
        tag = name_base(e.tag)
        if tag not in phases:
            continue
        a = int(np.floor((e.start_time - t0) / bin_s + 1e-9))
        if a < 0 or a + n_bin > n_tot:
            continue
        tr = int(trial_di(e.start_time))
        out.append(dict(cond=(tag, "odd" if tr % 2 == 1 else "even"), trial=tr, start=a, stop=a + n_bin))
    return out


def block_conditions(L, windows, choice: dict) -> np.ndarray:
    """Concatenate the chosen windows: choice[cond] = list of repetition indices of the condition; conditions
    in the order of the keys of `choice`. L: latents of the whole session (bins × d)."""
    per = defaultdict(list)
    for w in windows:
        per[w["cond"]].append(w)
    return np.concatenate([L[per[c][i]["start"]:per[c][i]["stop"]] for c in choice for i in choice[c]])


def means_conditions(L, windows, choice: dict) -> np.ndarray:
    """Like block_conditions, but for each condition the mean of the chosen repetitions (one window per condition)."""
    per = defaultdict(list)
    for w in windows:
        per[w["cond"]].append(w)
    return np.concatenate([np.mean([L[per[c][i]["start"]:per[c][i]["stop"]] for i in choice[c]], 0) for c in choice])


# --------------------------------------------------------------------------------------------
# Step 5d: GPFA with fewer channels (comparison with Yu 2009)
# --------------------------------------------------------------------------------------------
def sequences_gpfa(X, windows, columns=None):
    """Elephant format: for each window (T, square root of the counts channels × bins), as in notebook 02 (par. 10)."""
    X = np.asarray(X)
    columns = np.arange(X.shape[1]) if columns is None else np.asarray(columns)
    return np.array([(w["stop"] - w["start"], np.sqrt(X[w["start"]:w["stop"]][:, columns].T.astype(float)))
                     for w in windows], dtype=[("T", int), ("y", "O")])


def gpfa_fit(seqs, d: int, iters: int = 500, tau_init: float = 100.0, bin_ms: float = 20.0):
    """Elephant GPFA (gpfa_core.fit) with fixed seed and hidden messages, as fit_gpfa of notebook 02."""
    import contextlib
    import logging
    import os
    import warnings
    import elephant.utils
    from elephant.gpfa import gpfa_core
    elephant.utils.logger.setLevel(logging.ERROR)
    np.random.seed(0)
    with warnings.catch_warnings(), open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        warnings.simplefilter("ignore")
        params, _ = gpfa_core.fit(seqs, x_dim=d, bin_width=bin_ms, em_max_iters=iters, tau_init=tau_init)
    return params


def gpfa_ll_per_bin(params, seqs) -> float:
    """Mean log-likelihood per bin on whole windows (gpfa_ll_per_bin of notebook 02)."""
    import warnings
    from elephant.gpfa import gpfa_core
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ll = gpfa_core.exact_inference_with_ll(seqs, params, get_ll=True)
    return float(ll / sum(int(s[0]) for s in seqs))


def choose_d(ll_test, grid) -> tuple[int, int]:
    """Rule of notebook 02 (par. 10.2) on ll_test (len(grid) × blocks): (d with the highest mean score,
    smallest d ≤ that of the best that the best does not beat in all blocks)."""

    ll_test = np.asarray(ll_test, float)
    i_best = int(np.argmax(ll_test.mean(1)))
    beats = (ll_test[i_best][None, :] > ll_test).sum(1)
    d_gp = min(d for d, b in zip(grid, beats) if d <= grid[i_best] and b < ll_test.shape[1])
    return int(grid[i_best]), int(d_gp)
