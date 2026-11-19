"""Phase 2 of the literature comparison experiment (5/11): robustness of Parallel Analysis, intrinsic dimension, TME null.

Rules R8-R10 and clarifications P1-P8 in docs/literature/rules_decision.md (commit 1aa7c5d). Reuses literature.py
(loading with the FALCON bins, smoothing, Parallel Analysis with circular shift) and ports here, unchanged, the
functions of notebook 03 (random-phase twins, residual variance, advantage of Isomap over PCA).
Tests: tests/test_literature_phase2.py.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform

from . import literature

SIGMI = (20, 50, 100)                     # R8: grid of σ (ms)
K_NN = 10                                 # notebook 03: Isomap neighbors
K_PCA = 25                                # notebook 03: PCA before Isomap
D_MAX = 15                                # notebook 03: coordinates kept
STEP = 5                                  # notebook 03: one bin every 5 (100 ms)
GRID_N = (500, 1000, 2000, 5000)       # R9: points per draw (plus "all points")
N_DRAWS = 10                         # R9: draws per n (seeds 0-9)
ESTIMATORS = ("TwoNN", "MLE10", "MLE20", "CorrInt", "DANCo")


# --------------------------------------------------------------------------------------------
# R8: pipeline with variable σ
# --------------------------------------------------------------------------------------------
def pipeline_02_sigma(counts, sigma_ms: float) -> np.ndarray:
    """Notebook 02 with variable σ: counts, Gaussian, in Hz (the z-score, if any, is applied afterwards)."""
    return literature.smooth(counts, sigma_ms) / literature.BIN_S


def pipeline_gallego_sigma(counts, sigma_ms: float) -> np.ndarray:
    """Gallego 2018 with variable σ: square root of the counts, then Gaussian; without z-score."""
    return literature.smooth(np.sqrt(np.asarray(counts, float)), sigma_ms)


def grid_r8():
    """The 9 combinations of R8: (name, pipeline, σ, z-score, trial bins only)."""
    out = []
    for sigma in SIGMI:
        for z in (True, False):
            out.append((f"02_s{sigma}_{'z' if z else 'noz'}", "02", sigma, z, False))
    for sigma in SIGMI:
        out.append((f"gallego_s{sigma}", "gallego", sigma, False, True))
    return out


def k_pa(Y, mask, zscore: bool, allowed, n_surr: int = 200, seed: int = 0) -> int:
    """k_PA (first consecutive eigenvalues above the 95th percentile) with the circular shift, as in Step 2."""
    rng = np.random.default_rng(seed)
    ev, sur = literature.spectra(Y, mask, zscore, rng, allowed, n_surr, "shift")
    return literature.parallel_analysis(ev, sur)[0]


# --------------------------------------------------------------------------------------------
# Notebook 03: twins, residual variance, advantage of Isomap
# --------------------------------------------------------------------------------------------
def phases_random(Xt, seed: int) -> np.ndarray:
    """Linear Gaussian twin (phase_randomize of notebook 03): same Fourier amplitudes, random phases equal
    for all columns. Preserves the covariance between columns and the autocorrelation of each column."""
    rng_ = np.random.default_rng(seed)
    Xt = np.asarray(Xt, float)
    F = np.fft.rfft(Xt - Xt.mean(0), axis=0)
    phases = np.exp(2j * np.pi * rng_.uniform(size=F.shape[0]))
    phases[0] = 1
    if len(Xt) % 2 == 0:
        phases[-1] = 1
    return np.fft.irfft(F * phases[:, None], n=len(Xt), axis=0) + Xt.mean(0)


def variance_residual(d_target, Y) -> float:
    """1 − r² between the distances to preserve (pdist format) and those in the map Y (notebook 03)."""
    return 1 - np.corrcoef(d_target, pdist(Y))[0, 1] ** 2


def _above(M):
    return M[np.triu_indices(len(M), k=1)]


def curve_rv(Xp, k: int = K_NN, d_max: int = D_MAX):
    """rv_curves of notebook 03: residual variance of Isomap (geodesic distances) and of PCA (straight-line
    distances) for d = 1..d_max."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    Xp = np.asarray(Xp, float)
    D = squareform(pdist(Xp))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # graph with several pieces: Isomap connects them (as in 03)
        iso = Isomap(n_neighbors=k, n_components=d_max, metric="precomputed").fit(D)
    Yp = PCA(d_max, svd_solver="full").fit_transform(Xp)
    g, e = _above(iso.dist_matrix_), pdist(Xp)
    r_iso = np.array([variance_residual(g, iso.embedding_[:, :d]) for d in range(1, d_max + 1)])
    r_pca = np.array([variance_residual(e, Yp[:, :d]) for d in range(1, d_max + 1)])
    return r_iso, r_pca


def advantage_isomap(Xp, d: int = 3, k: int = K_NN) -> float:
    """Advantage of Isomap over PCA at d dimensions, in percentage points: 100 · (rv_PCA − rv_Isomap)."""
    r_iso, r_pca = curve_rv(Xp, k=k, d_max=max(d, 1))
    return float(100 * (r_pca[d - 1] - r_iso[d - 1]))


def pca25(X, k: int = K_PCA) -> np.ndarray:
    """Points reduced to k dimensions with PCA fitted on the same points (R10 statistic on the means)."""
    from sklearn.decomposition import PCA
    return PCA(n_components=k).fit_transform(np.asarray(X, float))


# --------------------------------------------------------------------------------------------
# R10: tensor of the means per condition (Step 4b)
# --------------------------------------------------------------------------------------------
def tensor_conditions(Z, windows, phases=literature.PHASES_MOVEMENT):
    """T × N × C tensor of the means per condition of Step 4b: conditions = (phase, odd/even trial), in the order
    phase by phase, odd first; Z = smoothed and z-scored data of the whole session; windows = io.phase_windows.
    Returns (tensor, repetitions per condition). The 800 × N PSTH of Step 4b is the tensor stacked by condition."""
    conds = [(ph, v) for ph in phases for v in ("odd", "even")]
    reps = {c: np.stack([Z[w["start"]:w["stop"]] for w in windows
                         if (w["tag"], "odd" if w["trial"] % 2 == 1 else "even") == c]) for c in conds}
    X = np.stack([reps[c].mean(0) for c in conds], axis=2)           # T × N × C
    return X, np.array([len(reps[c]) for c in conds])


def impila(X) -> np.ndarray:
    """From T × N × C to (C·T) × N, condition by condition (the same order as the PSTH of Step 4b)."""
    X = np.asarray(X)
    return np.concatenate([X[:, :, c] for c in range(X.shape[2])], axis=0)


def p_permutation(stat, null) -> float:
    """p = (1 + #{null ≥ stat}) / (1 + number of surrogates)."""
    null = np.asarray(null, float)
    return float((1 + (null >= stat).sum()) / (1 + len(null)))


# --------------------------------------------------------------------------------------------
# R9: intrinsic dimension estimators
# --------------------------------------------------------------------------------------------
def estimate_id(X, name: str) -> float:
    """One intrinsic dimension estimate with scikit-dimension 0.5.0 (clarification P6)."""
    import skdim
    X = np.asarray(X, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "TwoNN":
            return float(skdim.id.TwoNN(discard_fraction=0.1).fit(X).dimension_)
        if name in ("MLE10", "MLE20"):
            return float(skdim.id.MLE().fit(X, n_neighbors=int(name[3:]), comb="mean").dimension_)
        if name == "CorrInt":
            return float(skdim.id.CorrInt(k1=10, k2=20).fit(X).dimension_)
        if name == "DANCo":
            return float(skdim.id.DANCo(k=10, random_state=0).fit(X).dimension_)
    raise ValueError(f"unknown estimator: {name!r}")


def values_n(n_tot: int, grid=GRID_N) -> list:
    """Possible values of n for an input with n_tot points: those of the grid below n_tot, plus n_tot."""
    return [n for n in grid if n < n_tot] + [n_tot]


def draw(n_tot: int, n: int, seed: int) -> np.ndarray:
    """Indices of n points out of n_tot (all if n = n_tot), in increasing order; seed = draw number."""
    if n >= n_tot:
        return np.arange(n_tot)
    return np.sort(np.random.default_rng(seed).choice(n_tot, size=n, replace=False))


def stability(per_n: dict, n_min: int, literature_rel: float = 0.20) -> dict:
    """R9 criterion with clarification P2. per_n: {n: array of the estimates of the draws}; the last n (n_all
    points) has a single estimate. Stable if |med(n_all) − med(n_min)| / med(n_min) < 20% and IQR < 20% of the median at
    every n with several draws between n_min and the second to last."""
    ns = sorted(per_n)
    n_all = ns[-1]
    med = {n: float(np.median(per_n[n])) for n in ns}
    change = abs(med[n_all] - med[n_min]) / abs(med[n_min])
    iqr = {n: float(np.subtract(*np.percentile(per_n[n], [75, 25]))) / abs(med[n]) for n in ns
           if n >= n_min and len(per_n[n]) > 1}
    return {"change": change, "iqr_max": max(iqr.values()) if iqr else np.nan,
            "stable": bool(change < literature_rel and all(v < literature_rel for v in iqr.values()))}


def distinguishable(datum: float, twins) -> bool:
    """The value of the data (all points) falls outside [min, max] of the twins (all points)."""
    g = np.asarray(twins, float)
    return bool(datum < g.min() or datum > g.max())


def prediction_altan(stab_a: bool, dist_a: bool, stab_b: bool, stab_c: bool) -> bool:
    """Altan prediction for an estimator and a session: (a) unstable and not distinguishable, (b) and (c) stable."""
    return (not stab_a) and (not dist_a) and stab_b and stab_c


def estimates_job(path, n: int, seed: int, estimators) -> list:
    """One R9 job for the parallel processes: load the input saved in `path` (.npy), draw n points with
    the seed of the draw and apply the estimators. Returns [(estimator, estimate, seconds)]."""
    import time
    X = np.load(path, mmap_mode="r")
    Xs = np.asarray(X[draw(len(X), n, seed)], float)
    out = []
    for name in estimators:
        t = time.time()
        out.append((name, estimate_id(Xs, name), time.time() - t))
    return out


# --------------------------------------------------------------------------------------------
# R9, post-hoc check: Theiler window (Theiler 1986; notebook 03, par. 5.3)
# --------------------------------------------------------------------------------------------
WINDOWS_MS = (0, 100, 200, 500)          # 0 = as R9
ESTIMATORS_THEILER = ("TwoNN", "MLE10", "MLE20", "CorrInt")


def estimates_theiler(X, group, time, windows_bin, k_max: int = 20, block: int = 1000) -> dict:
    """TwoNN, Levina-Bickel MLE (k = 10, 20) and correlation dimension (k1 = 10, k2 = 20) with the Theiler
    window: among the neighbors of a point, the points of the same group (trial, or condition for the means)
    at a distance in time ≤ w bins are not counted. Same formulas as skdim 0.5.0 (with w = 0 the values coincide with estimate_id).
    Distances computed in blocks of rows, to fit in memory. Returns {(w, estimator): estimate}."""
    import skdim
    from sklearn.metrics import pairwise_distances
    X = np.asarray(X, float)
    g, t = np.asarray(group), np.asarray(time)
    n = len(X)
    knn = {w: np.empty((n, k_max)) for w in windows_bin}

    def blocks():
        for i0 in range(0, n, block):
            r = np.arange(i0, min(i0 + block, n))
            D = pairwise_distances(X[r], X)
            for w in windows_bin:
                Dw = D.copy()
                if w > 0:
                    Dw[(g[r, None] == g[None, :]) & (np.abs(t[r, None] - t[None, :]) <= w)] = np.inf
                Dw[np.arange(len(r)), r] = np.inf                   # the point itself
                yield w, r, Dw

    for w, r, Dw in blocks():
        knn[w][r] = np.sort(np.partition(Dw, k_max - 1, axis=1)[:, :k_max], axis=1)
    radii = {w: (np.median(knn[w][:, 9]), np.median(knn[w][:, 19])) for w in windows_bin}
    counts = {w: np.zeros(2) for w in windows_bin}
    for w, r, Dw in blocks():
        counts[w] += [(Dw < radii[w][0]).sum(), (Dw < radii[w][1]).sum()]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w in windows_bin:
            K = knn[w]
            out[(w, "TwoNN")] = float(skdim.id.TwoNN(discard_fraction=0.1, dist=True).fit(K[:, :2]).dimension_)
            for k in (10, 20):
                Rs = K[:, :k]
                out[(w, f"MLE{k}")] = float(np.mean((k - 1) / np.log(Rs[:, -1:] / Rs).sum(1)))
            r1, r2 = radii[w]
            s1, s2 = counts[w]
            out[(w, "CorrInt")] = float((np.log(s2 / n ** 2) - np.log(s1 / n ** 2)) / np.log(r2 / r1))
    return out


def estimates_job_theiler(path, path_meta, n: int, seed: int, windows_bin) -> list:
    """One job of the Theiler window check: same draw as R9 (draw(n_tot, n, seed)).
    path_meta: .npy 2 × n_tot with group and time (in bins) of each point. Returns [(w, estimator, estimate, s)]."""
    import time
    X = np.load(path, mmap_mode="r")
    gt = np.load(path_meta)
    idx = draw(len(X), n, seed)
    t0 = time.time()
    st = estimates_theiler(np.asarray(X[idx], float), gt[0, idx], gt[1, idx], windows_bin)
    sec = time.time() - t0
    return [(w, s, v, sec) for (w, s), v in st.items()]


# --------------------------------------------------------------------------------------------
# R9, post-hoc checks (steps 2-3): generator with known intrinsic dimension
# Rules in docs/literature/rules_r9_checks.md (commit 067e8f5).
# --------------------------------------------------------------------------------------------
GEN_T = 5507                              # bins of input (a) of S1
GEN_N = 160                               # channels
GEN_UNITS = 64                            # units of the tanh map
GEN_SIGMA_MS = 50                         # preprocessing (a): Gaussian 50 ms = 2.5 bins
QUOTE_CHECK = (0.167, 0.45, 0.58)     # FA share of H1 (S1, slide 5), 0.45, maximum attainable (0.65 is not)
GRID_FA = (5, 10, 15, 20, 30)          # slide 5 (soglia_step8.GRIGLIA_SMUSSATI)


def latents_sphere(T: int, d: int, sigma_bin: float, rng) -> np.ndarray:
    """d + 1 Gaussian signals smoothed in time (σ in bins), normalized point by point: T points on S^d."""
    from scipy.ndimage import gaussian_filter1d
    Z = gaussian_filter1d(rng.standard_normal((T, d + 1)), sigma_bin, axis=0, mode="wrap")
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def map_tanh(Z, rng, n_units: int = GEN_UNITS, n_channels: int = GEN_N) -> np.ndarray:
    """S = tanh(Z W + b) A, then z-score per channel. W ~ N(0, 1)·0.7·√(d+1), b ~ U(−0.5; 0.5), A ~ N(0, 1)."""
    m = Z.shape[1]
    W = rng.standard_normal((m, n_units)) * 0.7 * np.sqrt(m)
    b = rng.uniform(-0.5, 0.5, n_units)
    S = np.tanh(Z @ W + b) @ rng.standard_normal((n_units, n_channels))
    return (S - S.mean(0)) / S.std(0)


def zscore_channels(X) -> np.ndarray:
    """z-score per column; columns with zero variance are dropped."""
    X = np.asarray(X, float)
    sd = X.std(0)
    X = X[:, sd > 0]
    return (X - X.mean(0)) / sd[sd > 0]


def generate_check(freq_hz, g: float, seed: int, d: int = 6, sigma_lat: float = 5, T: int = GEN_T) -> dict:
    """Positive-control generator. Returns counts, expected rates (counts per bin) and S.
    λ = freq · exp(g S − g²/2) · 0.02 s: the mean of exp(g S − g²/2) is 1 if S is standard Gaussian."""
    rng = np.random.default_rng(seed)
    S = map_tanh(latents_sphere(T, d, sigma_lat, rng), rng, n_channels=len(freq_hz))
    lam = np.asarray(freq_hz, float) * np.exp(g * S - g ** 2 / 2) * literature.BIN_S
    return dict(counts=rng.poisson(lam).astype(float), rates=lam, S=S)


def preprocessa_a(X) -> np.ndarray:
    """Preprocessing of input (a): Gaussian σ = 50 ms (2.5 bins), z-score per channel."""
    return zscore_channels(literature.smooth(X, GEN_SIGMA_MS))


def share_fa_slide5(counts, n_jobs: int = 1) -> float:
    """Shared fraction per channel as in slide 5: σ = 50 ms, z-score, K by cross-validation, median."""
    Z = preprocessa_a(counts)
    cv = literature.fa_cv(Z, list(GRID_FA), n_jobs=n_jobs)
    f = literature.fa_fit(Z, GRID_FA[int(np.argmax(cv))])
    return float(np.median(literature.share_channels(f)))


def share_true(gen) -> float:
    """Per-channel median of var(smoothed expected rate) / var(smoothed counts)."""
    num = literature.smooth(gen["rates"], GEN_SIGMA_MS).var(0)
    den = literature.smooth(gen["counts"], GEN_SIGMA_MS).var(0)
    ok = den > 0
    return float(np.median(num[ok] / den[ok]))


def calibrate_g(freq_hz, share: float, sigma_lat: float, seed: int = 0, lo: float = 0.0, hi: float = 3.0,
              tol: float = 0.005, max_iter: int = 14, n_jobs: int = 1) -> tuple:
    """Bisection on g so that share_fa_slide5 equals `share` (the share grows with g). Returns (g, measured share,
    history [(g, share)])."""
    history = []
    g = (lo + hi) / 2
    for _ in range(max_iter):
        g = (lo + hi) / 2
        q = share_fa_slide5(generate_check(freq_hz, g, seed, sigma_lat=sigma_lat)["counts"], n_jobs=n_jobs)
        history.append((g, q))
        if abs(q - share) < tol:
            break
        lo, hi = (g, hi) if q < share else (lo, g)
    g, q = min(history, key=lambda t: abs(t[1] - share))
    return g, q, history


def thin(X, m: int, lag: int) -> np.ndarray:
    """One bin every m, starting from `lag` (0..m−1), along the order of the rows."""
    return np.asarray(X)[lag::m]


def estimates_thinning(X, m: int, estimators=("TwoNN", "MLE10")) -> dict:
    """{estimator: array of the estimates per lag 0..m−1} with one bin every m."""
    out = {s: [] for s in estimators}
    for o in range(m):
        Xo = thin(X, m, o)
        for s in estimators:
            out[s].append(estimate_id(Xo, s))
    return {s: np.array(v) for s, v in out.items()}


def reading_thinning(M10: float, M20: float, d: float, ratio: float = 1.25, band: float = 1.5) -> dict:
    """Readings of the rules: stable if max/min ≤ 1.25; around d if M10 and M20 are in [d − 1.5; d + 1.5]."""

    stable = max(M10, M20) / min(M10, M20) <= ratio
    around = all(d - band <= v <= d + band for v in (M10, M20))
    return dict(stable=bool(stable), around=bool(around))
