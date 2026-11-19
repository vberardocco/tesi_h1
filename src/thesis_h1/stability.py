"""Stability of the FA subspaces across sessions (docs/crosssession_protocol.md, section of 2/11/2026).

- Subspace of a block of 3 trials: FA with K = 15 as in the decoder; we keep the eigenvectors of the shared
  covariance S = L Lᵀ, ordered by shared variance (the FA loadings have no order).
- Alignment index (Elsayed et al. 2016, on S): I(A→B) = tr(Q_Bᵀ S_A Q_B) / Σ_{i≤k} λ_i(S_A), with Q_B the first k
  eigenvectors of S_B; index of the pair = ½ [I(A→B) + I(B→A)]. 1 = same subspace.
- Mean principal angle between the two subspaces (secondary).
- Chance: random subspaces drawn from a covariance C (Q = orth(U Λ^½ r), r Gaussian).
- Tests: Spearman with permutation (exact for n ≤ 8), partial Spearman, critical value of |ρ|, Mantel test with
  permutation of the days, slope with bootstrap over days.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

K = 15                      # FA factors, as in the decoder (02_new §9)
K_SENS = 5                  # sensitivity: first 5 directions by shared variance from the fit with K = 15
KS = (K, K_SENS)
SD_MIN = 0.1                # sd minima nello z-score del blocco: sd di Poisson a 0.5 Hz in bin da 20 ms, √(0.5 · 0.02)
BLOCK = 3                   # trial per blocco
N_DRAWS = 500               # chance draws per pair
CHANCE = ("allbins", "cond", "unif")


# ============================================================================================
# FA, subspaces and measures
# ============================================================================================
def zscore_block(X, sd_min=SD_MIN):
    """z-score within the block with minimum sd sd_min. Also returns how many channels have sd = 0 and sd < sd_min."""
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.maximum(sd, sd_min), int((sd == 0).sum()), int((sd < sd_min).sum())


def fit_loadings(X, k=K):
    """FA loadings (channels × k) as FAm of the decoder: svd_method="lapack", random_state=0."""
    from sklearn.decomposition import FactorAnalysis
    return FactorAnalysis(k, svd_method="lapack", random_state=0).fit(X).components_.T


@dataclass
class Sub:
    """FA subspace: U (channels × K) eigenvectors of S = L Lᵀ ordered by shared variance, lam their eigenvalues."""
    U: np.ndarray
    lam: np.ndarray

    @classmethod
    def from_loadings(cls, L):
        U, s, _ = np.linalg.svd(np.asarray(L, float), full_matrices=False)    # L = U s Vᵀ  ->  S = U s² Uᵀ
        return cls(U, s ** 2)

    def basis(self, k):
        return self.U[:, :k]


def captured(A: Sub, Q):
    """tr(Qᵀ S_A Q), for a basis Q (channels × k) or a stack of bases (n × channels × k)."""
    P = np.einsum("ci,...ck->...ik", A.U, Q)
    return np.einsum("i,...ik->...", A.lam, P ** 2)


def index_dir(A: Sub, Q, k):
    """I(A→Q): shared variance of A captured by Q, relative to the maximum possible with k directions."""
    return captured(A, Q) / A.lam[:k].sum()


def align_index(A: Sub, B: Sub, k):
    return 0.5 * (index_dir(A, B.basis(k), k) + index_dir(B, A.basis(k), k))


def mean_angle(Qa, Qb):
    """Mean principal angle in degrees between orthonormal bases (also stacks n × channels × k)."""
    s = np.linalg.svd(np.swapaxes(Qa, -1, -2) @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0))).mean(-1)


def pair_measures(A: Sub, B: Sub, ks=KS):
    out = {}
    for k in ks:
        out[f"index_k{k}"] = float(align_index(A, B, k))
        out[f"angle_k{k}"] = float(mean_angle(A.basis(k), B.basis(k)))
    return out


# ============================================================================================
# Elsayed-style chance
# ============================================================================================
def sampler(C):
    """Matrix M = U Λ^½ of the covariance C: the columns of M r (r Gaussian) follow the shape of the variance."""
    ev, U = np.linalg.eigh(np.asarray(C, float))
    return U * np.sqrt(np.clip(ev, 0.0, None))


def draw_bases(M, k, n, rng):
    """n random orthonormal bases (n × channels × k) from M (see sampler)."""
    Q, _ = np.linalg.qr(M @ rng.normal(size=(n, M.shape[0], k)))
    return Q


def chance(A: Sub, B: Sub, k, Q1, Q2):
    """Chance of the pair for each draw: index ½ [I(A→Q1) + I(B→Q2)] and mean angle between Q1 and Q2."""
    return 0.5 * (index_dir(A, Q1, k) + index_dir(B, Q2, k)), mean_angle(Q1, Q2)


def chance_summary(idx, ang, prefix):
    return {f"{prefix}_index_p5": float(np.percentile(idx, 5)), f"{prefix}_index_p50": float(np.percentile(idx, 50)),
            f"{prefix}_index_p95": float(np.percentile(idx, 95)), f"{prefix}_angle_p5": float(np.percentile(ang, 5)),
            f"{prefix}_angle_p50": float(np.percentile(ang, 50)), f"{prefix}_angle_p95": float(np.percentile(ang, 95))}


# ============================================================================================
# Tests
# ============================================================================================
def _spearman_null(n, n_perm, rng):
    """Spearman ρ of a random permutation (or of all of them, if n ≤ 8) against the ranks 1..n, without ties."""
    r = np.arange(n, dtype=float)
    P = np.array(list(itertools.permutations(range(n))), float) if n <= 8 else \
        rng.permuted(np.tile(r, (n_perm, 1)), axis=1)
    return 1 - 6 * ((P - r) ** 2).sum(1) / (n * (n ** 2 - 1))


def critical_rho(n, alpha=0.05, n_perm=200_000, seed=0):
    """The smallest two-tailed significant Spearman |ρ| (exact for n ≤ 8, otherwise Monte Carlo)."""
    null = np.abs(_spearman_null(n, n_perm, np.random.default_rng(seed)))
    vals = np.unique(np.round(null, 10))
    ok = [v for v in vals if np.mean(null >= v - 1e-9) <= alpha]
    return float(min(ok))


def _pearson_perm(a, b, n_perm, seed):
    """Pearson r between a and b and two-tailed p by permuting b (all permutations if n ≤ 8)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.corrcoef(a, b)[0, 1]
    if len(a) <= 8:
        B = b[np.array(list(itertools.permutations(range(len(b)))))]
        null = np.array([np.corrcoef(a, x)[0, 1] for x in B])
        return float(r), float(np.mean(np.abs(null) >= abs(r) - 1e-12))
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    return float(r), float((1 + np.sum(np.abs(null) >= abs(r) - 1e-12)) / (n_perm + 1))


def perm_spearman(x, y, n_perm=10_000, seed=0):
    """Spearman ρ and two-tailed p by permutation."""
    return _pearson_perm(stats.rankdata(x), stats.rankdata(y), n_perm, seed)


def partial_spearman(x, y, z, n_perm=10_000, seed=0):
    """Partial Spearman of x and y with respect to z: r between the residuals of the ranks after linear regression on the ranks of z;
    p by permutation of the residuals of y."""
    rz = stats.rankdata(z)

    def resid(v):
        rv = stats.rankdata(v)
        return rv - np.polyval(np.polyfit(rz, rv, 1), rz)

    return _pearson_perm(resid(x), resid(y), n_perm, seed)


def mantel(y, da, db, dates, n_perm=10_000, seed=0):
    """Mantel test: Spearman ρ between y (one measure per pair) and |Δ days|, with two-tailed p by permuting the dates
    among the days. da, db: day index of the two elements of each pair; dates: date (in days) of each day."""
    y, da, db, dates = (np.asarray(v) for v in (y, da, db, dates))
    ry = stats.rankdata(y)

    def stat(d):
        return np.corrcoef(ry, stats.rankdata(np.abs(d[da] - d[db])))[0, 1]

    rho = stat(dates)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(dates)) for _ in range(n_perm)])
    return float(rho), float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n_perm + 1))


def _daypair_sums(y, da, db, n_days):
    a, b = np.minimum(da, db), np.maximum(da, db)
    n, sy = np.zeros((n_days, n_days)), np.zeros((n_days, n_days))
    np.add.at(n, (a, b), 1)
    np.add.at(sy, (a, b), y)
    return n, sy


def _wls_slope(n, sy, X, m):
    """Least-squares slope with each pair of days (a < b) counted m_a · m_b times and the pairs within a
    day (a = b, different files) m_a times: this is the sample of pairs of a bootstrap with multiplicity m of the days."""
    W = np.triu(np.outer(m, m), 1) + np.diag(m)
    N, Sx, Sy = (W * n).sum(), (W * n * X).sum(), (W * sy).sum()
    Sxy, Sxx = (W * sy * X).sum(), (W * n * X ** 2).sum()
    return (N * Sxy - Sx * Sy) / (N * Sxx - Sx ** 2)


def slope_bootstrap(y, da, db, dates, n_boot=2_000, seed=0):
    """Slope of y on |Δ days| (per day) and 95% CI with bootstrap over days (see _wls_slope)."""

    y, da, db, dates = (np.asarray(v) for v in (y, da, db, dates))
    D = len(dates)
    n, sy = _daypair_sums(y.astype(float), da, db, D)
    X = np.abs(np.subtract.outer(dates, dates)).astype(float)
    est = _wls_slope(n, sy, X, np.ones(D))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        s = _wls_slope(n, sy, X, m)
        if np.isfinite(s):
            boot.append(s)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(est), float(lo), float(hi), len(boot)
