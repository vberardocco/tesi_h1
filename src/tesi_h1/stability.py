"""Stabilità dei sottospazi FA tra sessioni (docs/crosssession_protocol.md, sezione del 2/11/2026).

- Sottospazio di un blocco da 3 trial: FA con K = 15 come nel decoder; si tengono gli autovettori della covarianza
  condivisa S = L Lᵀ, ordinati per varianza condivisa (le loading FA non hanno un ordine).
- Indice di allineamento (Elsayed et al. 2016, su S): I(A→B) = tr(Q_Bᵀ S_A Q_B) / Σ_{i≤k} λ_i(S_A), con Q_B i primi k
  autovettori di S_B; indice della coppia = ½ [I(A→B) + I(B→A)]. 1 = stesso sottospazio.
- Angolo principale medio tra i due sottospazi (secondario).
- Caso: sottospazi casuali presi da una covarianza C (Q = orth(U Λ^½ r), r gaussiano).
- Test: Spearman con permutazione (esatta per n ≤ 8), Spearman parziale, soglia critica di |ρ|, test di Mantel con
  permutazione dei giorni, pendenza con bootstrap sui giorni.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

K = 15                      # fattori FA, come nel decoder (02_new §9)
K_SENS = 5                  # sensibilità: prime 5 direzioni per varianza condivisa dal fit con K = 15
KS = (K, K_SENS)
SD_MIN = 0.1                # sd minima nello z-score del blocco: sd di Poisson a 0.5 Hz in bin da 20 ms, √(0.5 · 0.02)
BLOCK = 3                   # trial per blocco
N_DRAWS = 500               # estrazioni del caso per coppia
CHANCE = ("allbins", "cond", "unif")


# ============================================================================================
# FA, sottospazi e misure
# ============================================================================================
def zscore_block(X, sd_min=SD_MIN):
    """z-score dentro il blocco con sd minima sd_min. Restituisce anche quanti canali hanno sd = 0 e sd < sd_min."""
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.maximum(sd, sd_min), int((sd == 0).sum()), int((sd < sd_min).sum())


def fit_loadings(X, k=K):
    """Loading FA (canali × k) come FAm del decoder: svd_method="lapack", random_state=0."""
    from sklearn.decomposition import FactorAnalysis
    return FactorAnalysis(k, svd_method="lapack", random_state=0).fit(X).components_.T


@dataclass
class Sub:
    """Sottospazio FA: U (canali × K) autovettori di S = L Lᵀ ordinati per varianza condivisa, lam i loro autovalori."""
    U: np.ndarray
    lam: np.ndarray

    @classmethod
    def from_loadings(cls, L):
        U, s, _ = np.linalg.svd(np.asarray(L, float), full_matrices=False)    # L = U s Vᵀ  ->  S = U s² Uᵀ
        return cls(U, s ** 2)

    def basis(self, k):
        return self.U[:, :k]


def captured(A: Sub, Q):
    """tr(Qᵀ S_A Q), per una base Q (canali × k) o una pila di basi (n × canali × k)."""
    P = np.einsum("ci,...ck->...ik", A.U, Q)
    return np.einsum("i,...ik->...", A.lam, P ** 2)


def index_dir(A: Sub, Q, k):
    """I(A→Q): varianza condivisa di A catturata da Q, rispetto al massimo possibile con k direzioni."""
    return captured(A, Q) / A.lam[:k].sum()


def align_index(A: Sub, B: Sub, k):
    return 0.5 * (index_dir(A, B.basis(k), k) + index_dir(B, A.basis(k), k))


def mean_angle(Qa, Qb):
    """Angolo principale medio in gradi tra basi ortonormali (anche pile n × canali × k)."""
    s = np.linalg.svd(np.swapaxes(Qa, -1, -2) @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0))).mean(-1)


def pair_measures(A: Sub, B: Sub, ks=KS):
    out = {}
    for k in ks:
        out[f"index_k{k}"] = float(align_index(A, B, k))
        out[f"angle_k{k}"] = float(mean_angle(A.basis(k), B.basis(k)))
    return out


# ============================================================================================
# Caso alla Elsayed
# ============================================================================================
def sampler(C):
    """Matrice M = U Λ^½ della covarianza C: le colonne di M r (r gaussiano) seguono la forma della varianza."""
    ev, U = np.linalg.eigh(np.asarray(C, float))
    return U * np.sqrt(np.clip(ev, 0.0, None))


def draw_bases(M, k, n, rng):
    """n basi casuali ortonormali (n × canali × k) da M (vedi sampler)."""
    Q, _ = np.linalg.qr(M @ rng.normal(size=(n, M.shape[0], k)))
    return Q


def chance(A: Sub, B: Sub, k, Q1, Q2):
    """Caso della coppia per ogni estrazione: indice ½ [I(A→Q1) + I(B→Q2)] e angolo medio tra Q1 e Q2."""
    return 0.5 * (index_dir(A, Q1, k) + index_dir(B, Q2, k)), mean_angle(Q1, Q2)


def chance_summary(idx, ang, prefix):
    return {f"{prefix}_index_p5": float(np.percentile(idx, 5)), f"{prefix}_index_p50": float(np.percentile(idx, 50)),
            f"{prefix}_index_p95": float(np.percentile(idx, 95)), f"{prefix}_angle_p5": float(np.percentile(ang, 5)),
            f"{prefix}_angle_p50": float(np.percentile(ang, 50)), f"{prefix}_angle_p95": float(np.percentile(ang, 95))}


# ============================================================================================
# Test
# ============================================================================================
def _spearman_null(n, n_perm, rng):
    """ρ di Spearman di una permutazione casuale (o di tutte, se n ≤ 8) contro i ranghi 1..n, senza pari merito."""
    r = np.arange(n, dtype=float)
    P = np.array(list(itertools.permutations(range(n))), float) if n <= 8 else \
        rng.permuted(np.tile(r, (n_perm, 1)), axis=1)
    return 1 - 6 * ((P - r) ** 2).sum(1) / (n * (n ** 2 - 1))


def critical_rho(n, alpha=0.05, n_perm=200_000, seed=0):
    """Il più piccolo |ρ| di Spearman significativo a due code (esatto per n ≤ 8, altrimenti Monte Carlo)."""
    null = np.abs(_spearman_null(n, n_perm, np.random.default_rng(seed)))
    vals = np.unique(np.round(null, 10))
    ok = [v for v in vals if np.mean(null >= v - 1e-9) <= alpha]
    return float(min(ok))


def _pearson_perm(a, b, n_perm, seed):
    """r di Pearson tra a e b e p a due code permutando b (tutte le permutazioni se n ≤ 8)."""
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
    """ρ di Spearman e p a due code per permutazione."""
    return _pearson_perm(stats.rankdata(x), stats.rankdata(y), n_perm, seed)


def partial_spearman(x, y, z, n_perm=10_000, seed=0):
    """Spearman parziale di x e y rispetto a z: r tra i residui dei ranghi dopo la regressione lineare sui ranghi di z;
    p per permutazione dei residui di y."""
    rz = stats.rankdata(z)

    def resid(v):
        rv = stats.rankdata(v)
        return rv - np.polyval(np.polyfit(rz, rv, 1), rz)

    return _pearson_perm(resid(x), resid(y), n_perm, seed)


def mantel(y, da, db, dates, n_perm=10_000, seed=0):
    """Test di Mantel: ρ di Spearman tra y (una misura per coppia) e |Δ giorni|, con p a due code permutando le date
    tra i giorni. da, db: indice del giorno dei due elementi di ogni coppia; dates: data (in giorni) di ogni giorno."""
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
    """Pendenza dei minimi quadrati con ogni coppia di giorni (a < b) contata m_a · m_b volte e le coppie dentro un
    giorno (a = b, file diversi) m_a volte: è il campione di coppie di un bootstrap con molteplicità m dei giorni."""
    W = np.triu(np.outer(m, m), 1) + np.diag(m)
    N, Sx, Sy = (W * n).sum(), (W * n * X).sum(), (W * sy).sum()
    Sxy, Sxx = (W * sy * X).sum(), (W * n * X ** 2).sum()
    return (N * Sxy - Sx * Sy) / (N * Sxx - Sx ** 2)


def slope_bootstrap(y, da, db, dates, n_boot=2_000, seed=0):
    """Pendenza di y su |Δ giorni| (per giorno) e IC 95% con bootstrap sui giorni (vedi _wls_slope)."""
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
