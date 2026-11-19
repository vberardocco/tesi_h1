"""Tensor Maximum Entropy (TME), Elsayed & Cunningham 2017, Nat Neurosci 20: 1310-1318.

Python port of the official MATLAB code (github.com/gamaleldin/TME, read on 5/11/2026):
- util/extractFeatures.m      -> marginals(): tensor of the means M.TNC (successive subtraction of the means
                                 along time, neurons, conditions) and the three marginal covariances (sums, not means);
- genTME/fitMaxEntropy.m      -> fit(): eigenvalues of the marginals rescaled (preScale), eigenvalues with
                                 log < -10 considered null (infinite multiplier), starting point
                                 log(sum of the ranks of the other modes) - log(eigenvalue);
- genTME/logObjectiveMaxEntropyTensor.m -> _objective_log(): same objective function (squared errors on the
                                 logarithms of the eigenvalues, divided by the norm of the log-eigenvalues) and same gradient;
                                 the optimizer is scipy's L-BFGS-B instead of Rasmussen's minimize.m;
- genTME/objectiveMaxEntropyTensor.m    -> cost_original() (in MATLAB: warning if > 1e-5);
- genTME/sampleTME.m          -> campiona(): variances 1 / (Kronecker sum of the multipliers), rotation with the
                                 eigenvectors of each mode, plus the tensor of the means.

The tensor is T × N × C (time × neurons × conditions). We use the 'surrogate-TNC' type of demo.m: the three marginal
covariances and the tensor of the means M.TNC. By construction TME preserves the covariance between neurons,
hence also the PCA spectrum: it is not useful for testing linear dimensionality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

LITERATURE_LOG = -10.0          # like threshold in fitMaxEntropy.m


def marginals(X):
    """extractFeatures.m: (marginal covariances [Σ_T, Σ_N, Σ_C], tensor of the means M.TNC).

    The means are removed one after the other: for each t the mean over (n, c), then for each n the mean over (t, c),
    then for each c the mean over (t, n). The covariances are sums: Σ_T = Σ_{n,c} x_{·nc} x_{·nc}ᵀ, etc."""
    X = np.asarray(X, float)
    X0 = X - X.mean(axis=(1, 2), keepdims=True)
    X0 = X0 - X0.mean(axis=(0, 2), keepdims=True)
    X0 = X0 - X0.mean(axis=(0, 1), keepdims=True)
    M = X - X0
    sig = [np.einsum("tnc,snc->ts", X0, X0), np.einsum("tnc,tmc->nm", X0, X0), np.einsum("tnc,tnd->cd", X0, X0)]
    return sig, M


def _sum_kron(L):
    """Tensor L_T[i] + L_N[j] + L_C[k] (diagKronSum.m, in the form of a T × N × C tensor)."""
    return L[0][:, None, None] + L[1][None, :, None] + L[2][None, None, :]


def _others(x):
    return tuple(a for a in range(3) if a != x)


def _expand(v, x):
    shape = [1, 1, 1]
    shape[x] = len(v)
    return v.reshape(shape)


def _objective_log(logL, logS, dims):
    """logObjectiveMaxEntropyTensor.m: value and gradient with respect to logL."""
    norm = float(sum((s ** 2).sum() for s in logS))
    pieces = np.split(logL, np.cumsum(dims)[:-1])
    L = [np.exp(p) for p in pieces]
    A = 1.0 / _sum_kron(L)
    B = A * A
    f = 0.0
    W = 0.0
    for x in range(3):
        m = A.sum(axis=_others(x))
        E = logS[x] - np.log(m)
        f += float((E ** 2).sum())
        W = W + _expand(E / m, x)
    BW = B * W
    grad = np.concatenate([2.0 * L[y] * BW.sum(axis=_others(y)) for y in range(3)])
    return f / norm, grad / norm


@dataclass
class MaxEnt:
    autovettori: list          # Q_T, Q_N, Q_C (columns in order of decreasing eigenvalue)
    eigenvalues: list           # eigenvalues of the marginals, decreasing
    lagrangiani: list          # multipliers (np.inf for the null eigenvalues), on the original scale
    mean: np.ndarray          # M.TNC
    objective_log: float       # final value of the transformed objective function (0 at the optimum)
    iterations: int
    convergence: bool


def fit(sig, mean, maxiter: int = 10000) -> MaxEnt:
    """fitMaxEntropy.m: Lagrange multipliers of the maximum-entropy distribution."""
    traces = np.array([np.trace(S) for S in sig])
    if np.ptp(traces) > 1e-8 * np.abs(traces).max():
        raise ValueError(f"the marginal covariances must have the same trace: {traces}")
    Q, ev = [], []
    for S in sig:
        e, V = np.linalg.eigh((S + S.T) / 2)
        o = np.argsort(e)[::-1]
        Q.append(V[:, o])
        ev.append(np.clip(e[o], 0, None))
    dims = [len(e) for e in ev]
    pre = ev[0].sum() / np.mean(dims)                                   # preScale
    logS, opt = [], []
    for e in ev:
        with np.errstate(divide="ignore"):
            le = np.log(e / pre)
        le = le[le > LITERATURE_LOG]                                       # sorted eigenvalues: keep the first ones
        logS.append(le)
        opt.append(len(le))
    x0 = np.concatenate([np.log(sum(opt[y] for y in _others(x))) - logS[x] for x in range(3)])
    res = minimize(_objective_log, x0, args=(logS, opt), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-16, gtol=1e-12, maxcor=20))
    pieces = np.split(np.exp(res.x), np.cumsum(opt)[:-1])
    lagr = [np.concatenate([p, np.full(d - o, np.inf)]) / pre for p, d, o in zip(pieces, dims, opt)]
    return MaxEnt(autovettori=Q, eigenvalues=ev, lagrangiani=lagr, mean=np.asarray(mean, float),
                  objective_log=float(res.fun), iterations=int(res.nit), convergence=bool(res.success))


def varianze(me: MaxEnt) -> np.ndarray:
    """Variances of the coordinates in the eigenvector basis: 1 / (L_T[i] + L_N[j] + L_C[k]); 0 if infinite."""
    return 1.0 / _sum_kron(me.lagrangiani)


def covarianze_expected(me: MaxEnt) -> list:
    """Expected marginal covariances of the fitted distribution (exact computation, without sampling)."""
    D = varianze(me)
    return [(Q * D.sum(axis=_others(x))) @ Q.T for x, Q in enumerate(me.autovettori)]


def cost_original(me: MaxEnt) -> float:
    """objectiveMaxEntropyTensor.m: squared errors on the eigenvalues / norm of the eigenvalues."""
    D = varianze(me)
    norm = float(sum((e ** 2).sum() for e in me.eigenvalues))
    return float(sum(((me.eigenvalues[x] - D.sum(axis=_others(x))) ** 2).sum() for x in range(3)) / norm)


def campiona(me: MaxEnt, rng) -> np.ndarray:
    """sampleTME.m: one surrogate tensor T × N × C."""
    D = varianze(me)
    Z = rng.standard_normal(D.shape) * np.sqrt(D)
    QT, QN, QC = me.autovettori
    X = np.einsum("ti,ijk->tjk", QT, Z)
    X = np.einsum("nj,tjk->tnk", QN, X)
    X = np.einsum("ck,tnk->tnc", QC, X)
    return X + me.mean


def error_relative(A, B) -> float:
    """||A − B||_F / ||B||_F."""
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B)) / np.linalg.norm(np.asarray(B)))


def marginals_centered(Y, M) -> list:
    """Marginal covariances (sums) of Y − M, with M fixed: this is the quantity constrained by TME (E[...] = Σ of the data).
    Recomputing the means on each surrogate (marginali(Y)) would also remove part of the noise and give
    slightly smaller covariances."""
    Y0 = np.asarray(Y, float) - M
    return [np.einsum("tnc,snc->ts", Y0, Y0), np.einsum("tnc,tmc->nm", Y0, Y0), np.einsum("tnc,tnd->cd", Y0, Y0)]


def validation(me: MaxEnt, sig, surrogates=()) -> dict:
    """Relative errors (Frobenius) with respect to the marginals of the data, mode by mode (T, N, C):
    - expected: expected covariances of the fitted distribution (rule P3: must be below 5%);
    - mean: mean of the marginals of the surrogates, centered on the tensor of the data means (M.TNC);
    - single: median over the surrogates of the error of the single surrogate (same center)."""

    out = {}
    for x, (a, s) in enumerate(zip(covarianze_expected(me), sig)):
        out[f"expected_{'TNC'[x]}"] = error_relative(a, s)
    if len(surrogates):
        marg = [marginals_centered(Y, me.mean) for Y in surrogates]
        for x in range(3):
            mean = np.mean([m[x] for m in marg], axis=0)
            out[f"mean_{'TNC'[x]}"] = error_relative(mean, sig[x])
            out[f"single_{'TNC'[x]}"] = float(np.median([error_relative(m[x], sig[x]) for m in marg]))
    return out
