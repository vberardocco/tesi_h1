"""
clock_gpfa_online.py — Clock + online GPFA for FALCON H1.

Contents
--------
1. Online (causal) GPFA. With the already fitted GPFA parameters (C, d, R, gamma, eps) the posterior
   mean E[x_t | y_{t-W+1..t}] is linear in the data of the window:
       x_hat_t = sum_{k=0}^{W-1} F_k (y_{t-k} - d)
   The filters F_k are computed once: no refit, cost per bin = W matrix-vector
   products. In the first W-1 bins of a continuous segment the exact filters
   for shorter windows are used (no fictitious padding).
2. Clock: time since the start of the trial, expanded on Gaussian bases (RBF), as a decoder feature.
3. Ridge decoder on [lagged latents | clock] and comparison of three variants:
     "clock"       control: how much R^2 comes from the task timing alone
     "gpfa"        online GPFA alone
     "clock+gpfa"  both
4. StreamingClockGPFA: the same math bin by bin, for a FALCON/EvalAI wrapper.

Kernel convention (Yu et al. 2009, elephant), time in bins:
    K_i(t1, t2) = (1 - eps_i) * exp(-gamma_i * (t1 - t2)^2 / 2) + eps_i * delta(t1, t2)
    gamma_i = (bin_size / tau_i)^2
Y must have the same preprocessing used to fit the GPFA (same bin, same
transformation, e.g. square root of the counts if you used it).
"""
from collections import deque

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VARIANTS = ("clock", "gpfa", "clock+gpfa")


# ----------------------------------------------------------------------------- parameters

def gamma_from_tau(tau_ms, bin_ms):
    return (float(bin_ms) / np.asarray(tau_ms, float)) ** 2


def tau_from_gamma(gamma, bin_ms):
    return float(bin_ms) / np.sqrt(np.asarray(gamma, float))


def params_from_elephant(gpfa):
    """Parameters from an already fitted elephant.gpfa.GPFA (dict params_estimated)."""
    p = gpfa.params_estimated
    return {"C": p["C"], "d": p["d"], "R": p["R"], "gamma": p["gamma"], "eps": p["eps"]}


def _unpack(params):
    C = np.asarray(params["C"], float)
    d = np.asarray(params["d"], float).ravel()
    R = np.asarray(params["R"], float)
    R = np.diag(R).copy() if R.ndim == 2 else R.ravel()
    gamma = np.asarray(params["gamma"], float).ravel()
    eps = np.asarray(params.get("eps", np.full(gamma.shape, 1e-3)), float).ravel()
    y_dim, x_dim = C.shape
    if d.shape != (y_dim,) or R.shape != (y_dim,):
        raise ValueError("d and R must have y_dim elements")
    if gamma.shape != (x_dim,) or eps.shape != (x_dim,):
        raise ValueError("gamma and eps must have x_dim elements")
    return C, d, R, gamma, eps


def recenter(params, Y_calib):
    """Recentering of the channels on a new session: d <- mean of the calibration bins."""
    out = dict(params)
    out["d"] = np.asarray(Y_calib, float).mean(axis=0)
    return out


def default_window(params, n_tau=3.0, max_bins=100):
    """Causal window = n_tau times the longest time constant, in bins."""
    _, _, _, gamma, _ = _unpack(params)
    tau_bins = 1.0 / np.sqrt(gamma.min())
    return int(min(max(np.ceil(n_tau * tau_bins), 1), max_bins))


# ------------------------------------------------------------------------ GPFA inference

def _prior_cov(gamma, eps, L):
    """GP prior covariance over L bins, time-major stacking (index = t * x_dim + i)."""
    x_dim = gamma.size
    t = np.arange(L)
    dt2 = (t[:, None] - t[None, :]) ** 2
    K = np.zeros((L * x_dim, L * x_dim))
    for i in range(x_dim):
        K[i::x_dim, i::x_dim] = (1 - eps[i]) * np.exp(-0.5 * gamma[i] * dt2) + eps[i] * np.eye(L)
    return K


def causal_gpfa_filters(params, window):
    """
    Exact causal filters. Returns a list: filters[L-1] has shape (L, x_dim, y_dim) and
    filters[L-1][k] is the weight of bin t-k (k = 0 current bin), for L = 1..window.

    Posterior: (K^-1 + Lam)^-1 = (I + K Lam)^-1 K, with Lam = I_L (x) C^T R^-1 C.
    We avoid inverting K (ill-conditioned for long tau).
    """
    C, _, R, gamma, eps = _unpack(params)
    y_dim, x_dim = C.shape
    CtRinv = C.T / R                     # (x, y)
    CtRinvC = CtRinv @ C                 # (x, x)
    filters = []
    for L in range(1, int(window) + 1):
        K = _prior_cov(gamma, eps, L)
        M = np.eye(L * x_dim) + K @ np.kron(np.eye(L), CtRinvC)
        G = np.linalg.solve(M, K[:, -x_dim:]).T      # rows of the last bin of the posterior cov
        F = np.empty((L, x_dim, y_dim))
        for tau in range(L):                          # tau = 0 is the oldest bin
            F[L - 1 - tau] = G[:, tau * x_dim:(tau + 1) * x_dim] @ CtRinv
        filters.append(F)
    return filters


def gpfa_online(Y, params, window=None, filters=None, segment_starts=None):
    """
    Causal latents (T, x_dim). x_hat_t uses only Y[max(s, t-W+1) : t+1], with s the start of the
    continuous segment. By default Y is a single segment; pass segment_starts if there are gaps.
    """
    C, d, _, _, _ = _unpack(params)
    if filters is None:
        filters = causal_gpfa_filters(params, window or default_window(params))
    W = len(filters)
    Yc = np.asarray(Y, float) - d
    T = Yc.shape[0]
    X = np.zeros((T, C.shape[1]))

    idx = np.arange(T)
    starts = np.zeros(T, bool)
    starts[0] = True
    if segment_starts is not None:
        starts[np.asarray(segment_starts, int)] = True
    seg_start = np.maximum.accumulate(np.where(starts, idx, 0))
    hist = np.minimum(idx - seg_start + 1, W)

    full = np.where(hist == W)[0]
    FW = filters[W - 1]
    for k in range(W):
        X[full] += Yc[full - k] @ FW[k].T
    for t in np.where(hist < W)[0]:
        F = filters[hist[t] - 1]
        for k in range(hist[t]):
            X[t] += F[k] @ Yc[t - k]
    return X


def gpfa_smooth(Y, params):
    """Offline posterior mean E[x_t | y_1..T] on a segment (for comparison)."""
    C, d, R, gamma, eps = _unpack(params)
    T, x_dim = len(Y), C.shape[1]
    CtRinv = C.T / R
    K = _prior_cov(gamma, eps, T)
    M = np.eye(T * x_dim) + K @ np.kron(np.eye(T), CtRinv @ C)
    b = ((np.asarray(Y, float) - d) @ CtRinv.T).ravel()
    return np.linalg.solve(M, K @ b).reshape(T, x_dim)


def orthonormalize(X, C):
    """Orthonormalized latents (Yu et al. 2009): C = U S V^T, x_orth = S V^T x."""
    _, s, Vt = np.linalg.svd(np.asarray(C, float), full_matrices=False)
    return X @ (np.diag(s) @ Vt).T


# ---------------------------------------------------------------------------------- clock

def time_in_trial(trial_starts, T):
    """Bins elapsed since the start of the current trial (0 at the first bin). Indices or bool mask."""
    ts = np.asarray(trial_starts)
    if ts.dtype == bool:
        mask = ts.copy()
    else:
        mask = np.zeros(T, bool)
        mask[ts.astype(int)] = True
    mask[0] = True
    idx = np.arange(T)
    return idx - np.maximum.accumulate(np.where(mask, idx, 0))


def clock_features(t_bins, t_max, n_basis=10):
    """Gaussian RBFs on the time in the trial; beyond t_max the time saturates. t_max must be fixed on the training set."""
    t = np.minimum(np.asarray(t_bins, float), float(t_max))
    centers = np.linspace(0.0, float(t_max), n_basis)
    width = centers[1] - centers[0] if n_basis > 1 else float(t_max)
    return np.exp(-0.5 * ((t[:, None] - centers[None, :]) / width) ** 2)


# -------------------------------------------------------------------------------- decoder

def add_lags(X, n_lags):
    """[x_t, x_{t-1}, ..., x_{t-n_lags}], causal, zeros before the start."""
    X = np.asarray(X, float)
    T, n = X.shape
    out = np.zeros((T, n * (n_lags + 1)))
    for j in range(n_lags + 1):
        out[j:, j * n:(j + 1) * n] = X[:T - j]
    return out


def design_matrix(variant, latents=None, clock=None, n_lags=4):
    parts = []
    if "gpfa" in variant:
        parts.append(add_lags(latents, n_lags))
    if "clock" in variant:
        parts.append(np.asarray(clock, float))
    return np.hstack(parts)


def fit_decoder(variant, latents, clock, kin, n_lags=4, alpha=1.0, mask=None):
    Xd = design_matrix(variant, latents, clock, n_lags)
    m = slice(None) if mask is None else np.asarray(mask, bool)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(Xd[m], np.asarray(kin, float)[m])
    return model


def evaluate(model, variant, latents, clock, kin, n_lags=4, mask=None,
             multioutput="uniform_average"):
    pred = model.predict(design_matrix(variant, latents, clock, n_lags))
    m = slice(None) if mask is None else np.asarray(mask, bool)
    return r2_score(np.asarray(kin, float)[m], pred[m], multioutput=multioutput), pred


def compare_variants(train, test, n_lags=4, alpha=1.0, variants=VARIANTS,
                     multioutput="uniform_average"):
    """train/test: dict with 'latents', 'clock', 'kin' and optional 'mask'. Returns {variant: R^2}."""
    out = {}
    for v in variants:
        model = fit_decoder(v, train.get("latents"), train.get("clock"), train["kin"],
                            n_lags, alpha, train.get("mask"))
        out[v], _ = evaluate(model, v, test.get("latents"), test.get("clock"), test["kin"],
                             n_lags, test.get("mask"), multioutput)
    return out


# ------------------------------------------------------------------------------ streaming

class StreamingClockGPFA:
    """
    Causal decoder bin by bin, identical to gpfa_online + clock_features + ridge in batch.
    In a FALCON wrapper: reset() at the start of the session, step(y) for each bin, new_trial()
    when the evaluator signals the end of a trial (the next bin is t = 0).
    """

    def __init__(self, params, model, variant="clock+gpfa", filters=None, window=None,
                 n_lags=4, t_max=None, n_basis=10):
        C, d, _, _, _ = _unpack(params)
        self.d, self.x_dim = d, C.shape[1]
        self.variant, self.n_lags = variant, n_lags
        self.t_max, self.n_basis = t_max, n_basis
        if "clock" in variant and t_max is None:
            raise ValueError("t_max is required (the same used in training)")

        if "gpfa" in variant and filters is None:
            filters = causal_gpfa_filters(params, window or default_window(params))
        self.filters = filters
        scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
        self._mu, self._sd = scaler.mean_, scaler.scale_
        self._w, self._b = ridge.coef_, ridge.intercept_
        self.reset()

    def reset(self):
        W = len(self.filters) if self.filters is not None else 1
        self._y = deque(maxlen=W)
        self._x = deque(maxlen=self.n_lags + 1)
        self._t = 0

    def new_trial(self):
        self._t = 0

    def step(self, y):
        feats = []
        if "gpfa" in self.variant:
            self._y.appendleft(np.asarray(y, float) - self.d)
            F = self.filters[len(self._y) - 1]
            x = np.einsum("kxy,ky->x", F, np.asarray(self._y))
            self._x.appendleft(x)
            feats += [self._x[j] if j < len(self._x) else np.zeros(self.x_dim)
                      for j in range(self.n_lags + 1)]
        if "clock" in self.variant:
            feats.append(clock_features([self._t], self.t_max, self.n_basis)[0])
        self._t += 1
        f = (np.concatenate(feats) - self._mu) / self._sd
        return f @ self._w.T + self._b
