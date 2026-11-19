"""
Debug tests for clock_gpfa_online.py on synthetic GPFA data (true parameters known).

    pytest -q test_clock_gpfa_online.py      # all tests
    python test_clock_gpfa_online.py         # prints the numbers to compare with the expected ones

Expected outputs (fixed seeds):
  1. filters vs exact GP (full window and warm-up)    max |diff| < 1e-8
  2. last online bin == offline smoother               max |diff| < 1e-8 ; first bin different (> 1e-3)
  3. causality: perturb Y[300] -> X[:300] identical    (array_equal True), X[300] changes (> 0.1)
  4. latent recovery: mean corr online > 0.85, offline > online
  5. time_in_trial([0,3,7], T=10) == [0,1,2,0,1,2,3,0,1,2]
  6. clock-only: kinematics tied to the time in the trial R^2 > 0.95 ; unrelated kinematics R^2 < 0.05
  7. clock+gpfa beats both by at least 0.05 when the kinematics = time profile + latents
  8. streaming == batch                                max |diff| < 1e-9

Values obtained with `python test_clock_gpfa_online.py` (numpy 2.5, scipy 1.18, sklearn 1.9):
  1. ~8e-15   2. last ~1e-13, first ~0.16   4. 0.994 / 0.998
  7. R^2 clock -0.083, gpfa 0.876, clock+gpfa 0.974
"""
import numpy as np

from clock_gpfa_online import (
    StreamingClockGPFA, _prior_cov, _unpack, causal_gpfa_filters, clock_features,
    compare_variants, fit_decoder, evaluate, gamma_from_tau, gpfa_online, gpfa_smooth,
    time_in_trial,
)

BIN_MS = 20
TRIAL_LEN = 50


def simulate(T=1000, x_dim=3, y_dim=30, taus_ms=(100, 200, 300), seed=0, eps=1e-3):
    rng = np.random.default_rng(seed)
    gamma = gamma_from_tau(taus_ms, BIN_MS)
    t = np.arange(T)
    X = np.empty((T, x_dim))
    for i in range(x_dim):
        K = (1 - eps) * np.exp(-0.5 * gamma[i] * (t[:, None] - t[None]) ** 2) + eps * np.eye(T)
        X[:, i] = np.linalg.cholesky(K) @ rng.standard_normal(T)
    C = rng.standard_normal((y_dim, x_dim))
    d = rng.uniform(1, 3, y_dim)
    R = rng.uniform(0.5, 1.5, y_dim)
    Y = X @ C.T + d + rng.standard_normal((T, y_dim)) * np.sqrt(R)
    return X, Y, dict(C=C, d=d, R=R, gamma=gamma, eps=np.full(x_dim, eps))


def exact_causal(Y, params, t, W):
    """Independent reference: GP conditioning in the observation space."""
    C, d, R, gamma, eps = _unpack(params)
    x_dim = C.shape[1]
    lo = max(0, t - W + 1)
    L = t - lo + 1
    K = _prior_cov(gamma, eps, L)
    B = np.kron(np.eye(L), C)
    Sy = B @ K @ B.T + np.diag(np.tile(R, L))
    return K[-x_dim:] @ B.T @ np.linalg.solve(Sy, (Y[lo:t + 1] - d).ravel())


def corr_per_dim(A, B):
    return np.array([np.corrcoef(A[:, i], B[:, i])[0, 1] for i in range(A.shape[1])])


def make_task(seed=1):
    """Trials of 50 bins, 20 trials: 12 train, 8 test. Kinematics with 2 DoF."""
    X, Y, p = simulate(T=1000, seed=seed)
    starts = np.arange(0, 1000, TRIAL_LEN)
    tt = time_in_trial(starts, 1000)
    profile = np.c_[np.sin(np.pi * tt / TRIAL_LEN), np.cos(np.pi * tt / TRIAL_LEN)]
    rng = np.random.default_rng(seed + 100)
    b = rng.standard_normal((3, 2))
    noise = 0.05 * rng.standard_normal((1000, 2))
    return X, Y, p, starts, tt, profile, b, noise


# ----------------------------------------------------------------------------------- test

def test_filters_vs_gp_exact():
    _, Y, p = simulate(T=600)
    W = 40
    X = gpfa_online(Y, p, window=W)
    diffs = [np.abs(X[t] - exact_causal(Y, p, t, W)).max() for t in (0, 5, 20, 100, 333, 599)]
    assert max(diffs) < 1e-8, diffs


def test_last_bin_equal_smoother():
    _, Y, p = simulate(T=600)
    Ys = Y[:60]
    on, off = gpfa_online(Ys, p, window=60), gpfa_smooth(Ys, p)
    assert np.abs(on[-1] - off[-1]).max() < 1e-8
    assert np.abs(on[0] - off[0]).max() > 1e-3


def test_causalita():
    _, Y, p = simulate(T=600)
    filt = causal_gpfa_filters(p, 40)
    Y2 = Y.copy()
    Y2[300] += 50.0
    X1, X2 = gpfa_online(Y, p, filters=filt), gpfa_online(Y2, p, filters=filt)
    assert np.array_equal(X1[:300], X2[:300])
    assert np.abs(X2[300] - X1[300]).max() > 0.1


def test_recovery_latents():
    Xtrue, Y, p = simulate(T=600)
    c_on = corr_per_dim(Xtrue, gpfa_online(Y, p)).mean()
    c_off = corr_per_dim(Xtrue, gpfa_smooth(Y, p)).mean()
    assert c_on > 0.85 and c_off > c_on, (c_on, c_off)


def test_time_in_trial():
    assert time_in_trial([0, 3, 7], 10).tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]
    mask = np.zeros(10, bool); mask[[3, 7]] = True
    assert time_in_trial(mask, 10).tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]


def test_clock_features():
    F = clock_features([0, 10, 49, 80], t_max=49, n_basis=10)
    assert F.shape == (4, 10)
    assert np.isclose(F[0, 0], 1.0) and np.isclose(F[2, -1], 1.0)
    assert np.allclose(F[2], F[3])          # saturation beyond t_max


def test_clock_only():
    _, _, _, _, tt, profile, _, noise = make_task()
    clk = clock_features(tt, TRIAL_LEN - 1)
    tr, te = slice(0, 600), slice(600, 1000)
    r2_time = compare_variants({"clock": clk[tr], "kin": (profile + noise)[tr]},
                               {"clock": clk[te], "kin": (profile + noise)[te]},
                               variants=("clock",))["clock"]
    Xu, _, _ = simulate(T=1000, seed=7)      # smooth kinematics but unrelated to the trial
    r2_free = compare_variants({"clock": clk[tr], "kin": Xu[tr, :2]},
                               {"clock": clk[te], "kin": Xu[te, :2]},
                               variants=("clock",))["clock"]
    assert r2_time > 0.95 and r2_free < 0.05, (r2_time, r2_free)


def test_clock_more_gpfa():
    X, Y, p, _, tt, profile, b, noise = make_task()
    lat = gpfa_online(Y, p)
    clk = clock_features(tt, TRIAL_LEN - 1)
    kin = profile + X @ b + noise
    tr, te = slice(0, 600), slice(600, 1000)
    r2 = compare_variants({"latents": lat[tr], "clock": clk[tr], "kin": kin[tr]},
                          {"latents": lat[te], "clock": clk[te], "kin": kin[te]})
    assert r2["clock+gpfa"] > max(r2["clock"], r2["gpfa"]) + 0.05, r2


def test_streaming_equal_batch():
    X, Y, p, starts, tt, profile, b, noise = make_task()
    filt = causal_gpfa_filters(p, 45)
    lat = gpfa_online(Y, p, filters=filt)
    clk = clock_features(tt, TRIAL_LEN - 1)
    kin = profile + X @ b + noise
    model = fit_decoder("clock+gpfa", lat[:600], clk[:600], kin[:600])
    _, pred_batch = evaluate(model, "clock+gpfa", lat, clk, kin)
    dec = StreamingClockGPFA(p, model, filters=filt, t_max=TRIAL_LEN - 1)
    start_set = set(starts.tolist())
    pred_stream = []
    for t in range(len(Y)):
        if t in start_set:
            dec.new_trial()
        pred_stream.append(dec.step(Y[t]))
    assert np.abs(np.array(pred_stream) - pred_batch).max() < 1e-9


# ------------------------------------------------------------------------------- numbers

if __name__ == "__main__":
    _, Y, p = simulate(T=600)
    X = gpfa_online(Y, p, window=40)
    print("1. max diff vs exact GP:",
          max(np.abs(X[t] - exact_causal(Y, p, t, 40)).max() for t in (0, 5, 20, 100, 333, 599)))
    Ys = Y[:60]
    on, off = gpfa_online(Ys, p, window=60), gpfa_smooth(Ys, p)
    print("2. last bin online-smoother:", np.abs(on[-1] - off[-1]).max(),
          "| first bin:", np.abs(on[0] - off[0]).max())
    Xtrue, Y, p = simulate(T=600)
    print("4. mean corr online / offline:",

          corr_per_dim(Xtrue, gpfa_online(Y, p)).mean().round(3),
          corr_per_dim(Xtrue, gpfa_smooth(Y, p)).mean().round(3))
    X, Y, p, _, tt, profile, b, noise = make_task()
    lat, clk = gpfa_online(Y, p), clock_features(tt, TRIAL_LEN - 1)
    kin = profile + X @ b + noise
    r2 = compare_variants({"latents": lat[:600], "clock": clk[:600], "kin": kin[:600]},
                          {"latents": lat[600:], "clock": clk[600:], "kin": kin[600:]})
    print("7. R^2 test:", {k: round(v, 3) for k, v in r2.items()})
