"""Tests of src/thesis_h1/tme.py and src/thesis_h1/literature2.py (phase 2 of the literature comparison experiment, rules R8-R10).

Synthetic data with the expected result written next to it. Usage:
NUMBA_THREADING_LAYER=workqueue PYTHONPATH=~/thesis_h1_git/src pytest tests/test_literature_phase2.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import approx_fprime

sys.path.insert(0, str(Path(__file__).parent))
from thesis_h1 import literature, literature2, tme                                   # noqa: E402


def smooth(rng, shape, s=3.0):
    return gaussian_filter1d(rng.standard_normal(shape), s, axis=0)


# --------------------------------------------------------------------------------------------
# R8
# --------------------------------------------------------------------------------------------
def test_T13_pipeline_sigma50_equal_step2():
    """With σ = 50 ms the R8 pipelines coincide bit by bit with those of Step 2; the grid has 9 combinations."""
    rng = np.random.default_rng(0)
    C = rng.poisson(0.2, size=(3000, 40))
    assert np.array_equal(literature2.pipeline_02_sigma(C, 50), literature.pipeline_02(C))
    assert np.array_equal(literature2.pipeline_gallego_sigma(C, 50), literature.pipeline_gallego(C))
    g = literature2.grid_r8()
    assert len(g) == 9 and len({x[0] for x in g}) == 9


def test_T14_k_pa_latents_e_noise():
    """5 smooth latents + noise: k_PA = 5 with every σ; independent smoothed noise: k_PA = 0."""
    rng = np.random.default_rng(1)
    T, N = 8000, 60
    L = smooth(rng, (T, 5), 10)
    L /= L.std(0)
    X = L @ rng.standard_normal((5, N)) * 0.6 + rng.standard_normal((T, N))
    allowed, _ = literature.shifts_allowed(T, 780)
    m = np.ones(T, bool)
    for sigma in literature2.SIGMI:
        assert literature2.k_pa(literature.smooth(X, sigma), m, True, allowed, n_surr=60) == 5
    R = rng.standard_normal((T, N))
    assert literature2.k_pa(literature.smooth(R, 50), m, True, allowed, n_surr=60) == 0


# --------------------------------------------------------------------------------------------
# Twins and Isomap (notebook 03)
# --------------------------------------------------------------------------------------------
def test_T15_twin_keep_covarianza():
    """The random-phase twin has the same mean and the same covariance (error < 1e-8) but different values."""
    rng = np.random.default_rng(2)
    X = smooth(rng, (4001, 12)) @ rng.standard_normal((12, 12)) + 3
    G = literature2.phases_random(X, 0)
    assert np.allclose(G.mean(0), X.mean(0))
    assert tme.error_relative(np.cov(G.T), np.cov(X.T)) < 1e-8
    assert np.abs(G - X).mean() > 0.1 * X.std()


def test_T16_isomap_roll_e_cloud():
    """Rolled sheet: Isomap beats PCA at d = 2 by more than 10 points. 25-dimensional Gaussian cloud with
    decreasing spectrum: at d = 3 Isomap does not beat PCA (advantage < 0)."""
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=800, noise=0.3, random_state=0)
    assert literature2.advantage_isomap(X, d=2) > 10
    rng = np.random.default_rng(3)
    G = rng.standard_normal((800, 25)) * (1 / np.arange(1, 26) ** 0.5)
    assert literature2.advantage_isomap(G, d=3) < 0


def test_T17_tensor_conditions_e_impila():
    """The tensor of the means per condition has the order of Step 4b (phase by phase, odd trials first)."""
    Z = np.arange(40 * 3, dtype=float).reshape(40, 3)
    fin = [dict(tag="Reach", trial=1, start=0, stop=5), dict(tag="Reach", trial=3, start=10, stop=15),
           dict(tag="Reach", trial=2, start=20, stop=25), dict(tag="Grasp", trial=1, start=30, stop=35),
           dict(tag="Grasp", trial=2, start=35, stop=40)]
    X, rip = literature2.tensor_conditions(Z, fin, phases=("Reach", "Grasp"))
    assert X.shape == (5, 3, 4) and list(rip) == [2, 1, 1, 1]
    assert np.allclose(X[:, :, 0], (Z[0:5] + Z[10:15]) / 2) and np.allclose(X[:, :, 3], Z[35:40])
    P = literature2.impila(X)
    assert P.shape == (20, 3) and np.allclose(P[5:10], Z[20:25])


# --------------------------------------------------------------------------------------------
# TME
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tensor():
    rng = np.random.default_rng(4)
    T, N, C = 20, 15, 6
    lat = smooth(rng, (T, 3, C), 2)
    X = np.einsum("tkc,nk->tnc", lat, rng.standard_normal((N, 3))) + 0.3 * rng.standard_normal((T, N, C))
    return X + rng.standard_normal(N)[None, :, None]


def test_T18_tme_gradient(tensor):
    """The gradient of the objective function matches the finite differences."""
    sig, _ = tme.marginals(tensor)
    ev = [np.sort(np.linalg.eigvalsh(S))[::-1] for S in sig]
    pre = ev[0].sum() / np.mean([len(e) for e in ev])
    logS = [np.log(e / pre) for e in ev]
    dims = [len(x) for x in logS]
    x0 = np.random.default_rng(5).standard_normal(sum(dims))
    num = approx_fprime(x0, lambda z: tme._objective_log(z, logS, dims)[0], 1e-7)
    an = tme._objective_log(x0, logS, dims)[1]
    assert np.abs(num - an).max() < 1e-5 * max(1, np.abs(an).max())


def test_T19_tme_constraints_e_samples(tensor):
    """The fitted distribution reproduces the three marginals (error < 1e-6); the mean of 3000 surrogates reproduces
    the marginals within 3% and the tensor of the means within 3%; the single surrogate does not (sampling noise)."""
    sig, M = tme.marginals(tensor)
    me = tme.fit(sig, M)
    assert me.convergence and tme.cost_original(me) < 1e-5
    assert all(tme.error_relative(a, s) < 1e-6 for a, s in zip(tme.covarianze_expected(me), sig))
    rng = np.random.default_rng(6)
    sur = [tme.campiona(me, rng) for _ in range(3000)]
    v = tme.validation(me, sig, sur)
    assert max(v[f"mean_{m}"] for m in "TNC") < 0.03
    assert min(v[f"single_{m}"] for m in "TNC") > 0.05
    assert tme.error_relative(np.mean(sur, 0), M) < 0.03


def test_T20_tme_rank_reduced():
    """Rank-reduced time marginal (T = 30 > N · C = 12): the null eigenvalues have an infinite multiplier
    and the distribution still reproduces the marginals (error < 1e-4)."""
    rng = np.random.default_rng(7)
    X = smooth(rng, (30, 4, 3), 2)
    sig, M = tme.marginals(X)
    me = tme.fit(sig, M)
    assert np.isinf(me.lagrangiani[0]).sum() >= 30 - 12
    assert all(tme.error_relative(a, s) < 1e-4 for a, s in zip(tme.covarianze_expected(me), sig))
    assert np.isfinite(tme.campiona(me, rng)).all()


def test_T21_tme_keep_spectrum_pca(tensor):
    """TME preserves the covariance between neurons: the expected PCA spectrum of the surrogates is that of the data."""
    sig, M = tme.marginals(tensor)
    me = tme.fit(sig, M)
    a = np.linalg.eigvalsh(tme.covarianze_expected(me)[1])
    d = np.linalg.eigvalsh(sig[1])
    assert np.allclose(a, d, rtol=1e-6, atol=1e-8 * d.max())


# --------------------------------------------------------------------------------------------
# R9
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["TwoNN", "MLE10", "MLE20", "CorrInt", "DANCo"])
def test_T22_estimators_su_5_dims(name):
    """Gaussian with 5 dimensions linearly embedded in 25 (2000 points, no noise): estimate between 4 and 6.5."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((2000, 5)) @ rng.standard_normal((5, 25))
    assert 4 <= literature2.estimate_id(X, name) <= 6.5


def test_T23_mle_depends_from_k():
    """MLE with k = 10 and k = 20 gives different numbers (skdim takes k from fit(n_neighbors=k), not from the constructor)."""
    rng = np.random.default_rng(9)
    X = smooth(rng, (1500, 30), 2)
    assert literature2.estimate_id(X, "MLE10") != literature2.estimate_id(X, "MLE20")


def test_T24_criteria_r9():
    """Draws, values of n and criteria of stability, distinguishability and Altan prediction on hand-made examples."""
    assert literature2.values_n(5495) == [500, 1000, 2000, 5000, 5495]
    assert literature2.values_n(800) == [500, 800]
    i = literature2.draw(5495, 500, 3)
    assert len(np.unique(i)) == 500 and np.array_equal(i, literature2.draw(5495, 500, 3))
    assert np.array_equal(literature2.draw(800, 800, 0), np.arange(800))
    flat = {1000: np.full(10, 10.0) + np.linspace(-0.5, 0.5, 10), 2000: np.full(10, 10.5), 5495: np.array([11.0])}
    assert literature2.stability(flat, 1000)["stable"]
    sale = {1000: np.full(10, 10.0), 2000: np.full(10, 12.0), 5495: np.array([14.0])}
    assert not literature2.stability(sale, 1000)["stable"]
    wide = {1000: np.linspace(5, 15, 10), 5495: np.array([10.0])}
    assert not literature2.stability(wide, 1000)["stable"]
    assert literature2.distinguishable(20, [10, 12, 13]) and not literature2.distinguishable(11, [10, 12, 13])
    assert literature2.prediction_altan(False, False, True, True)
    assert not literature2.prediction_altan(False, True, True, True)
    assert literature2.p_permutation(5.0, [1, 2, 5, 6]) == 3 / 5


def test_T25_estimates_job(tmp_path):
    """The job of the parallel processes uses the same draw as draw() and the same estimators as estimate_id()."""
    rng = np.random.default_rng(10)
    X = rng.standard_normal((900, 4)) @ rng.standard_normal((4, 20))
    np.save(tmp_path / "x.npy", X)
    out = literature2.estimates_job(tmp_path / "x.npy", 500, 2, ("TwoNN", "MLE10"))
    assert [o[0] for o in out] == ["TwoNN", "MLE10"]
    assert out[0][1] == literature2.estimate_id(X[literature2.draw(900, 500, 2)], "TwoNN")


# --------------------------------------------------------------------------------------------
# R9, post-hoc check with the Theiler window
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("D", [30, 10])
def test_T26_theiler_zero_equal_a_skdim(D):
    """With window 0 the estimates coincide with skdim (estimate_id), even when computing the distances in blocks."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((700, 6)) @ rng.standard_normal((6, D)) + 0.2 * rng.standard_normal((700, D))
    th = literature2.estimates_theiler(X, np.zeros(700), np.arange(700), (0,), block=256)
    for s in literature2.ESTIMATORS_THEILER:
        assert th[(0, s)] == pytest.approx(literature2.estimate_id(X, s), rel=1e-6)


def test_T27_theiler_removes_the_chain_of_time():
    """40 trials × 100 bins of smooth 8-dimensional noise: without a window MLE with k = 10 sees a curve (< 4); with
    a window of 10 bins (beyond the smoothing scale) it rises by more than 5."""
    rng = np.random.default_rng(12)
    Z = gaussian_filter1d(rng.standard_normal((4000, 8)), 5, axis=0) @ rng.standard_normal((8, 30))
    g, t = np.repeat(np.arange(40), 100), np.tile(np.arange(100), 40)
    th = literature2.estimates_theiler(Z, g, t, (0, 10))
    assert th[(0, "MLE10")] < 4 and th[(10, "MLE10")] - th[(0, "MLE10")] > 5


def test_T28_theiler_job(tmp_path):
    """The job of the parallel processes uses the R9 draw and returns one estimate per window and estimator."""

    rng = np.random.default_rng(13)
    X = rng.standard_normal((900, 20))
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "m.npy", np.stack([np.repeat(np.arange(9), 100), np.tile(np.arange(100), 9)]))
    out = literature2.estimates_job_theiler(tmp_path / "x.npy", tmp_path / "m.npy", 500, 3, (0, 5))
    assert len(out) == 2 * len(literature2.ESTIMATORS_THEILER)
    v0 = {s: v for w, s, v, _ in out if w == 0}
    assert v0["MLE10"] == pytest.approx(literature2.estimate_id(X[literature2.draw(900, 500, 3)], "MLE10"), rel=1e-6)
