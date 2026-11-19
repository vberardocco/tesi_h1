"""Test of src/thesis_h1/literature.py on synthetic data (docs/literature/prompt.md, Steps 1 and 2).

T1-T4, T9-T12 (Step 1) and T5-T6 (Step 2) with the expected results written in the prompt; the "extra" tests check
the pieces on hand-made examples. Usage: PYTHONPATH=~/thesis_h1_git/src pytest tests/test_literature.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))
import synthetic_literature as ss                                          # noqa: E402
from thesis_h1 import literature                                             # noqa: E402

DURATION = 780                                                           # "trial duration" for the data without trials (15.6 s)


def data_fattoriali(seed, N=90, T=6000, K=5, share=0.30):
    """K smooth latents (Gaussian of 5 bins, variance 1) + white private noise; population shared fraction
    trace(W Wᵀ) / trace(W Wᵀ + Ψ) = share."""
    rng = np.random.default_rng(seed)
    L = gaussian_filter1d(rng.normal(size=(T, K)), 5, axis=0)
    L = (L - L.mean(0)) / L.std(0)
    W = rng.normal(size=(N, K))
    psi = rng.uniform(0.5, 1.5, N)
    W *= np.sqrt(share / (1 - share) * psi.sum() / (W ** 2).sum())
    return L @ W.T + rng.normal(size=(T, N)) * np.sqrt(psi)


def poisson_independent(seed, N=160, T=12000):
    rng = np.random.default_rng(seed)
    return rng.poisson(rng.uniform(1, 30, N)[None, :] * literature.BIN_S, size=(T, N))


@pytest.fixture(scope="module")
def t2():
    X = data_fattoriali(0)
    Xz = (X - X.mean(0)) / X.std(0)
    return X, Xz


# --------------------------------------------------------------------------------------------
# Step 1
# --------------------------------------------------------------------------------------------
def test_T1_noise_white():
    X = np.random.default_rng(0).normal(size=(12000, 66))
    v = 100 * literature.share_first(literature.eigenvalues(X, zscore=True), 12)
    print(f"T1: variance of the first 12 PCs {v:.2f}%")
    assert 18 <= v <= 22, v


def test_T2_fa(t2):
    X, Xz = t2
    grid = list(range(2, 21, 2))                                    # the grid of 1c
    cv = literature.fa_cv(Xz, grid, n_jobs=-1)
    K = grid[int(np.argmax(cv))]
    assert 4 <= K <= 6, (K, cv)
    end = list(range(1, 11))
    K1 = end[int(np.argmax(literature.fa_cv(Xz, end, n_jobs=-1)))]
    assert 4 <= K1 <= 6, K1
    d = literature.dim_shared(literature.fa_fit(Xz, K))
    assert d in (4, 5), d
    q = literature.share_population(literature.fa_fit(X, K))                  # counts not z-scored
    print(f"T2: K* {K} (grid 2-20), {K1} (grid 1-10); dimensions for 95% {d}; population share {q:.3f}")
    assert abs(q - 0.30) <= 0.03, q


def test_T3_ten_latents():
    rng = np.random.default_rng(0)
    N, T = 96, 10000
    W = np.linalg.qr(rng.normal(size=(N, 10)))[0]                      # 10 orthonormal directions, same variance
    a = 0.9 / 0.1 * N / 10                                             # 10 a / (10 a + N) = 90%
    X = rng.normal(size=(T, 10)) * np.sqrt(a) @ W.T + rng.normal(size=(T, N))
    n = literature.n_per(literature.eigenvalues(X), 0.80)
    print(f"T3: components for 80% {n}")
    assert n <= 10


def test_T4_ribinning_keep_the_spike():
    rng = np.random.default_rng(0)
    st = [np.sort(rng.uniform(0, 60, rng.integers(10, 2000))) for _ in range(40)]
    counts, edges, complete = literature.ribinna(st, 0.0, 60.0, 0.045)
    assert counts.sum() == sum(len(s) for s in st)
    assert np.allclose(np.diff(edges), 0.045) and edges[0] == 0 and edges[-1] >= 60
    assert complete[:-1].all() and not complete[-1]                    # 60 / 0.045 is not an integer: the last one is partial


def test_extra_mask_ribinnata():
    m20 = np.zeros(100, bool)
    m20[10:50] = True                                                  # 0.2-1.0 s after t0
    edges = -0.02 + np.arange(46) * 0.045
    m = literature.mask_ribinnata(m20, -0.02, edges)
    centers = edges[:-1] + 0.0225
    assert np.array_equal(m, (centers - -0.02 >= 0.2) & (centers - -0.02 < 1.0))


def test_T9_poisson_independent_null_approx_zero():
    counts = poisson_independent(1)
    m = np.ones(len(counts), bool)
    a = literature.draws_con_null(literature.pipeline_gallego(counts), m, 66, range(20), DURATION)
    b = literature.draws_con_null(literature.pipeline_silvernagel(counts), m, 96, range(20), DURATION)
    da = np.median([r["var12_diff"] for r in a])
    db = np.median([r["n80_diff"] for r in b])
    print(f"T9: median data - null: 1a {da:+.2f} points, 1b {db:+.1f} components")
    assert abs(da) <= 1, da
    assert abs(db) <= 1, db


def test_T10_task_pure():
    sess = ss.sessions_1g(task=True)
    recs = [literature.frankenstein_1g(sess, s, n_surr=10) for s in range(6)]
    med = {k: np.median([r[k] for r in recs]) for k in recs[0]}
    assert med["S1_share"] >= 0.9, med["S1_share"]
    assert abs(med["frank"] - med["S1_data"]) <= 3, (med["frank"], med["S1_data"])
    d = sess["S1"]
    phase = np.median(literature.check_even_odd(d.Y, d.segs, "phase"))
    start = np.median(literature.check_even_odd(d.Y, d.segs, "start"))
    print(f"T10: share {med['S1_share']:.3f}; data {med['S1_data']:.2f}, Frankenstein {med['frank']:.2f}, "
          f"null {med['S1_null']:.2f}; phase check {phase:.3f}, start {start:.3f}")
    assert phase > 3 * max(start, 0) and phase - start > 0.05, (phase, start)
    assert med["S1_cases_phase"] > 0                                     # "Presentation>end" appears in a single trial


def test_T11_latents_out_from_task():
    sess = ss.sessions_1g(task=False)
    recs = [literature.frankenstein_1g(sess, s, n_surr=10) for s in range(6)]
    med = {k: np.median([r[k] for r in recs]) for k in recs[0]}
    print(f"T11: share {med['S1_share']:.3f}; Frankenstein {med['frank']:.2f}, its null {med['frank_null']:.2f}; "
          f"data {med['S1_data']:.2f}, null {med['S1_null']:.2f}")
    assert med["S1_share"] <= 0.10, med["S1_share"]
    assert abs(med["frank"] - med["frank_null"]) <= 1, (med["frank"], med["frank_null"])
    assert med["S1_data"] - med["S1_null"] > 3                       # the structure is there, but it does not depend on the task


def test_T12_cheater_su_channels_independent():
    counts = poisson_independent(2)
    out = literature.cheater(literature.pipeline_gallego(counts), np.ones(len(counts), bool), DURATION)
    print(f"T12: group A data {out['A_data']:.2f}, null {out['A_null']:.2f}; median of the random ones {out['random_data_median']:.2f}")
    assert abs(out["A_data"] - out["A_null"]) <= 1, out
    assert out["A_data"] >= out["random_data_median"] + 5, out


# --------------------------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------------------------
def test_T5_parallel_analysis_noise_smoothed():
    k_shift, k_shuffle = [], []
    for r in range(20):
        rng = np.random.default_rng(r)
        Y = literature.smooth(rng.normal(size=(12000, 160)), 50)
        m = np.ones(len(Y), bool)
        allowed, _ = literature.shifts_allowed(len(Y), DURATION)
        ev, sur = literature.spectra(Y, m, True, rng, allowed, 200, "shift")
        k_shift.append(literature.parallel_analysis(ev, sur)[0])
        ev, sur = literature.spectra(Y, m, True, rng, allowed, 200, "shuffle")
        k_shuffle.append(literature.parallel_analysis(ev, sur)[0])
    print(f"T5: k_PA with circular shift {k_shift}; with shuffle {k_shuffle}")
    assert sum(k == 0 for k in k_shift) >= 19, k_shift
    assert all(k > 0 for k in k_shuffle), k_shuffle


def test_T6_parallel_analysis_five_latents(t2):
    _, Xz = t2
    rng = np.random.default_rng(0)
    allowed, _ = literature.shifts_allowed(len(Xz), DURATION)
    ev, sur = literature.spectra(Xz, np.ones(len(Xz), bool), True, rng, allowed, 200, "shift")
    k = literature.parallel_analysis(ev, sur)[0]
    print(f"T6: k_PA {k}")
    assert k == 5


# --------------------------------------------------------------------------------------------
# Step 3
# --------------------------------------------------------------------------------------------
def test_T7_labels_only_from_time():
    clock, lat, phase, trial = ss.session_clock(0, informative=False)
    out = literature.clock_neurons(clock, {k: lat for k in range(5)}, phase, trial, n_perm=200)
    print(f"T7: (a) {out['acc_a']:.1f}, (b) {out['acc_b']:.1f}, (c) {out['acc_c']:.1f}, (c) - (a) {out['diff_c_a']:+.2f}, "
          f"p {out['p']:.3f}; weights {[out[f'weight_block{k}'] for k in range(5)]}")
    assert abs(out["diff_c_a"]) <= 2, out["diff_c_a"]
    assert out["p"] > 0.05, out["p"]


def test_extra_T7_con_labels_noisy():
    """Like T7, but with 30% of the labels drawn at random: the clock does not reach 100% and the latents, which
    carry only the time, must not add anything."""
    clock, lat, phase, trial = ss.session_clock(0, informative=False)
    rng = np.random.default_rng(10)
    phase = phase.copy()
    via = rng.random(len(phase)) < 0.3
    phase[via] = rng.choice(literature.PHASES_MOVEMENT, via.sum())
    out = literature.clock_neurons(clock, {k: lat for k in range(5)}, phase, trial, n_perm=200)
    print(f"T7 with noisy labels: (a) {out['acc_a']:.1f}, (c) {out['acc_c']:.1f}, p {out['p']:.3f}")
    assert abs(out["diff_c_a"]) <= 2 and out["p"] > 0.05, (out["diff_c_a"], out["p"])


def test_T8_information_in_more_in_latents():
    clock, lat, phase, trial = ss.session_clock(0, informative=True)
    out = literature.clock_neurons(clock, {k: lat for k in range(5)}, phase, trial, n_perm=200)
    print(f"T8: (a) {out['acc_a']:.1f}, (b) {out['acc_b']:.1f}, (c) {out['acc_c']:.1f}, (c) - (a) {out['diff_c_a']:+.2f}, "
          f"p {out['p']:.3f}; weights {[out[f'weight_block{k}'] for k in range(5)]}")
    assert out["diff_c_a"] > 0 and out["p"] < 0.05, (out["diff_c_a"], out["p"])


def test_extra_latents_permutati():
    trial = np.array([1] * 4 + [2] * 3 + [3] * 5)
    clock = np.concatenate([np.arange(4), np.arange(3), np.arange(5)]) * literature.BIN_S
    lat = np.arange(12, dtype=float)[:, None]
    out = literature.latents_permutati(lat, trial, clock, {1: 2, 2: 3, 3: 1})
    assert out[:, 0].tolist() == [4, 5, 6, 6, 7, 8, 9, 0, 1, 2, 3, 3]    # same instant; the short trial repeats the last bin


def test_extra_points_e_combine():
    phase = np.array(["Presentation"] * 7 + ["Reach"] * 13, object)
    assert literature.points_phase(phase).tolist() == [10, 15]
    clock, lat = np.arange(20.0), np.random.default_rng(0).normal(size=(20, 25))
    X = literature.combine(clock, lat, np.arange(20), 2.0)
    assert X.shape == (20, 26) and np.isclose(X[:, 0].std(), 2) and np.isclose((X[:, 1:] ** 2).sum(1).mean(), 1, atol=0.05)
    assert literature.combine(clock, lat, np.arange(20), np.inf).shape == (20, 1)


# --------------------------------------------------------------------------------------------
# Step 5a
# --------------------------------------------------------------------------------------------
def test_5a_cca():
    rng = np.random.default_rng(0)
    A = gaussian_filter1d(rng.normal(size=(2000, 10)), 3, axis=0)
    R = np.linalg.qr(rng.normal(size=(10, 10)))[0]
    B = A @ R + 0.05 * rng.normal(size=A.shape)                        # same rotated dynamics: aligned ≈ 1
    cc = literature.correlations_canonical(A, B)
    assert cc.min() > 0.95 and literature.correlations_non_aligned(A, B).mean() < 0.7
    C = gaussian_filter1d(rng.normal(size=(2000, 10)), 3, axis=0)     # independent dynamics
    assert literature.correlations_canonical(A, C).mean() < 0.3
    assert np.allclose(literature.correlations_canonical(A, -A), 1) and np.allclose(literature.correlations_non_aligned(A, -A), 1)


def test_5a_windows_e_blocks():
    import pandas as pd
    ep = pd.DataFrame({"start_time": [0.1, 0.5, 1.0, 1.3], "tag": ["Presentation3", "Reach", "Grasp", "Reach"]})
    w = literature.windows_phase(ep, lambda t: 1 if t < 1 else 2, 0.0, 0.03, 10, 50)
    assert [(x["cond"], x["start"]) for x in w] == [(("Reach", "odd"), 16), (("Grasp", "even"), 33)]
    L = np.arange(50.0)[:, None]                                       # the 1.3 s Reach window falls outside: discarded
    out = literature.block_conditions(L, w, {("Grasp", "even"): [0], ("Reach", "odd"): [0]})
    assert out[:, 0].tolist() == list(range(33, 43)) + list(range(16, 26))
    w2 = w + [dict(cond=("Reach", "odd"), trial=3, start=20, stop=30)]
    m = literature.means_conditions(L, w2, {("Reach", "odd"): [0, 1], ("Grasp", "even"): [0]})
    assert m[:, 0].tolist() == list(np.arange(18, 28)) + list(range(33, 43))  # mean of 16-25 and 20-29


# --------------------------------------------------------------------------------------------
# Step 5d
# --------------------------------------------------------------------------------------------
def test_5d_choose_d():
    grid = [2, 5, 10, 15]
    ll = np.array([[1.0, 1.0, 1.0], [2.0, 2.1, 1.9], [2.2, 2.0, 2.3], [2.1, 1.9, 2.0]])
    assert literature.choose_d(ll, grid) == (10, 5)                     # 10 does not beat 5 in the second block


def _cv_gpfa_synthetic(X, grid=(1, 2, 3, 5, 8)):
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    fin = [dict(start=i * 50, stop=(i + 1) * 50) for i in range(60)]
    folds = list(KFold(n_splits=5, shuffle=False).split(np.arange(60)))

    def one(d, tr, te):
        p = literature.gpfa_fit(literature.sequences_gpfa(X, [fin[i] for i in tr]), d, iters=200)
        return literature.gpfa_ll_per_bin(p, literature.sequences_gpfa(X, [fin[i] for i in te]))
    ll = np.array(Parallel(n_jobs=-1)(delayed(one)(d, tr, te) for d in grid for tr, te in folds))
    return literature.choose_d(ll.reshape(len(grid), 5), list(grid))


def _three_latents(seed):
    rng = np.random.default_rng(seed)
    lat = gaussian_filter1d(rng.normal(size=(3000, 3)), 6, axis=0)
    return rng, lat / lat.std(0), rng.normal(size=(3, 30))


def test_5d_gpfa_three_latents():
    """The GPFA model: square root of the counts = d + C x + Gaussian noise (30 channels, 3 smooth latents, 60 windows
    of 50 bins). Cross-validation must choose 3. (The first version of the test used Poisson counts with
    an exponential and expected 2-4: it failed with 8 because the model is linear; see the next test.)"""
    rng, lat, C = _three_latents(0)
    Y = 3 + 0.5 * lat @ C + 0.5 * rng.normal(size=(3000, 30))
    best, d = _cv_gpfa_synthetic(np.clip(Y, 0, None) ** 2)
    print(f"5d, GPFA on linear-Gaussian data with 3 latents: best d {best}, chosen {d}")
    assert best == 3 and d == 3, (best, d)


def test_5d_gpfa_overestimate_con_data_not_linear():
    """With Poisson counts and an exponential of the 3 latents, the GPFA (linear on the square root) chooses more than 3
    factors: the d chosen by cross-validation is a ceiling, not the dimension of the latents."""
    rng, lat, C = _three_latents(0)
    best, d = _cv_gpfa_synthetic(rng.poisson(0.3 * np.exp(0.6 * lat @ C)))
    print(f"5d, GPFA on Poisson-exponential with 3 latents: best d {best}, chosen {d}")
    assert d > 3, (best, d)


# --------------------------------------------------------------------------------------------
# Extra: pieces on hand-made examples
# --------------------------------------------------------------------------------------------
def test_extra_shifts_allowed():
    s, frac = literature.shifts_allowed(12000, 780)
    assert s.min() >= 100 and s.max() <= 11900
    multiple = np.arange(1, 17) * 780
    assert np.abs(s[:, None] - multiple[None, :]).min() > 100
    assert 0.2 < frac < 0.3                                            # about 201 / 780


def test_extra_circular_shift_keep_every_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 10))
    allowed, _ = literature.shifts_allowed(5000, 780)
    Xs = literature.circular_shift(X, rng, allowed)
    assert np.allclose(np.sort(X, 0), np.sort(Xs, 0))
    for c in range(10):                                                # every column is the same, rotated
        k = np.flatnonzero(np.isclose(Xs[:, c], X[0, c]))[0]
        assert np.allclose(np.roll(X[:, c], k), Xs[:, c]) and k in allowed


def test_extra_n_per_e_share():
    ev = np.array([4.0, 3, 2, 1])
    assert literature.n_per(ev, 0.4) == 1 and literature.n_per(ev, 0.7) == 2 and literature.n_per(ev, 0.71) == 3
    assert literature.share_first(ev, 2) == 0.7


def test_extra_parallel_analysis_sequential():
    null = np.ones((100, 4)) + np.linspace(0, 0.1, 100)[:, None]
    k_seq, k_all, _ = literature.parallel_analysis(np.array([5.0, 0.5, 5.0, 0.5]), null)
    assert (k_seq, k_all) == (1, 2)


def test_extra_segments_e_keys():
    f = ["", "Intertrial", "Presentation", "Presentation", "Reach", "Presentation", "Orient", "Release",
         "Release", "Presentation", "Reach", "Reach", "Presentation", "Intertrial"]
    t = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    segs = literature.segments(np.array(f, object), np.array(t), ~np.isin(f, literature.ESCLUSE))
    assert [g.key for g in segs] == [("Presentation>Reach", 1), ("Reach", 1), ("Presentation>Orient", 1),
                                        ("Orient", 1), ("Release", 1),
                                        ("Presentation>Reach", 1), ("Reach", 1), ("Presentation>end", 1)]
    assert segs[5].start == 9                                         # the initial Release of trial 2 is discarded
    assert list(literature.bin_of_segments(segs)) == list(range(2, 8)) + list(range(9, 13))


def test_extra_ricampionamento():
    M = literature.matrix_ricampionamento(5, 5)
    assert np.allclose(M, np.eye(5))
    x = np.linspace(0, 1, 7)
    assert np.allclose(literature.matrix_ricampionamento(7, 13) @ x, np.linspace(0, 1, 13), atol=0.05)
    assert np.allclose(literature.matrix_ricampionamento(1, 4) @ np.array([3.0]), 3.0)
    assert np.allclose(literature.matrix_ricampionamento(9, 3).sum(1), 1)


def test_extra_ripieghi_of_shuffle_e_frankenstein():
    f = np.array(["Reach"] * 3 + ["Grasp"] * 2 + ["Reach"] * 4 + ["Carry"] * 2, object)
    t = np.array([1] * 5 + [2] * 6)
    Y = np.arange(len(f), dtype=float)[:, None] * np.ones((1, 3))
    segs = literature.segments(f, t, np.ones(len(f), bool))
    mix, cases = literature.shuffle_trial(Y, segs, np.random.default_rng(0))
    assert cases == {"key": 6, "phase": 0, "same": 6}               # Grasp and Carry are in a single trial only
    assert np.allclose(mix[3:5], Y[3:5]) and np.allclose(mix[9:11], Y[9:11])
    other = {"A": segs, "B": literature.segments(np.array(["Reach"] * 3 + ["Shape"] * 3, object), np.ones(6, int),
                                             np.ones(6, bool))}
    Ys = {"A": Y, "B": 100 + np.arange(6.0)[:, None] * np.ones((1, 2))}
    F, cases = literature.frankenstein(Ys, other, {"A": [0], "B": [1]}, "A", np.random.default_rng(0))
    assert cases == {"key": 6, "phase": 0, "mean": 2}                # in B there are neither Grasp nor Carry

    assert F.shape == (11, 2) and np.allclose(F.mean(0), 0)


def test_extra_ledermann():
    assert literature.ledermann(20, 14) and not literature.ledermann(20, 15) and literature.ledermann(40, 20)


def test_extra_saturation():
    N = np.arange(20, 161, 20.0)
    assert literature.models_saturation(N, 50 - 45 * np.exp(-N / 30) + 0.05 * np.sin(N))["saturates"]
    assert not literature.models_saturation(N, 3 + 0.4 * N + 0.3 * np.cos(N))["saturates"]
    assert not literature.models_saturation(N, 2 * N ** 0.8 * (1 + 0.01 * np.sin(N)))["saturates"]
