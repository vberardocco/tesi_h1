"""Tests of the alignment index, the angles, the chance level and the tests of the stability of the FA subspaces
(src/thesis_h1/stability.py).

Usage: python tests/test_stability.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.linalg import subspace_angles  # noqa: E402

from thesis_h1 import stability as st  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'ERROR'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


rng = np.random.default_rng(0)
p, K = 170, 15
L = rng.normal(size=(p, K)) * np.linspace(3, 0.5, K)          # loadings with decreasing shared variance
A = st.Sub.from_loadings(L)

# 1. same subspace, rotation of the loadings and sign change: index 1, angle 0
R, _ = np.linalg.qr(rng.normal(size=(K, K)))
for name, L2 in [("same loadings", L), ("rotated loadings", L @ R), ("signs changed", L * np.sign(rng.normal(size=K)))]:
    B = st.Sub.from_loadings(L2)
    i15, a15 = st.align_index(A, B, 15), st.mean_angle(A.basis(15), B.basis(15))
    check(f"1. {name}: index k=15 = 1, angle = 0", abs(i15 - 1) < 1e-9 and a15 < 1e-3, f"{i15:.12f}, {a15:.2e}°")
B = st.Sub.from_loadings(L @ R)
i5 = st.align_index(A, B, 5)
check("1. rotated loadings: index k=5 = 1 (directions ordered by shared variance)", abs(i5 - 1) < 1e-9, f"{i5:.12f}")

# 2. orthogonal subspaces: index 0, angle 90
Q, _ = np.linalg.qr(rng.normal(size=(p, 2 * K)))
A0, B0 = st.Sub.from_loadings(Q[:, :K] * 2), st.Sub.from_loadings(Q[:, K:] * 3)
i0, a0 = st.align_index(A0, B0, 15), st.mean_angle(A0.basis(15), B0.basis(15))
check("2. orthogonal: index 0, angle 90°", i0 < 1e-12 and abs(a0 - 90) < 1e-6, f"{i0:.2e}, {a0:.6f}°")

# 3. bounds, symmetry and consistency with scipy on random subspaces
ok_b, ok_s, ok_a = True, True, True
for _ in range(50):
    X1, X2 = (st.Sub.from_loadings(rng.normal(size=(p, K)) * rng.uniform(0.2, 3, K)) for _ in range(2))
    for k in (15, 5):
        v = st.align_index(X1, X2, k)
        ok_b &= 0 <= v <= 1 + 1e-12
        ok_s &= abs(v - st.align_index(X2, X1, k)) < 1e-12
        ok_a &= abs(st.mean_angle(X1.basis(k), X2.basis(k))
                    - np.degrees(subspace_angles(X1.basis(k), X2.basis(k))).mean()) < 1e-6
check("3. index in [0, 1] (k = 15 and 5)", bool(ok_b))
check("3. symmetric index", bool(ok_s))
check("3. mean angle = scipy.linalg.subspace_angles", bool(ok_a))

# 4. definition: I(A→B) = tr(Q_Bᵀ L_A L_Aᵀ Q_B) / sum of the first k eigenvalues of L_A L_Aᵀ
L1, L2 = rng.normal(size=(p, K)), rng.normal(size=(p, K))
S1 = L1 @ L1.T
ev1 = np.sort(np.linalg.eigvalsh(S1))[::-1]
U2 = np.linalg.eigh(L2 @ L2.T)[1][:, ::-1]
for k in (15, 5):
    ref = np.trace(U2[:, :k].T @ S1 @ U2[:, :k]) / ev1[:k].sum()
    got = st.index_dir(st.Sub.from_loadings(L1), st.Sub.from_loadings(L2).basis(k), k)
    check(f"4. I(A→B) from the definition, k = {k}", abs(got - ref) < 1e-10, f"{got:.10f} vs {ref:.10f}")

# 5. sensitivity at 5 directions: B shares with A only the 5 strongest directions
Qa, _ = np.linalg.qr(rng.normal(size=(p, 25)))
La = Qa[:, :15] * np.r_[np.full(5, 5.0), np.full(10, 0.5)]
Lb = np.c_[Qa[:, :5] * 5.0, Qa[:, 15:25] * 0.5]
Aa, Bb = st.Sub.from_loadings(La), st.Sub.from_loadings(Lb)
i5, i15 = st.align_index(Aa, Bb, 5), st.align_index(Aa, Bb, 15)
check("5. first 5 directions in common: index k=5 ≈ 1, k=15 < 1", abs(i5 - 1) < 1e-9 and i15 < 0.99, f"k5 {i5:.4f}, k15 {i15:.4f}")

# 6. uniform chance: E[I(A→Q)] = k / p for uniform Q
M = st.sampler(np.eye(p))
Q1 = st.draw_bases(M, 15, 2000, np.random.default_rng(1))
check("6. random orthonormal bases", np.allclose(np.swapaxes(Q1, 1, 2) @ Q1, np.eye(15), atol=1e-10))
m = float(st.index_dir(A, Q1, 15).mean())
check("6. uniform chance: mean index ≈ k/p", abs(m - 15 / p) < 0.005, f"{m:.4f} vs {15 / p:.4f}")
Q2 = st.draw_bases(M, 15, 2000, np.random.default_rng(2))
ang_u = st.mean_angle(Q1, Q2)
check("6. uniform chance: mean angle close to 90° (k = 15 out of 170)", 70 < np.median(ang_u) < 90, f"{np.median(ang_u):.1f}°")

# 7. Elsayed-style chance: with C concentrated in the subspace of A the chance level is much more aligned than the uniform one
C = A.U @ np.diag(A.lam) @ A.U.T + 0.01 * np.eye(p)
Qe = st.draw_bases(st.sampler(C), 15, 500, np.random.default_rng(3))
me = float(st.index_dir(A, Qe, 15).mean())
check("7. chance from the concentrated covariance: high chance index", me > 0.8, f"{me:.3f}")

# 8. z-score of the block with minimum sd
X = rng.poisson(0.3, size=(2400, 6)).astype(float)
X[:, 0] = 0
X[:, 1] = 0
X[7, 1] = 1
Z, n0, nlow = st.zscore_block(X)
check("8. minimum sd: silent channel -> zeros, correct counts", np.all(Z[:, 0] == 0) and n0 == 1 and nlow == 2,
      f"sd=0: {n0}, sd<0.1: {nlow}")
check("8. channel with 1 spike: z ≤ 1/0.1", Z[:, 1].max() <= 10 + 1e-9, f"max z {Z[:, 1].max():.2f}")
check("8. normal channels: standard z-score", np.allclose(Z[:, 2:].std(0), 1))

# 9. critical values of Spearman (tables: n = 7 -> 0.786, n = 14 -> 0.538)
c7, c14 = st.critical_rho(7), st.critical_rho(14)
check("9. critical value n = 7 ≈ 0.79", abs(c7 - 0.786) < 0.01, f"{c7:.3f}")
check("9. critical value n = 14 ≈ 0.54", abs(c14 - 0.538) < 0.02, f"{c14:.3f}")

# 10. Spearman with permutation and partial
x = rng.normal(size=14)
y = x + 0.3 * rng.normal(size=14)
r, pv = st.perm_spearman(x, y, n_perm=2000)
check("10. Spearman: same ρ as scipy, small p with a strong relationship",
      abs(r - stats.spearmanr(x, y).statistic) < 1e-12 and pv < 0.01, f"ρ {r:.3f}, p {pv:.4f}")
r7, p7 = st.perm_spearman(np.arange(7), np.arange(7))
check("10. n = 7, exact permutation: p = 2/5040 for ρ = 1", abs(p7 - 2 / 5040) < 1e-12, f"p {p7:.6f}")
z = np.arange(14)
xz, yz = z + rng.normal(size=14), z + rng.normal(size=14)
rp, pp = st.partial_spearman(xz, yz, z, n_perm=2000)
check("10. partial: relationship only through z -> small partial ρ", abs(rp) < abs(stats.spearmanr(xz, yz).statistic),
      f"partial {rp:.3f} vs simple {stats.spearmanr(xz, yz).statistic:.3f}")

# 11. Mantel and slope: blocks over 6 days, index that decreases with the distance
dates = np.array([0, 3, 7, 12, 20, 30])
blk_day = np.repeat(np.arange(6), 4)
ii, jj = np.triu_indices(len(blk_day), 1)
da, db = blk_day[ii], blk_day[jj]
dist = np.abs(dates[da] - dates[db])
yy = 0.8 - 0.01 * dist + 0.02 * rng.normal(size=len(dist))
rho, pm = st.mantel(yy, da, db, dates, n_perm=2000)
check("11. Mantel: decrease with days detected", rho < -0.5 and pm < 0.05, f"ρ {rho:.3f}, p {pm:.4f}")
rho0, pm0 = st.mantel(0.02 * rng.normal(size=len(dist)), da, db, dates, n_perm=2000)
check("11. Mantel: no relationship -> p not small", pm0 > 0.05, f"ρ {rho0:.3f}, p {pm0:.3f}")
est, lo, hi, nb = st.slope_bootstrap(yy, da, db, dates, n_boot=500)
ols = np.polyfit(dist, yy, 1)[0]
check("11. slope = least squares on the pairs", abs(est - ols) < 1e-12, f"{est:.5f} vs {ols:.5f}")
check("11. bootstrap CI contains the true slope", lo < -0.01 < hi, f"[{lo:.4f}, {hi:.4f}], {nb} resamples")

print()
print("ALL OK" if not FAIL else f"{len(FAIL)} TESTS FAILED: {FAIL}")

sys.exit(1 if FAIL else 0)
