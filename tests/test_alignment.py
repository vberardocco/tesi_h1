"""Tests of moment matching, of the ridge towards the held-in weights, of the metric and of the adaptation without leakage
(src/thesis_h1/crosssession.py).

Uso: python tests/test_alignment.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="thesis_xs_"))
os.environ["THESIS_XS"] = str(TMP)                                # model cache in a temporary folder
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
from thesis_h1 import crosssession as xs  # noqa: E402
from thesis_h1.decoding import r2_vw  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'ERROR'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel_err(Z, ref):
    return float(np.linalg.norm(Z - ref) / np.linalg.norm(ref - ref.mean(0)))


# --- synthetic data: seed 0, d = 8, z_src of 5000 samples from an anisotropic Gaussian -----------------
rng = np.random.default_rng(0)
d, n = 8, 5000
A = rng.normal(size=(d, d))
C_true = A @ A.T + 0.5 * np.eye(d)
mu_true = rng.normal(size=d)
L = np.linalg.cholesky(C_true)
z_src = mu_true + rng.normal(size=(n, d)) @ L.T
mu_s, C_s = xs.moments(z_src)

# 1. shift + gain: mean and covariance brought onto those of z_src
gain = rng.uniform(0.5, 2.0, size=d)
shift = rng.normal(scale=3.0, size=d)
z_t = z_src * gain + shift
f = xs.MomentMatch(z_t, mu_s, C_s)
mu_f, C_f = xs.moments(f(z_t))
dm, dC = float(np.abs(mu_f - mu_s).max()), float(np.abs(C_f - C_s).max())
check("1. shift + gain: |Δmean| < 1e-6", dm < 1e-6, f"{dm:.2e}")
check("1. shift + gain: |Δcovariance| < 1e-6", dC < 1e-6, f"{dC:.2e}")
print(f"     (relative pointwise error after shift + gain: {rel_err(f(z_t), z_src):.3f}; "
      "the non-isotropic gain is not exactly inverted by the symmetric roots)")

# 2. same session: another sample of the same distribution -> f almost the identity
z_new = mu_true + rng.normal(size=(n, d)) @ L.T
f2 = xs.MomentMatch(z_new, mu_s, C_s)
e2 = rel_err(f2(z_new), z_new)
check("2. same session: relative pointwise error ~0.03", 0.01 < e2 < 0.06, f"{e2:.4f}")

# 3. pure rotation: the covariance matches, the points do not (known limitation: moment matching is blind to rotations)
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
z_rot = (z_src - mu_s) @ Q + mu_s
f3 = xs.MomentMatch(z_rot, mu_s, C_s)
_, C3 = xs.moments(f3(z_rot))
dC3, e3 = float(np.abs(C3 - C_s).max()), rel_err(f3(z_rot), z_src)
check("3. rotation: |Δcovariance| < 1e-6", dC3 < 1e-6, f"{dC3:.2e}")
check("3. rotation: high relative pointwise error (> 0.3)", e3 > 0.3, f"{e3:.3f} (without alignment: {rel_err(z_rot, z_src):.3f})")

# 4. ridge towards the held-in weights: λ -> ∞ returns the held-in weights, λ -> 0 least squares
X = rng.normal(size=(400, 30))
Y = X @ rng.normal(size=(30, 7)) + rng.normal(size=(400, 7))
B0 = rng.normal(size=(31, 7))
sol = xs.ridge_solve(X, Y, [1e-10, 1e14], B0, intercept="prior")
W_inf, b_inf = sol[1e14]
check("4. λ -> ∞: held-in weights and intercept", np.allclose(W_inf, B0[:-1], atol=1e-6) and np.allclose(b_inf, B0[-1], atol=1e-6))
Xa = np.hstack([X, np.ones((400, 1))])
ols = np.linalg.lstsq(Xa, Y, rcond=None)[0]
W0, b0 = sol[1e-10]
check("4. λ -> 0: least squares", np.allclose(W0, ols[:-1], atol=1e-6) and np.allclose(b0, ols[-1], atol=1e-6))
Xw = rng.normal(size=(50, 300))                                   # p > n: dual form
Yw = rng.normal(size=(50, 7))
B0w = rng.normal(size=(301, 7))
lam = 3.0
Xwa = np.hstack([Xw, np.ones((50, 1))])
direct = B0w + np.linalg.solve(Xwa.T @ Xwa + lam * np.eye(301), Xwa.T @ (Yw - Xwa @ B0w))
Wd, bd = xs.ridge_solve(Xw, Yw, [lam], B0w, intercept="prior")[lam]
check("4. dual form = direct solution", np.allclose(np.vstack([Wd, bd]), direct, atol=1e-8))
Wf, bf = xs.ridge_solve(Xw, Yw, [lam], intercept="free")[lam]
mx, my = Xw.mean(0), Yw.mean(0)
direct_f = np.linalg.solve((Xw - mx).T @ (Xw - mx) + lam * np.eye(300), (Xw - mx).T @ (Yw - my))
check("4. free intercept, dual form = direct", np.allclose(Wf, direct_f, atol=1e-8) and np.allclose(bf, my - mx @ direct_f))

# 5. the metric is that of the FALCON evaluator: sklearn r2_score(multioutput="variance_weighted")
from sklearn.metrics import r2_score  # noqa: E402
yt = rng.normal(size=(800, 7)) * rng.uniform(0.1, 3, size=7)
yp = yt + rng.normal(size=yt.shape)
check("5. r2_vw = sklearn variance_weighted", abs(r2_vw(yt, yp) - r2_score(yt, yp, multioutput="variance_weighted")) < 1e-12)
from falcon_challenge.evaluator import FalconEvaluator  # noqa: E402
m = rng.random(800) > 0.1
r_falcon = FalconEvaluator.compute_metrics_regression(yp, yt[m], m, {"S9": [800]})["R2 Mean"]
check("5. = compute_metrics_regression of falcon_challenge (one file, eval_mask)",
      abs(r_falcon - r2_vw(yt[m], yp[m])) < 1e-12, f"{r_falcon:.6f} vs {r2_vw(yt[m], yp[m]):.6f}")

# 6. no leakage: the adaptation does not see the test trial; changing the test trial changes nothing in the fold
T, C = 700, 40
def fake_trial(seed):
    r = np.random.default_rng(seed)
    lat = np.cumsum(r.normal(size=(T, 3)), 0) / 10
    rate = np.exp(0.5 + lat @ r.normal(size=(3, C)) * 0.3)
    return xs.Trial(seed, r.poisson(rate * 0.02 * 20).astype(np.int16), np.gradient(lat, axis=0) @ r.normal(size=(3, 7)),
                    r.random(T) > 0.05, r.random(T) > 0.3, [(a, a + 50) for a in range(0, T - 50, 90)])
train = [fake_trial(s) for s in range(12)]
sess_like = [xs.Sess(f"S{d_}_set_1", f"S{d_}", "x", "held_in", train[4 * d_:4 * d_ + 4]) for d_ in range(3)]
b = xs.fit_bundle("PCA", 0, sess_like, "trial", use_cache=False)
tr = [fake_trial(100 + i) for i in range(3)]
s1 = xs.Sess("S9_set_1", "S9", "x", "held_out", tr)
alt = fake_trial(999)                                             # different test trial: neural data and kinematics
s2 = xs.Sess("S9_set_1", "S9", "x", "held_out", [tr[0], tr[1], alt])
p1 = xs.eval_session(b, s1, [10.0], [10.0])
p2 = xs.eval_session(b, s2, [10.0], [10.0])
A1 = xs.adapt(b, [tr[0], tr[1]], [10.0], [10.0])
same = all(np.array_equal(xs.predict_trial(b, A1, tr[2])[k], p1[k][2]) for k in p1)
check("6. eval_session = adapt(2 trials) + predict_trial(test)", same)
pa = xs.predict_trial(b, A1, alt)
check("6. fold 2 with another test trial = same adaptation applied to the new trial",
      all(np.allclose(pa[k], p2[k][2]) for k in p2))
tr_v = xs.Trial(tr[2].num, tr[2].counts, tr[2].vel * 5 + 1, tr[2].mask, tr[2].mov, tr[2].windows)
p3 = xs.eval_session(b, xs.Sess("S9_set_1", "S9", "x", "held_out", [tr[0], tr[1], tr_v]), [10.0], [10.0])
check("6. the kinematics of the test trial does not enter its predictions", all(np.array_equal(p1[k][2], p3[k][2]) for k in p1))
check("6. ...but it enters the other folds (where it is an adaptation trial)",
      not np.array_equal(p1[("fewshot", 10.0)][0], p3[("fewshot", 10.0)][0]))
check("6. expected conditions for PCA", sorted({k[0] for k in p1}) == sorted(xs.expected_conditions("PCA")))

# 7. fewshot_gain: gain and offset per DoF with least squares
yh7 = rng.normal(size=(500, 7))
g_true, o_true = rng.uniform(0.5, 2, 7), rng.normal(size=7)
g7, o7 = xs.fit_gain(yh7, yh7 * g_true + o_true)
check("7. fit_gain recovers gain and offset", np.allclose(g7, g_true) and np.allclose(o7, o_true))
A7 = xs.adapt(b, [tr[0], tr[1]], [10.0], [10.0])
pr7 = xs.predict_trial(b, A7, tr[2])
check("7. fewshot_gain = align × gain + offset", np.allclose(pr7[("fewshot_gain", None)],
                                                                 pr7[("align", None)] * A7.gain[0] + A7.gain[1]))

# 8. channel decoder: the fit of ChannelDecoder.fit = the recipe of 05 (RidgeHistory), with small H
from thesis_h1 import decoding as dec_mod  # noqa: E402
for t in train + tr + [alt, tr_v]:
    t.counts_all = t.counts
H8 = 3
cd = xs.ChannelDecoder.fit(sess_like, H=H8, alpha=10.0)
files = [np.concatenate([t.counts for t in s_.trials]).astype(np.float32) for s_ in sess_like]
N_CH0 = dec_mod.N_CH
dec_mod.N_CH = C                                                  # the recipe of 05 is written for 176 channels
rh = dec_mod.RidgeHistory(H8).fit_scaler(files)
rh.accumulate(files, [np.concatenate([t.vel for t in s_.trials]) for s_ in sess_like],
              [np.concatenate([t.mask for t in s_.trials]) for s_ in sess_like])
(W8, b8), = rh.solve(H8, [10.0]).values()
dec_mod.N_CH = N_CH0
check("8. ChannelDecoder.fit = RidgeHistory of 05", np.allclose(cd.W, W8, atol=1e-8) and np.allclose(cd.b, b8, atol=1e-8)
      and np.allclose(cd.mu, rh.mu) and np.allclose(cd.sd, rh.sd))
pc1 = xs.eval_channels(cd, s1, [10.0, 1e12])
pc3 = xs.eval_channels(cd, xs.Sess("S9_set_1", "S9", "x", "held_out", [tr[0], tr[1], tr_v]), [10.0, 1e12])
check("8. ridge_channels: the kinematics of the test trial does not enter its predictions",
      all(np.array_equal(pc1[k][2], pc3[k][2]) for k in pc1))
check("8. ridge_channels: λ -> ∞ = recenter", np.allclose(pc1[("fewshot", 1e12)][2], pc1[("recenter", None)][2], atol=1e-6))
check("8. expected conditions for ridge_channels",
      sorted({k[0] for k in pc1}) == sorted(xs.expected_conditions("ridge_channels")))

# 9. diagnosis within the session: only the trials of the blocks, three conditions
sess9 = xs.Sess("S1_set_1", "S1", "x", "held_in", [fake_trial(200 + i) for i in range(8)])
d9 = xs.diag_within(b, sess9)
check("9. diag_within: no_adapt, recenter, align on the 6 trials of the 2 blocks",
      sorted(k[0] for k in d9) == ["align", "no_adapt", "recenter"]
      and all([y is not None for y in v] == [True] * 6 + [False] * 2 for v in d9.values()))

print()
shutil.rmtree(TMP, ignore_errors=True)
if FAIL:
    print(f"{len(FAIL)} tests failed: {FAIL}")
    sys.exit(1)
print("all tests passed")

